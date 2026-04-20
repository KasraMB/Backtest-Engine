"""
Multi-config data collection for the ICT/SMC ML pipeline.

Steps
-----
1. Sample N configs via Latin Hypercube Sampling from the param space.
2. Run a single backtest per config that simultaneously checks sensitivity
   (base metric) and captures trade data.  For Round 2+ only, also run
   perturbation backtests for non-LHS params to validate robustness.
3. Attach config features to each trade row.
4. Merge into data/ml_dataset.parquet (appends on Round 2+).

Parallelism
-----------
Every individual backtest is its own task in the pool.  The pool is created
once and shared across both the sensitivity and collection phases, so workers
are not re-initialised between phases.

In Round 1 (full LHS coverage) no perturbation params fall outside the
LHS axes, so only 150 base runs are needed.  All are submitted at once and
run in parallel across all available cores.

Rounds
------
Set ROUND = 1 for the initial broad exploration.
After reviewing Round 1 feature importance / partial-dependence, tighten
PARAM_RANGES_V2 in backtest/ml/configs.py and re-run with ROUND = 2.
Each round appends to the existing dataset — the model trains on all rounds.

Delete cache/sens_runs_cache.json + cache/sensitivity_cache.json to force a
full re-run, or delete data/ml_dataset.parquet to rebuild from scratch.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import time as dtime
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from backtest.ml.configs import (
    sample_configs, normalize_config, ROUND_RANGES,
    PHASE1_PARAMS, PHASE2_PARAMS, CONFIG_FEATURE_NAMES,
)
from backtest.ml.features import ALL_FEATURE_NAMES
from backtest.regime.vol_regime import compute_vol_regime_map

# ---------------------------------------------------------------------------
# Configuration — edit before each run
# ---------------------------------------------------------------------------
ROUND         = 1          # 1 = fresh broad LHS; 2+ = append tighter ranges
N_CONFIGS     = 150        # configs to sample per round
LHS_SEED      = 42         # reproducibility

SENSITIVITY_CFG = dict(
    perturbation_pct    = 0.15,
    max_degradation_pct = 30.0,
    min_base_metric     = 0.05,
    min_trades          = 10,
)

BASE_EXEC = dict(
    starting_capital         = 100_000,
    slippage_points          = 0.25,
    commission_per_contract  = 4.50,
    eod_exit_time            = dtime(17, 0),
    order_cancel_time        = dtime(11, 0),
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CACHE_1M          = ROOT / "data" / "NQ_1m.parquet"
CACHE_5M          = ROOT / "data" / "NQ_5m.parquet"
CACHE_BAR_MAP     = ROOT / "data" / "NQ_bar_map.npy"
SENS_RUNS_CACHE   = ROOT / "cache" / "sens_runs_cache.json"   # per-run results
SENSITIVITY_CACHE = ROOT / "cache" / "sensitivity_cache.json" # per-config aggregated
TRADES_CACHE      = ROOT / "cache" / "trades_cache.json"
VAL_TRADES_CACHE  = ROOT / "cache" / "val_trades_cache.json"
DATASET_OUT       = ROOT / "data" / "ml_dataset.parquet"
VALID_CONFIGS_OUT = ROOT / "data" / "validated_configs.json"
REGIME_MAP_CACHE  = ROOT / "cache" / "regime_map_cache.pkl"
ATR_RANK_CACHE    = ROOT / "cache" / "atr_rank_map_cache.pkl"

# ---------------------------------------------------------------------------
# Map caching helpers — keyed by data file mtime+size and param fingerprint
# ---------------------------------------------------------------------------

import pickle as _pickle


def _map_cache_key(path: Path, **params) -> str:
    stat = path.stat()
    parts = [str(stat.st_mtime), str(stat.st_size)] + [f"{k}={v}" for k, v in sorted(params.items())]
    return "|".join(parts)


def _load_map_cache(cache_path: Path, key: str):
    if cache_path.exists():
        try:
            with open(cache_path, 'rb') as f:
                entry = _pickle.load(f)
            if entry.get('key') == key:
                return entry['data']
        except Exception:
            pass
    return None


def _save_map_cache(cache_path: Path, key: str, data) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, 'wb') as f:
        _pickle.dump({'key': key, 'data': data}, f)


# ---------------------------------------------------------------------------
# Daily ATR percentile rank map — forward-safe helper
# ---------------------------------------------------------------------------

def _compute_daily_atr_rank_map(
    df_1m: pd.DataFrame,
    lookback: int = 60,
) -> dict:
    """
    For each trading day D, compute percentile rank of D's ATR vs prior
    `lookback` days.  Reference ATR = 14-period Wilder ATR of daily ranges,
    evaluated at market open (so signal-time data for day D is not used).

    All values are in [0, 1] where 0 = lowest ATR day in lookback window.
    Returns dict[date, float].  Days with insufficient history return 0.5.
    """
    from datetime import date as _date, time as _time
    from backtest.ml.splits import VALIDATION_END

    cutoff = _date.fromisoformat(VALIDATION_END)

    # RTH daily OHLC (9:30–16:00 ET)
    rth = df_1m[(df_1m.index.time >= _time(9, 30)) & (df_1m.index.time <= _time(16, 0))]
    daily = rth.groupby(rth.index.date).agg(
        high=('high', 'max'),
        low=('low',   'min'),
        close=('close', 'last'),
    )
    daily = daily[(daily.index <= cutoff) & (daily['close'] > 0)].copy()

    # True range (simplified to H-L range for daily bars; prev close gap included)
    prev_close = daily['close'].shift(1)
    tr = pd.concat([
        daily['high'] - daily['low'],
        (daily['high'] - prev_close).abs(),
        (daily['low']  - prev_close).abs(),
    ], axis=1).max(axis=1)

    # 14-period Wilder ATR (EWM with alpha=1/14, adjust=False)
    atr = tr.ewm(alpha=1.0 / 14, min_periods=14, adjust=False).mean()
    atr_pct = (atr / daily['close'] * 100.0).values
    dates_arr = list(daily.index)

    n = len(atr_pct)
    ranks = np.full(n, 0.5)

    # Vectorised sliding-window rank using stride tricks
    from numpy.lib.stride_tricks import sliding_window_view
    # Replace NaN with -inf so they sort below any real value
    safe = np.where(np.isfinite(atr_pct), atr_pct, -np.inf)
    # For each position i (i >= 1): rank safe[i] against safe[max(0,i-lookback):i]
    # Build windows of size (lookback+1); last element is the current day
    win_size = lookback + 1
    if n >= win_size:
        windows = sliding_window_view(safe, win_size)  # shape (n - lookback, lookback+1)
        prior_w = windows[:, :-1]                      # (n-lookback, lookback)
        curr_w  = windows[:, -1:]                      # (n-lookback, 1)
        valid   = prior_w > -np.inf                    # mask out padded -inf
        n_valid = valid.sum(axis=1)                    # count of real prior values
        n_below = ((prior_w < curr_w) & valid).sum(axis=1)  # only count real priors
        # Avoid division by zero for windows with no valid prior
        with np.errstate(invalid='ignore'):
            w_ranks = np.where(n_valid > 0, n_below / n_valid, 0.5)
        ranks[lookback:] = w_ranks

    # Short head (fewer than lookback prior days): leave as 0.5 default
    rank_map: dict = {d: float(ranks[i]) for i, d in enumerate(dates_arr)}
    return rank_map


# ---------------------------------------------------------------------------
# Numba warmup — called in the main process before spawning workers so that
# all JIT kernels are compiled and cached to disk.  Workers then load from
# __pycache__ instead of all competing to compile at startup.
# ---------------------------------------------------------------------------
def _warmup_numba() -> None:
    try:
        from strategies.ict_smc import (
            _wilder_atr, _detect_swings, _detect_swings_confirmed_at,
            _detect_accum_zones_nb, _detect_ob_nb, _detect_fvg_nb, _cisd_scan_nb,
        )
        z4 = np.zeros(4, dtype=np.float64)
        z8 = np.ones(8, dtype=np.float64)
        lb = 6
        n  = lb + 4
        zn = np.ones(n, dtype=np.float64)
        xs_dev = np.arange(lb, dtype=np.float64) - (lb - 1) / 2.0
        xs_sq  = float((xs_dev * xs_dev).sum())
        _wilder_atr(z4, z4, z4, 14)
        _detect_swings(z8, z8, 1)
        _detect_swings_confirmed_at(z8, z8, 1)
        _detect_accum_zones_nb(
            zn, zn, zn, zn, zn, xs_dev, xs_sq,
            1.0, 0.01, 0.5, 1.0, 2, 3, lb,
        )
        _detect_ob_nb(z8, z8, z8, z8)
        _detect_fvg_nb(z8, z8, z8, z8, 0.5)
        _cisd_scan_nb(z8, z8, 1, 2, 0.5, 0, 7)
        print("Numba warmup complete.")
    except Exception as exc:
        print(f"Numba warmup skipped: {exc}")


# ---------------------------------------------------------------------------
# Worker globals — initialised once per worker process
# ---------------------------------------------------------------------------
_g_data        = None
_g_exec_kwargs = None


def _init_worker(
    cache_1m: str, cache_5m: str, cache_bar_map: str,
    exec_kwargs: dict,
    split: str = "train",
) -> None:
    global _g_data, _g_exec_kwargs
    # 1 Phase 1 thread per worker: Phase 1A is cached after first config, so
    # subsequent configs are GIL-bound Python work — extra threads add overhead,
    # not parallelism. Plain-loop path in _precompute_phase1_parallel handles n_workers=1.
    os.environ["BACKTEST_PHASE1_THREADS"] = "1"
    from backtest.data.loader import DataLoader
    from backtest.data.market_data import MarketData
    from backtest.ml.splits import filter_market_data

    df_1m   = pd.read_parquet(cache_1m)
    df_5m   = pd.read_parquet(cache_5m)
    bar_map = np.load(cache_bar_map)

    rth = (df_1m.index.time >= dtime(9, 30)) & (df_1m.index.time <= dtime(16, 0))
    a1  = {c: df_1m[c].to_numpy(dtype="float64") for c in ["open","high","low","close","volume"]}
    a5  = {c: df_5m[c].to_numpy(dtype="float64") for c in ["open","high","low","close","volume"]}

    loader    = DataLoader()
    full_data = MarketData(
        df_1m=df_1m, df_5m=df_5m,
        open_1m=a1["open"],  high_1m=a1["high"],  low_1m=a1["low"],
        close_1m=a1["close"], volume_1m=a1["volume"],
        open_5m=a5["open"],  high_5m=a5["high"],  low_5m=a5["low"],
        close_5m=a5["close"], volume_5m=a5["volume"],
        bar_map=bar_map,
        trading_dates=sorted(set(df_1m[rth].index.date)),
    )
    _g_data        = filter_market_data(full_data, split, loader)
    _g_exec_kwargs = exec_kwargs


def _task_run_config(args: tuple) -> tuple:
    """
    Run ONE backtest and return results.

    Parameters
    ----------
    args : (config_hash, perturb_key, params, capture_trades)
        perturb_key : 'base' for the canonical run, 'param_name+'/'-' for perturbs
        capture_trades : only True for 'base' runs — avoids allocating trade dicts
                         for perturbation runs that only need the scalar metric.

    Returns
    -------
    (config_hash, perturb_key, metric, trades_list_or_none)
    """
    from backtest.runner.runner import run_backtest
    from backtest.runner.config import RunConfig
    from backtest.ml.evaluate import sortino_r
    from strategies.ict_smc import ICTSMCStrategy

    h, key, params, capture_trades = args

    config = RunConfig(
        starting_capital        = _g_exec_kwargs["starting_capital"],
        slippage_points         = _g_exec_kwargs["slippage_points"],
        commission_per_contract = _g_exec_kwargs["commission_per_contract"],
        eod_exit_time           = _g_exec_kwargs["eod_exit_time"],
        order_cancel_time       = _g_exec_kwargs["order_cancel_time"],
        params                  = {**params, "ml_model": None},
    )
    result = run_backtest(ICTSMCStrategy, config, _g_data, validate=False)

    r_arr = np.array(
        [t.r_multiple for t in result.trades if t.r_multiple is not None],
        dtype=float,
    )
    n = len(r_arr)
    metric = float(sortino_r(r_arr)) if n >= SENSITIVITY_CFG["min_trades"] else -999.0

    trades = None
    if capture_trades:
        trades = []
        for t in result.trades:
            if not t.signal_features:
                continue
            sl_pts = (
                float(abs(t.entry_price - t.initial_sl_price))
                if t.initial_sl_price is not None else 0.0
            )
            tp_pts = (
                float(abs(t.entry_price - t.initial_tp_price))
                if t.initial_tp_price is not None else 0.0
            )
            trades.append({
                "signal_features": t.signal_features,
                "r_multiple":      float(t.r_multiple) if t.r_multiple is not None else 0.0,
                "is_winner":       int(t.is_winner),
                "entry_bar":       t.entry_bar,
                "exit_bar":        t.exit_bar,
                "sl_pts":          sl_pts,
                "tp_pts":          tp_pts,
                "net_pnl_dollars": float(t.net_pnl_dollars),
                "entry_price":     float(t.entry_price),
                "exit_price":      float(t.exit_price),
            })

    return h, key, metric, trades


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def _save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _hash_params(params: dict) -> str:
    key = json.dumps(
        {k: v for k, v in sorted(params.items()) if k != "ml_model"},
        sort_keys=True, default=str,
    )
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _perturb_params(params: dict, key: str, pct: float, sign: int) -> dict:
    val = params[key]
    p   = dict(params)
    p[key] = val * (1.0 + sign * pct)
    return p


def _build_perturb_list(params: dict, lhs_axes: set, pct: float) -> list[tuple]:
    """
    Return list of (perturb_key, perturbed_params) for all non-LHS continuous params.
    Returns empty list for Round 1 when all params are LHS axes.
    """
    # Default continuous params that are candidates for perturbation
    candidates = [
        'confluence_tolerance_atr_mult',
        'tp_confluence_tolerance_atr_mult',
        'level_penetration_atr_mult',
        'min_rr',
        'tick_offset_atr_mult',
        'po3_atr_mult',
        'po3_band_pct',
        'po3_vol_sens',
        'po3_min_manipulation_size_atr_mult',
        'min_ote_size_atr_mult',
        'cancel_pct_to_tp',
    ]
    tasks = []
    for k in candidates:
        if k in lhs_axes:
            continue   # already sampled — no need to perturb
        val = params.get(k)
        if val is None or not isinstance(val, (int, float)) or val == 0:
            continue
        tasks.append((f'{k}+', _perturb_params(params, k, pct, +1)))
        tasks.append((f'{k}-', _perturb_params(params, k, pct, -1)))
    return tasks


def _aggregate_sensitivity(
    h: str,
    base_metric: float,
    run_results: dict,   # {perturb_key: metric}
    cfg: dict,
) -> tuple[bool, float, float, str]:
    """Return (is_valid, base_metric, degradation_pct, worst_param)."""
    if not run_results:
        # No perturbations ran — only check base metric
        is_stable = True
        return (base_metric >= cfg["min_base_metric"]), base_metric, 0.0, ''

    worst_metric = base_metric
    worst_key    = ''
    for k, m in run_results.items():
        if m < worst_metric:
            worst_metric = m
            worst_key    = k

    if abs(base_metric) < 1e-9:
        degrad = 0.0
    else:
        degrad = (base_metric - worst_metric) / abs(base_metric) * 100.0

    is_stable  = degrad <= cfg["max_degradation_pct"]
    is_valid   = is_stable and base_metric >= cfg["min_base_metric"]
    worst_param = worst_key.rstrip('+-')
    return is_valid, base_metric, degrad, worst_param


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"=== ICT/SMC ML Data Collection  [Round {ROUND}] ===\n")

    if not CACHE_1M.exists():
        print(f"ERROR: {CACHE_1M} not found. Run run.py once to build the cache.")
        return

    ranges   = ROUND_RANGES.get(ROUND, ROUND_RANGES[1])
    lhs_axes = set(ranges.keys())

    n_cpu     = os.cpu_count() or 1
    n_workers = min(2, max(1, n_cpu - 1))
    print(f"CPU cores: {n_cpu}  ->  using {n_workers} worker(s) (capped at 2)")
    _warmup_numba()
    print(f"Sampling {N_CONFIGS} configs via LHS (Round {ROUND} ranges)\n")

    all_configs = sample_configs(N_CONFIGS, ranges, seed=LHS_SEED + ROUND - 1)

    # Persist all LHS configs (including 0-trade ones) so tearsheet/threshold-opt
    # can run the same universe without re-sampling.
    LHS_CONFIGS_OUT = ROOT / "data" / "lhs_configs.json"
    _lhs_to_save = [
        {"config_idx": i, "config_hash": _hash_params(p),
         "params": {k: v for k, v in p.items() if not callable(v)}}
        for i, p in enumerate(all_configs)
    ]
    _save_json(LHS_CONFIGS_OUT, _lhs_to_save)
    print(f"LHS manifest saved -> {LHS_CONFIGS_OUT}  ({len(all_configs)} configs)\n")

    # Load caches
    sens_runs_cache  = _load_json(SENS_RUNS_CACHE)   # {f"{h}:{key}": metric}
    sens_agg_cache   = _load_json(SENSITIVITY_CACHE)  # {h: aggregated result}
    trades_cache     = _load_json(TRADES_CACHE)        # {h: {round, n_trades}}

    init_args = (
        str(CACHE_1M), str(CACHE_5M), str(CACHE_BAR_MAP),
        {k: v for k, v in BASE_EXEC.items()},
    )

    # Pre-build perturb config maps so we can check trades_cache before submitting.
    # Keyed by the perturbed params hash (not the base config hash).
    # Skips any perturb that coincidentally hashes to an LHS config (avoid duplicates).
    lhs_hash_set = {_hash_params(p) for p in all_configs}
    perturb_config_map: dict[str, dict]          = {}  # h_perturb -> perturbed params
    perturb_run_map: dict[str, tuple[str, str]]  = {}  # h_perturb -> (h_base, key)
    for params in all_configs:
        h = _hash_params(params)
        for key, p_params in _build_perturb_list(params, lhs_axes,
                                                  SENSITIVITY_CFG['perturbation_pct']):
            h_p = _hash_params(p_params)
            if h_p not in lhs_hash_set:
                perturb_config_map.setdefault(h_p, p_params)
                perturb_run_map.setdefault(h_p, (h, key))

    # -----------------------------------------------------------------------
    # Build task list — one task per (config, perturbation) pair
    # Both base and perturbation runs capture trade data (free data from runs
    # we're doing anyway; perturb configs get cfg_is_valid=0 in the dataset).
    # -----------------------------------------------------------------------
    # Tasks to submit: (config_hash, perturb_key, params, capture_trades)
    pending_tasks: list[tuple] = []

    for params in all_configs:
        h = _hash_params(params)

        # Skip if aggregated sensitivity already known AND trades already collected
        if h in sens_agg_cache and h in trades_cache:
            continue

        # Base run needed if not yet in per-run cache
        base_run_key = f"{h}:base"
        if base_run_key not in sens_runs_cache:
            capture = h not in trades_cache  # capture trades only if not already collected
            pending_tasks.append((h, 'base', params, capture))
        elif h not in trades_cache:
            # Sensitivity already cached but trades not yet captured — add trade-only run.
            # Without this, these configs fall through to a separate pool (cold start).
            pending_tasks.append((h, 'base', params, True))

        # Perturbation runs (only for non-LHS params; empty in Round 1)
        if h not in sens_agg_cache:
            for key, p_params in _build_perturb_list(params, lhs_axes,
                                                      SENSITIVITY_CFG['perturbation_pct']):
                run_key = f"{h}:{key}"
                if run_key not in sens_runs_cache:
                    h_p = _hash_params(p_params)
                    capture_p = h_p not in lhs_hash_set and h_p not in trades_cache
                    pending_tasks.append((h, key, p_params, capture_p))

    n_base   = sum(1 for t in pending_tasks if t[1] == 'base')
    n_perturb = len(pending_tasks) - n_base
    print(f"Tasks to run:  {n_base} base  +  {n_perturb} perturbation  =  {len(pending_tasks)} total")

    if not pending_tasks:
        print("All configs already in cache.\n")
    else:
        estimated_min = len(pending_tasks) / n_workers * 20 / 60
        print(f"Estimated wall clock: ~{estimated_min:.0f} min  ({n_workers} workers)\n")

    # -----------------------------------------------------------------------
    # Run all tasks in one pool
    # -----------------------------------------------------------------------
    new_trade_rows: dict[str, list] = {}   # h -> list of trade dicts (from base runs)

    if pending_tasks:
        with ProcessPoolExecutor(
            max_workers = n_workers,
            initializer = _init_worker,
            initargs    = init_args,
        ) as pool:
            future_map = {
                pool.submit(_task_run_config, task): task
                for task in pending_tasks
            }
            done = 0
            for fut in as_completed(future_map):
                task = future_map[fut]
                h, key, params, capture = task
                try:
                    _, _, metric, trades = fut.result()
                except Exception as exc:
                    print(f"  [error] {h}:{key}: {exc}")
                    metric, trades = -999.0, None

                # Store per-run result
                run_key = f"{h}:{key}"
                sens_runs_cache[run_key] = metric
                _save_json(SENS_RUNS_CACHE, sens_runs_cache)

                # Store trade data — base runs under h, perturb runs under their own hash
                if key == 'base' and trades is not None:
                    new_trade_rows[h] = trades
                elif key != 'base' and trades is not None:
                    h_p = _hash_params(params)  # params IS the perturbed params
                    if h_p not in lhs_hash_set:
                        new_trade_rows[h_p] = trades

                done += 1
                tag = 'base' if key == 'base' else 'perturb'
                print(f"  [{done:>4}/{len(pending_tasks)}] {h}:{key:<20}  "
                      f"metric={metric:>7.3f}  [{tag}]")

    # -----------------------------------------------------------------------
    # Aggregate sensitivity per config
    # -----------------------------------------------------------------------
    print("\n--- Aggregating sensitivity ---")
    validated_configs: list[dict] = []

    for params in all_configs:
        h = _hash_params(params)

        if h in sens_agg_cache:
            cached = sens_agg_cache[h]
            if cached["is_valid"]:
                validated_configs.append(params)
            print(f"  [cache] {h}  base={cached['base_metric']:.3f}  "
                  f"degrad={cached['degradation_pct']:.1f}%  valid={cached['is_valid']}")
            continue

        base_metric  = sens_runs_cache.get(f"{h}:base", -999.0)
        perturb_keys = {k[len(h)+1:]: v for k, v in sens_runs_cache.items()
                        if k.startswith(h + ':') and not k.endswith(':base')}

        is_valid, base_m, degrad, worst = _aggregate_sensitivity(
            h, base_metric, perturb_keys, SENSITIVITY_CFG)

        sens_agg_cache[h] = {
            "is_valid":        is_valid,
            "base_metric":     base_m,
            "degradation_pct": degrad,
            "worst_param":     worst,
            "round":           ROUND,
        }
        _save_json(SENSITIVITY_CACHE, sens_agg_cache)

        tag = "PASS" if is_valid else "FAIL"
        n_perturbs = len(perturb_keys)
        print(f"  [{tag}] {h}  base={base_m:.3f}  "
              f"degrad={degrad:.1f}%  worst={worst or 'n/a'}  "
              f"perturbs={n_perturbs}")

        if is_valid:
            validated_configs.append(params)

    print(f"\n  {len(validated_configs)} / {len(all_configs)} configs passed.\n")

    # Build validity map for ALL configs (LHS + perturb) — used to populate
    # cfg_is_valid / cfg_base_metric features so the model sees trades from
    # bad configs too (avoids survivorship bias).
    validity_map: dict[str, tuple[bool, float]] = {}
    for params in all_configs:
        h = _hash_params(params)
        agg = sens_agg_cache.get(h, {})
        validity_map[h] = (bool(agg.get("is_valid", False)), float(agg.get("base_metric", -999.0)))
    # Perturb configs are never independently validated — cfg_is_valid=False,
    # cfg_base_metric = their own backtest metric.
    for h_p, (h_base, p_key) in perturb_run_map.items():
        metric_p = sens_runs_cache.get(f"{h_base}:{p_key}", -999.0)
        validity_map[h_p] = (False, float(metric_p))

    # Combined list of all configs whose trades we collect (LHS + perturb).
    # Used for missing-trades checks, val collection, cache updates, and dataset build.
    all_config_entries: list[dict] = all_configs + [
        perturb_config_map[h] for h in perturb_config_map
    ]

    if not validated_configs:
        print("No configs passed. Widen ranges or lower min_base_metric.")
        print("Trade data will still be collected for all configs.\n")

    # Save validated config list (append new across rounds)
    existing_valid = _load_json(VALID_CONFIGS_OUT) if VALID_CONFIGS_OUT.exists() else []
    if isinstance(existing_valid, list):
        existing_hashes = {_hash_params(c) for c in existing_valid}
    else:
        existing_valid, existing_hashes = [], set()

    new_valid = [
        {k: v for k, v in p.items() if k != "ml_model" and not callable(v)}
        for p in validated_configs
        if _hash_params(p) not in existing_hashes
    ]
    _save_json(VALID_CONFIGS_OUT, existing_valid + new_valid)
    print(f"  Validated configs saved -> {VALID_CONFIGS_OUT}  "
          f"(+{len(new_valid)} new, {len(existing_valid)} existing)\n")

    # -----------------------------------------------------------------------
    # Collect trade data for ALL configs whose trades weren't captured this run.
    # We collect from all configs (not just validated) to avoid survivorship
    # bias: the ML model needs to see trades from "bad" configs to learn the
    # contrast.  cfg_is_valid / cfg_base_metric are passed as features so
    # the model knows which configs the sensitivity filter accepted.
    # -----------------------------------------------------------------------
    existing_train_hashes: set = set()
    if DATASET_OUT.exists():
        _edf = pd.read_parquet(DATASET_OUT, columns=["config_hash", "split"])
        if "split" in _edf.columns:
            existing_train_hashes = set(_edf.loc[_edf["split"] == "train", "config_hash"])
        else:
            existing_train_hashes = set(_edf["config_hash"])

    # Schema migration: if sl_pts (or other new columns) are missing from the
    # existing dataset, force re-collection so all rows get the new fields.
    _new_required_cols = {"sl_pts", "tp_pts", "exit_bar", "net_pnl_dollars", "entry_price", "exit_price"}
    _force_schema_rebuild = False
    if DATASET_OUT.exists():
        _existing_cols = set(pd.read_parquet(DATASET_OUT, columns=[]).columns)
        if not _new_required_cols.issubset(_existing_cols):
            print("Schema update needed (new columns missing) — forcing re-collection.\n")
            existing_train_hashes = set()   # triggers train re-collection
            _force_schema_rebuild = True    # val_trades_cache cleared after it loads

    missing_trades = [
        (_hash_params(p), p)
        for p in all_config_entries
        if _hash_params(p) not in new_trade_rows
        and (
            _hash_params(p) not in trades_cache
            or _hash_params(p) not in existing_train_hashes
        )
    ]

    if missing_trades:
        print(f"--- Collecting trade data for {len(missing_trades)} config(s) "
              f"(cache miss) ---")
        with ProcessPoolExecutor(
            max_workers = n_workers,
            initializer = _init_worker,
            initargs    = init_args,
        ) as pool:
            future_map = {
                pool.submit(_task_run_config, (h, 'base', params, True)): (h, params)
                for h, params in missing_trades
            }
            for fut in as_completed(future_map):
                h, params = future_map[fut]
                try:
                    _, _, metric, trades = fut.result()
                except Exception as exc:
                    print(f"  [error] {h}: {exc}")
                    trades = []
                new_trade_rows[h] = trades or []
                print(f"  [done] {h}  trades={len(new_trade_rows[h])}")

    # Mark all configs (LHS + perturb, valid and invalid) as collected
    for params in all_config_entries:
        h = _hash_params(params)
        if h in new_trade_rows:
            trades_cache[h] = {"round": ROUND, "n_trades": len(new_trade_rows[h])}
    _save_json(TRADES_CACHE, trades_cache)

    # -----------------------------------------------------------------------
    # Collect validation split trades for all validated configs
    # -----------------------------------------------------------------------
    print("\n--- Collecting validation split trades ---")
    val_trades_cache = _load_json(VAL_TRADES_CACHE)
    if _force_schema_rebuild:
        val_trades_cache = {}  # triggers re-collection of all val trades

    val_missing = [
        (_hash_params(p), p)
        for p in all_config_entries
        if _hash_params(p) not in val_trades_cache
    ]

    val_trade_rows: dict[str, list] = {}

    if val_missing:
        print(f"  Running {len(val_missing)} config(s) on validation split...")
        val_init_args = (
            str(CACHE_1M), str(CACHE_5M), str(CACHE_BAR_MAP),
            {k: v for k, v in BASE_EXEC.items()},
            "validation",
        )
        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_init_worker,
            initargs=val_init_args,
        ) as pool:
            future_map = {
                pool.submit(_task_run_config, (h, 'base', params, True)): (h, params)
                for h, params in val_missing
            }
            for fut in as_completed(future_map):
                h, params = future_map[fut]
                try:
                    _, _, metric, trades = fut.result()
                except Exception as exc:
                    print(f"  [error] {h}: {exc}")
                    trades = []
                val_trade_rows[h] = trades or []
                val_trades_cache[h] = {"round": ROUND, "n_trades": len(val_trade_rows[h])}
                print(f"  [done] {h}  val_trades={len(val_trade_rows[h])}  metric={metric:.3f}")
        _save_json(VAL_TRADES_CACHE, val_trades_cache)
    else:
        print("  All configs already in validation cache.\n")

    # -----------------------------------------------------------------------
    # Build dataset rows (with context features and config features)
    # -----------------------------------------------------------------------
    print("\n--- Building dataset ---")

    from collections import deque
    from backtest.ml.splits import split_bounds

    def _build_trade_list(
        trade_rows: dict,
        split_name: str,
        configs_list: list,
        val_map: dict,   # hash -> (is_valid, base_metric)
    ) -> list:
        result = []
        for idx, params in enumerate(configs_list):
            h                = _hash_params(params)
            is_v, base_m     = val_map.get(h, (False, -999.0))
            cfg_feat         = normalize_config(params, ranges,
                                                is_valid=is_v, base_metric=base_m)
            for t in trade_rows.get(h, []):
                result.append({**t,
                                "config_hash":     h,
                                "config_idx":      idx,
                                "round":           ROUND,
                                "split":           split_name,
                                "config_features": cfg_feat})
        return result

    def _build_rows(
        trade_list: list,
        df_1m_slice: "pd.DataFrame",
        regime_map: dict,
        atr_rank_map: dict,
    ) -> list:
        from collections import defaultdict
        from datetime import date as _date
        from backtest.ml.evaluate import sortino_r as _sortino_r

        # Group by config so rolling features never cross config boundaries.
        # Within one config trades are sequential (one position at a time), so
        # each trade's context only uses already-resolved outcomes — no look-ahead.
        by_config: dict[str, list] = defaultdict(list)
        for trade in trade_list:
            by_config[trade.get("config_hash", "")].append(trade)

        all_rows: list[dict] = []

        for config_trades in by_config.values():
            history            = deque(maxlen=10)
            consecutive_losses = 0
            day_counts: dict   = {}
            running_r: list    = []   # all prior R-multiples for this config
            # Per-day session R accumulator (reset each day)
            session_r_by_date: dict[_date, float] = {}
            # Date of last winning trade for this config
            last_win_date: _date | None = None

            for trade in sorted(config_trades, key=lambda t: t["entry_bar"]):
                sf               = trade["signal_features"]
                entry_bar        = trade["entry_bar"]
                exit_bar_val     = trade.get("exit_bar", -1)
                sl_pts_val       = trade.get("sl_pts", 0.0)
                tp_pts_val       = trade.get("tp_pts", 0.0)
                net_pnl_val      = trade.get("net_pnl_dollars", 0.0)
                entry_price_val  = trade.get("entry_price", 0.0)
                exit_price_val   = trade.get("exit_price", 0.0)
                r_multiple       = float(trade["r_multiple"])
                # copy so we can override cfg_base_metric without mutating the source
                cfg_feat   = dict(trade.get("config_features", {}))

                try:
                    ts         = df_1m_slice.index[entry_bar] if entry_bar < len(df_1m_slice) else None
                    trade_date = ts.date() if ts is not None else None
                except (IndexError, AttributeError):
                    trade_date = None

                daily_idx              = day_counts.get(trade_date, 0)
                day_counts[trade_date] = daily_idx + 1

                if len(history) >= 1:
                    recent_r = list(history)
                    rwr      = float(sum(r > 0 for r in recent_r) / len(recent_r))
                    rex      = float(sum(recent_r) / len(recent_r))
                else:
                    rwr, rex = 0.5, 0.0

                # Replace static cfg_base_metric with rolling Sortino of prior trades
                # for this config only — eliminates full-train-period look-ahead.
                # Default 0.5 (neutral) until 5 trades have resolved.
                if len(running_r) >= 5:
                    roll_m = float(_sortino_r(np.array(running_r, dtype=float)))
                    cfg_feat['cfg_base_metric'] = float(
                        np.clip(roll_m, -2.0, 2.0) / 4.0 + 0.5
                    )
                else:
                    cfg_feat['cfg_base_metric'] = 0.5

                vol_p = regime_map.get(trade_date, 1.0 / 3.0) if trade_date is not None else 1.0 / 3.0

                # atr_pct_rank — forward-safe daily percentile, 0.5 default
                atr_rank = atr_rank_map.get(trade_date, 0.5) if trade_date is not None else 0.5

                # session_r_so_far — cumulative R earned today BEFORE this trade
                sess_r = session_r_by_date.get(trade_date, 0.0) if trade_date is not None else 0.0

                # days_since_last_win — calendar days since last win, capped at 30
                if last_win_date is None or trade_date is None:
                    days_win = 30.0
                else:
                    days_win = float(min((trade_date - last_win_date).days, 30))

                row = {
                    **sf,
                    **cfg_feat,
                    "daily_trade_idx":        daily_idx,
                    "recent_win_rate_10":     rwr,
                    "recent_expectancy_r_10": rex,
                    "consecutive_losses":     consecutive_losses,
                    "drawdown_pct":           0.0,
                    "vol_regime_p_high":      vol_p,
                    "atr_pct_rank":           atr_rank,
                    "session_r_so_far":       sess_r,
                    "days_since_last_win":    days_win,
                    "r_multiple":             r_multiple,
                    "is_winner":              int(r_multiple > 0),
                    "date":                   trade_date,
                    "entry_bar":              entry_bar,
                    "exit_bar":               exit_bar_val,
                    "sl_pts":                 sl_pts_val,
                    "tp_pts":                 tp_pts_val,
                    "net_pnl_dollars":        net_pnl_val,
                    "entry_price":            entry_price_val,
                    "exit_price":             exit_price_val,
                    "config_hash":            trade.get("config_hash", ""),
                    "config_idx":             trade.get("config_idx", -1),
                    "round":                  trade.get("round", ROUND),
                    "split":                  trade.get("split", "train"),
                }

                for col in ALL_FEATURE_NAMES:
                    if col not in row:
                        row[col] = 0

                all_rows.append(row)
                history.append(r_multiple)
                running_r.append(r_multiple)
                consecutive_losses = consecutive_losses + 1 if r_multiple <= 0 else 0
                # Update session accumulator AFTER recording (before = what we observed)
                if trade_date is not None:
                    session_r_by_date[trade_date] = sess_r + r_multiple
                # Update last win date AFTER recording
                if r_multiple > 0 and trade_date is not None:
                    last_win_date = trade_date

        return all_rows

    df_1m_full = pd.read_parquet(CACHE_1M)

    # Compute volatility regime map once — forward-safe, hard-stopped at VALIDATION_END.
    # Never reads test-period price data.
    from backtest.ml.splits import VALIDATION_END as _VEND
    from backtest.regime.vol_regime import _WARMUP_END as _WUE
    _regime_key = _map_cache_key(CACHE_1M, validation_end=_VEND, warmup_end=str(_WUE), n_states=3, seed=42)
    regime_map = _load_map_cache(REGIME_MAP_CACHE, _regime_key)
    if regime_map is None:
        print("\n--- Computing volatility regime map ---")
        regime_map = compute_vol_regime_map(df_1m_full)
        _save_map_cache(REGIME_MAP_CACHE, _regime_key, regime_map)
        print(f"  Regime map computed for {len(regime_map)} trading days.")
    else:
        print(f"\n--- Regime map loaded from cache ({len(regime_map)} days) ---")

    # Compute daily ATR percentile rank map — forward-safe.
    # For each day D: rank of that day's ATR vs prior 60 days (09:30 open bar,
    # 14-period Wilder ATR of daily ranges). Uses only pre-cutoff data.
    _atr_key = _map_cache_key(CACHE_1M, validation_end=_VEND, lookback=60)
    atr_rank_map = _load_map_cache(ATR_RANK_CACHE, _atr_key)
    if atr_rank_map is None:
        print("\n--- Computing daily ATR rank map ---")
        atr_rank_map = _compute_daily_atr_rank_map(df_1m_full)
        _save_map_cache(ATR_RANK_CACHE, _atr_key, atr_rank_map)
        print(f"  ATR rank map computed for {len(atr_rank_map)} trading days.")
    else:
        print(f"\n--- ATR rank map loaded from cache ({len(atr_rank_map)} days) ---")

    # Train rows
    all_train_trades = _build_trade_list(new_trade_rows, "train", all_config_entries, validity_map)
    s_ts, e_ts = split_bounds("train")
    df_1m_train = df_1m_full[(df_1m_full.index >= s_ts) & (df_1m_full.index <= e_ts)]
    train_rows = _build_rows(all_train_trades, df_1m_train, regime_map, atr_rank_map)

    # Validation rows
    all_val_trades = _build_trade_list(val_trade_rows, "validation", all_config_entries, validity_map)
    s_ts, e_ts = split_bounds("validation")
    df_1m_val = df_1m_full[(df_1m_full.index >= s_ts) & (df_1m_full.index <= e_ts)]
    val_rows = _build_rows(all_val_trades, df_1m_val, regime_map, atr_rank_map)

    all_rows = train_rows + val_rows

    meta_cols = ["date", "entry_bar", "exit_bar", "r_multiple", "is_winner",
                 "sl_pts", "tp_pts", "net_pnl_dollars", "entry_price", "exit_price",
                 "config_hash", "config_idx", "round", "split"]
    cols = ALL_FEATURE_NAMES + meta_cols

    if all_rows:
        df_new = pd.DataFrame(all_rows)
        for c in cols:
            if c not in df_new.columns:
                df_new[c] = 0
        df_new = df_new[cols].reset_index(drop=True)
    else:
        df_new = pd.DataFrame(columns=cols)

    # Always merge with existing dataset — preserves rows from prior runs/rounds
    # and deduplicates by (config_hash, entry_bar, split) so re-runs are idempotent.
    if DATASET_OUT.exists():
        existing_df = pd.read_parquet(DATASET_OUT)
        for c in df_new.columns:
            if c not in existing_df.columns:
                existing_df[c] = 0
        for c in existing_df.columns:
            if c not in df_new.columns:
                df_new[c] = 0
        combined = pd.concat([existing_df, df_new], ignore_index=True)
        df = combined.drop_duplicates(
            subset=["config_hash", "entry_bar", "split"], keep="last"
        ).reset_index(drop=True)
    else:
        df = df_new

    if len(df) == 0:
        print("ERROR: No trade data collected and no existing dataset to preserve.")
        return

    DATASET_OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DATASET_OUT, index=False)

    print(f"  New rows this run:   {len(df_new)}")
    print(f"  Total dataset rows:  {len(df)}")
    if 'split' in df.columns:
        for sp, cnt in df['split'].value_counts().items():
            print(f"    {sp}: {cnt}")
    print(f"  Win rate:            {df['is_winner'].mean():.1%}")
    print(f"  Mean R:              {df['r_multiple'].mean():.3f}")
    print(f"  Rounds in dataset:   {sorted(df['round'].unique())}")
    print(f"  Dataset saved -> {DATASET_OUT}")


if __name__ == "__main__":
    import time as _time
    _t0 = _time.perf_counter()
    main()
    print(f"\nTotal time: {_time.perf_counter() - _t0:.1f}s")
