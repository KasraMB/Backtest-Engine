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
DATASET_OUT       = ROOT / "data" / "ml_dataset.parquet"
VALID_CONFIGS_OUT = ROOT / "data" / "validated_configs.json"

# ---------------------------------------------------------------------------
# Worker globals — initialised once per worker process
# ---------------------------------------------------------------------------
_g_data        = None
_g_exec_kwargs = None


def _init_worker(
    cache_1m: str, cache_5m: str, cache_bar_map: str,
    exec_kwargs: dict,
) -> None:
    global _g_data, _g_exec_kwargs
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
    _g_data        = filter_market_data(full_data, "train", loader)
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
    result = run_backtest(ICTSMCStrategy, config, _g_data)

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
            trades.append({
                "signal_features": t.signal_features,
                "r_multiple":      float(t.r_multiple) if t.r_multiple is not None else 0.0,
                "is_winner":       int(t.is_winner),
                "entry_bar":       t.entry_bar,
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
    n_workers = max(1, n_cpu - 1)
    print(f"CPU cores: {n_cpu}  →  using {n_workers} worker(s)")
    print(f"Sampling {N_CONFIGS} configs via LHS (Round {ROUND} ranges)\n")

    all_configs = sample_configs(N_CONFIGS, ranges, seed=LHS_SEED + ROUND - 1)

    # Load caches
    sens_runs_cache  = _load_json(SENS_RUNS_CACHE)   # {f"{h}:{key}": metric}
    sens_agg_cache   = _load_json(SENSITIVITY_CACHE)  # {h: aggregated result}
    trades_cache     = _load_json(TRADES_CACHE)        # {h: {round, n_trades}}

    init_args = (
        str(CACHE_1M), str(CACHE_5M), str(CACHE_BAR_MAP),
        {k: v for k, v in BASE_EXEC.items()},
    )

    # -----------------------------------------------------------------------
    # Build task list — one task per (config, perturbation) pair
    # base runs also capture trade data (capture_trades=True)
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

        # Perturbation runs (only for non-LHS params; empty in Round 1)
        if h not in sens_agg_cache:
            for key, p_params in _build_perturb_list(params, lhs_axes,
                                                      SENSITIVITY_CFG['perturbation_pct']):
                run_key = f"{h}:{key}"
                if run_key not in sens_runs_cache:
                    pending_tasks.append((h, key, p_params, False))

    n_base   = sum(1 for t in pending_tasks if t[1] == 'base')
    n_perturb = len(pending_tasks) - n_base
    print(f"Tasks to run:  {n_base} base  +  {n_perturb} perturbation  =  {len(pending_tasks)} total")

    if not pending_tasks:
        print("All configs already in cache.\n")
    else:
        estimated = len(pending_tasks) / n_workers * 1150 / 3600
        print(f"Estimated wall clock: ~{estimated:.1f} h  ({n_workers} workers)\n")

    # -----------------------------------------------------------------------
    # Run all tasks in one pool
    # -----------------------------------------------------------------------
    new_trade_rows: dict[str, list] = {}   # h → list of trade dicts (from base runs)

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

                # Store trade data from base runs
                if key == 'base' and trades is not None:
                    new_trade_rows[h] = trades

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

    if not validated_configs:
        print("No configs passed. Widen ranges or lower min_base_metric.")
        return

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
    print(f"  Validated configs saved → {VALID_CONFIGS_OUT}  "
          f"(+{len(new_valid)} new, {len(existing_valid)} existing)\n")

    # -----------------------------------------------------------------------
    # Collect any validated configs whose trade data wasn't captured
    # (shouldn't happen in normal flow, but handles the case where base run
    #  was restored from cache without trade capture)
    # -----------------------------------------------------------------------
    missing_trades = [
        (_hash_params(p), p)
        for p in validated_configs
        if _hash_params(p) not in new_trade_rows
        and _hash_params(p) not in trades_cache
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

    # Mark all valid configs as collected
    for params in validated_configs:
        h = _hash_params(params)
        if h in new_trade_rows:
            trades_cache[h] = {"round": ROUND, "n_trades": len(new_trade_rows[h])}
    _save_json(TRADES_CACHE, trades_cache)

    # -----------------------------------------------------------------------
    # Build dataset rows (with context features and config features)
    # -----------------------------------------------------------------------
    print("\n--- Building dataset ---")

    all_new_trades = []
    for params in validated_configs:
        h        = _hash_params(params)
        cfg_feat = normalize_config(params, ranges)
        idx      = next(
            (i for i, c in enumerate(validated_configs) if _hash_params(c) == h), -1)
        for t in new_trade_rows.get(h, []):
            all_new_trades.append({**t,
                                    "config_hash":     h,
                                    "config_idx":      idx,
                                    "round":           ROUND,
                                    "config_features": cfg_feat})

    if not all_new_trades and ROUND > 1 and DATASET_OUT.exists():
        print("  No new trades to add. Existing dataset unchanged.")
        df = pd.read_parquet(DATASET_OUT)
        print(f"  Existing rows: {len(df)}")
        return

    if not all_new_trades:
        print("ERROR: No trade data collected.")
        return

    # Load 1m index for date lookup
    df_1m = pd.read_parquet(CACHE_1M)
    from backtest.ml.splits import split_bounds
    s_ts, e_ts = split_bounds("train")
    df_1m = df_1m[(df_1m.index >= s_ts) & (df_1m.index <= e_ts)]

    from collections import deque
    rows: list[dict]   = []
    history: deque     = deque(maxlen=10)
    consecutive_losses = 0
    day_counts: dict   = {}

    for trade in sorted(all_new_trades, key=lambda t: t["entry_bar"]):
        sf         = trade["signal_features"]
        entry_bar  = trade["entry_bar"]
        r_multiple = float(trade["r_multiple"])
        cfg_feat   = trade.get("config_features", {})

        try:
            ts         = df_1m.index[entry_bar] if entry_bar < len(df_1m) else None
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

        row = {
            **sf,
            **cfg_feat,
            "daily_trade_idx":        daily_idx,
            "recent_win_rate_10":     rwr,
            "recent_expectancy_r_10": rex,
            "consecutive_losses":     consecutive_losses,
            "drawdown_pct":           0.0,
            "r_multiple":             r_multiple,
            "is_winner":              int(r_multiple > 0),
            "date":                   trade_date,
            "entry_bar":              entry_bar,
            "config_hash":            trade.get("config_hash", ""),
            "config_idx":             trade.get("config_idx", -1),
            "round":                  trade.get("round", ROUND),
        }

        for col in ALL_FEATURE_NAMES:
            if col not in row:
                row[col] = 0

        rows.append(row)
        history.append(r_multiple)
        consecutive_losses = consecutive_losses + 1 if r_multiple <= 0 else 0

    meta_cols = ["date", "entry_bar", "r_multiple", "is_winner",
                 "config_hash", "config_idx", "round"]
    cols   = ALL_FEATURE_NAMES + meta_cols
    df_new = pd.DataFrame(rows)
    for c in cols:
        if c not in df_new.columns:
            df_new[c] = 0
    df_new = df_new[cols].reset_index(drop=True)

    # Append to existing dataset on Round 2+
    existing_df = pd.read_parquet(DATASET_OUT) if DATASET_OUT.exists() and ROUND > 1 else None
    if existing_df is not None:
        for c in df_new.columns:
            if c not in existing_df.columns:
                existing_df[c] = 0
        for c in existing_df.columns:
            if c not in df_new.columns:
                df_new[c] = 0
        df = pd.concat([existing_df, df_new], ignore_index=True)
    else:
        df = df_new

    DATASET_OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DATASET_OUT, index=False)

    print(f"  New rows this round: {len(df_new)}")
    print(f"  Total dataset rows:  {len(df)}")
    print(f"  Win rate:            {df['is_winner'].mean():.1%}")
    print(f"  Mean R:              {df['r_multiple'].mean():.3f}")
    print(f"  Rounds in dataset:   {sorted(df['round'].unique())}")
    print(f"  Dataset saved → {DATASET_OUT}")


if __name__ == "__main__":
    main()
