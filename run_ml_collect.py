"""
Multi-config data collection for the ICT/SMC ML pipeline.

Steps
-----
1. Build a parameter grid.
2. For each candidate config, run a sensitivity check (parallelised across CPU cores).
   Results are cached so re-runs skip already-checked configs.
3. For every config that passes sensitivity, run a clean data-collection backtest
   (also parallelised + cached).
4. Merge all trade rows into a single dataset and save to data/ml_dataset.parquet.

After this script finishes, run  run_ml_train.py  to train the model.

Workers
-------
Uses ProcessPoolExecutor with one worker per logical CPU (minus one for the OS).
Each worker loads the training-split market data ONCE via the initializer, so
data is not re-read for every backtest.  On Windows, ensure you run this script
under the  if __name__ == '__main__':  guard (already done at the bottom).

Caching
-------
cache/sensitivity_cache.json  — sensitivity check results keyed by config hash
cache/trades_cache.json       — set of config hashes already collected

Both caches survive between runs; delete them to force a full re-run.
"""
from __future__ import annotations

import json
import os
import sys
import hashlib
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import time as dtime
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CACHE_1M      = ROOT / "data" / "NQ_1m.parquet"
CACHE_5M      = ROOT / "data" / "NQ_5m.parquet"
CACHE_BAR_MAP = ROOT / "data" / "NQ_bar_map.npy"
SENSITIVITY_CACHE = ROOT / "cache" / "sensitivity_cache.json"
TRADES_CACHE      = ROOT / "cache" / "trades_cache.json"
DATASET_OUT       = ROOT / "data" / "ml_dataset.parquet"
VALID_CONFIGS_OUT = ROOT / "data" / "validated_configs.json"

# ---------------------------------------------------------------------------
# Base execution config (strategy params are overridden per run)
# ---------------------------------------------------------------------------
BASE_EXEC = dict(
    starting_capital=100_000,
    slippage_points=0.25,
    commission_per_contract=4.50,
    eod_exit_time=dtime(17, 0),
    order_cancel_time=dtime(11, 0),
)

# Base strategy params — grid values override these
BASE_PARAMS = dict(
    contracts=1,
    swing_n=1,
    cisd_min_series_candles=2,
    cisd_min_body_ratio=0.5,
    rb_min_wick_ratio=0.3,
    confluence_tolerance_atr_mult=0.18,
    tp_confluence_tolerance_atr_mult=0.18,
    level_penetration_atr_mult=0.5,
    min_rr=5.0,
    tick_offset_atr_mult=0.035,
    order_expiry_bars=10,
    session_level_validity_days=2,
    cancel_pct_to_tp=0.75,
    min_ote_size_atr_mult=0.0,
    max_ote_per_session=1,
    max_stdv_per_session=1,
    max_session_ote_per_session=1,
    po3_lookback=6,
    po3_atr_mult=0.95,
    po3_atr_len=14,
    po3_band_pct=0.3,
    po3_vol_sens=1.0,
    po3_max_r2=0.4,
    po3_min_dir_changes=2,
    po3_min_candles=3,
    po3_max_accum_gap_bars=10,
    po3_min_manipulation_size_atr_mult=0.0,
    manip_leg_timeframe="5m",
    manip_leg_swing_depth=1,
    max_trades_per_day=2,
    allowed_setup_types=["OTE", "STDV", "SESSION_OTE"],
    ml_model=None,
)

# ---------------------------------------------------------------------------
# Parameter grid — values that override BASE_PARAMS
# Tier 1 (high impact) and Tier 2 (medium impact).
# Add more tiers carefully — grid size grows multiplicatively.
# ---------------------------------------------------------------------------
PARAM_GRID: dict = {
    # Tier 1
    "confluence_tolerance_atr_mult": [0.12, 0.18, 0.25],
    "min_rr":                        [3.0, 5.0, 7.0],
    # Tier 2
    "cancel_pct_to_tp":              [0.60, 0.75, 1.0],
    "swing_n":                       [1, 2],
    "manip_leg_timeframe":           ["1m", "5m"],
}

# Sensitivity check settings
SENSITIVITY_CFG = dict(
    perturbation_pct=0.15,
    max_degradation_pct=30.0,
    min_base_metric=0.05,   # minimum sortino to even consider a config
    min_trades=10,          # skip configs with too few trades
)

# ---------------------------------------------------------------------------
# Worker globals — set once per process by _init_worker
# ---------------------------------------------------------------------------
_g_data        = None
_g_exec_kwargs = None


def _init_worker(
    cache_1m: str, cache_5m: str, cache_bar_map: str,
    exec_kwargs: dict,
) -> None:
    """
    Runs once per worker process.  Loads market data (train split only) into
    a module-level global so it is not re-read for every task.
    """
    global _g_data, _g_exec_kwargs

    from backtest.data.loader import DataLoader
    from backtest.data.market_data import MarketData
    from backtest.ml.splits import filter_market_data

    df_1m   = pd.read_parquet(cache_1m)
    df_5m   = pd.read_parquet(cache_5m)
    bar_map = np.load(cache_bar_map)

    rth  = (df_1m.index.time >= dtime(9, 30)) & (df_1m.index.time <= dtime(16, 0))
    a1   = {c: df_1m[c].to_numpy(dtype="float64") for c in ["open","high","low","close","volume"]}
    a5   = {c: df_5m[c].to_numpy(dtype="float64") for c in ["open","high","low","close","volume"]}

    loader   = DataLoader()
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


def _run_metric(params: dict) -> float:
    """Run a single backtest and return sortino of R-multiples."""
    from backtest.runner.runner import run_backtest
    from backtest.runner.config import RunConfig
    from backtest.ml.evaluate import sortino_r
    from strategies.ict_smc import ICTSMCStrategy

    config = RunConfig(
        starting_capital=_g_exec_kwargs["starting_capital"],
        slippage_points=_g_exec_kwargs["slippage_points"],
        commission_per_contract=_g_exec_kwargs["commission_per_contract"],
        eod_exit_time=_g_exec_kwargs["eod_exit_time"],
        order_cancel_time=_g_exec_kwargs["order_cancel_time"],
        params={**params, "ml_model": None},
    )
    result = run_backtest(ICTSMCStrategy, config, _g_data)
    r_arr  = np.array(
        [t.r_multiple for t in result.trades if t.r_multiple is not None],
        dtype=float,
    )
    if len(r_arr) < SENSITIVITY_CFG["min_trades"]:
        return -999.0
    return sortino_r(r_arr)


def _task_sensitivity(args: tuple) -> tuple:
    """
    Worker task: run sensitivity check for one config.

    Returns (config_hash, base_metric, is_stable)
    """
    params, sensitivity_cfg = args
    from backtest.ml.sensitivity import check_sensitivity

    result = check_sensitivity(
        params,
        _run_metric,
        perturbation_pct=sensitivity_cfg["perturbation_pct"],
        max_degradation_pct=sensitivity_cfg["max_degradation_pct"],
    )
    is_valid = result.is_stable and result.base_metric >= sensitivity_cfg["min_base_metric"]
    return is_valid, result.base_metric, result.degradation_pct, result.worst_param


def _task_collect(params: dict) -> list:
    """
    Worker task: run a clean feature-collection backtest for validated params.
    Returns a list of serialisable trade dicts.
    """
    from backtest.runner.runner import run_backtest
    from backtest.runner.config import RunConfig
    from strategies.ict_smc import ICTSMCStrategy

    config = RunConfig(
        starting_capital=_g_exec_kwargs["starting_capital"],
        slippage_points=_g_exec_kwargs["slippage_points"],
        commission_per_contract=_g_exec_kwargs["commission_per_contract"],
        eod_exit_time=_g_exec_kwargs["eod_exit_time"],
        order_cancel_time=_g_exec_kwargs["order_cancel_time"],
        params={**params, "ml_model": None},
    )
    result = run_backtest(ICTSMCStrategy, config, _g_data)

    trades = []
    for t in result.trades:
        if not t.signal_features:
            continue
        trades.append({
            "signal_features": t.signal_features,
            "r_multiple":      t.r_multiple if t.r_multiple is not None else 0.0,
            "is_winner":       int(t.is_winner),
            "entry_bar":       t.entry_bar,
        })
    return trades


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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=== ICT/SMC ML Data Collection ===\n")

    if not CACHE_1M.exists():
        print(f"ERROR: {CACHE_1M} not found. Run run.py once to build the cache.")
        return

    # How many workers to use
    n_cpu     = os.cpu_count() or 1
    n_workers = max(1, n_cpu - 1)
    print(f"CPU cores: {n_cpu}  →  using {n_workers} worker(s)\n")

    # Build full config list from grid
    keys   = list(PARAM_GRID.keys())
    values = list(PARAM_GRID.values())
    all_combos = [
        {**BASE_PARAMS, **dict(zip(keys, combo))}
        for combo in itertools.product(*values)
    ]
    print(f"Grid size: {len(all_combos)} config(s)\n")

    # Load caches
    sens_cache   = _load_json(SENSITIVITY_CACHE)   # hash → {base_metric, is_stable, ...}
    trades_cache = _load_json(TRADES_CACHE)         # hash → True (already collected)

    # -----------------------------------------------------------------------
    # Phase 1: Sensitivity check (parallel)
    # -----------------------------------------------------------------------
    print("--- Phase 1: Sensitivity check ---")
    validated_configs = []
    to_check = []

    for params in all_combos:
        h = _hash_params(params)
        if h in sens_cache:
            cached = sens_cache[h]
            if cached["is_stable"] and cached["base_metric"] >= SENSITIVITY_CFG["min_base_metric"]:
                validated_configs.append(params)
            print(f"  [cache] {h}  metric={cached['base_metric']:.3f}  stable={cached['is_stable']}")
        else:
            to_check.append((h, params))

    if to_check:
        print(f"  Running sensitivity for {len(to_check)} new config(s) "
              f"(~{len(to_check) * 23} backtests)...")

        init_args = (
            str(CACHE_1M), str(CACHE_5M), str(CACHE_BAR_MAP),
            {k: v for k, v in BASE_EXEC.items()},
        )

        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_init_worker,
            initargs=init_args,
        ) as pool:
            future_map = {
                pool.submit(_task_sensitivity, (params, SENSITIVITY_CFG)): (h, params)
                for h, params in to_check
            }
            for fut in as_completed(future_map):
                h, params = future_map[fut]
                try:
                    is_valid, base_metric, degrad_pct, worst_param = fut.result()
                except Exception as exc:
                    print(f"  [error] {h}: {exc}")
                    is_valid, base_metric, degrad_pct, worst_param = False, -999.0, 0.0, ""

                sens_cache[h] = {
                    "is_stable":      is_valid,
                    "base_metric":    base_metric,
                    "degradation_pct": degrad_pct,
                    "worst_param":    worst_param,
                }
                _save_json(SENSITIVITY_CACHE, sens_cache)

                tag = "PASS" if is_valid else "FAIL"
                print(f"  [{tag}] {h}  metric={base_metric:.3f}  degrad={degrad_pct:.1f}%"
                      f"  worst={worst_param}")
                if is_valid:
                    validated_configs.append(params)

    print(f"\n  {len(validated_configs)} / {len(all_combos)} configs passed sensitivity.\n")

    if not validated_configs:
        print("No configs passed sensitivity. Widen the grid or lower min_base_metric.")
        return

    # Save validated config list
    _save_json(VALID_CONFIGS_OUT, [
        {k: v for k, v in p.items() if k != "ml_model" and not callable(v)}
        for p in validated_configs
    ])
    print(f"  Validated configs saved → {VALID_CONFIGS_OUT}\n")

    # -----------------------------------------------------------------------
    # Phase 2: Data collection for validated configs (parallel)
    # -----------------------------------------------------------------------
    print("--- Phase 2: Trade collection ---")
    to_collect = [
        (_hash_params(p), p)
        for p in validated_configs
        if _hash_params(p) not in trades_cache
    ]

    all_trade_rows: list[dict] = []

    # Load already-collected trades from disk if dataset exists
    if DATASET_OUT.exists() and not to_collect:
        print("  All configs already collected. Loading existing dataset.")
        df_existing = pd.read_parquet(DATASET_OUT)
        all_trade_rows = df_existing.to_dict("records")
    elif DATASET_OUT.exists() and to_collect:
        df_existing = pd.read_parquet(DATASET_OUT)
        all_trade_rows = df_existing.to_dict("records")
        print(f"  {len(validated_configs) - len(to_collect)} already cached, "
              f"{len(to_collect)} new to collect.")
    else:
        print(f"  Collecting {len(to_collect)} config(s)...")

    if to_collect:
        init_args = (
            str(CACHE_1M), str(CACHE_5M), str(CACHE_BAR_MAP),
            {k: v for k, v in BASE_EXEC.items()},
        )
        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_init_worker,
            initargs=init_args,
        ) as pool:
            future_map = {
                pool.submit(_task_collect, params): (h, params)
                for h, params in to_collect
            }
            for fut in as_completed(future_map):
                h, params = future_map[fut]
                try:
                    trades = fut.result()
                except Exception as exc:
                    print(f"  [error] {h}: {exc}")
                    trades = []

                all_trade_rows.extend(trades)
                trades_cache[h] = True
                _save_json(TRADES_CACHE, trades_cache)
                print(f"  [done] {h}  trades={len(trades)}")

    # -----------------------------------------------------------------------
    # Phase 3: Build and save dataset
    # -----------------------------------------------------------------------
    print("\n--- Phase 3: Building dataset ---")

    if not all_trade_rows:
        print("ERROR: No trades collected. Check strategy is recording signal_features.")
        return

    from backtest.data.market_data import MarketData
    from backtest.data.loader import DataLoader
    from backtest.ml.splits import filter_market_data
    from backtest.ml.dataset import build_dataset
    from backtest.runner.runner import RunResult

    # We need to reconstruct a RunResult-like object to pass to build_dataset.
    # Instead, build the DataFrame directly from the collected trade dicts.
    from collections import deque
    from backtest.ml.features import ALL_FEATURE_NAMES

    # Load data for context features (entry_bar → equity curve)
    # For multi-config collection we don't have a single equity curve, so
    # context features are approximated from per-trade order (sorted by entry_bar).
    rows = []
    history: deque = deque(maxlen=10)
    consecutive_losses = 0
    day_counts: dict = {}

    # Sort by entry_bar so rolling context is chronologically consistent
    sorted_trades = sorted(all_trade_rows, key=lambda t: t["entry_bar"])

    # Load 1m data to get dates (train split only)
    df_1m = pd.read_parquet(CACHE_1M)
    from backtest.ml.splits import split_bounds
    s_ts, e_ts = split_bounds("train")
    df_1m = df_1m[(df_1m.index >= s_ts) & (df_1m.index <= e_ts)]

    for trade in sorted_trades:
        sf         = trade["signal_features"]
        entry_bar  = trade["entry_bar"]
        r_multiple = float(trade["r_multiple"])

        try:
            ts         = df_1m.index[entry_bar] if entry_bar < len(df_1m) else None
            trade_date = ts.date() if ts is not None else None
        except (IndexError, AttributeError):
            trade_date = None

        daily_idx          = day_counts.get(trade_date, 0)
        day_counts[trade_date] = daily_idx + 1

        if len(history) >= 1:
            recent_r = list(history)
            rwr      = float(sum(r > 0 for r in recent_r) / len(recent_r))
            rex      = float(sum(recent_r) / len(recent_r))
        else:
            rwr, rex = 0.5, 0.0

        row = {**sf,
               "daily_trade_idx":        daily_idx,
               "recent_win_rate_10":     rwr,
               "recent_expectancy_r_10": rex,
               "consecutive_losses":     consecutive_losses,
               "drawdown_pct":           0.0,   # not available in multi-config mode
               "r_multiple":             r_multiple,
               "is_winner":              int(r_multiple > 0),
               "date":                   trade_date,
               "entry_bar":              entry_bar}

        for col in ALL_FEATURE_NAMES:
            row.setdefault(col, 0)

        rows.append(row)

        history.append(r_multiple)
        consecutive_losses = consecutive_losses + 1 if r_multiple <= 0 else 0

    cols = ALL_FEATURE_NAMES + ["date", "entry_bar", "r_multiple", "is_winner"]
    df   = pd.DataFrame(rows)
    for c in cols:
        df.setdefault(c, 0)
    df = df[cols].reset_index(drop=True)

    DATASET_OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DATASET_OUT, index=False)

    print(f"  Total trades: {len(df)}")
    print(f"  Win rate:     {df['is_winner'].mean():.1%}")
    print(f"  Mean R:       {df['r_multiple'].mean():.3f}")
    print(f"  Dataset saved → {DATASET_OUT}")


if __name__ == "__main__":
    main()
