"""
Multi-config data collection for the ICT/SMC ML pipeline.

Steps
-----
1. Sample N configs via Latin Hypercube Sampling (LHS) from the param space.
2. For each candidate, run a sensitivity check (parallelised, cached).
3. For every config that passes sensitivity, run a clean data-collection
   backtest (also parallelised, cached).
4. Attach config features to each trade row.
5. Merge into data/ml_dataset.parquet (appends on Round 2+).

Rounds
------
Set ROUND = 1 for the initial broad exploration.
After reviewing Round 1 feature importance/partial-dependence plots, tighten
PARAM_RANGES_V2 in backtest/ml/configs.py and re-run with ROUND = 2.
Each round appends to the existing dataset — the model trains on all rounds.

Set ROUND = 1 and delete cache files to start completely fresh.
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
SENSITIVITY_CACHE = ROOT / "cache" / "sensitivity_cache.json"
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


def _run_metric(params: dict) -> float:
    from backtest.runner.runner import run_backtest
    from backtest.runner.config import RunConfig
    from backtest.ml.evaluate import sortino_r
    from strategies.ict_smc import ICTSMCStrategy

    config = RunConfig(
        starting_capital        = _g_exec_kwargs["starting_capital"],
        slippage_points         = _g_exec_kwargs["slippage_points"],
        commission_per_contract = _g_exec_kwargs["commission_per_contract"],
        eod_exit_time           = _g_exec_kwargs["eod_exit_time"],
        order_cancel_time       = _g_exec_kwargs["order_cancel_time"],
        params                  = {**params, "ml_model": None},
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
    """Worker: sensitivity check for one config. Returns (is_valid, base_metric, degrad_pct, worst_param)."""
    params, sens_cfg, lhs_axes = args
    from backtest.ml.sensitivity import check_sensitivity

    # Only perturb params that are NOT LHS axes (those are already sampled)
    perturb = [
        p for p in [
            'tp_confluence_tolerance_atr_mult',
            'level_penetration_atr_mult',
            'po3_atr_mult',
            'po3_band_pct',
            'po3_vol_sens',
            'po3_min_manipulation_size_atr_mult',
            'min_ote_size_atr_mult',
            'tick_offset_atr_mult',
            'cancel_pct_to_tp',
        ]
        if p not in lhs_axes
    ]

    result   = check_sensitivity(
        params, _run_metric,
        perturbation_pct    = sens_cfg["perturbation_pct"],
        max_degradation_pct = sens_cfg["max_degradation_pct"],
        params_to_perturb   = perturb if perturb else None,
    )
    is_valid = result.is_stable and result.base_metric >= sens_cfg["min_base_metric"]
    return is_valid, result.base_metric, result.degradation_pct, result.worst_param


def _task_collect(params: dict) -> list:
    """Worker: clean data-collection backtest. Returns list of trade dicts."""
    from backtest.runner.runner import run_backtest
    from backtest.runner.config import RunConfig
    from strategies.ict_smc import ICTSMCStrategy

    config = RunConfig(
        starting_capital        = _g_exec_kwargs["starting_capital"],
        slippage_points         = _g_exec_kwargs["slippage_points"],
        commission_per_contract = _g_exec_kwargs["commission_per_contract"],
        eod_exit_time           = _g_exec_kwargs["eod_exit_time"],
        order_cancel_time       = _g_exec_kwargs["order_cancel_time"],
        params                  = {**params, "ml_model": None},
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
    print(f"=== ICT/SMC ML Data Collection  [Round {ROUND}] ===\n")

    if not CACHE_1M.exists():
        print(f"ERROR: {CACHE_1M} not found. Run run.py once to build the cache.")
        return

    ranges  = ROUND_RANGES.get(ROUND, ROUND_RANGES[1])
    lhs_axes = set(ranges.keys())

    n_cpu     = os.cpu_count() or 1
    n_workers = max(1, n_cpu - 1)
    print(f"CPU cores: {n_cpu}  →  using {n_workers} worker(s)")
    print(f"Sampling {N_CONFIGS} configs via LHS (Round {ROUND} ranges)\n")

    all_configs = sample_configs(N_CONFIGS, ranges, seed=LHS_SEED + ROUND - 1)

    # Load caches
    sens_cache   = _load_json(SENSITIVITY_CACHE)
    trades_cache = _load_json(TRADES_CACHE)

    init_args = (
        str(CACHE_1M), str(CACHE_5M), str(CACHE_BAR_MAP),
        {k: v for k, v in BASE_EXEC.items()},
    )

    # -----------------------------------------------------------------------
    # Phase 1: Sensitivity check
    # -----------------------------------------------------------------------
    print("--- Phase 1: Sensitivity check ---")
    validated_configs: list[dict] = []
    to_check: list[tuple] = []

    for params in all_configs:
        h = _hash_params(params)
        if h in sens_cache:
            cached = sens_cache[h]
            if cached["is_stable"] and cached["base_metric"] >= SENSITIVITY_CFG["min_base_metric"]:
                validated_configs.append(params)
            print(f"  [cache] {h}  metric={cached['base_metric']:.3f}  stable={cached['is_stable']}")
        else:
            to_check.append((h, params))

    if to_check:
        print(f"  Running sensitivity for {len(to_check)} new config(s)...")
        with ProcessPoolExecutor(
            max_workers  = n_workers,
            initializer  = _init_worker,
            initargs     = init_args,
        ) as pool:
            future_map = {
                pool.submit(_task_sensitivity, (params, SENSITIVITY_CFG, lhs_axes)): (h, params)
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
                    "is_stable":       is_valid,
                    "base_metric":     base_metric,
                    "degradation_pct": degrad_pct,
                    "worst_param":     worst_param,
                    "round":           ROUND,
                }
                _save_json(SENSITIVITY_CACHE, sens_cache)

                tag = "PASS" if is_valid else "FAIL"
                print(f"  [{tag}] {h}  metric={base_metric:.3f}"
                      f"  degrad={degrad_pct:.1f}%  worst={worst_param}")
                if is_valid:
                    validated_configs.append(params)

    print(f"\n  {len(validated_configs)} / {len(all_configs)} configs passed sensitivity.\n")

    if not validated_configs:
        print("No configs passed. Widen ranges or lower min_base_metric.")
        return

    # Save validated config list (append across rounds)
    existing_valid = _load_json(VALID_CONFIGS_OUT) if VALID_CONFIGS_OUT.exists() else []
    existing_hashes = {_hash_params(c) for c in existing_valid}
    new_valid = [
        {k: v for k, v in p.items() if k != "ml_model" and not callable(v)}
        for p in validated_configs
        if _hash_params(p) not in existing_hashes
    ]
    _save_json(VALID_CONFIGS_OUT, existing_valid + new_valid)
    print(f"  Validated configs saved → {VALID_CONFIGS_OUT}  "
          f"(+{len(new_valid)} new, {len(existing_valid)} existing)\n")

    # -----------------------------------------------------------------------
    # Phase 2: Trade collection
    # -----------------------------------------------------------------------
    print("--- Phase 2: Trade collection ---")
    to_collect = [
        (_hash_params(p), p)
        for p in validated_configs
        if _hash_params(p) not in trades_cache
    ]

    new_trade_rows: list[dict] = []

    if not to_collect:
        print("  All validated configs already collected.")
    else:
        print(f"  Collecting {len(to_collect)} config(s)...")
        with ProcessPoolExecutor(
            max_workers  = n_workers,
            initializer  = _init_worker,
            initargs     = init_args,
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

                cfg_feat = normalize_config(params, ranges)
                idx      = next(
                    (i for i, c in enumerate(validated_configs)
                     if _hash_params(c) == h), -1
                )
                for t in trades:
                    t["config_hash"] = h
                    t["config_idx"]  = idx
                    t["round"]       = ROUND
                    t["config_features"] = cfg_feat

                new_trade_rows.extend(trades)
                trades_cache[h] = {"round": ROUND, "n_trades": len(trades)}
                _save_json(TRADES_CACHE, trades_cache)
                print(f"  [done] {h}  trades={len(trades)}")

    # -----------------------------------------------------------------------
    # Phase 3: Build dataset and save (append to existing on Round 2+)
    # -----------------------------------------------------------------------
    print("\n--- Phase 3: Building dataset ---")

    if not new_trade_rows and ROUND > 1 and DATASET_OUT.exists():
        print("  No new trades. Existing dataset unchanged.")
        df = pd.read_parquet(DATASET_OUT)
        print(f"  Existing rows: {len(df)}")
        return

    if not new_trade_rows:
        print("ERROR: No trades collected.")
        return

    # Load existing dataset if appending
    existing_df = pd.read_parquet(DATASET_OUT) if DATASET_OUT.exists() and ROUND > 1 else None

    # Build context features from new rows
    from collections import deque
    rows: list[dict]  = []
    history: deque    = deque(maxlen=10)
    consecutive_losses = 0
    day_counts: dict  = {}

    df_1m = pd.read_parquet(CACHE_1M)
    from backtest.ml.splits import split_bounds
    s_ts, e_ts = split_bounds("train")
    df_1m = df_1m[(df_1m.index >= s_ts) & (df_1m.index <= e_ts)]

    sorted_trades = sorted(new_trade_rows, key=lambda t: t["entry_bar"])

    for trade in sorted_trades:
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
            row.setdefault(col, 0)

        rows.append(row)
        history.append(r_multiple)
        consecutive_losses = consecutive_losses + 1 if r_multiple <= 0 else 0

    meta_cols = ["date", "entry_bar", "r_multiple", "is_winner",
                 "config_hash", "config_idx", "round"]
    cols      = ALL_FEATURE_NAMES + meta_cols
    df_new    = pd.DataFrame(rows)
    for c in cols:
        df_new.setdefault(c, 0)
    df_new = df_new[cols].reset_index(drop=True)

    if existing_df is not None:
        # Align columns before concat
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
