"""
ML training entry point for the ICT/SMC strategy.

Prerequisites
-------------
Run  run_ml_collect.py  first to produce  data/ml_dataset.parquet.
If the dataset is missing, this script falls back to a single-config
feature-collection backtest (slower, less data).

Steps
-----
1. Load the training dataset (train split only — no validation/test leakage).
2. Walk-forward training inside the train split.
3. Tune the final skip threshold on the validation split.
4. Print both train (OOS walk-forward) and validation metrics.
5. Save model to  models/ict_smc.pkl.

Data splits (defined in backtest/ml/splits.py)
-----------------------------------------------
  train       2019-01-01 → 2022-12-31   walk-forward lives here
  validation  2023-01-01 → 2023-12-31   threshold tuning only
  test1/test2                            NEVER touched here
"""
from __future__ import annotations

import sys
from pathlib import Path
from datetime import time as dtime

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from backtest.ml.splits import filter_df, SPLITS
from backtest.ml.train import WalkForwardTrainer, WalkForwardConfig
from backtest.ml.evaluate import sortino_r, profit_factor_r, search_threshold

DATASET_PATH = ROOT / "data" / "ml_dataset.parquet"
MODEL_PATH   = ROOT / "models" / "ict_smc.pkl"

WALK_FORWARD_CONFIG = WalkForwardConfig(
    train_months=24,
    test_months=3,
    embargo_months=1,
    metric="sortino",
    min_train_trades=30,
    min_test_trades=5,
    threshold_search=True,
    min_take_rate=0.20,
)


# ---------------------------------------------------------------------------
# Fallback: single-config feature collection (used if dataset not found)
# ---------------------------------------------------------------------------

def _collect_single_run() -> pd.DataFrame:
    """Run one backtest (train split only) to collect features."""
    from backtest.data.loader import DataLoader
    from backtest.data.market_data import MarketData
    from backtest.runner.config import RunConfig
    from backtest.runner.runner import run_backtest
    from backtest.ml.dataset import build_dataset
    from backtest.ml.splits import filter_market_data
    from strategies.ict_smc import ICTSMCStrategy

    print("  Dataset not found — running single-config feature collection...")

    CACHE_1M      = ROOT / "data" / "NQ_1m.parquet"
    CACHE_5M      = ROOT / "data" / "NQ_5m.parquet"
    CACHE_BAR_MAP = ROOT / "data" / "NQ_bar_map.npy"

    if not CACHE_1M.exists():
        raise FileNotFoundError(
            f"{CACHE_1M} not found. Run run.py once to build the data cache, "
            "or run run_ml_collect.py to generate a richer dataset."
        )

    loader  = DataLoader()
    df_1m   = pd.read_parquet(CACHE_1M)
    df_5m   = pd.read_parquet(CACHE_5M)
    bar_map = np.load(CACHE_BAR_MAP)

    rth  = (df_1m.index.time >= dtime(9, 30)) & (df_1m.index.time <= dtime(16, 0))
    a1   = {c: df_1m[c].to_numpy(dtype="float64") for c in ["open","high","low","close","volume"]}
    a5   = {c: df_5m[c].to_numpy(dtype="float64") for c in ["open","high","low","close","volume"]}
    full_data = MarketData(
        df_1m=df_1m, df_5m=df_5m,
        open_1m=a1["open"],  high_1m=a1["high"],  low_1m=a1["low"],
        close_1m=a1["close"], volume_1m=a1["volume"],
        open_5m=a5["open"],  high_5m=a5["high"],  low_5m=a5["low"],
        close_5m=a5["close"], volume_5m=a5["volume"],
        bar_map=bar_map,
        trading_dates=sorted(set(df_1m[rth].index.date)),
    )
    data = filter_market_data(full_data, "train", loader)

    config = RunConfig(
        starting_capital=100_000,
        slippage_points=0.25,
        commission_per_contract=4.50,
        eod_exit_time=dtime(17, 0),
        order_cancel_time=dtime(11, 0),
        params={
            "contracts": 1, "swing_n": 1,
            "cisd_min_series_candles": 2, "cisd_min_body_ratio": 0.5,
            "rb_min_wick_ratio": 0.3,
            "confluence_tolerance_atr_mult": 0.18,
            "tp_confluence_tolerance_atr_mult": 0.18,
            "level_penetration_atr_mult": 0.5,
            "min_rr": 5.0, "tick_offset_atr_mult": 0.035,
            "order_expiry_bars": 10, "session_level_validity_days": 2,
            "cancel_pct_to_tp": 0.75, "min_ote_size_atr_mult": 0.0,
            "max_ote_per_session": 1, "max_stdv_per_session": 1,
            "max_session_ote_per_session": 1,
            "po3_lookback": 6, "po3_atr_mult": 0.95, "po3_atr_len": 14,
            "po3_band_pct": 0.3, "po3_vol_sens": 1.0, "po3_max_r2": 0.4,
            "po3_min_dir_changes": 2, "po3_min_candles": 3,
            "po3_max_accum_gap_bars": 10, "po3_min_manipulation_size_atr_mult": 0.0,
            "manip_leg_timeframe": "5m", "manip_leg_swing_depth": 1,
            "max_trades_per_day": 2,
            "allowed_setup_types": ["OTE", "STDV", "SESSION_OTE"],
            "ml_model": None,
        },
    )

    result = run_backtest(ICTSMCStrategy, config, data)
    print(f"  Collected {result.n_trades} trades "
          f"({sum(1 for t in result.trades if t.signal_features)} with features)")
    return build_dataset(result, data)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=== ICT/SMC ML Training ===\n")

    # 1. Load dataset
    if DATASET_PATH.exists():
        print(f"Loading dataset from {DATASET_PATH}...")
        df_full = pd.read_parquet(DATASET_PATH)
        print(f"  Total rows: {len(df_full):,}\n")
    else:
        df_full = _collect_single_run()
        print()

    # 2. Split into train and validation — NEVER touch test here
    df_train = filter_df(df_full, "train")
    df_val   = filter_df(df_full, "validation")

    print(f"Train split:      {len(df_train):,} trades "
          f"({SPLITS['train'][0]} → {SPLITS['train'][1]})")
    print(f"Validation split: {len(df_val):,} trades "
          f"({SPLITS['validation'][0]} → {SPLITS['validation'][1]})")
    print()

    if len(df_train) < 20:
        print("ERROR: Too few training trades. "
              "Run run_ml_collect.py or check your date ranges.")
        return

    # 3. Walk-forward training on train split
    print("Running walk-forward training (train split only)...")
    trainer   = WalkForwardTrainer(WALK_FORWARD_CONFIG)
    wf_result = trainer.fit(df_train)

    print(f"\n=== Walk-Forward OOS Results (train split) ===")
    print(f"  Folds:              {wf_result.summary['n_folds']}")
    print(f"  Total OOS trades:   {wf_result.summary['total_oos_trades']}")
    print(f"  OOS take rate:      {wf_result.summary['oos_take_rate']:.1%}")
    print()
    print(f"  --- All trades (baseline) ---")
    print(f"  Sortino:            {wf_result.summary['oos_sortino_all']:.3f}")
    print(f"  Profit factor:      {wf_result.summary['oos_pf_all']:.3f}")
    print(f"  Win rate:           {wf_result.summary['oos_win_rate_all']:.1%}")
    print(f"  Expectancy R:       {wf_result.summary['oos_expectancy_r_all']:.3f}")
    print()
    print(f"  --- ML-filtered trades ---")
    print(f"  Sortino:            {wf_result.summary['oos_sortino_taken']:.3f}")
    print(f"  Profit factor:      {wf_result.summary['oos_pf_taken']:.3f}")
    print(f"  Win rate:           {wf_result.summary['oos_win_rate_taken']:.1%}")
    print(f"  Expectancy R:       {wf_result.summary['oos_expectancy_r_taken']:.3f}")
    print(f"  WF threshold:       {wf_result.summary['final_threshold']:.4f}")
    print()

    # Per-fold table
    print(f"  {'Fold':>4}  {'Train':>12}  {'Test':>12}  {'N':>5}  "
          f"{'SortAll':>8}  {'SortTaken':>10}  {'TakeRate':>9}")
    for f in wf_result.folds:
        print(f"  {f.fold_idx:>4}  {str(f.train_start):>12}  {str(f.test_start):>12}"
              f"  {f.n_test:>5}  {f.oos_sortino_all:>8.3f}  {f.oos_sortino_taken:>10.3f}"
              f"  {f.oos_take_rate:>8.1%}")
    print()

    # 4. Tune threshold on validation split (separate from train — no leakage)
    if len(df_val) >= 5:
        print("Tuning skip threshold on validation split...")
        from backtest.ml.features import ALL_FEATURE_NAMES
        X_val  = df_val[ALL_FEATURE_NAMES]
        y_val  = df_val["r_multiple"].values
        pred_v = wf_result.final_model.predict_r(X_val)
        val_thresh, val_stats = search_threshold(
            pred_v, y_val,
            metric=WALK_FORWARD_CONFIG.metric,
            min_take_rate=WALK_FORWARD_CONFIG.min_take_rate,
        )
        wf_result.final_model.threshold = val_thresh

        r_all   = np.array(y_val, dtype=float)
        mask    = pred_v >= val_thresh
        r_taken = r_all[mask]

        print(f"\n=== Validation Split Metrics ===")
        print(f"  Threshold (from validation): {val_thresh:.4f}")
        print(f"  Take rate:                   {mask.mean():.1%}")
        print()
        print(f"  --- All trades ---")
        print(f"  Sortino:      {sortino_r(r_all):.3f}")
        print(f"  Profit factor:{profit_factor_r(r_all):.3f}")
        print(f"  Win rate:     {float(np.mean(r_all > 0)):.1%}")
        print()
        print(f"  --- ML-filtered trades ---")
        print(f"  Sortino:      {sortino_r(r_taken):.3f}")
        print(f"  Profit factor:{profit_factor_r(r_taken):.3f}")
        print(f"  Win rate:     {float(np.mean(r_taken > 0)) if len(r_taken) else 0.0:.1%}")
        print()
    else:
        print(f"  Validation split has too few trades ({len(df_val)}). "
              "Using walk-forward threshold.")
        val_thresh = wf_result.final_threshold

    # Feature importance
    print("  Top-10 feature importances:")
    imp = wf_result.final_model.feature_importance().head(10)
    for feat, val in imp.items():
        print(f"    {feat:<35} {val:.4f}")
    print()

    # 5. Save model
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    wf_result.final_model.save(MODEL_PATH)
    print(f"Model saved → {MODEL_PATH}")
    print(f"Skip threshold: {wf_result.final_model.threshold:.4f}")


if __name__ == "__main__":
    main()
