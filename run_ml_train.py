"""
ML training entry point for the ICT/SMC strategy.

Steps:
  1. Run a full backtest with feature recording (no ML model — collects all signals).
  2. Build a training dataset from the run result.
  3. Run walk-forward training and print OOS metrics.
  4. Save the final model to models/ict_smc.pkl.

Usage:
  python run_ml_train.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from datetime import time as dtime

# ---------------------------------------------------------------------------
# Make sure the project root is on the path
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from backtest.data.loader import DataLoader
from backtest.data.market_data import MarketData
from backtest.runner.config import RunConfig
from backtest.runner.runner import run_backtest
from backtest.ml.dataset import build_dataset
from backtest.ml.train import WalkForwardTrainer, WalkForwardConfig
from strategies.ict_smc import ICTSMCStrategy


# ---------------------------------------------------------------------------
# Date range — set to None for full history, or "YYYY-MM-DD" to restrict
# ---------------------------------------------------------------------------
DATE_FROM = "2019-01-01"   # start of training data
DATE_TO   = None           # None = up to latest available bar

# ---------------------------------------------------------------------------
# Configuration — mirrors run.py but with ml_model=None for data collection
# ---------------------------------------------------------------------------

CONFIG = RunConfig(
    starting_capital=100_000,
    slippage_points=0.25,
    commission_per_contract=4.50,
    eod_exit_time=dtime(17, 0),
    order_cancel_time=dtime(11, 0),
    params={
        # Core ICT/SMC params (keep in sync with run.py)
        "contracts":                     1,
        "swing_n":                       1,
        "cisd_min_series_candles":       2,
        "cisd_min_body_ratio":           0.5,
        "rb_min_wick_ratio":             0.3,
        "confluence_tolerance_atr_mult": 0.18,
        "level_penetration_atr_mult":    0.5,
        "min_rr":                        5.0,
        "tick_offset_atr_mult":          0.035,
        "order_expiry_bars":             10,
        "session_level_validity_days":   2,
        "po3_lookback":                  6,
        "po3_atr_mult":                  0.95,
        "po3_atr_len":                   14,
        "po3_band_pct":                  0.3,
        "po3_vol_sens":                  1.0,
        "po3_max_r2":                    0.4,
        "po3_min_dir_changes":           2,
        "po3_min_candles":               3,
        "po3_max_accum_gap_bars":        10,
        "po3_min_manipulation_size_atr_mult": 0.0,
        "max_trades_per_day":            2,
        "manip_leg_timeframe":           "5m",
        "manip_leg_swing_depth":         1,
        "allowed_setup_types":           ["OTE", "STDV", "SESSION_OTE"],
        "cancel_pct_to_tp":              0.75,
        "min_ote_size_atr_mult":         0.0,
        "max_ote_per_session":           1,
        "max_stdv_per_session":          1,
        "max_session_ote_per_session":   1,
        # No ml_model — feature collection mode
        "ml_model": None,
    },
)

WALK_FORWARD_CONFIG = WalkForwardConfig(
    train_months=24,
    test_months=3,
    embargo_months=1,
    metric='sortino',
    min_train_trades=30,
    min_test_trades=5,
    threshold_search=True,
    min_take_rate=0.20,
)

MODEL_PATH = ROOT / "models" / "ict_smc.pkl"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=== ICT/SMC ML Training Pipeline ===\n")

    # 1. Load data (uses parquet cache if available, same as run.py)
    print("Loading market data...")
    CACHE_1M      = ROOT / "data" / "NQ_1m.parquet"
    CACHE_5M      = ROOT / "data" / "NQ_5m.parquet"
    CACHE_BAR_MAP = ROOT / "data" / "NQ_bar_map.npy"
    loader = DataLoader()

    if CACHE_1M.exists() and CACHE_5M.exists() and CACHE_BAR_MAP.exists():
        df_1m   = pd.read_parquet(CACHE_1M)
        df_5m   = pd.read_parquet(CACHE_5M)
        bar_map = np.load(CACHE_BAR_MAP)
        arrays_1m = {c: df_1m[c].to_numpy(dtype="float64") for c in ["open","high","low","close","volume"]}
        arrays_5m = {c: df_5m[c].to_numpy(dtype="float64") for c in ["open","high","low","close","volume"]}
        rth_mask      = (df_1m.index.time >= dtime(9, 30)) & (df_1m.index.time <= dtime(16, 0))
        trading_dates = sorted(set(df_1m[rth_mask].index.date))
        data = MarketData(
            df_1m=df_1m, df_5m=df_5m,
            open_1m=arrays_1m["open"], high_1m=arrays_1m["high"],
            low_1m=arrays_1m["low"],   close_1m=arrays_1m["close"],
            volume_1m=arrays_1m["volume"],
            open_5m=arrays_5m["open"], high_5m=arrays_5m["high"],
            low_5m=arrays_5m["low"],   close_5m=arrays_5m["close"],
            volume_5m=arrays_5m["volume"],
            bar_map=bar_map, trading_dates=trading_dates,
        )
    else:
        data = loader.load(path_1m="NQ_1m.txt", path_5m="NQ_5m.txt")

    print(f"  1m bars: {len(data.df_1m):,} | 5m bars: {len(data.df_5m):,}")
    print(f"  {data.trading_dates[0]}  →  {data.trading_dates[-1]}\n")

    # Apply date range filter (set DATE_FROM / DATE_TO at top of file)
    if DATE_FROM is not None or DATE_TO is not None:
        start_ts = pd.Timestamp(DATE_FROM, tz="America/New_York") if DATE_FROM else data.df_1m.index[0]
        end_ts   = (pd.Timestamp(DATE_TO, tz="America/New_York") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
                    if DATE_TO else data.df_1m.index[-1])
        mask_1m = (data.df_1m.index >= start_ts) & (data.df_1m.index <= end_ts)
        mask_5m = (data.df_5m.index >= start_ts) & (data.df_5m.index <= end_ts)
        df_1m_f = data.df_1m[mask_1m]
        df_5m_f = data.df_5m[mask_5m]
        rth_f   = (df_1m_f.index.time >= dtime(9, 30)) & (df_1m_f.index.time <= dtime(16, 0))
        a1 = {c: df_1m_f[c].to_numpy(dtype="float64") for c in ["open","high","low","close","volume"]}
        a5 = {c: df_5m_f[c].to_numpy(dtype="float64") for c in ["open","high","low","close","volume"]}
        data = MarketData(
            df_1m=df_1m_f, df_5m=df_5m_f,
            open_1m=a1["open"], high_1m=a1["high"], low_1m=a1["low"],
            close_1m=a1["close"], volume_1m=a1["volume"],
            open_5m=a5["open"], high_5m=a5["high"], low_5m=a5["low"],
            close_5m=a5["close"], volume_5m=a5["volume"],
            bar_map=loader._build_bar_map(df_1m_f, df_5m_f),
            trading_dates=sorted(set(df_1m_f[rth_f].index.date)),
        )
        print(f"  Date filter: {DATE_FROM or 'start'} → {DATE_TO or 'end'}")
        print(f"  {len(data.df_1m):,} 1m bars | {len(data.trading_dates)} trading days\n")

    # 2. Run backtest (feature collection — no ML filter)
    print("Running backtest (feature collection mode)...")
    result = run_backtest(ICTSMCStrategy, CONFIG, data)
    print(f"  Trades: {result.n_trades}")
    n_with_features = sum(1 for t in result.trades if t.signal_features)
    print(f"  Trades with signal features: {n_with_features}\n")

    if n_with_features < 20:
        print("ERROR: Too few trades with signal features. Check strategy is recording features.")
        return

    # 3. Build dataset
    print("Building training dataset...")
    df = build_dataset(result, data)
    print(f"  Dataset shape: {df.shape}")
    print(f"  Date range: {df['date'].min()} → {df['date'].max()}")
    print(f"  Win rate: {df['is_winner'].mean():.1%}")
    print(f"  Mean R: {df['r_multiple'].mean():.3f}")
    print()

    # 4. Walk-forward training
    print("Running walk-forward training...")
    trainer = WalkForwardTrainer(WALK_FORWARD_CONFIG)
    wf_result = trainer.fit(df)

    print(f"\n=== Walk-Forward Results ===")
    print(f"  Folds completed:        {wf_result.summary['n_folds']}")
    print(f"  Total OOS trades:       {wf_result.summary['total_oos_trades']}")
    print(f"  OOS take rate:          {wf_result.summary['oos_take_rate']:.1%}")
    print()
    print(f"  --- ALL trades (baseline) ---")
    print(f"  Sortino (R):            {wf_result.summary['oos_sortino_all']:.3f}")
    print(f"  Profit factor:          {wf_result.summary['oos_pf_all']:.3f}")
    print(f"  Win rate:               {wf_result.summary['oos_win_rate_all']:.1%}")
    print(f"  Expectancy (R):         {wf_result.summary['oos_expectancy_r_all']:.3f}")
    print()
    print(f"  --- TAKEN trades (ML filtered) ---")
    print(f"  Sortino (R):            {wf_result.summary['oos_sortino_taken']:.3f}")
    print(f"  Profit factor:          {wf_result.summary['oos_pf_taken']:.3f}")
    print(f"  Win rate:               {wf_result.summary['oos_win_rate_taken']:.1%}")
    print(f"  Expectancy (R):         {wf_result.summary['oos_expectancy_r_taken']:.3f}")
    print(f"  Final threshold:        {wf_result.summary['final_threshold']:.4f}")
    print()

    # Per-fold table
    print("  Per-fold OOS Sortino:")
    print(f"  {'Fold':>4}  {'Train':>12}  {'Test':>12}  {'N':>5}  {'SortAll':>8}  {'SortTaken':>10}  {'TakeRate':>9}")
    for f in wf_result.folds:
        print(f"  {f.fold_idx:>4}  {str(f.train_start):>12}  {str(f.test_start):>12}"
              f"  {f.n_test:>5}  {f.oos_sortino_all:>8.3f}  {f.oos_sortino_taken:>10.3f}"
              f"  {f.oos_take_rate:>8.1%}")
    print()

    # Feature importance
    print("  Top-10 feature importances:")
    imp = wf_result.final_model.feature_importance().head(10)
    for feat, val in imp.items():
        print(f"    {feat:<35} {val:.4f}")
    print()

    # 5. Save model
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    wf_result.final_model.save(MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == '__main__':
    main()
