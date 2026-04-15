"""
Minimal run: backtest only, saves trade log, prints summary.
Used to verify that optimisations produce identical trades.
"""
import os
import time
import numpy as np
import pandas as pd
from datetime import time as dtime
from backtest.data.loader import DataLoader
from backtest.runner.runner import run_backtest
from backtest.runner.config import RunConfig
from backtest.performance.trade_log import save_trade_log
from backtest.data.market_data import MarketData
from strategies.ict_smc import ICTSMCStrategy

CACHE_1M      = "data/NQ_1m.parquet"
CACHE_5M      = "data/NQ_5m.parquet"
CACHE_BAR_MAP = "data/NQ_bar_map.npy"
DATE_FROM     = "2024-01-01"
OUT_CSV       = "trade_logs/ICTSMCStrategy_verify.csv"

_t_total = time.perf_counter()
loader = DataLoader()
df_1m   = pd.read_parquet(CACHE_1M)
df_5m   = pd.read_parquet(CACHE_5M)
bar_map = np.load(CACHE_BAR_MAP)

arrays_1m = {col: df_1m[col].to_numpy(dtype="float64") for col in ["open","high","low","close","volume"]}
arrays_5m = {col: df_5m[col].to_numpy(dtype="float64") for col in ["open","high","low","close","volume"]}
rth_mask  = (df_1m.index.time >= dtime(9, 30)) & (df_1m.index.time <= dtime(16, 0))
trading_dates = sorted(set(df_1m[rth_mask].index.date))

data = MarketData(
    df_1m=df_1m, df_5m=df_5m,
    open_1m=arrays_1m["open"], high_1m=arrays_1m["high"],
    low_1m=arrays_1m["low"],  close_1m=arrays_1m["close"],
    volume_1m=arrays_1m["volume"],
    open_5m=arrays_5m["open"], high_5m=arrays_5m["high"],
    low_5m=arrays_5m["low"],  close_5m=arrays_5m["close"],
    volume_5m=arrays_5m["volume"],
    bar_map=bar_map, trading_dates=trading_dates,
)

start_ts = pd.Timestamp(DATE_FROM, tz="America/New_York")
mask_1m = data.df_1m.index >= start_ts
mask_5m = data.df_5m.index >= start_ts
df_1m_f = data.df_1m[mask_1m]
df_5m_f = data.df_5m[mask_5m]
rth_mask_f    = (df_1m_f.index.time >= dtime(9, 30)) & (df_1m_f.index.time <= dtime(16, 0))
trading_dates_f = sorted(set(df_1m_f[rth_mask_f].index.date))
arrays_1m_f = {col: df_1m_f[col].to_numpy(dtype="float64") for col in ["open","high","low","close","volume"]}
arrays_5m_f = {col: df_5m_f[col].to_numpy(dtype="float64") for col in ["open","high","low","close","volume"]}
bar_map_f = loader._build_bar_map(df_1m_f, df_5m_f)
data = MarketData(
    df_1m=df_1m_f, df_5m=df_5m_f,
    open_1m=arrays_1m_f["open"], high_1m=arrays_1m_f["high"],
    low_1m=arrays_1m_f["low"],  close_1m=arrays_1m_f["close"],
    volume_1m=arrays_1m_f["volume"],
    open_5m=arrays_5m_f["open"], high_5m=arrays_5m_f["high"],
    low_5m=arrays_5m_f["low"],  close_5m=arrays_5m_f["close"],
    volume_5m=arrays_5m_f["volume"],
    bar_map=bar_map_f, trading_dates=trading_dates_f,
)

config = RunConfig(
    starting_capital=100_000,
    slippage_points=0.5,
    commission_per_contract=4.50,
    eod_exit_time=dtime(11, 0),
    params={
        "contracts": 1, "swing_n": 1,
        "cisd_min_series_candles": 2, "cisd_min_body_ratio": 0.5,
        "rb_min_wick_ratio": 0.3, "confluence_tolerance_atr_mult": 0.18,
        "level_penetration_atr_mult": 0.5, "min_rr": 5.0,
        "tick_offset_atr_mult": 0.035, "order_expiry_bars": 10,
        "session_level_validity_days": 2,
        "po3_lookback": 6, "po3_atr_mult": 0.95, "po3_atr_len": 14,
        "po3_band_pct": 0.3, "po3_vol_sens": 1.0, "po3_max_r2": 0.4,
        "po3_min_dir_changes": 2, "po3_min_candles": 3,
        "po3_max_accum_gap_bars": 10, "po3_min_manipulation_size_atr_mult": 0.0,
        "max_trades_per_day": 2,
    },
)

t0 = time.perf_counter()
print("Running backtest...")
result = run_backtest(ICTSMCStrategy, config, data)
elapsed = time.perf_counter() - t0
result.print_summary()
print(f"Backtest time: {elapsed:.2f}s")
save_trade_log(result, data, OUT_CSV)
print(f"Trade log saved -> {OUT_CSV}")
print(f"Total time: {time.perf_counter() - _t_total:.1f}s")
