"""
Run this once before starting the sweep to build md_cache.pkl from local data files.
Usage: python build_md_cache.py
"""
import pickle, numpy as np, pandas as pd
from datetime import time as dtime

print("Loading NQ data (2019+)...")
df_1m_full = pd.read_parquet("data/NQ_1m.parquet")
df_5m_full = pd.read_parquet("data/NQ_5m.parquet")

start  = pd.Timestamp("2019-01-01", tz=df_1m_full.index.tz)
df_1m  = df_1m_full[df_1m_full.index >= start].copy()
df_5m  = df_5m_full[df_5m_full.index >= start].copy()

bar_map_dummy = np.zeros(len(df_1m), dtype=np.int32)  # AnchoredMeanReversion doesn't use bar_map

from backtest.data.market_data import MarketData
a1 = {c: df_1m[c].to_numpy(np.float64) for c in ["open","high","low","close","volume"]}
a5 = {c: df_5m[c].to_numpy(np.float64) for c in ["open","high","low","close","volume"]}

rth           = (df_1m.index.time >= dtime(9,30)) & (df_1m.index.time <= dtime(16,0))
trading_dates = sorted(set(df_1m[rth].index.date))
n_days        = len(trading_dates)

md = MarketData(
    df_1m=df_1m, df_5m=df_5m,
    open_1m=a1["open"], high_1m=a1["high"], low_1m=a1["low"],
    close_1m=a1["close"], volume_1m=a1["volume"],
    open_5m=a5["open"], high_5m=a5["high"], low_5m=a5["low"],
    close_5m=a5["close"], volume_5m=a5["volume"],
    bar_map=bar_map_dummy, trading_dates=trading_dates,
)

with open("md_cache.pkl", "wb") as f:
    pickle.dump({"md": md, "n_days": n_days, "trading_dates": trading_dates}, f)

print(f"md_cache.pkl built: {n_days} days, {trading_dates[0]} to {trading_dates[-1]}, {len(df_1m):,} 1m bars")
