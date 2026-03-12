import os
import pandas as pd
from backtest.data.loader import DataLoader
from backtest.runner.runner import run_backtest
from backtest.runner.config import RunConfig
from strategies.dummy import DummyLongStrategy
from datetime import time

# ── Load Data (cached) ─────────────────────────────────────────────────────
CACHE_1M = "data/NQ_1m.parquet"
CACHE_5M = "data/NQ_5m.parquet"

loader = DataLoader()

if os.path.exists(CACHE_1M) and os.path.exists(CACHE_5M):
    print("Loading from cache...")
    df_1m = pd.read_parquet(CACHE_1M)
    df_5m = pd.read_parquet(CACHE_5M)
    data = loader.build_market_data(df_1m, df_5m)
else:
    print("Loading from raw files (first time — building cache)...")
    data = loader.load(path_1m="NQ_1m.txt", path_5m="NQ_5m.txt")
    data.df_1m.to_parquet(CACHE_1M)
    data.df_5m.to_parquet(CACHE_5M)
    print("Saved to parquet — future loads will be instant.")

print(f"Loaded {len(data.df_1m):,} 1m bars | {len(data.df_5m):,} 5m bars")
print(f"Date range: {data.trading_dates[0]} -> {data.trading_dates[-1]}")
print(f"Trading days: {len(data.trading_dates):,}\n")

# ── Config ─────────────────────────────────────────────────────────────────
config = RunConfig(
    starting_capital=100_000,
    slippage_points=0.25,
    commission_per_contract=4.50,
    eod_exit_time=time(15, 30),
    params={
        "entry_every": 30,
        "sl_offset": 20.0,
        "tp_offset": 40.0,
        "contracts": 1,
    },
)

# ── Run ────────────────────────────────────────────────────────────────────
print("Running backtest...")
result = run_backtest(DummyLongStrategy, config, data)

# ── Output ─────────────────────────────────────────────────────────────────
result.print_summary()
result.print_trades(max_trades=30)