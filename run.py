import os
import numpy as np
import pandas as pd
from backtest.data.loader import DataLoader
from backtest.runner.runner import run_backtest
from backtest.runner.config import RunConfig
from backtest.performance.engine import PerformanceEngine
from backtest.performance.tearsheet import TearsheetRenderer
from strategies.dummy import DummyLongStrategy
from datetime import time

# ── Config ─────────────────────────────────────────────────────────────────
AUTO_OPEN_TEARSHEET = True

# ── Load Data (cached) ─────────────────────────────────────────────────────
CACHE_1M      = "data/NQ_1m.parquet"
CACHE_5M      = "data/NQ_5m.parquet"
CACHE_BAR_MAP = "data/NQ_bar_map.npy"

loader = DataLoader()

if os.path.exists(CACHE_1M) and os.path.exists(CACHE_5M) and os.path.exists(CACHE_BAR_MAP):
    print("Loading from cache...")
    df_1m   = pd.read_parquet(CACHE_1M)
    df_5m   = pd.read_parquet(CACHE_5M)
    bar_map = np.load(CACHE_BAR_MAP)

    from backtest.data.market_data import MarketData
    arrays_1m = {col: df_1m[col].to_numpy(dtype="float64") for col in ["open","high","low","close","volume"]}
    arrays_5m = {col: df_5m[col].to_numpy(dtype="float64") for col in ["open","high","low","close","volume"]}
    trading_dates = sorted(ts.date() for ts in df_1m.index.normalize().unique())

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

else:
    print("Loading from raw files (first time — building cache)...")
    data = loader.load(path_1m="NQ_1m.txt", path_5m="NQ_5m.txt")
    data.df_1m.to_parquet(CACHE_1M)
    data.df_5m.to_parquet(CACHE_5M)
    np.save(CACHE_BAR_MAP, data.bar_map)
    print("Cache saved — future loads will be near-instant.")

print(f"Loaded {len(data.df_1m):,} 1m bars | {len(data.df_5m):,} 5m bars")
print(f"Date range: {data.trading_dates[0]} -> {data.trading_dates[-1]}")
print(f"Trading days: {len(data.trading_dates):,}\n")

# ── Run Config ─────────────────────────────────────────────────────────────
# Slippage: 0.5pt per side is a realistic assumption for NQ.
# NQ bid/ask spread = 0.25pt. Add ~0.25pt of market impact / queue position
# on a MARKET order = 0.5pt total. Use 0.25pt if you're always limit-filling.
# At 0.94pt slippage the dummy strategy's edge goes to zero — good stress test.
config = RunConfig(
    starting_capital=100_000,
    slippage_points=0.5,
    commission_per_contract=4.50,   # NinjaTrader / Tradovate approx round-trip halved
    eod_exit_time=time(15, 30),
    params={
        "entry_every": 30,
        "sl_offset": 20.0,
        "tp_offset": 40.0,
        "contracts": 1,
    },
)

# ── Backtest ───────────────────────────────────────────────────────────────
print("Running backtest...")
result = run_backtest(DummyLongStrategy, config, data)
result.print_summary()

# ── Performance ────────────────────────────────────────────────────────────
print("Computing performance metrics...")
perf = PerformanceEngine().compute(result, data)

# ── Tearsheet ──────────────────────────────────────────────────────────────
print("Rendering tearsheet...")
TearsheetRenderer().render(
    perf,
    output_path="tearsheet.html",
    auto_open=AUTO_OPEN_TEARSHEET,
)