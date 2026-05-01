"""
Diagnostic: trace signal generation and order processing directly.
"""
import sys
import pandas as pd
import numpy as np
from datetime import time as dtime

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from backtest.data.loader import DataLoader
from backtest.data.market_data import MarketData
from backtest.runner.runner import build_active_bar_set, build_required_bar_set, build_eod_bar_set
from backtest.runner.config import RunConfig
from strategies.asia_breakout_strategy import AsiaBreakoutStrategy, _compute_asia_range, _atr_at

CACHE_1M = "data/NQ_1m.parquet"
CACHE_5M = "data/NQ_5m.parquet"

TICK = 0.25

def main():
    loader = DataLoader()
    df_1m = pd.read_parquet(CACHE_1M)
    df_5m = pd.read_parquet(CACHE_5M)

    start_ts = pd.Timestamp("2019-01-01", tz="America/New_York")
    end_ts   = pd.Timestamp("2019-03-31", tz="America/New_York") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    m1 = (df_1m.index >= start_ts) & (df_1m.index <= end_ts)
    m5 = (df_5m.index >= start_ts) & (df_5m.index <= end_ts)
    d1, d5 = df_1m[m1], df_5m[m5]

    rth    = (d1.index.time >= dtime(3, 0)) & (d1.index.time <= dtime(16, 0))
    tdates = sorted(set(d1[rth].index.date))
    a1 = {c: d1[c].to_numpy(dtype="float64") for c in ["open","high","low","close","volume"]}
    a5 = {c: d5[c].to_numpy(dtype="float64") for c in ["open","high","low","close","volume"]}
    bm = loader._build_bar_map(d1, d5)
    data = MarketData(
        df_1m=d1, df_5m=d5,
        open_1m=a1["open"], high_1m=a1["high"], low_1m=a1["low"],
        close_1m=a1["close"], volume_1m=a1["volume"],
        open_5m=a5["open"], high_5m=a5["high"], low_5m=a5["low"],
        close_5m=a5["close"], volume_5m=a5["volume"],
        bar_map=bm, trading_dates=tdates,
    )

    print(f"Data: {len(data.df_1m):,} bars, {len(tdates)} trading days")
    print(f"First bar: {d1.index[0]}  Last: {d1.index[-1]}\n")

    params = {
        'entry_mode': 'breakout',
        'asia_start_hour': 18,
        'entry_hour': 3,
        'direction': 'short',
        'sl_type': 'atr',
        'rr_ratio': 1.5,
        'sl_atr_multiplier': 0.75,
        'atr_period': 14,
        'min_range_atr': 0.0,
        'max_range_atr': 5.0,
        'risk_per_trade': 0.01,
    }

    strategy = AsiaBreakoutStrategy(params)
    strategy._setup(data)

    times_min = strategy._times_min

    # Build active_bars and required_mask
    active_bars  = build_active_bar_set(data, strategy.trading_hours)
    req_set      = build_required_bar_set(data, AsiaBreakoutStrategy)
    required_mask = np.zeros(len(data.open_1m), dtype=bool)
    for idx in req_set:
        required_mask[idx] = True

    print(f"trading_hours: {strategy.trading_hours}")
    print(f"entry_min: {strategy._entry_min}  ({strategy._entry_min//60}:{strategy._entry_min%60:02d})")
    print(f"Active bars count: {len(active_bars)}")

    # Check first 3 AM bar
    entry_bars = [i for i in range(len(times_min)) if times_min[i] == strategy._entry_min]
    print(f"3:00 AM bars: {len(entry_bars)} found")
    if entry_bars:
        i0 = entry_bars[0]
        print(f"  First 3 AM bar: i={i0}  ts={data.df_1m.index[i0]}")
        print(f"  i0 >= min_lookback ({strategy.min_lookback}): {i0 >= strategy.min_lookback}")
        print(f"  i0 in active_bars: {i0 in active_bars}")
        print(f"  required_mask[i0]: {required_mask[i0]}")
    print()

    # Manually call generate_signals on first 3 AM bar
    if entry_bars:
        i0 = entry_bars[0]
        print(f"--- Calling generate_signals at bar {i0} ---")
        order = strategy.generate_signals(data, i0)
        if order is None:
            print("  RETURNED None!")
        else:
            print(f"  Returned Order: dir={order.direction} type={order.order_type} "
                  f"stop={order.stop_price:.2f} sl={order.sl_price:.2f} tp={order.tp_price:.2f} "
                  f"expiry={order.expiry_bars}")

        # Now manually check next bars for fill
        print(f"\n--- Checking next 5 bars for stop fill ---")
        sp = order.stop_price if order else None
        if sp is not None:
            for j in range(i0 + 1, min(i0 + 6, len(times_min))):
                bar_ts = data.df_1m.index[j]
                h, l, o, c = data.high_1m[j], data.low_1m[j], data.open_1m[j], data.close_1m[j]
                t = times_min[j]
                in_req = required_mask[j]
                cancel_min = 11 * 60
                cancel = t >= cancel_min
                print(f"  bar {j} ({bar_ts.time()}) OHLC={o:.2f}/{h:.2f}/{l:.2f}/{c:.2f}  "
                      f"t={t} req={in_req} cancel={cancel}")
                # Check stop fill
                if order.direction == -1:  # short
                    if l <= sp:
                        fill_px = min(o, sp)
                        print(f"    -> WOULD FILL SHORT at {fill_px:.2f} (sp={sp:.2f})")
                else:  # long
                    if h >= sp:
                        fill_px = max(o, sp)
                        print(f"    -> WOULD FILL LONG at {fill_px:.2f} (sp={sp:.2f})")


if __name__ == "__main__":
    main()
