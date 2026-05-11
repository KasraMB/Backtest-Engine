"""
IS parameter sensitivity sweep for SessionMeanRevStrategy
Period: 2019-01-01 to 2022-12-31 (correct in-sample)

Sweeping:
  disp_min_atr_mult ∈ {0.5, 1.0, 1.5, 2.0, 2.5}
  require_bos       ∈ {True, False}
  momentum_only     ∈ {True, False}
  → 20 combinations total
"""
from __future__ import annotations

import time
import numpy as np
import pandas as pd
from datetime import time as dtime

from backtest.data.loader import DataLoader
from backtest.data.market_data import MarketData
from backtest.runner.runner import run_backtest
from backtest.runner.config import RunConfig
from strategies.session_mean_rev import SessionMeanRevStrategy

# ── Load data once ────────────────────────────────────────────────────────────
CACHE_1M      = "data/NQ_1m.parquet"
CACHE_5M      = "data/NQ_5m.parquet"
CACHE_BAR_MAP = "data/NQ_bar_map.npy"

loader = DataLoader()

df_1m   = pd.read_parquet(CACHE_1M)
df_5m   = pd.read_parquet(CACHE_5M)
bar_map = np.load(CACHE_BAR_MAP)

arrays_1m = {col: df_1m[col].to_numpy(dtype="float64") for col in ["open","high","low","close","volume"]}
arrays_5m = {col: df_5m[col].to_numpy(dtype="float64") for col in ["open","high","low","close","volume"]}
rth_mask      = (df_1m.index.time >= dtime(9, 30)) & (df_1m.index.time <= dtime(16, 0))
trading_dates = sorted(set(df_1m[rth_mask].index.date))

data_full = MarketData(
    df_1m=df_1m, df_5m=df_5m,
    open_1m=arrays_1m["open"], high_1m=arrays_1m["high"],
    low_1m=arrays_1m["low"],  close_1m=arrays_1m["close"],
    volume_1m=arrays_1m["volume"],
    open_5m=arrays_5m["open"], high_5m=arrays_5m["high"],
    low_5m=arrays_5m["low"],  close_5m=arrays_5m["close"],
    volume_5m=arrays_5m["volume"],
    bar_map=bar_map, trading_dates=trading_dates,
)

# ── Filter to IS period ───────────────────────────────────────────────────────
DATE_FROM = "2019-01-01"
DATE_TO   = "2022-12-31"

start_ts = pd.Timestamp(DATE_FROM, tz="America/New_York")
end_ts   = pd.Timestamp(DATE_TO,   tz="America/New_York") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

mask_1m = (df_1m.index >= start_ts) & (df_1m.index <= end_ts)
mask_5m = (df_5m.index >= start_ts) & (df_5m.index <= end_ts)

df_1m_f = df_1m[mask_1m]
df_5m_f = df_5m[mask_5m]

rth_mask_f      = (df_1m_f.index.time >= dtime(9, 30)) & (df_1m_f.index.time <= dtime(16, 0))
trading_dates_f = sorted(set(df_1m_f[rth_mask_f].index.date))

arrays_1m_f = {col: df_1m_f[col].to_numpy(dtype="float64") for col in ["open","high","low","close","volume"]}
arrays_5m_f = {col: df_5m_f[col].to_numpy(dtype="float64") for col in ["open","high","low","close","volume"]}
bar_map_f   = loader._build_bar_map(df_1m_f, df_5m_f)

data = MarketData(
    df_1m=df_1m_f, df_5m=df_5m_f,
    open_1m=arrays_1m_f["open"], high_1m=arrays_1m_f["high"],
    low_1m=arrays_1m_f["low"],   close_1m=arrays_1m_f["close"],
    volume_1m=arrays_1m_f["volume"],
    open_5m=arrays_5m_f["open"], high_5m=arrays_5m_f["high"],
    low_5m=arrays_5m_f["low"],   close_5m=arrays_5m_f["close"],
    volume_5m=arrays_5m_f["volume"],
    bar_map=bar_map_f,
    trading_dates=trading_dates_f,
)
print(f"IS data: {DATE_FROM} to {DATE_TO}  |  {len(data.df_1m):,} 1m bars  |  {len(data.trading_dates):,} trading days\n")

# ── Fixed params ──────────────────────────────────────────────────────────────
FIXED = {
    "atr_period":          10,
    "wick_threshold":      0.15,
    "rr_ratio":            1.25,
    "sl_atr_multiplier":   1.0,
    "risk_per_trade":      0.01,
    "equity_mode":         "dynamic",
    "starting_equity":     100_000,
    "point_value":         20.0,
    "max_trades_per_day":  3,
    "allowed_sessions":    ['NY'],
}

# ── Grid ──────────────────────────────────────────────────────────────────────
DISP_MULTS    = [0.5, 1.0, 1.5, 2.0, 2.5]
REQUIRE_BOS   = [True, False]
MOMENTUM_ONLY = [True, False]

results = []

total = len(DISP_MULTS) * len(REQUIRE_BOS) * len(MOMENTUM_ONLY)
n = 0
t0 = time.perf_counter()

for disp in DISP_MULTS:
    for bos in REQUIRE_BOS:
        for mom in MOMENTUM_ONLY:
            n += 1
            params = {
                **FIXED,
                "disp_min_atr_mult": disp,
                "require_bos":       bos,
                "momentum_only":     mom,
            }
            config = RunConfig(
                starting_capital=100_000,
                slippage_points=0.25,
                commission_per_contract=4.50,
                eod_exit_time=dtime(23, 59),
                params=params,
            )
            result = run_backtest(SessionMeanRevStrategy, config, data)

            trades = result.trades
            n_trades = len(trades)
            if n_trades == 0:
                wr = 0.0
                total_pnl = 0.0
                avg_win = 0.0
                avg_loss = 0.0
            else:
                pnls  = np.array([t.net_pnl_dollars for t in trades])
                wins  = pnls[pnls > 0]
                losses = pnls[pnls <= 0]
                wr        = 100.0 * len(wins) / n_trades
                total_pnl = float(pnls.sum())
                avg_win   = float(wins.mean())   if len(wins)   > 0 else 0.0
                avg_loss  = float(losses.mean()) if len(losses) > 0 else 0.0

            results.append({
                "disp_mult":    disp,
                "bos":          bos,
                "momentum_only": mom,
                "trades":       n_trades,
                "WR%":          round(wr, 1),
                "total_PnL":    round(total_pnl, 0),
                "avg_win":      round(avg_win, 0),
                "avg_loss":     round(avg_loss, 0),
            })
            elapsed = time.perf_counter() - t0
            rate    = n / elapsed
            eta     = (total - n) / rate if rate > 0 else 0
            print(
                f"  [{n:2d}/{total}] disp={disp:.1f} bos={str(bos):<5} mom={str(mom):<5}  "
                f"trades={n_trades:3d}  WR={wr:5.1f}%  PnL={total_pnl:>10,.0f}  "
                f"ETA {eta:.0f}s",
                flush=True,
            )

# ── Sort and print table ──────────────────────────────────────────────────────
results.sort(key=lambda r: r["total_PnL"], reverse=True)

print()
print("=" * 85)
print(f"{'IS Sweep Results':^85}")
print(f"{'Period: 2019-01-01 to 2022-12-31  |  NY session only':^85}")
print("=" * 85)
print(
    f"{'disp_mult':>9} {'bos':>5} {'mom_only':>8} {'trades':>7} {'WR%':>7} "
    f"{'total_PnL':>12} {'avg_win':>9} {'avg_loss':>9}"
)
print("-" * 85)
for r in results:
    print(
        f"{r['disp_mult']:>9.1f} {str(r['bos']):>5} {str(r['momentum_only']):>8} "
        f"{r['trades']:>7d} {r['WR%']:>7.1f} "
        f"{r['total_PnL']:>12,.0f} {r['avg_win']:>9,.0f} {r['avg_loss']:>9,.0f}"
    )
print("=" * 85)
print(f"\nTotal time: {time.perf_counter() - t0:.1f}s")
