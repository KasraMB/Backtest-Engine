"""Test afternoon session on 2025 OOS."""
import sys
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
from datetime import time as dtime

from backtest.data.loader import DataLoader
from backtest.data.market_data import MarketData
from backtest.runner.runner import run_backtest
from backtest.runner.config import RunConfig
from backtest.propfirm.lucidflex import (
    LUCIDFLEX_ACCOUNTS,
    run_propfirm_grid,
    EVAL_RISK_PCT_OF_MLL,
    FUNDED_RISK_PCT_OF_MLL,
    RISK_GEOMETRIES,
)
from strategies.session_mean_rev import SessionMeanRevStrategy

DATE_FROM = '2025-01-01'
DATE_TO   = '2025-12-31'

CACHE_1M      = "data/NQ_1m.parquet"
CACHE_5M      = "data/NQ_5m.parquet"
CACHE_BAR_MAP = "data/NQ_bar_map.npy"

BASE_PARAMS = dict(
    rr_ratio=1.5, sl_atr_multiplier=0.75, atr_period=10,
    wick_threshold=0.15, disp_min_atr_mult=2.0,
    require_bos=True, momentum_only=True,
    allowed_sessions=['NY'],
    max_trades_per_day=3,
    risk_per_trade=0.01, equity_mode='dynamic', starting_equity=100_000,
    point_value=20.0,
)

AFT_PARAMS = dict(
    rr_ratio=1.5, sl_atr_multiplier=0.75, atr_period=10,
    wick_threshold=0.15, disp_min_atr_mult=2.0,
    require_bos=True, momentum_only=True,
    allowed_sessions=['NY', 'Afternoon'],
    max_trades_per_day=3,
    risk_per_trade=0.01, equity_mode='dynamic', starting_equity=100_000,
    point_value=20.0,
)

# -- Load from cache --
loader = DataLoader()
print("Loading from cache...")
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

# -- Filter to 2025 --
start_ts = pd.Timestamp(DATE_FROM, tz="America/New_York")
end_ts   = pd.Timestamp(DATE_TO, tz="America/New_York") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

mask_1m = (df_1m.index >= start_ts) & (df_1m.index <= end_ts)
mask_5m = (df_5m.index >= start_ts) & (df_5m.index <= end_ts)

df_1m_f = df_1m[mask_1m]
df_5m_f = df_5m[mask_5m]

rth_f = (df_1m_f.index.time >= dtime(9, 30)) & (df_1m_f.index.time <= dtime(16, 0))
trading_dates_f = sorted(set(df_1m_f[rth_f].index.date))

arrays_1m_f = {col: df_1m_f[col].to_numpy(dtype="float64") for col in ["open","high","low","close","volume"]}
arrays_5m_f = {col: df_5m_f[col].to_numpy(dtype="float64") for col in ["open","high","low","close","volume"]}
bar_map_f = loader._build_bar_map(df_1m_f, df_5m_f)

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

print(f"Data loaded: {DATE_FROM} to {DATE_TO}, {len(trading_dates_f)} trading days\n")

account = LUCIDFLEX_ACCOUNTS['150K']


def run_and_analyze(label: str, params: dict):
    config = RunConfig(
        starting_capital=100_000,
        slippage_points=0.25,
        commission_per_contract=4.50,
        eod_exit_time=dtime(23, 59),
        params=params,
    )
    result = run_backtest(SessionMeanRevStrategy, config, data, validate=False)
    trades = result.trades
    n = len(trades)
    wins = sum(1 for t in trades if t.is_winner)
    wr = wins / n * 100 if n else 0
    pnl = sum(t.net_pnl_dollars for t in trades)

    print(f"\n{'='*60}")
    print(f"=== {label} ===")
    print(f"  Trades: {n}, WR: {wr:.1f}%, PnL: ${pnl:,.0f}")

    if n == 0:
        print("  No trades — skipping propfirm analysis")
        return result, {}

    grid = run_propfirm_grid(
        trades=trades,
        account=account,
        n_sims=2_000,
        sizing_mode='micros',
        n_workers=1,
    )

    # Find best EV/day cell across all scheme/erp/frp
    best_ev = -999.0
    best_cell = {}
    best_key = {}

    for scheme in RISK_GEOMETRIES:
        if scheme not in grid:
            continue
        for erp in EVAL_RISK_PCT_OF_MLL:
            frp_dict = grid[scheme].get(erp, {})
            for frp in FUNDED_RISK_PCT_OF_MLL:
                cell = frp_dict.get(frp)
                if cell is None:
                    continue
                ev = cell.get('ev_per_day', -999.0)
                if ev is not None and ev > best_ev:
                    best_ev = ev
                    best_cell = cell
                    best_key = {'scheme': scheme, 'erp': erp, 'frp': frp}

    print(f"  Best EV/day: ${best_ev:.2f}")
    print(f"    scheme={best_key.get('scheme')}, erp={best_key.get('erp'):.0%}, frp={best_key.get('frp'):.0%}")
    print(f"    pass_rate={best_cell.get('pass_rate', 0)*100:.1f}%, optimal_k={best_cell.get('optimal_k')}")
    print(f"    net_ev=${best_cell.get('net_ev', 0):,.0f}, mean_withdrawal=${best_cell.get('mean_withdrawal', 0):,.0f}")

    # Standard config: fixed_dollar, erp=0.2, frp=0.4
    print(f"\n  Standard config (fixed_dollar, erp=0.20, frp=0.40):")
    try:
        std_cell = grid['fixed_dollar'][0.20][0.40]
        if std_cell:
            print(f"    EV/day: ${std_cell.get('ev_per_day', 0):.2f}")
            print(f"    pass_rate: {std_cell.get('pass_rate', 0)*100:.1f}%")
            print(f"    optimal_k: {std_cell.get('optimal_k')}")
            print(f"    net_ev: ${std_cell.get('net_ev', 0):,.0f}")
        else:
            print("    Cell is None")
    except (KeyError, TypeError) as e:
        print(f"    Lookup failed: {e}")

    return result, best_key


run_and_analyze("NY only (baseline)", BASE_PARAMS)
run_and_analyze("NY + Afternoon", AFT_PARAMS)

print("\nDone.")
