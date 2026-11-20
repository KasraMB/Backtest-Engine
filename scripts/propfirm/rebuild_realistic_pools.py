"""
Rebuild AMR trade pools with REALISTIC execution for live-money validation.

Per audit findings + 2-agent verification:
  - fill_mode='next_open': market orders fill at NEXT bar open (real live timing)
  - slippage_points=0.5: 2 ticks/side adverse slippage (covers entry + exit)
  - commission_per_contract=3.50 (NQ mini RT $7)
  - order_cancel_time=23:59 (no premature cancel for 8 PM AMR session)

For each config (cfg_406, cfg_495, cfg_152), output per-day arrays needed
by the bootstrap propfirm sim:
  - R_per_slot[day]      sum_R per day (drives daily $PnL)
  - sl_dists_per_slot[d] mean SL per day (for mini/micro decision)
  - fired_per_slot[d]    bool: did strategy fire any trade that day
  - n_trades_per_slot[d] trade count per day (for commissions)
  - R_per_trade_per_slot[d, k]   per-trade R (for intraday-blow check)
  - MAE_R_per_trade_per_slot[d, k] per-trade MAE in R (for intraday-blow)

Save to cache/amr_realistic_pools.pkl.
"""
from __future__ import annotations

import datetime
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from backtest.propfirm.lucidflex import extract_normalised_trades
from backtest.runner.config import RunConfig
from backtest.runner.runner import run_backtest
from strategies.amr_strategy import AnchoredMeanReversionStrategy

CONFIGS_TO_REBUILD = ['cfg_406', 'cfg_495', 'cfg_152']
SLIPPAGE = 0.5
COMMISSION = 3.50
FILL_MODE = 'next_open'
ORDER_CANCEL_TIME = datetime.time(23, 59)


def build_per_day_arrays(trades: list, md, max_trades_per_day: int = 10):
    """Aggregate trades to per-day arrays for bootstrap sim."""
    if len(trades) == 0:
        return None
    pnl_pts, sl_dists = extract_normalised_trades(trades)
    R = pnl_pts / np.maximum(sl_dists, 1e-6)
    entry_dates = np.array([md.df_1m.index[t.entry_bar].date() for t in trades])
    mae_pts = np.array([float(getattr(t, 'mae_points', 0.0) or 0.0) for t in trades], dtype=np.float64)
    MAE_R = mae_pts / np.maximum(sl_dists, 1e-6)

    # Group by date
    df = pd.DataFrame({'date': entry_dates, 'pnl_pts': pnl_pts, 'sl_dist': sl_dists,
                       'R': R, 'MAE_R': MAE_R})
    unique_days = pd.to_datetime(df['date'].unique())
    n_days = len(unique_days)
    R_per_day  = np.zeros(n_days, dtype=np.float64)
    sl_per_day = np.zeros(n_days, dtype=np.float64)
    fired      = np.ones(n_days, dtype=bool)
    n_trades   = np.zeros(n_days, dtype=np.int64)
    R_per_trade   = np.zeros((n_days, max_trades_per_day), dtype=np.float64)
    MAE_per_trade = np.zeros((n_days, max_trades_per_day), dtype=np.float64)

    date_to_idx = {d.date() if hasattr(d, 'date') else d: i for i, d in enumerate(unique_days)}
    for date_val, sub in df.groupby('date'):
        i = date_to_idx[date_val]
        R_per_day[i]  = sub['R'].sum()
        sl_per_day[i] = sub['sl_dist'].mean()
        n_trades[i]   = len(sub)
        k = min(len(sub), max_trades_per_day)
        R_per_trade[i, :k]   = sub['R'].values[:k]
        MAE_per_trade[i, :k] = sub['MAE_R'].values[:k]
    return {
        'dates':         unique_days,
        'R':             R_per_day,
        'sl_dists':      sl_per_day,
        'fired':         fired,
        'n_trades':      n_trades,
        'R_per_trade':   R_per_trade,
        'MAE_R_per_trade': MAE_per_trade,
        'mean_R_trade':  float(R.mean()),
        'wr':            float((R > 0).mean()),
        'total_trades':  int(len(trades)),
    }


def main():
    print("Loading MarketData from md_cache.pkl...", flush=True)
    md = pickle.load(open(REPO / 'md_cache.pkl', 'rb'))['md']
    print(f"  {len(md.df_1m)} 1m bars", flush=True)

    amr = pickle.load(open(REPO / 'cache' / 'amr_per_day_trades.pkl', 'rb'))

    results: dict = {
        'meta': {
            'fill_mode':     FILL_MODE,
            'slippage_pts':  SLIPPAGE,
            'commission_per_contract': COMMISSION,
            'order_cancel_time': str(ORDER_CANCEL_TIME),
            'rebuilt_at':    datetime.datetime.now().isoformat(),
            'note':          'realistic execution: next-bar-open fills + 2-tick/side slippage',
        },
    }
    for cfg_id in CONFIGS_TO_REBUILD:
        if cfg_id not in amr:
            print(f"Skipping {cfg_id} — not in cache", flush=True)
            continue
        params = amr[cfg_id]['params']
        print(f"\nRebuilding {cfg_id} (session={params['sessions']})...", flush=True)
        t0 = time.time()
        cfg = RunConfig(
            starting_capital=100_000,
            commission_per_contract=COMMISSION,
            slippage_points=SLIPPAGE,
            fill_mode=FILL_MODE,
            order_cancel_time=ORDER_CANCEL_TIME,
            params=params,
        )
        res = run_backtest(AnchoredMeanReversionStrategy, cfg, data=md, validate=False)
        per_day = build_per_day_arrays(res.trades, md)
        if per_day is None:
            print(f"  no trades, skipping")
            continue
        results[cfg_id] = {
            'params': params,
            'per_day': per_day,
        }
        print(f"  {per_day['total_trades']} trades, "
              f"mean_R/trade={per_day['mean_R_trade']:+.4f}, WR={per_day['wr']*100:.1f}%, "
              f"{(per_day['fired']).sum()} trading days, took {time.time()-t0:.1f}s")

    out_path = REPO / 'cache' / 'amr_realistic_pools.pkl'
    with open(out_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nSaved {out_path}", flush=True)


if __name__ == '__main__':
    main()
