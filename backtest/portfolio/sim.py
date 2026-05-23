"""Portfolio-level propfirm simulation utilities.

Equal-weight per-day R aggregation, then audited bootstrap propfirm sim.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from backtest.portfolio.cell import CellResult
from backtest.propfirm.lucidflex import LUCIDFLEX_ACCOUNTS, LucidFlexAccount
from backtest.propfirm.three_phase_mc import three_phase_reinvestment_mc


def _per_day_R(trades: pd.DataFrame) -> pd.Series:
    """Sum trade R per calendar date for one strategy. Returns Series indexed by date."""
    if len(trades) == 0:
        return pd.Series([], dtype=np.float64)
    return trades.groupby(trades['date'].dt.date)['R'].sum()


def portfolio_daily_R(
    results: Iterable[CellResult],
    date_lo: 'pd.Timestamp | None' = None,
    date_hi: 'pd.Timestamp | None' = None,
) -> pd.DataFrame:
    """Equal-weight combined daily R across selected cells, restricted to [date_lo, date_hi].

    Output DataFrame indexed by date with columns per cell + 'portfolio'
    (= MEAN across cells per day; days where cell didn't trade get 0).
    """
    series = {}
    for r in results:
        if r.trades.empty:
            continue
        s = _per_day_R(r.trades)
        s.index = pd.to_datetime(s.index)
        series[r.cell.name] = s
    if not series:
        return pd.DataFrame()
    df = pd.concat(series, axis=1).sort_index().fillna(0.0)
    if date_lo is not None:
        df = df[df.index >= pd.Timestamp(date_lo)]
    if date_hi is not None:
        df = df[df.index <= pd.Timestamp(date_hi)]
    df['portfolio'] = df.drop(columns=[]).mean(axis=1)
    return df


def build_per_day_arrays(
    trades: pd.DataFrame,
    max_trades_per_day: int = 10,
) -> dict:
    """Build per-day arrays expected by the bootstrap propfirm sim.

    Inputs are a per-trade DataFrame (cols: date, R, sl_dist, MAE_R).
    Returns dict with R, sl_dist, fired, n_trades, R_per_trade, MAE_R_per_trade.
    """
    if len(trades) == 0:
        return {}
    df = trades.copy()
    df['date'] = pd.to_datetime(df['date'])
    unique = pd.to_datetime(df['date'].unique())
    nd = len(unique)
    d2i = {d: i for i, d in enumerate(unique)}
    R       = np.zeros(nd, dtype=np.float64)
    sl_arr  = np.zeros(nd, dtype=np.float64)
    fired   = np.zeros(nd, dtype=bool)
    nt      = np.zeros(nd, dtype=np.int64)
    Rt      = np.zeros((nd, max_trades_per_day), dtype=np.float64)
    MAE_t   = np.zeros((nd, max_trades_per_day), dtype=np.float64)
    for dt, sub in df.groupby('date'):
        i = d2i[dt]
        R[i]      = sub['R'].sum()
        sl_arr[i] = sub['sl_dist'].mean()
        fired[i]  = True
        nt[i]     = len(sub)
        k = min(len(sub), max_trades_per_day)
        Rt[i, :k]    = sub['R'].values[:k]
        MAE_t[i, :k] = sub['MAE_R'].values[:k]
    return {'dates': unique, 'R': R, 'sl_dist': sl_arr,
            'fired': fired, 'n_trades': nt,
            'R_per_trade': Rt, 'MAE_R_per_trade': MAE_t}


def propfirm_p_profit(
    daily_R:      np.ndarray,            # (n_days,) combined daily R
    sl_dist:      np.ndarray,            # (n_days,) mean SL distance per day
    n_trades:     np.ndarray,            # (n_days,) integer trade count
    R_per_trade:  np.ndarray,            # (n_days, max_trades)
    MAE_per_trade: np.ndarray,           # (n_days, max_trades)
    account:      LucidFlexAccount = LUCIDFLEX_ACCOUNTS['50K'],
    budget:       float = 1000.0,
    horizon:      int   = 84,
    erp:          float = 0.5,
    n_sims:       int   = 15000,
    seed:         int   = 2026,
    min_payout:   float = 500.0,
    max_concurrent: int = 1,
) -> dict:
    """Run audited bootstrap propfirm sim on a single combined-portfolio slot.

    Returns dict with p_profit, p_zero, mean_growth, median_growth, sortino.
    """
    fired = np.ones_like(daily_R, dtype=bool)
    sl_arr = np.where(sl_dist > 0, sl_dist, 16.5)

    out = three_phase_reinvestment_mc(
        account=account, budget=budget, horizon=horizon,
        max_concurrent=max_concurrent, n_sims=n_sims, seed=seed,
        eval_grind_risk=erp, bootstrap_block_len=5.0,
        bootstrap_oos_R_per_slot=daily_R[None, :],
        bootstrap_oos_sl_dists_per_slot=sl_arr[None, :],
        bootstrap_oos_fired_per_slot=fired[None, :],
        bootstrap_oos_n_trades_per_slot=n_trades[None, :],
        bootstrap_oos_R_per_trade_per_slot=R_per_trade[None, :, :],
        bootstrap_oos_MAE_R_per_trade_per_slot=MAE_per_trade[None, :, :],
        bootstrap_intraday_blow_check=True,
        enforce_consistency_rule=True,
    )
    cash = out['cash'].astype(np.float64)
    n_evals = out['eval_fees_paid'] / account.eval_fee
    reopens = np.maximum(n_evals - max_concurrent, 0.0)
    cash_adj = cash - (account.reset_fee - account.eval_fee) * reopens
    growth = cash_adj - budget
    downside_std = growth[growth < 0].std() if (growth < 0).any() else 1.0
    return {
        'p_profit':      float((cash_adj >= budget + min_payout).mean()),
        'p_zero':        float((cash_adj <= 0).mean()),
        'mean_growth':   float(growth.mean()),
        'median_growth': float(np.median(growth)),
        'sortino':       float(growth.mean() / max(downside_std, 1e-9)),
        'mean_payouts':  float(out['n_payouts'].mean()),
    }
