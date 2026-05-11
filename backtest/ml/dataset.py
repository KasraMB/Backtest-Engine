"""
Build a training DataFrame from a completed backtest RunResult.

Each row = one trade.
Columns = ALL_FEATURE_NAMES + ['date', 'entry_bar', 'r_multiple', 'is_winner'].

Context features (rolling account state) are computed here from the full
trade list so they are always derived from prior trades only (no lookahead).
"""
from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from backtest.ml.features import ALL_FEATURE_NAMES

if TYPE_CHECKING:
    from backtest.data.market_data import MarketData
    from backtest.runner.runner import RunResult


def build_dataset(run_result: 'RunResult', data: 'MarketData') -> pd.DataFrame:
    """
    Convert a RunResult into a feature DataFrame suitable for ML training.

    Trades without signal_features (e.g. from strategies that don't record them)
    are silently skipped.

    Parameters
    ----------
    run_result : RunResult
        Output of run_backtest().
    data : MarketData
        The same MarketData object used for the backtest (needed for dates,
        equity curve lookup).

    Returns
    -------
    pd.DataFrame
        Columns: ALL_FEATURE_NAMES + ['date', 'entry_bar', 'r_multiple', 'is_winner']
        Index: range index.
    """
    trades = [t for t in run_result.trades if t.signal_features]
    _meta = ['date', 'entry_bar', 'exit_bar', 'r_multiple', 'is_winner',
             'sl_pts', 'net_pnl_dollars', 'entry_price', 'exit_price']
    if not trades:
        return pd.DataFrame(columns=ALL_FEATURE_NAMES + _meta)

    # Build equity curve indexed by bar for drawdown lookup
    eq_arr = np.array(run_result.equity_curve, dtype=float)
    peak = np.maximum.accumulate(eq_arr)
    dd_pct = np.where(peak > 0, (peak - eq_arr) / peak * 100.0, 0.0)

    # Rolling context state — updated as we process trades in order
    history: deque = deque(maxlen=10)     # (r_multiple,) of recent trades
    consecutive_losses: int = 0
    day_counts: dict = {}                 # date → count of trades placed so far

    rows = []
    for trade in sorted(trades, key=lambda t: t.entry_bar):
        sf = trade.signal_features

        # Date of this trade from the 1m index
        try:
            ts = data.df_1m.index[trade.entry_bar]
            trade_date = ts.date()
        except (IndexError, AttributeError):
            trade_date = None

        # Context features (derived from history BEFORE this trade)
        daily_idx = day_counts.get(trade_date, 0)
        day_counts[trade_date] = daily_idx + 1

        if len(history) >= 1:
            recent_r = list(history)
            rwr = float(np.mean([r > 0 for r in recent_r]))
            rex = float(np.mean(recent_r))
        else:
            rwr = 0.5
            rex = 0.0

        # Drawdown at entry bar
        entry_bar = trade.entry_bar
        dd = float(dd_pct[entry_bar]) if entry_bar < len(dd_pct) else 0.0

        ctx: dict = {
            'daily_trade_idx':        daily_idx,
            'recent_win_rate_10':     rwr,
            'recent_expectancy_r_10': rex,
            'consecutive_losses':     consecutive_losses,
            'drawdown_pct':           dd,
        }

        # Merge signal + context features
        row = {**sf, **ctx}

        # Ensure all feature columns present (fill 0 for any missing)
        for col in ALL_FEATURE_NAMES:
            if col not in row:
                row[col] = 0

        # Labels + trade metadata
        r_mult = trade.r_multiple
        row['r_multiple']      = float(r_mult) if r_mult is not None else 0.0
        row['is_winner']       = int(trade.is_winner)
        row['date']            = trade_date
        row['entry_bar']       = entry_bar
        row['exit_bar']        = trade.exit_bar
        row['sl_pts']          = (float(abs(trade.entry_price - trade.initial_sl_price))
                                  if trade.initial_sl_price is not None else 0.0)
        row['net_pnl_dollars'] = float(trade.net_pnl_dollars)
        row['entry_price']     = float(trade.entry_price)
        row['exit_price']      = float(trade.exit_price)

        rows.append(row)

        # Update rolling state AFTER recording (so it doesn't leak into this row)
        r = row['r_multiple']
        history.append(r)
        if r <= 0:
            consecutive_losses += 1
        else:
            consecutive_losses = 0

    df = pd.DataFrame(rows)

    # Enforce column order
    cols = ALL_FEATURE_NAMES + _meta
    for c in cols:
        if c not in df.columns:
            df[c] = 0
    return df[cols].reset_index(drop=True)


def reverse_ml_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform an ML dataset into its "reversed trade" equivalent.

    Each trade is flipped: SL placed at original TP level, TP at original SL.
    This gives the model a dataset of trades that profit when the original
    strategy loses and vice versa.

    Transformations applied:
      - direction      : flipped  (-1 ↔ 1)
      - sl_risk_atr    : *= tp_r  (new risk = old TP distance)
      - tp_r           : 1 / tp_r (new reward = old SL distance)
      - tp_next_r      : set to -1 (no meaningful next candidate)
      - n_tp_candidates: set to 1
      - r_multiple     : -r / tp_r
      - is_winner      : sign of new r_multiple
      - sl_pts (meta)  : *= tp_r
      - net_pnl_dollars: negated (approximate)
    """
    out = df.copy()
    tp_r = df['tp_r'].values.copy()

    # Guard: tp_r == 0 would divide by zero — shouldn't happen but clip to small positive
    tp_r = np.where(tp_r == 0.0, 1e-6, tp_r)

    out['direction']       = -df['direction']
    out['sl_risk_atr']     = df['sl_risk_atr'] * tp_r
    out['tp_r']            = 1.0 / tp_r
    out['tp_next_r']       = -1.0
    out['n_tp_candidates'] = 1
    out['r_multiple']      = -df['r_multiple'].values / tp_r
    out['is_winner']       = (out['r_multiple'] > 0).astype(int)
    out['sl_pts']          = df['sl_pts'].values * tp_r
    out['net_pnl_dollars'] = -df['net_pnl_dollars'].values
    return out
