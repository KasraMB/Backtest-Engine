"""
Daily ATR computed from 1-minute bars.

Used by IVB v9-final as the universal distance scaler (α, γ × ATR_20_daily).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def daily_ohlc_from_1m(df_1m: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 1-minute bars to daily OHLC keyed by session date.

    Session date = trading day in the index's local tz (no calendar shift).
    Expects columns: open, high, low, close.
    """
    idx = pd.DatetimeIndex(df_1m.index)
    day = idx.normalize()
    grp = df_1m.groupby(day, sort=True)
    daily = pd.DataFrame({
        'open':  grp['open'].first(),
        'high':  grp['high'].max(),
        'low':   grp['low'].min(),
        'close': grp['close'].last(),
    })
    return daily


def atr_daily(df_1m: pd.DataFrame, window: int = 20) -> pd.Series:
    """Wilder True Range smoothed by simple rolling mean over `window` days.

    Returns a daily Series indexed by session date (DatetimeIndex normalized).
    Day 0..window-2 produce NaN (insufficient lookback).
    """
    daily = daily_ohlc_from_1m(df_1m)
    h = daily['high'].to_numpy(np.float64)
    l = daily['low'].to_numpy(np.float64)
    c = daily['close'].to_numpy(np.float64)
    prev_c = np.concatenate([[np.nan], c[:-1]])
    tr = np.maximum.reduce([
        h - l,
        np.abs(h - prev_c),
        np.abs(l - prev_c),
    ])
    atr = pd.Series(tr, index=daily.index).rolling(window=window, min_periods=window).mean()
    atr.name = f'atr_{window}_daily'
    return atr


def broadcast_daily_to_1m(daily_series: pd.Series, df_1m_index: pd.DatetimeIndex) -> np.ndarray:
    """Map a daily-indexed series to a per-1m-bar array using each bar's session date.

    Uses **prior day's** value (no same-day look-ahead): bar on day D gets daily value at D-1.
    Returns np.ndarray length len(df_1m_index); leading bars before first valid day get NaN.
    """
    daily_idx = pd.DatetimeIndex(daily_series.index).normalize()
    daily_shifted = pd.Series(daily_series.to_numpy(), index=daily_idx).shift(1)
    bar_day = pd.DatetimeIndex(df_1m_index).normalize()
    out = daily_shifted.reindex(bar_day).to_numpy(np.float64)
    return out
