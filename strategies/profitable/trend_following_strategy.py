"""
TradingWithRayner Trend Following Strategy — NQ Futures

Adapted from the TradingWithRayner MQ5 EA.

Entry (LONG):
  - D1 SMA(sma_fast) > D1 SMA(sma_slow)   (uptrend)
  - Prior D1 close broke above the lookback-bar highest high

Entry (SHORT):
  - D1 SMA(sma_fast) < D1 SMA(sma_slow)   (downtrend)
  - Prior D1 close broke below the lookback-bar lowest low

Exit:
  - ATR(14) × atr_mult trailing stop — updated each bar in manage_position
  - No fixed TP (pure trend following)
  - Set RunConfig.eod_exit_time = time(23, 59) to allow overnight holds

Entry timing: checked at RTH open (9:30 ET) each day.
One position at a time; once entered, held until trailing stop hit.

Daily data: data/NQ_1day_full_data.parquet
"""
from __future__ import annotations

import os
from datetime import time as _time
from typing import Optional

import numpy as np
import pandas as pd

from backtest.data.market_data import MarketData
from backtest.strategy.base import BaseStrategy
from backtest.strategy.enums import OrderType, SizeType
from backtest.strategy.order import Order
from backtest.strategy.update import OpenPosition, PositionUpdate

POINT_VALUE = 20.0
TICK        = 0.25

_RTH_OPEN_MIN = 9 * 60 + 30

_DAILY_DATA_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'NQ_1day_full_data.parquet')


def _tick(price: float) -> float:
    return round(round(price / TICK) * TICK, 10)


def _load_daily_arrays(sma_fast: int, sma_slow: int, lookback: int) -> tuple:
    """
    Returns (date_arr, fast_sma_arr, slow_sma_arr, hh_arr, ll_arr, atr14_arr, close_arr).
    hh_arr[i] = highest high of the prior `lookback` bars (not including bar i).
    ll_arr[i] = lowest  low  of the prior `lookback` bars.
    """
    df = pd.read_parquet(_DAILY_DATA_PATH)
    df = df.rename(columns={'Last': 'Close'})
    df['Date'] = pd.to_datetime(df['Date']).dt.normalize()
    df = df.sort_values('Date').reset_index(drop=True)

    df['sma_fast'] = df['Close'].rolling(sma_fast, min_periods=sma_fast).mean()
    df['sma_slow'] = df['Close'].rolling(sma_slow, min_periods=sma_slow).mean()

    # Highest high / lowest low of prior `lookback` bars (shift 1 so we use completed bars)
    df['hh'] = df['High'].shift(1).rolling(lookback, min_periods=lookback).max()
    df['ll'] = df['Low'].shift(1).rolling(lookback, min_periods=lookback).min()

    prev_close = df['Close'].shift(1)
    tr = pd.concat([
        df['High'] - df['Low'],
        (df['High'] - prev_close).abs(),
        (df['Low']  - prev_close).abs(),
    ], axis=1).max(axis=1)
    df['atr14'] = tr.rolling(14, min_periods=14).mean()

    return (
        df['Date'].values.astype('datetime64[D]'),
        df['sma_fast'].values.astype(np.float64),
        df['sma_slow'].values.astype(np.float64),
        df['hh'].values.astype(np.float64),
        df['ll'].values.astype(np.float64),
        df['atr14'].values.astype(np.float64),
        df['Close'].values.astype(np.float64),
    )


# Module-level cache keyed by (sma_fast, sma_slow, lookback)
_DAILY_CACHE: dict = {}


def _get_daily_arrays(sma_fast: int, sma_slow: int, lookback: int) -> tuple:
    key = (sma_fast, sma_slow, lookback)
    if key not in _DAILY_CACHE:
        _DAILY_CACHE[key] = _load_daily_arrays(sma_fast, sma_slow, lookback)
    return _DAILY_CACHE[key]


class TrendFollowingStrategy(BaseStrategy):
    """
    Multi-day trend following on NQ using D1 SMA crossover + N-bar breakout.
    Positions can be held overnight; set eod_exit_time=time(23,59) in RunConfig.
    """

    trading_hours = None   # positions held overnight — no session restriction
    min_lookback  = 1

    def __init__(self, params: dict = None) -> None:
        super().__init__(params)
        p = params or {}

        self.sma_fast       = int(p.get('sma_fast',      50))
        self.sma_slow       = int(p.get('sma_slow',      100))
        self.lookback       = int(p.get('lookback',      200))
        self.atr_mult       = float(p.get('atr_mult',    6.0))
        self.risk_per_trade = float(p.get('risk_per_trade', 0.01))
        self.starting_equity = float(p.get('starting_equity', 100_000))
        self.equity_mode    = str(p.get('equity_mode',   'dynamic'))

        self._times_min:    Optional[np.ndarray] = None
        self._bar_day:      Optional[np.ndarray] = None
        self._bar_sma_fast: Optional[np.ndarray] = None
        self._bar_sma_slow: Optional[np.ndarray] = None
        self._bar_hh:       Optional[np.ndarray] = None
        self._bar_ll:       Optional[np.ndarray] = None
        self._bar_atr14:    Optional[np.ndarray] = None
        self._bar_prev_cls: Optional[np.ndarray] = None
        self._sig_mask:     Optional[np.ndarray] = None

        # Trailing stop state
        self._trail_sl:  float = 0.0
        self._direction: int   = 0

        self._today        = None
        self._traded_today = False

        self._eq_cache   = self.starting_equity
        self._eq_cache_n = 0

    def _setup(self, data: MarketData) -> None:
        if self._times_min is not None:
            return

        idx = data.df_1m.index

        if data.bar_times_1m_min is not None:
            self._times_min = data.bar_times_1m_min
        else:
            self._times_min = (idx.hour * 60 + idx.minute).to_numpy(np.int32)

        if data.bar_day_int_1m is not None:
            self._bar_day = data.bar_day_int_1m
        else:
            self._bar_day = idx.normalize().view(np.int64) // (86_400 * 10**9)

        d_dates, d_sma_f, d_sma_s, d_hh, d_ll, d_atr14, d_close = _get_daily_arrays(
            self.sma_fast, self.sma_slow, self.lookback
        )

        bar_dates = idx.normalize().values.astype('datetime64[D]')
        pos = np.searchsorted(d_dates, bar_dates, side='left') - 1

        valid    = pos >= 0
        idx_safe = np.clip(pos, 0, len(d_dates) - 1)

        self._bar_sma_fast = np.where(valid, d_sma_f[idx_safe],  np.nan)
        self._bar_sma_slow = np.where(valid, d_sma_s[idx_safe],  np.nan)
        self._bar_hh       = np.where(valid, d_hh[idx_safe],     np.nan)
        self._bar_ll       = np.where(valid, d_ll[idx_safe],     np.nan)
        self._bar_atr14    = np.where(valid, d_atr14[idx_safe],  np.nan)
        self._bar_prev_cls = np.where(valid, d_close[idx_safe],  np.nan)

        # signal_bar_mask: only 9:30 RTH open bars
        self._sig_mask = (self._times_min == _RTH_OPEN_MIN)

    def signal_bar_mask(self, data: MarketData) -> np.ndarray:
        self._setup(data)
        return self._sig_mask

    def _current_equity(self) -> float:
        if self.equity_mode == 'fixed':
            return self.starting_equity
        n = len(self.closed_trades)
        while self._eq_cache_n < n:
            self._eq_cache += self.closed_trades[self._eq_cache_n].net_pnl_dollars
            self._eq_cache_n += 1
        return self._eq_cache

    def generate_signals(self, data: MarketData, i: int) -> Optional[Order]:
        self._setup(data)

        today = int(self._bar_day[i])
        if today != self._today:
            self._today        = today
            self._traded_today = False

        if self._traded_today:
            return None

        sma_f   = float(self._bar_sma_fast[i])
        sma_s   = float(self._bar_sma_slow[i])
        hh      = float(self._bar_hh[i])
        ll      = float(self._bar_ll[i])
        atr14   = float(self._bar_atr14[i])
        prev_cls = float(self._bar_prev_cls[i])

        if any(np.isnan(v) for v in [sma_f, sma_s, hh, ll, atr14, prev_cls]):
            return None
        if atr14 <= 0:
            return None

        entry = float(data.open_1m[i])
        sl_dist = self.atr_mult * atr14

        direction = 0
        if sma_f > sma_s and prev_cls > hh:
            direction = 1   # uptrend + breakout above N-bar high
        elif sma_f < sma_s and prev_cls < ll:
            direction = -1  # downtrend + breakout below N-bar low

        if direction == 0:
            return None

        if direction == 1:
            sl_price = _tick(entry - sl_dist)
            tp_price = None   # no TP — trailing stop only
        else:
            sl_price = _tick(entry + sl_dist)
            tp_price = None

        equity    = self._current_equity()
        contracts = max(1, int(equity * self.risk_per_trade / (sl_dist * POINT_VALUE)))

        self._traded_today = True
        return Order(
            direction    = direction,
            order_type   = OrderType.MARKET,
            size_type    = SizeType.CONTRACTS,
            size_value   = float(contracts),
            sl_price     = sl_price,
            tp_price     = tp_price,
            trade_reason = 'tf_long' if direction == 1 else 'tf_short',
        )

    def on_fill(self, position: OpenPosition, data: MarketData, bar_index: int) -> None:
        # Store initial trailing stop distance; trail_sl tracks the stop level
        self._trail_sl  = position.sl_price
        self._direction = position.direction
        position.set_initial_sl_tp(position.sl_price, None)

    def manage_position(
        self, data: MarketData, i: int, position: OpenPosition,
    ) -> Optional[PositionUpdate]:
        self._setup(data)

        atr14 = float(self._bar_atr14[i])
        if np.isnan(atr14) or atr14 <= 0:
            return None

        close = float(data.close_1m[i])
        sl_dist = self.atr_mult * atr14

        if self._direction == 1:
            new_sl = _tick(close - sl_dist)
            if new_sl > self._trail_sl:
                self._trail_sl = new_sl
                return PositionUpdate(new_sl_price=new_sl)
        else:
            new_sl = _tick(close + sl_dist)
            if new_sl < self._trail_sl:
                self._trail_sl = new_sl
                return PositionUpdate(new_sl_price=new_sl)

        return None
