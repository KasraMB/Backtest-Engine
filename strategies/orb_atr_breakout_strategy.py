"""
ORB ATR Breakout Strategy (NQ Futures, M30 timeframe)

Wait for first 30-min bar after RTH open (9:30-10:00 ET) to close.
Set breakout levels using D1 ATR(5) * atr_space_multiplier above/below the
9:30 open price. Enter on 1m close crossing either level between 10:00-15:00.

SL: sl_pct * entry_price (percentage of price)
TP: rr_ratio * sl_distance from entry

Optional: D1 200 SMA filter — only trade in direction of trend.

Daily data source: data/NQ_1day_full_data.parquet
  Columns: Date, Time, Open, High, Low, Last (close), Volume, ...
  Convention: Date = session START date (18:00 ET); the session corresponds
  to the NEXT calendar day's RTH. For a 1m bar at RTH date D, we use the
  daily row with the largest Date < D (last completed/in-progress daily bar).
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

_RTH_OPEN_MIN  = 9  * 60 + 30  # 09:30
_ORB_END_MIN   = 10 * 60        # 10:00 — first 30-min candle closes here
_WIN_END_MIN   = 15 * 60        # 15:00 — last entry allowed

_DAILY_DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'NQ_1day_full_data.parquet')


def _tick(price: float) -> float:
    return round(round(price / TICK) * TICK, 10)


def _load_daily_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load NQ_1day_full_data.parquet and return:
        (date_arr, atr5_arr, sma200_arr, close_arr)
    date_arr is numpy datetime64[D], sorted ascending.
    ATR uses Wilder's method (simple rolling mean of TR — standard for D1 ATR).
    """
    df = pd.read_parquet(_DAILY_DATA_PATH)
    df = df.rename(columns={'Last': 'Close'})
    df['Date'] = pd.to_datetime(df['Date']).dt.normalize()
    df = df.sort_values('Date').reset_index(drop=True)

    prev_close = df['Close'].shift(1)
    tr = pd.concat([
        df['High']  - df['Low'],
        (df['High'] - prev_close).abs(),
        (df['Low']  - prev_close).abs(),
    ], axis=1).max(axis=1)

    df['atr5']   = tr.rolling(5,   min_periods=5).mean()
    df['sma200'] = df['Close'].rolling(200, min_periods=200).mean()

    date_arr  = df['Date'].values.astype('datetime64[D]')
    atr5_arr  = df['atr5'].values.astype(np.float64)
    sma200_arr = df['sma200'].values.astype(np.float64)
    close_arr  = df['Close'].values.astype(np.float64)
    return date_arr, atr5_arr, sma200_arr, close_arr


# Module-level cache so multiple backtest runs reuse the loaded arrays
_DAILY_CACHE: Optional[tuple] = None


def _get_daily_arrays() -> tuple:
    global _DAILY_CACHE
    if _DAILY_CACHE is None:
        _DAILY_CACHE = _load_daily_arrays()
    return _DAILY_CACHE


class ORBATRBreakoutStrategy(BaseStrategy):
    """
    Opening Range Breakout with D1 ATR(5) space filter.
    Trades 10:00-15:00 ET. Up to max_trades_per_day (default 2, one per side).
    """

    trading_hours = [(_time(9, 30), _time(14, 59))]
    min_lookback  = 1  # daily data handles the lookback

    # ── Construction ─────────────────────────────────────────────────────────

    def __init__(self, params: dict = None) -> None:
        super().__init__(params)
        p = params or {}

        self.atr_space_mult  = float(p.get('atr_space_multiplier', 0.3))
        self.sl_pct          = float(p.get('sl_pct',               0.005))
        self.rr_ratio        = float(p.get('rr_ratio',             2.0))
        self.use_sma_filter  = bool(p.get('use_200sma_filter',     False))
        self.max_trades_day  = int(p.get('max_trades_per_day',     2))
        self.risk_per_trade  = float(p.get('risk_per_trade',       0.01))
        self.starting_equity = float(p.get('starting_equity',      100_000))
        self.equity_mode     = str(p.get('equity_mode',            'dynamic'))

        # Per-bar precomputed arrays (lazy, set in _setup)
        self._times_min:    Optional[np.ndarray] = None
        self._bar_d1_atr5:  Optional[np.ndarray] = None
        self._bar_sma200:   Optional[np.ndarray] = None
        self._bar_prev_cls: Optional[np.ndarray] = None

        # 9:30 open per date (date -> float)
        self._orb_open: dict = {}

        # Daily state
        self._today:        object = None
        self._long_level:   float  = 0.0
        self._short_level:  float  = 0.0
        self._levels_set:   bool   = False
        self._long_taken:   bool   = False
        self._short_taken:  bool   = False
        self._trades_today: int    = 0

        # Equity cache
        self._eq_cache   = self.starting_equity
        self._eq_cache_n = 0

    # ── Setup ─────────────────────────────────────────────────────────────────

    def _setup(self, data: MarketData) -> None:
        if self._times_min is not None:
            return

        idx = data.df_1m.index

        # Use cached times array from MarketData if available; compute once otherwise
        if data.bar_times_1m_min is not None:
            self._times_min = data.bar_times_1m_min
        else:
            self._times_min = (idx.hour * 60 + idx.minute).to_numpy(np.int32)

        # Integer day index per bar (days since epoch).  Reuse cached array when
        # the sweep pre-populates data.bar_day_int_1m — avoids 1.5s normalize() call.
        if data.bar_day_int_1m is not None:
            self._bar_day = data.bar_day_int_1m
        else:
            self._bar_day = idx.normalize().view(np.int64) // (86_400 * 10**9)

        # Build day-int → 9:30 open dict.  Iterate with index arithmetic only;
        # no idx.date call (eliminates the slow 0.6s pandas date extraction).
        rth_mask = (self._times_min == _RTH_OPEN_MIN)
        self._orb_open_int: dict = {}
        day_arr = self._bar_day
        rth_indices = np.where(rth_mask)[0]
        for j in rth_indices:
            k = int(day_arr[j])
            if k not in self._orb_open_int:
                self._orb_open_int[k] = float(data.open_1m[j])

        # Load daily arrays
        d_dates, d_atr5, d_sma200, d_close = _get_daily_arrays()

        # For each 1m bar at date D, find the last daily row with Date < D.
        # searchsorted('left') returns the first position >= D, so -1 gives last < D.
        bar_dates = idx.normalize().values.astype('datetime64[D]')
        pos = np.searchsorted(d_dates, bar_dates, side='left') - 1

        # Clamp
        valid = pos >= 0
        idx_safe = np.clip(pos, 0, len(d_dates) - 1)

        self._bar_d1_atr5  = np.where(valid, d_atr5[idx_safe],  np.nan)
        self._bar_sma200   = np.where(valid, d_sma200[idx_safe], np.nan)
        self._bar_prev_cls = np.where(valid, d_close[idx_safe],  np.nan)

        # Precompute signal bar mask: M30 closes (t_min % 30 == 29) within the
        # entry window.  Exposed via signal_bar_mask() so the runner can skip
        # all other bars when flat, avoiding ~322K redundant loop iterations.
        t = self._times_min
        self._sig_mask = (
            (t % 30 == 29) &
            (t >= _ORB_END_MIN) &
            (t < _WIN_END_MIN)
        )

    # ── Signal bar mask (runner hook) ─────────────────────────────────────────

    def signal_bar_mask(self, data: MarketData) -> np.ndarray:
        self._setup(data)
        return self._sig_mask

    # ── Equity ────────────────────────────────────────────────────────────────

    def _current_equity(self) -> float:
        if self.equity_mode == 'fixed':
            return self.starting_equity
        n = len(self.closed_trades)
        while self._eq_cache_n < n:
            self._eq_cache += self.closed_trades[self._eq_cache_n].net_pnl_dollars
            self._eq_cache_n += 1
        return self._eq_cache

    # ── Signal generation ─────────────────────────────────────────────────────

    def generate_signals(self, data: MarketData, i: int) -> Optional[Order]:
        self._setup(data)

        t_min = int(self._times_min[i])
        cl    = float(data.close_1m[i])
        today = int(self._bar_day[i])

        # Daily reset
        if today != self._today:
            self._today        = today
            self._levels_set   = False
            self._long_taken   = False
            self._short_taken  = False
            self._trades_today = 0
            self._long_level   = 0.0
            self._short_level  = 0.0

        # Set breakout levels once the ORB period closes (at 10:00)
        if not self._levels_set and t_min >= _ORB_END_MIN:
            orb_open = self._orb_open_int.get(today)
            d1_atr   = float(self._bar_d1_atr5[i])
            if orb_open is not None and not np.isnan(d1_atr) and d1_atr > 0:
                space = d1_atr * self.atr_space_mult
                self._long_level  = _tick(orb_open + space)
                self._short_level = _tick(orb_open - space)
                self._levels_set  = True

        # Entry window guard (runner's signal_bar_mask already ensures M30 closes only)
        if not self._levels_set or t_min < _ORB_END_MIN or t_min >= _WIN_END_MIN:
            return None

        if self._trades_today >= self.max_trades_day:
            return None

        sma200   = float(self._bar_sma200[i])
        prev_cls = float(self._bar_prev_cls[i])

        # Long: price closes above long level
        if not self._long_taken and cl >= self._long_level:
            skip = self.use_sma_filter and not np.isnan(sma200) and prev_cls < sma200
            if not skip:
                order = self._build_order(1, cl, 'orb_long')
                if order is not None:
                    self._long_taken   = True
                    self._trades_today += 1
                    return order

        # Short: price closes below short level
        if not self._short_taken and cl <= self._short_level:
            skip = self.use_sma_filter and not np.isnan(sma200) and prev_cls > sma200
            if not skip:
                order = self._build_order(-1, cl, 'orb_short')
                if order is not None:
                    self._short_taken  = True
                    self._trades_today += 1
                    return order

        return None

    def _build_order(self, direction: int, entry: float, reason: str) -> Optional[Order]:
        sl_dist = self.sl_pct * entry
        if sl_dist <= 0.0:
            return None
        if direction == 1:
            sl_price = _tick(entry - sl_dist)
            tp_price = _tick(entry + self.rr_ratio * sl_dist)
        else:
            sl_price = _tick(entry + sl_dist)
            tp_price = _tick(entry - self.rr_ratio * sl_dist)

        equity    = self._current_equity()
        contracts = max(1, int(equity * self.risk_per_trade / (sl_dist * POINT_VALUE)))

        return Order(
            direction    = direction,
            order_type   = OrderType.MARKET,
            size_type    = SizeType.CONTRACTS,
            size_value   = float(contracts),
            sl_price     = sl_price,
            tp_price     = tp_price,
            trade_reason = reason,
        )

    # ── Position management ───────────────────────────────────────────────────

    def on_fill(self, position: OpenPosition, data: MarketData, bar_index: int) -> None:
        position.set_initial_sl_tp(position.sl_price, position.tp_price)

    def manage_position(
        self, data: MarketData, i: int, position: OpenPosition,
    ) -> Optional[PositionUpdate]:
        self._setup(data)

        t_min = int(self._times_min[i])

        # Force-close at 15:00
        if t_min >= _WIN_END_MIN:
            cl = float(data.close_1m[i])
            return PositionUpdate(new_tp_price=cl)

        return None
