"""
Turtle Soup Strategy — NQ Futures (Linda Raschke)

Concept: fade N-bar range breakouts on a 4H timeframe.

4H bar construction from 1m data:
  Bars align to 4-hour boundaries: 00:00, 04:00, 08:00, 12:00, 16:00, 20:00 ET.
  A 4H bar is "complete" when the next 4H boundary starts.

Signal (SHORT — fading a top breakout):
  1. The most recently completed 4H bar's HIGH is a new N-bar (4H) highest high.
  2. The current 1m close drops back BELOW that breakout high.
  → Enter SHORT at close, SL = breakout high + sl_buffer (ATR × sl_atr_mult)

Signal (LONG — fading a bottom breakout):
  1. The most recently completed 4H bar's LOW is a new N-bar (4H) lowest low.
  2. The current 1m close rallies back ABOVE that breakout low.
  → Enter LONG at close, SL = breakout low - sl_buffer

TP = SL distance × rr_ratio.
Max 1 trade per 4H session (4-hour window).
Session filter optional.

Operate during RTH by default (9:30–16:00 ET).
"""
from __future__ import annotations

from datetime import time as _time
from typing import Optional

import numpy as np

from backtest.data.market_data import MarketData
from backtest.strategy.base import BaseStrategy
from backtest.strategy.enums import OrderType, SizeType
from backtest.strategy.order import Order
from backtest.strategy.update import OpenPosition, PositionUpdate

POINT_VALUE = 20.0
TICK        = 0.25

_RTH_START_MIN = 9 * 60 + 30   # 570
_RTH_END_MIN   = 16 * 60        # 960
_4H_MINS       = 4 * 60         # 240


def _tick(price: float) -> float:
    return round(round(price / TICK) * TICK, 10)


class TurtleSoupStrategy(BaseStrategy):
    """
    Contrarian 4H breakout fade on NQ.
    Checks every 1m bar within the RTH window for a fade setup.
    """

    trading_hours = [(_time(9, 30), _time(15, 59))]
    min_lookback  = 1

    def __init__(self, params: dict = None) -> None:
        super().__init__(params)
        p = params or {}

        self.lookback_4h    = int(p.get('lookback_4h',    20))
        self.sl_atr_mult    = float(p.get('sl_atr_mult',  0.5))
        self.rr_ratio       = float(p.get('rr_ratio',     2.0))
        self.atr_period     = int(p.get('atr_period',     14))
        self.risk_per_trade = float(p.get('risk_per_trade', 0.01))
        self.starting_equity = float(p.get('starting_equity', 100_000))
        self.equity_mode    = str(p.get('equity_mode',    'dynamic'))

        self._times_min:  Optional[np.ndarray] = None
        self._bar_day:    Optional[np.ndarray] = None
        self._atr_arr:    Optional[np.ndarray] = None   # per-1m ATR(atr_period) from 1m closes

        # Rolling window of completed 4H bar highs/lows — persists across days
        self._4h_highs:   list  = []
        self._4h_lows:    list  = []
        # Current (in-progress) 4H bar OHLC accumulators
        self._4h_high:    float = -np.inf
        self._4h_low:     float = np.inf
        # Global 4H bin index: (day_int * 1440 + t_min) // 240
        self._cur_4h_bin: int   = -1

        # Breakout fade levels, set when a new N-bar extreme is detected
        self._long_breakout:  float = np.nan
        self._short_breakout: float = np.nan
        self._traded_4h:      bool  = False

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

        # ATR(atr_period) on 1m closes — simple rolling mean of |close - prev_close|
        closes = data.close_1m
        diffs  = np.abs(np.diff(closes, prepend=closes[0]))
        kernel = np.ones(self.atr_period) / self.atr_period
        self._atr_arr = np.convolve(diffs, kernel, mode='full')[:len(closes)]

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

        t_min   = int(self._times_min[i])
        day_int = int(self._bar_day[i])
        cl      = float(data.close_1m[i])
        hi      = float(data.high_1m[i])
        lo      = float(data.low_1m[i])

        # Global 4H bin: unique across all days so the window persists
        bin_idx = (day_int * 1440 + t_min) // _4H_MINS

        # On 4H boundary: close the previous 4H bar, record high/low
        if bin_idx != self._cur_4h_bin:
            if self._cur_4h_bin >= 0 and self._4h_high > -np.inf:
                # Completed 4H bar
                self._4h_highs.append(self._4h_high)
                self._4h_lows.append(self._4h_low)
                if len(self._4h_highs) > self.lookback_4h:
                    self._4h_highs.pop(0)
                    self._4h_lows.pop(0)

                # Check for breakout on the just-completed 4H bar
                if len(self._4h_highs) >= self.lookback_4h:
                    prev_highs = self._4h_highs[:-1]
                    prev_lows  = self._4h_lows[:-1]
                    bar_high   = self._4h_highs[-1]
                    bar_low    = self._4h_lows[-1]

                    if bar_high >= max(prev_highs):
                        # New N-bar high breakout → set short fade level
                        self._short_breakout = bar_high
                    else:
                        self._short_breakout = np.nan

                    if bar_low <= min(prev_lows):
                        # New N-bar low breakout → set long fade level
                        self._long_breakout = bar_low
                    else:
                        self._long_breakout = np.nan

            # Start new 4H bar
            self._cur_4h_bin = bin_idx
            self._4h_high    = hi
            self._4h_low     = lo
            self._traded_4h  = False
        else:
            # Update current 4H bar running H/L
            if hi > self._4h_high:
                self._4h_high = hi
            if lo < self._4h_low:
                self._4h_low = lo

        if self._traded_4h:
            return None

        # RTH window check
        if t_min < _RTH_START_MIN or t_min >= _RTH_END_MIN:
            return None

        atr = float(self._atr_arr[i])
        if atr <= 0:
            return None

        sl_buf = self.sl_atr_mult * atr

        # Fade the short breakout: price drops back below breakout high
        if not np.isnan(self._short_breakout) and cl < self._short_breakout:
            sl_dist  = sl_buf + (self._short_breakout - cl)
            sl_price = _tick(self._short_breakout + sl_buf)
            tp_price = _tick(cl - self.rr_ratio * sl_dist)
            equity   = self._current_equity()
            contracts = max(1, int(equity * self.risk_per_trade / (sl_dist * POINT_VALUE)))
            self._traded_4h = True
            self._short_breakout = np.nan
            return Order(
                direction    = -1,
                order_type   = OrderType.MARKET,
                size_type    = SizeType.CONTRACTS,
                size_value   = float(contracts),
                sl_price     = sl_price,
                tp_price     = tp_price,
                trade_reason = 'ts_short',
            )

        # Fade the long breakout: price rallies back above breakout low
        if not np.isnan(self._long_breakout) and cl > self._long_breakout:
            sl_dist  = sl_buf + (cl - self._long_breakout)
            sl_price = _tick(self._long_breakout - sl_buf)
            tp_price = _tick(cl + self.rr_ratio * sl_dist)
            equity   = self._current_equity()
            contracts = max(1, int(equity * self.risk_per_trade / (sl_dist * POINT_VALUE)))
            self._traded_4h = True
            self._long_breakout = np.nan
            return Order(
                direction    = 1,
                order_type   = OrderType.MARKET,
                size_type    = SizeType.CONTRACTS,
                size_value   = float(contracts),
                sl_price     = sl_price,
                tp_price     = tp_price,
                trade_reason = 'ts_long',
            )

        return None

    def on_fill(self, position: OpenPosition, data: MarketData, bar_index: int) -> None:
        position.set_initial_sl_tp(position.sl_price, position.tp_price)

    def manage_position(
        self, data: MarketData, i: int, position: OpenPosition,
    ) -> Optional[PositionUpdate]:
        return None
