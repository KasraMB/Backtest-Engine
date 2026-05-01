"""
Gap Fill Strategy (NQ Futures)

When NQ opens with a gap vs. prior RTH close, fade the gap.
Entry: MARKET at 9:30 open bar
TP:    prior close (the gap fill level)
SL:    entry +/- gap_size * sl_frac
Expiry: expiry_mins bars (~1 bar per minute)

Filters:
  min_gap_atr  — minimum gap / ATR ratio to trade
  max_gap_atr  — maximum gap / ATR ratio (huge gaps rarely fill intraday)
  direction    — 'fade' (always fade), 'long' (only buy gap-downs), 'short' (only sell gap-ups)

Prior close = close of 15:59 bar of the prior session (stored via RTH-end state).
"""
from __future__ import annotations

from datetime import time as _time, date
from typing import Optional

import numpy as np

from backtest.data.market_data import MarketData
from backtest.strategy.base import BaseStrategy
from backtest.strategy.enums import OrderType, SizeType
from backtest.strategy.order import Order
from backtest.strategy.update import OpenPosition, PositionUpdate

POINT_VALUE = 20.0
TICK        = 0.25
_RTH_OPEN   = 9 * 60 + 30   # 09:30 ET in minutes
_RTH_CLOSE  = 15 * 60 + 59  # 15:59 ET in minutes


def _tick(price: float) -> float:
    return round(round(price / TICK) * TICK, 10)


def _atr_at(high: np.ndarray, low: np.ndarray, close: np.ndarray,
            period: int, i: int) -> float:
    if i < 1:
        return 0.0
    start = max(1, i - period * 3)
    trs = np.maximum(
        high[start:i + 1] - low[start:i + 1],
        np.abs(high[start:i + 1] - close[start - 1:i]),
        np.abs(low[start:i + 1]  - close[start - 1:i]),
    )
    if len(trs) == 0:
        return 0.0
    atr = float(trs[0])
    for tr in trs[1:]:
        atr = (atr * (period - 1) + float(tr)) / period
    return atr


class GapFillStrategy(BaseStrategy):
    """
    Fades the overnight gap at 9:30 RTH open.
    TP is anchored to prior RTH close; SL is sl_frac × gap_size beyond entry.
    """

    trading_hours = [(_time(9, 30), _time(15, 59))]
    min_lookback  = 20

    def __init__(self, params: dict = None) -> None:
        super().__init__(params)
        p = params or {}

        self.min_gap_atr    = float(p.get("min_gap_atr",    0.5))
        self.max_gap_atr    = float(p.get("max_gap_atr",    3.0))
        self.sl_frac        = float(p.get("sl_frac",        0.5))
        self.atr_period     = int(p.get("atr_period",       14))
        self.expiry_mins    = int(p.get("expiry_mins",       60))
        self.risk_per_trade = float(p.get("risk_per_trade", 0.01))
        # 'fade': short gap-ups, long gap-downs  'long': only gap-downs  'short': only gap-ups
        self.direction      = str(p.get("direction",       "fade"))

        self._times_min: Optional[np.ndarray] = None

        # Per-day state
        self._cur_date:        Optional[date] = None
        self._prev_close:      float = 0.0   # prior RTH session close (15:59)
        self._session_close:   float = 0.0   # tracks last close seen this session
        self._order_placed:    bool  = False

        # Passed to on_fill
        self._pending_prior_close: float = 0.0
        self._pending_sl_dist:     float = 0.0

    def _setup(self, data: MarketData) -> None:
        if self._times_min is not None:
            return
        idx             = data.df_1m.index
        self._times_min = (idx.hour * 60 + idx.minute).to_numpy(np.int32)
        if data.bar_times_1m_min is None:
            data.bar_times_1m_min = self._times_min

    def _reset_day(self) -> None:
        self._order_placed         = False
        self._pending_prior_close  = 0.0
        self._pending_sl_dist      = 0.0

    def generate_signals(self, data: MarketData, i: int) -> Optional[Order]:
        self._setup(data)

        t_min    = int(self._times_min[i])
        bar_date = data.df_1m.index[i].date()

        # Day rollover FIRST so _session_close still holds the prior session's last close.
        if bar_date != self._cur_date:
            if self._cur_date is not None and self._session_close > 0:
                self._prev_close = self._session_close
            self._reset_day()
            self._cur_date = bar_date

        # Update AFTER rollover so _session_close always = last close seen this session.
        self._session_close = float(data.close_1m[i])

        # Only trade the 9:30 bar (first bar of RTH) and only once per day
        if t_min != _RTH_OPEN or self._order_placed:
            return None
        if self._prev_close <= 0:
            return None  # no prior close available (first day in dataset)

        self._order_placed = True

        curr_open = float(data.open_1m[i])
        gap       = curr_open - self._prev_close   # positive = gap up, negative = gap down
        gap_abs   = abs(gap)

        atr = _atr_at(data.high_1m, data.low_1m, data.close_1m, self.atr_period, i)
        if atr <= 0 or gap_abs < 1e-6:
            return None

        gap_atr = gap_abs / atr
        if gap_atr < self.min_gap_atr or gap_atr > self.max_gap_atr:
            return None

        # Direction filter
        gap_up = gap > 0
        if self.direction == "long" and gap_up:
            return None   # only trade gap-downs for long
        if self.direction == "short" and not gap_up:
            return None   # only trade gap-ups for short

        # SL distance = gap_size * sl_frac
        sl_dist = _tick(max(gap_abs * self.sl_frac, TICK * 4))
        self._pending_sl_dist     = sl_dist
        self._pending_prior_close = self._prev_close

        if gap_up:
            # Fade gap up → SHORT
            tp_px = _tick(self._prev_close)
            sl_px = _tick(curr_open + sl_dist)
            return Order(
                direction    = -1,
                order_type   = OrderType.MARKET,
                size_type    = SizeType.PCT_RISK,
                size_value   = self.risk_per_trade,
                stop_price   = 0.0,
                sl_price     = sl_px,
                tp_price     = tp_px,
                expiry_bars  = self.expiry_mins,
                trade_reason = "gap_fade_short",
            )
        else:
            # Fade gap down → LONG
            tp_px = _tick(self._prev_close)
            sl_px = _tick(curr_open - sl_dist)
            return Order(
                direction    = 1,
                order_type   = OrderType.MARKET,
                size_type    = SizeType.PCT_RISK,
                size_value   = self.risk_per_trade,
                stop_price   = 0.0,
                sl_price     = sl_px,
                tp_price     = tp_px,
                expiry_bars  = self.expiry_mins,
                trade_reason = "gap_fade_long",
            )

    def on_fill(self, position: OpenPosition, data: MarketData, i: int) -> None:
        """Recompute SL from actual fill; TP anchored to prior close."""
        fill    = position.entry_price
        prior   = self._pending_prior_close
        sl_dist = _tick(max(self._pending_sl_dist, TICK))

        if position.is_long():
            tp_dist = prior - fill
            sl = _tick(fill - sl_dist)
        else:
            tp_dist = fill - prior
            sl = _tick(fill + sl_dist)

        if tp_dist < 1.0:
            # Gap already filled by the time we entered; extend TP past prior close
            # by half the SL distance so the trade can still produce a small gain.
            if position.is_long():
                tp = _tick(fill + sl_dist * 0.5)
            else:
                tp = _tick(fill - sl_dist * 0.5)
        else:
            tp = _tick(prior)

        position.set_initial_sl_tp(sl, tp)

    def manage_position(self, data: MarketData, i: int,
                        position: OpenPosition) -> Optional[PositionUpdate]:
        return None
