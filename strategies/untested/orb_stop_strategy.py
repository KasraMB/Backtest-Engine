"""
Opening Range Breakout — Stop Order Version (NQ Futures)

Range window:  9:30 ET → 9:30 + or_minutes (default 15 min)
At range close: places a STOP buy above range_high + tick OR sell below range_low - tick.
Direction: momentum of first or_minutes (close vs open of range window).
SL/TP computed in on_fill from actual fill price → proper RR guaranteed.

Expiry bars cancels the pending stop if not triggered by 11:00 ET.
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


def _tick(price: float) -> float:
    return round(round(price / TICK) * TICK, 10)


def _atr_at(high: np.ndarray, low: np.ndarray, close: np.ndarray,
            period: int, i: int) -> float:
    """Wilder ATR at bar i using only bars up to i (no lookahead)."""
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


class ORBStopStrategy(BaseStrategy):
    """
    Stop-order ORB: fills AT the breakout level, SL/TP from fill price.
    Momentum direction: close of last range bar vs open of first range bar.
    """

    trading_hours = [(_time(9, 30), _time(15, 59))]
    min_lookback  = 20

    def __init__(self, params: dict = None) -> None:
        super().__init__(params)
        p = params or {}

        self.or_minutes     = int(p.get("or_minutes",        15))
        self.rr_ratio       = float(p.get("rr_ratio",        1.5))
        self.sl_atr_mult    = float(p.get("sl_atr_multiplier", 1.0))
        self.atr_period     = int(p.get("atr_period",        14))
        self.risk_per_trade = float(p.get("risk_per_trade",  0.01))
        self.max_range_atr  = float(p.get("max_range_atr",   3.0))  # skip if range > N×ATR
        self.min_range_atr  = float(p.get("min_range_atr",   0.1))  # skip if range < N×ATR
        # 'momentum': follow range direction  'long': always long  'short': always short
        self.direction_mode = str(p.get("direction_mode",  "momentum"))

        # Computed range end time
        total_min = 9 * 60 + 30 + self.or_minutes
        self._range_end = _time(total_min // 60, total_min % 60)
        # Expiry: bars until 11:00 ET from range_end (≈75 min × 1min bars)
        _cutoff_min = 11 * 60
        _range_min  = total_min
        self._expiry_bars = max(30, _cutoff_min - _range_min)

        # Pre-computed time array (no lookahead — just bar metadata)
        self._times_min:  Optional[np.ndarray] = None

        # Per-day state
        self._cur_date:       Optional[date] = None
        self._range_high:     float = 0.0
        self._range_low:      float = float("inf")
        self._range_open:     float = 0.0   # first bar close in range
        self._range_close:    float = 0.0   # last bar close in range
        self._range_complete: bool  = False
        self._order_placed:   bool  = False
        self._in_range:       bool  = False
        self._pending_sl_dist: float = 0.0  # SL distance set in generate_signals, used in on_fill

    def _setup(self, data: MarketData) -> None:
        if self._times_min is not None:
            return
        idx             = data.df_1m.index
        self._times_min = (idx.hour * 60 + idx.minute).to_numpy(np.int32)
        if data.bar_times_1m_min is None:
            data.bar_times_1m_min = self._times_min

    def _reset_day(self) -> None:
        self._range_high     = 0.0
        self._range_low      = float("inf")
        self._range_open     = 0.0
        self._range_close    = 0.0
        self._range_complete = False
        self._order_placed   = False
        self._in_range       = False
        self._pending_sl_dist = 0.0

    def generate_signals(self, data: MarketData, i: int) -> Optional[Order]:
        self._setup(data)

        t_min     = int(self._times_min[i])
        bar_date  = data.df_1m.index[i].date()

        # Day rollover
        if bar_date != self._cur_date:
            self._reset_day()
            self._cur_date = bar_date

        start_min = 9 * 60 + 30
        end_min   = start_min + self.or_minutes  # first bar AFTER range

        # Build the opening range
        if start_min <= t_min < end_min:
            self._in_range = True
            h = data.high_1m[i]
            l = data.low_1m[i]
            c = data.close_1m[i]
            if self._range_open == 0.0:
                self._range_open = c
            self._range_high  = max(self._range_high, h)
            self._range_low   = min(self._range_low,  l)
            self._range_close = c
            return None

        # Mark range complete at the first bar after range window
        if self._in_range and t_min >= end_min:
            self._in_range       = False
            self._range_complete = True

        if not self._range_complete or self._order_placed:
            return None

        # Only place order on the first bar after range completion
        # (next bars: order_placed=True, no signal)
        self._order_placed = True

        atr        = _atr_at(data.high_1m, data.low_1m, data.close_1m, self.atr_period, i)
        range_size = self._range_high - self._range_low

        # Skip degenerate ranges
        if atr <= 0 or range_size <= 0:
            return None
        if range_size > self.max_range_atr * atr:
            return None
        if range_size < self.min_range_atr * atr:
            return None

        # Direction
        if self.direction_mode == "long":
            go_long = True
        elif self.direction_mode == "short":
            go_long = False
        else:  # momentum
            go_long = self._range_close >= self._range_open

        # sl_price in Order is used for PCT_RISK sizing from stop_px.
        # tp_price is a wide placeholder — on_fill overwrites both from actual fill.
        sl_dist = _tick(max(self.sl_atr_mult * atr, TICK * 4))  # min 1pt SL
        tp_dist = _tick(sl_dist * self.rr_ratio)
        self._pending_sl_dist = sl_dist

        if go_long:
            stop_px = _tick(self._range_high + TICK)
            # sl_price used for PCT_RISK sizing; tp_price set wide (on_fill overrides both)
            sl_px   = _tick(stop_px - sl_dist)
            tp_px   = _tick(stop_px + tp_dist + 100.0)  # wide placeholder, on_fill corrects
            return Order(
                direction   = 1,
                order_type  = OrderType.STOP,
                size_type   = SizeType.PCT_RISK,
                size_value  = self.risk_per_trade,
                stop_price  = stop_px,
                sl_price    = sl_px,
                tp_price    = tp_px,
                expiry_bars = self._expiry_bars,
                trade_reason= "orb_long",
            )
        else:
            stop_px = _tick(self._range_low - TICK)
            sl_px   = _tick(stop_px + sl_dist)
            tp_px   = _tick(stop_px - tp_dist - 100.0)  # wide placeholder, on_fill corrects
            return Order(
                direction   = -1,
                order_type  = OrderType.STOP,
                size_type   = SizeType.PCT_RISK,
                size_value  = self.risk_per_trade,
                stop_price  = stop_px,
                sl_price    = sl_px,
                tp_price    = tp_px,
                expiry_bars = self._expiry_bars,
                trade_reason= "orb_short",
            )

    def on_fill(self, position: OpenPosition, data: MarketData, i: int) -> None:
        """Recompute SL/TP from actual fill price to guarantee exact RR."""
        fill    = position.entry_price
        sl_dist = _tick(max(self._pending_sl_dist, TICK))
        tp_dist = _tick(sl_dist * self.rr_ratio)
        if position.is_long():
            sl = _tick(fill - sl_dist)
            tp = _tick(fill + tp_dist)
        else:
            sl = _tick(fill + sl_dist)
            tp = _tick(fill - tp_dist)
        position.set_initial_sl_tp(sl, tp)

    def manage_position(self, data: MarketData, i: int,
                        position: OpenPosition) -> Optional[PositionUpdate]:
        return None  # rely on RunConfig.eod_exit_time
