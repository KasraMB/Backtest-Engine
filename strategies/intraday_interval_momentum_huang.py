"""
Intraday Interval Momentum — Huang, Luo & Ye (2023)
"Intraday Momentum in the VIX Futures Market"

Mechanism
─────────
The session is divided into equal intervals. The return during each completed
interval predicts the next interval's return.

Signal:
    r_interval = (interval_close / interval_open) - 1

Trade rule (fired at the start of the next interval):
    r_interval >  threshold → long
    r_interval < -threshold → short

Entry: MARKET at the first bar of the next interval.
Exit:  EOD (engine-enforced) or at the end of the next interval via
       manage_position closing the position with a MARKET signal.
       No SL or TP.

Optional vol_gate: only trade when today's session range so far exceeds
atr_mult × ATR — the effect is stronger in volatile sessions.

Params dict keys:
    interval_m          : int   — minutes per interval. Default 30
    threshold_pct       : float — min |r_interval| % to enter. Default 0.0
    exit_at_interval_end: bool  — close via signal at next interval end. Default True
    vol_gate            : bool  — session range > atr_mult × ATR required. Default False
    atr_length          : int   — ATR period. Default 14
    atr_mult            : float — session range multiplier for vol gate. Default 1.0
    max_trades_per_day  : int   — cap on daily entries. Default 4
    contracts           : int   — fixed contracts. Default 1
"""
from __future__ import annotations

from datetime import time
from typing import Optional

import numpy as np

from backtest.data.market_data import MarketData
from backtest.strategy.base import BaseStrategy
from backtest.strategy.enums import OrderType, SizeType
from backtest.strategy.order import Order
from backtest.strategy.update import OpenPosition, PositionUpdate

_RTH_START_MIN = 9 * 60 + 30   # 09:30 ET in minutes since midnight
_RTH_LENGTH_MIN = 450           # 09:30–17:00 ET = 450 minutes (futures daily close 16:00 CT)


def _atr(highs, lows, closes, period, i) -> float:
    if i < 1:
        return 0.0
    start = max(1, i - period * 3)
    trs = np.maximum(
        highs[start:i+1] - lows[start:i+1],
        np.abs(highs[start:i+1] - closes[start-1:i]),
        np.abs(lows[start:i+1]  - closes[start-1:i]),
    )
    if len(trs) == 0:
        return 0.0
    atr = float(trs[0])
    for tr in trs[1:]:
        atr = (atr * (period - 1) + float(tr)) / period
    return atr


def _build_intervals(interval_m: int) -> list[tuple[time, time]]:
    """List of (start, end) time pairs covering RTH 09:30–17:00 ET."""
    result = []
    t = _RTH_START_MIN
    rth_end = _RTH_START_MIN + _RTH_LENGTH_MIN
    while t < rth_end:
        s_h, s_m = divmod(t, 60)
        e = min(t + interval_m, rth_end)
        e_h, e_m = divmod(e, 60)
        result.append((time(s_h, s_m), time(e_h, e_m)))
        t += interval_m
    return result


class IntradayIntervalMomentumHuang(BaseStrategy):
    """
    Rolling short-interval momentum.
    Each completed interval signals direction for the next.
    Exit at next interval end (or EOD). No SL or TP.
    """

    trading_hours = [(time(8, 30), time(16, 0))]
    min_lookback  = 20

    def __init__(self, params: dict = None):
        super().__init__(params)
        p = params or {}

        self.interval_m:          int   = p.get("interval_m",          30)
        self.threshold_pct:       float = p.get("threshold_pct",        0.0)
        self.exit_at_interval_end: bool = p.get("exit_at_interval_end", True)
        self.vol_gate:            bool  = p.get("vol_gate",             False)
        self.atr_length:          int   = p.get("atr_length",           14)
        self.atr_mult:            float = p.get("atr_mult",             1.0)
        self.max_trades_per_day:  int   = p.get("max_trades_per_day",   4)
        self.contracts:           int   = p.get("contracts",            1)

        self._intervals   = _build_intervals(self.interval_m)
        # O(1) lookup: which times are interval boundaries
        self._end_times: set[time] = {end for _, end in self._intervals}
        # Map: interval start time → its end time (for manage_position exit)
        self._start_to_end: dict[time, time] = {s: e for s, e in self._intervals}

        # Per-day state
        self._current_date        = None
        self._trades_today:   int = 0
        self._interval_open: float = 0.0
        self._interval_start_time: Optional[time] = None
        self._pending_direction: Optional[int] = None   # carry signal one bar
        self._session_high:  float = 0.0
        self._session_low:   float = float("inf")
        # Track end time of the current open interval (for exit logic)
        self._current_interval_end: Optional[time] = None

    def _reset_day(self) -> None:
        self._trades_today          = 0
        self._interval_open         = 0.0
        self._interval_start_time   = None
        self._pending_direction     = None
        self._session_high          = 0.0
        self._session_low           = float("inf")
        self._current_interval_end  = None

    def generate_signals(self, data: MarketData, i: int) -> Optional[Order]:
        ts       = data.df_1m.index[i]
        bar_time = ts.time()
        bar_date = ts.date()
        close_i  = float(data.close_1m[i])
        high_i   = float(data.high_1m[i])
        low_i    = float(data.low_1m[i])

        if bar_date != self._current_date:
            self._reset_day()
            self._current_date = bar_date

        # Track session range for vol gate
        if bar_time >= time(8, 30):
            self._session_high = max(self._session_high, high_i)
            self._session_low  = min(self._session_low,  low_i)

        # Capture interval open on first bar of each interval
        for start, end in self._intervals:
            if bar_time == start:
                self._interval_open       = float(data.open_1m[i])
                self._interval_start_time = start
                break

        # At interval end: compute signal for next interval
        if bar_time in self._end_times and self._interval_open > 0:
            r = (close_i / self._interval_open) - 1.0
            self._pending_direction = (1 if r > 0 else -1) if abs(r) >= self.threshold_pct / 100.0 else None
            self._interval_open = 0.0

        # Fire signal on bar immediately after an interval boundary
        if self._pending_direction is None:
            return None
        if self._trades_today >= self.max_trades_per_day:
            self._pending_direction = None
            return None

        # Vol gate
        if self.vol_gate:
            session_range = self._session_high - self._session_low
            atr = _atr(data.high_1m, data.low_1m, data.close_1m, self.atr_length, i)
            if atr > 0 and session_range < self.atr_mult * atr:
                self._pending_direction = None
                return None

        direction = self._pending_direction
        self._pending_direction = None

        # Record which interval end we should exit at
        # Find the next interval whose start is at or after bar_time
        self._current_interval_end = None
        for start, end in self._intervals:
            if start >= bar_time:
                self._current_interval_end = end
                break

        self._trades_today += 1
        return Order(
            direction  = direction,
            order_type = OrderType.MARKET,
            size_type  = SizeType.CONTRACTS,
            size_value = self.contracts,
        )

    def manage_position(self, data: MarketData, i: int,
                        position: OpenPosition) -> Optional[PositionUpdate]:
        """
        Exit at the end of the current interval by forcing SL beyond price.
        Setting new_sl_price beyond current price triggers a FORCED_EXIT at
        bar.close via apply_position_update's enforcement logic.
        """
        if not self.exit_at_interval_end:
            return None
        if self._current_interval_end is None:
            return None
        bar_time = data.df_1m.index[i].time()
        if bar_time >= self._current_interval_end:
            current_price = float(data.close_1m[i])
            # For long: set SL above current price → forced exit
            # For short: set SL below current price → forced exit
            if position.is_long():
                return PositionUpdate(new_sl_price=current_price + 0.25)
            else:
                return PositionUpdate(new_sl_price=current_price - 0.25)
        return None