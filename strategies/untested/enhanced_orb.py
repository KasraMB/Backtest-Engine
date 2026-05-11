"""
Enhanced Opening Range Breakout (ORB) Strategy
───────────────────────────────────────────────
Exact Python replica of the PineScript "Enhanced Opening Range Strategy".

Pine entry logic:
  - When close > session_high and consecutive_bullish_closes >= breakout_candles:
    enter MARKET at bar close, SL/TP anchored to session_high (not fill price)
  - When close < session_low and consecutive_bearish_closes >= breakout_candles:
    enter MARKET at bar close, SL/TP anchored to session_low (not fill price)
  - Reverse logic: entry at close, SL/TP anchored to close (same as Pine)
  - Second chance: detected by price re-entering the range when position closes
    (Pine: close < session_high for long exit, close > session_low for short exit)

Params dict keys:
    or_minutes          : int   — opening range duration in minutes (5/15/30/45/60). Default 15
    breakout_candles    : int   — consecutive closes needed to confirm breakout. Default 2
    reverse_logic       : bool  — short on bull breakout, long on bear breakout. Default False
    sl_type             : str   — "atr_multiple"|"range_pct"|"fixed_pct"|"fixed_points"|"opposite_range"
    sl_value            : float — multiplier/value for sl_type. Default 2.0
    tp_type             : str   — "risk_reward"|"range_pct"|"atr_multiple"|"fixed_pct"|"fixed_points"
    tp_value            : float — multiplier/value for tp_type. Default 1.0
    atr_length          : int   — ATR period. Default 14
    enable_second_chance: bool  — allow one more trade if first hits SL. Default False
    contracts           : int   — fixed contracts. Default 1
"""
from __future__ import annotations

from datetime import time, date
from typing import Optional

import numpy as np

from backtest.data.market_data import MarketData
from backtest.strategy.base import BaseStrategy
from backtest.strategy.enums import OrderType, SizeType
from backtest.strategy.order import Order
from backtest.strategy.update import OpenPosition, PositionUpdate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
         period: int, i: int) -> float:
    """Wilder ATR up to bar i (inclusive). Returns 0.0 if not enough bars."""
    if i < 1:
        return 0.0
    start = max(1, i - period * 3)
    trs = np.maximum(
        highs[start:i + 1] - lows[start:i + 1],
        np.abs(highs[start:i + 1] - closes[start - 1:i]),
        np.abs(lows[start:i + 1]  - closes[start - 1:i]),
    )
    if len(trs) == 0:
        return 0.0
    atr = float(trs[0])
    for tr in trs[1:]:
        atr = (atr * (period - 1) + float(tr)) / period
    return atr


def _calc_sl(entry: float, is_long: bool, sl_type: str, sl_value: float,
             range_size: float, atr: float,
             session_high: float, session_low: float) -> float:
    if sl_type == "range_pct":
        offset = range_size * sl_value / 100.0
        return entry - offset if is_long else entry + offset
    elif sl_type == "atr_multiple":
        offset = atr * sl_value
        return entry - offset if is_long else entry + offset
    elif sl_type == "fixed_pct":
        return entry * (1 - sl_value / 100.0) if is_long else entry * (1 + sl_value / 100.0)
    elif sl_type == "fixed_points":
        return entry - sl_value if is_long else entry + sl_value
    elif sl_type == "opposite_range":
        return session_low if is_long else session_high
    else:
        raise ValueError(f"Unknown sl_type: {sl_type!r}")


def _calc_tp(entry: float, sl: float, is_long: bool, tp_type: str, tp_value: float,
             range_size: float, atr: float) -> float:
    risk = abs(entry - sl)
    if tp_type == "risk_reward":
        return entry + risk * tp_value if is_long else entry - risk * tp_value
    elif tp_type == "range_pct":
        offset = range_size * tp_value / 100.0
        return entry + offset if is_long else entry - offset
    elif tp_type == "atr_multiple":
        offset = atr * tp_value
        return entry + offset if is_long else entry - offset
    elif tp_type == "fixed_pct":
        return entry * (1 + tp_value / 100.0) if is_long else entry * (1 - tp_value / 100.0)
    elif tp_type == "fixed_points":
        return entry + tp_value if is_long else entry - tp_value
    else:
        raise ValueError(f"Unknown tp_type: {tp_type!r}")


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

class EnhancedORBStrategy(BaseStrategy):
    """
    Exact Python replica of the PineScript Enhanced Opening Range Strategy.
    """

    trading_hours = [(time(9, 30), time(16, 0))]
    min_lookback  = 14

    def __init__(self, params: dict = None):
        super().__init__(params)
        params = params or {}

        self.or_minutes:           int   = params.get("or_minutes",           15)
        self.breakout_candles:     int   = params.get("breakout_candles",      2)
        self.reverse_logic:        bool  = params.get("reverse_logic",         False)
        self.sl_type:              str   = params.get("sl_type",               "atr_multiple")
        self.sl_value:             float = params.get("sl_value",              2.0)
        self.tp_type:              str   = params.get("tp_type",               "risk_reward")
        self.tp_value:             float = params.get("tp_value",              1.0)
        self.atr_length:           int   = params.get("atr_length",            14)
        self.enable_second_chance: bool  = params.get("enable_second_chance",  False)
        self.contracts:            int   = params.get("contracts",             1)

        # OR end time: 09:30 + or_minutes
        # e.g. 15 min → 09:45, 60 min → 10:30
        or_end_h = 9 + (30 + self.or_minutes) // 60
        or_end_m = (30 + self.or_minutes) % 60
        self._or_end_time = time(or_end_h, or_end_m)

        # Per-day state
        self._current_date:        Optional[date] = None
        self._in_or:               bool  = False
        self._or_complete:         bool  = False
        self._session_high:        float = 0.0
        self._session_low:         float = float("inf")

        self._traded_today:        bool  = False
        self._first_trade_dir:     int   = 0       # 1=long, -1=short
        self._first_trade_loss:    bool  = False   # Pine: price back inside range on close
        self._second_chance_taken: bool  = False

        self._consec_bull:         int   = 0
        self._consec_bear:         int   = 0

        self._was_in_position:     bool  = False

        # SL/TP anchor — Pine anchors to session_high/low, not fill price
        # Stored at signal time so on_fill can use the correct anchor
        self._sl_anchor:           float = 0.0

    def _reset_day(self) -> None:
        self._in_or              = False
        self._or_complete        = False
        self._session_high       = 0.0
        self._session_low        = float("inf")
        self._traded_today       = False
        self._first_trade_dir    = 0
        self._first_trade_loss   = False
        self._second_chance_taken= False
        self._consec_bull        = 0
        self._consec_bear        = 0
        self._was_in_position    = False
        self._sl_anchor          = 0.0

    def generate_signals(self, data: MarketData, i: int) -> Optional[Order]:
        ts       = data.df_1m.index[i]
        bar_time = ts.time()
        bar_date = ts.date()
        high_i   = data.high_1m[i]
        low_i    = data.low_1m[i]
        close_i  = data.close_1m[i]

        # ── Detect position close — check for first_trade_loss ───────────────
        # Pine: when position closes, check if close is back inside the range.
        # (first_trade_direction == "long" and close < session_high) OR
        # (first_trade_direction == "short" and close > session_low)
        if self._was_in_position and len(self.closed_trades) > 0:
            if not self._second_chance_taken:
                last_dir = self._first_trade_dir
                if last_dir == 1 and close_i < self._session_high:
                    self._first_trade_loss = True
                elif last_dir == -1 and close_i > self._session_low:
                    self._first_trade_loss = True
            self._was_in_position = False

        # ── Day rollover ──────────────────────────────────────────────────────
        if bar_date != self._current_date:
            self._reset_day()
            self._current_date = bar_date

        # ── Opening range building ────────────────────────────────────────────
        # Pine: in_session covers 09:30 up to (but not including) or_end_time
        if time(9, 30) <= bar_time < self._or_end_time:
            self._in_or = True
            self._session_high = max(self._session_high, high_i)
            self._session_low  = min(self._session_low,  low_i)
            return None

        # ── OR completion ─────────────────────────────────────────────────────
        # Pine fires `if in_session[1] and not in_session` at the first bar
        # after the session — that is or_end_time bar. OR does NOT include it.
        if self._in_or and bar_time >= self._or_end_time:
            self._in_or       = False
            self._or_complete = True
            # Do NOT update session_high/low — this bar is not part of the OR

        if not self._or_complete:
            return None

        # ── Consecutive close tracking ────────────────────────────────────────
        if close_i > self._session_high:
            self._consec_bull += 1
            self._consec_bear  = 0
        elif close_i < self._session_low:
            self._consec_bear += 1
            self._consec_bull  = 0
        else:
            self._consec_bull = 0
            self._consec_bear = 0

        # ── Trade gate ────────────────────────────────────────────────────────
        can_second_chance = (
            self.enable_second_chance
            and self._first_trade_loss
            and not self._second_chance_taken
        )
        if self._traded_today and not can_second_chance:
            return None

        atr = _atr(data.high_1m, data.low_1m, data.close_1m, self.atr_length, i)

        # ── Bullish breakout ──────────────────────────────────────────────────
        if close_i > self._session_high and self._consec_bull >= self.breakout_candles:
            if self._traded_today:
                # Second chance: first trade must have been short
                if self._first_trade_dir != -1:
                    return None
                direction = 1 if not self.reverse_logic else -1
            else:
                direction = 1 if not self.reverse_logic else -1

            # Pine: normal entry at session_high (anchor for SL/TP)
            #       reverse entry at close (anchor for SL/TP)
            if not self.reverse_logic:
                self._sl_anchor = self._session_high
            else:
                self._sl_anchor = float(close_i)

            if self._traded_today:
                self._second_chance_taken = True
            else:
                self._traded_today    = True
                self._first_trade_dir = direction
            self._was_in_position = True

            return Order(
                direction  = direction,
                order_type = OrderType.MARKET,
                size_type  = SizeType.CONTRACTS,
                size_value = self.contracts,
            )

        # ── Bearish breakout ──────────────────────────────────────────────────
        if close_i < self._session_low and self._consec_bear >= self.breakout_candles:
            if self._traded_today:
                # Second chance: first trade must have been long
                if self._first_trade_dir != 1:
                    return None
                direction = -1 if not self.reverse_logic else 1
            else:
                direction = -1 if not self.reverse_logic else 1

            if not self.reverse_logic:
                self._sl_anchor = self._session_low
            else:
                self._sl_anchor = float(close_i)

            if self._traded_today:
                self._second_chance_taken = True
            else:
                self._traded_today    = True
                self._first_trade_dir = direction
            self._was_in_position = True

            return Order(
                direction  = direction,
                order_type = OrderType.MARKET,
                size_type  = SizeType.CONTRACTS,
                size_value = self.contracts,
            )

        return None

    def on_fill(self, position: OpenPosition, data: MarketData, bar_index: int) -> None:
        """
        Set SL/TP anchored to session_high/low (not fill price) — matching Pine.
        Pine: entry_price = session_high/low, SL/TP calculated from that anchor.
        """
        anchor  = self._sl_anchor if self._sl_anchor != 0.0 else position.entry_price
        is_long = position.is_long()
        atr     = _atr(data.high_1m, data.low_1m, data.close_1m,
                       self.atr_length, bar_index)
        rng_size = self._session_high - self._session_low

        sl = _calc_sl(anchor, is_long,
                      self.sl_type, self.sl_value,
                      rng_size, atr,
                      self._session_high, self._session_low)
        tp = _calc_tp(anchor, sl, is_long,
                      self.tp_type, self.tp_value,
                      rng_size, atr)

        position.set_initial_sl_tp(sl, tp)

    def manage_position(self, data: MarketData, i: int,
                        position: OpenPosition) -> Optional[PositionUpdate]:
        self._was_in_position = True
        return None