"""
Intraday Time-Series Momentum — Jin, Lai & Zheng (2020)
"Intraday Time-Series Momentum: Evidence from China"

Mechanism
─────────
The return during the first open_window_m minutes of the session positively
predicts the return during the last window before close.

Signal:
    r_open = (close_at_open_window_end / session_open) - 1

Trade rule:
    r_open >  threshold → long  at close_entry_time
    r_open < -threshold → short at close_entry_time

Entry: MARKET order at close_entry_time (default 15:00 ET).
Exit:  EOD only — engine closes at eod_exit_time. No SL or TP.

Optional vol filter (Jin et al.): only trade when the open-window range
exceeds atr_multiplier × ATR — the effect is stronger in volatile sessions.

Params dict keys:
    open_window_m   : int   — minutes in open window. Default 30
    close_entry_h   : int   — entry hour. Default 15
    close_entry_m   : int   — entry minute. Default 0
    threshold_pct   : float — min |r_open| % to trade. Default 0.0
    vol_filter      : bool  — require above-ATR range in open window. Default False
    atr_length      : int   — ATR period for vol filter. Default 14
    atr_multiplier  : float — open range must exceed atr_mult × ATR. Default 0.5
    contracts       : int   — fixed contracts. Default 1
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


class IntradayMomentumJin(BaseStrategy):
    """
    First-window return predicts last-window return.
    One trade per day. Entry MARKET at close_entry_time, exit EOD only.
    """

    trading_hours = [(time(8, 30), time(16, 0))]
    min_lookback  = 35

    def __init__(self, params: dict = None):
        super().__init__(params)
        p = params or {}

        self.open_window_m:  int   = p.get("open_window_m",  30)
        self.close_entry_h:  int   = p.get("close_entry_h",  14)
        self.close_entry_m:  int   = p.get("close_entry_m",   0)
        self.threshold_pct:  float = p.get("threshold_pct",  0.0)
        self.vol_filter:     bool  = p.get("vol_filter",      False)
        self.atr_length:     int   = p.get("atr_length",      14)
        self.atr_multiplier: float = p.get("atr_multiplier",  0.5)
        self.contracts:      int   = p.get("contracts",        1)

        ow_total_m = 30 + self.open_window_m
        self._open_window_end = time(8 + ow_total_m // 60, ow_total_m % 60)
        self._close_entry     = time(self.close_entry_h, self.close_entry_m)

        self._current_date    = None
        self._session_open:   float = 0.0
        self._open_win_close: float = 0.0
        self._open_win_high:  float = 0.0
        self._open_win_low:   float = float("inf")
        self._open_win_done:  bool  = False
        self._traded_today:   bool  = False

    def _reset_day(self) -> None:
        self._session_open    = 0.0
        self._open_win_close  = 0.0
        self._open_win_high   = 0.0
        self._open_win_low    = float("inf")
        self._open_win_done   = False
        self._traded_today    = False

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

        if bar_time == time(8, 30) and self._session_open == 0.0:
            self._session_open = float(data.open_1m[i])

        # Build open window
        if time(8, 30) <= bar_time < self._open_window_end and not self._open_win_done:
            self._open_win_high  = max(self._open_win_high, high_i)
            self._open_win_low   = min(self._open_win_low,  low_i)
            self._open_win_close = close_i

        if bar_time >= self._open_window_end and not self._open_win_done:
            self._open_win_done  = True
            self._open_win_close = close_i

        if self._traded_today or bar_time != self._close_entry:
            return None
        if not self._open_win_done or self._session_open <= 0:
            return None

        r_open = (self._open_win_close / self._session_open) - 1.0

        if abs(r_open) < self.threshold_pct / 100.0:
            return None

        if self.vol_filter:
            open_range = self._open_win_high - self._open_win_low
            atr = _atr(data.high_1m, data.low_1m, data.close_1m, self.atr_length, i)
            if atr > 0 and open_range < self.atr_multiplier * atr:
                return None

        self._traded_today = True
        return Order(
            direction  = 1 if r_open > 0 else -1,
            order_type = OrderType.MARKET,
            size_type  = SizeType.CONTRACTS,
            size_value = self.contracts,
        )

    def manage_position(self, data: MarketData, i: int,
                        position: OpenPosition) -> Optional[PositionUpdate]:
        return None