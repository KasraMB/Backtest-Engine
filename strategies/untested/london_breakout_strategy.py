from __future__ import annotations

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


def _wilder_atr_full(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int
) -> np.ndarray:
    n = len(close)
    atr = np.zeros(n)
    if n < period + 1:
        return atr
    seed = 0.0
    for k in range(period):
        hl = high[k] - low[k]
        if k == 0:
            seed = hl
        else:
            hc = abs(high[k] - close[k - 1])
            lc = abs(low[k] - close[k - 1])
            seed += max(hl, hc, lc)
    atr[period - 1] = seed / period
    alpha = 1.0 / period
    inv = 1.0 - alpha
    for i in range(period, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        atr[i] = atr[i - 1] * inv + max(hl, hc, lc) * alpha
    return atr


def _tick_round(price: float) -> float:
    return round(round(price / 0.25) * 0.25, 10)


class LondonBreakoutStrategy(BaseStrategy):
    """
    London / Pre-market Breakout Strategy for NQ futures.

    Builds a pre-market reference range (from `pre_session_start` to 09:29 ET)
    for each trading day. At NY open (09:30), enters long if price breaks above
    the pre-market high or short if price breaks below the pre-market low.
    Only trades within the first `entry_window_minutes` of the NY session.
    """

    trading_hours = [(_time(9, 30), _time(15, 59))]
    min_lookback = 1000

    def __init__(self, params: dict = None) -> None:
        super().__init__(params)
        p = params or {}

        self.pre_session_start_hour: int = int(p.get("pre_session_start_hour", 4))
        self.pre_session_start_minute: int = int(p.get("pre_session_start_minute", 0))
        self.entry_window_minutes: int = int(p.get("entry_window_minutes", 60))
        self.rr_ratio: float = float(p.get("rr_ratio", 1.5))
        self.sl_atr_multiplier: float = float(p.get("sl_atr_multiplier", 1.0))
        self.atr_period: int = int(p.get("atr_period", 14))
        self.risk_pct: float = float(p.get("risk_pct", 0.01))

        # Compute entry cutoff time: 9:30 + entry_window_minutes
        total_min = 9 * 60 + 30 + self.entry_window_minutes
        self._entry_cutoff = _time(total_min // 60, total_min % 60)

        # Bar-indexed arrays populated by _setup
        self._pre_high: Optional[np.ndarray] = None
        self._pre_low: Optional[np.ndarray] = None
        self._atr: Optional[np.ndarray] = None

    def _setup(self, data: MarketData) -> None:
        if self._atr is not None:
            return

        idx = data.df_1m.index
        t_arr = idx.time
        pre_start = _time(self.pre_session_start_hour, self.pre_session_start_minute)
        pre_end = _time(9, 29, 59)
        in_pre = np.array([(pre_start <= t <= pre_end) for t in t_arr])
        dates = idx.date

        df_pre = pd.DataFrame(
            {
                "date": dates,
                "high": data.high_1m,
                "low": data.low_1m,
                "in_pre": in_pre,
            }
        )
        df_pre_filtered = df_pre[df_pre["in_pre"]]
        pre_range = df_pre_filtered.groupby("date").agg(
            pre_high=("high", "max"),
            pre_low=("low", "min"),
        )
        date_to_pre_high = pre_range["pre_high"].to_dict()
        date_to_pre_low = pre_range["pre_low"].to_dict()

        self._pre_high = np.array([date_to_pre_high.get(d, np.nan) for d in dates])
        self._pre_low = np.array([date_to_pre_low.get(d, np.nan) for d in dates])
        self._atr = _wilder_atr_full(
            data.high_1m, data.low_1m, data.close_1m, self.atr_period
        )

    def generate_signals(self, data: MarketData, i: int) -> Optional[Order]:
        self._setup(data)

        t = data.df_1m.index[i]
        t_time = t.time()

        if t_time < _time(9, 30) or t_time >= self._entry_cutoff:
            return None

        if i < 1:
            return None

        pre_high = self._pre_high[i]
        pre_low = self._pre_low[i]
        if np.isnan(pre_high) or np.isnan(pre_low):
            return None

        close = float(data.close_1m[i])
        close_prev = float(data.close_1m[i - 1])
        atr = float(self._atr[i])
        if atr <= 0.0:
            return None

        # Long: just broke above pre-market high
        if close > pre_high and close_prev <= pre_high:
            sl = close - self.sl_atr_multiplier * atr
            tp = close + self.rr_ratio * self.sl_atr_multiplier * atr
            return Order(
                direction=1,
                size_value=self.risk_pct,
                size_type=SizeType.PCT_RISK,
                order_type=OrderType.MARKET,
                sl_price=sl,
                tp_price=tp,
                trade_reason="london_breakout_long",
            )

        # Short: just broke below pre-market low
        if close < pre_low and close_prev >= pre_low:
            sl = close + self.sl_atr_multiplier * atr
            tp = close - self.rr_ratio * self.sl_atr_multiplier * atr
            return Order(
                direction=-1,
                size_value=self.risk_pct,
                size_type=SizeType.PCT_RISK,
                order_type=OrderType.MARKET,
                sl_price=sl,
                tp_price=tp,
                trade_reason="london_breakout_short",
            )

        return None

    def on_fill(self, position: OpenPosition, data: MarketData, bar_index: int) -> None:
        self._setup(data)

        entry = position.entry_price
        atr = float(self._atr[bar_index])
        if atr <= 0.0:
            return

        if position.is_long():
            sl = _tick_round(entry - self.sl_atr_multiplier * atr)
            tp = _tick_round(entry + self.rr_ratio * self.sl_atr_multiplier * atr)
        else:
            sl = _tick_round(entry + self.sl_atr_multiplier * atr)
            tp = _tick_round(entry - self.rr_ratio * self.sl_atr_multiplier * atr)

        position.set_initial_sl_tp(sl, tp)

    def manage_position(
        self, data: MarketData, i: int, position: OpenPosition
    ) -> Optional[PositionUpdate]:
        return None
