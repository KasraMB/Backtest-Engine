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


class DualThrustStrategy(BaseStrategy):
    """
    Dual Thrust breakout strategy for NQ futures.

    Computes daily high/low/open/close from prior `lookback_days` days to build
    upper and lower breakout levels. Enters long on a close above the upper level,
    short on a close below the lower level, within the first `entry_window_minutes`
    of the NY session.
    """

    trading_hours = [(_time(9, 30), _time(15, 59))]
    min_lookback = 5000

    def __init__(self, params: dict = None) -> None:
        super().__init__(params)
        p = params or {}

        self.lookback_days: int = int(p.get("lookback_days", 1))
        self.k_upper: float = float(p.get("k_upper", 0.5))
        self.k_lower: float = float(p.get("k_lower", 0.5))
        self.rr_ratio: float = float(p.get("rr_ratio", 1.5))
        self.sl_atr_multiplier: float = float(p.get("sl_atr_multiplier", 1.0))
        self.atr_period: int = int(p.get("atr_period", 14))
        self.risk_pct: float = float(p.get("risk_pct", 0.01))
        self.entry_window_minutes: int = int(p.get("entry_window_minutes", 240))

        # Compute entry cutoff time: 9:30 + entry_window_minutes
        total_min = 9 * 60 + 30 + self.entry_window_minutes
        self._entry_cutoff = _time(total_min // 60, total_min % 60)

        # Bar-indexed arrays populated by _setup
        self._upper: Optional[np.ndarray] = None
        self._lower: Optional[np.ndarray] = None
        self._atr: Optional[np.ndarray] = None

    def _setup(self, data: MarketData) -> None:
        if self._atr is not None:
            return

        idx = data.df_1m.index
        dates = idx.date

        df_daily = pd.DataFrame(
            {
                "high": data.high_1m,
                "low": data.low_1m,
                "open": data.open_1m,
                "close": data.close_1m,
                "date": dates,
            }
        )
        daily = df_daily.groupby("date").agg(
            day_high=("high", "max"),
            day_low=("low", "min"),
            day_open=("open", "first"),
            day_close=("close", "last"),
        )

        lb = self.lookback_days
        hh = daily["day_high"].rolling(lb).max()
        lc = daily["day_close"].rolling(lb).min()
        hc = daily["day_close"].rolling(lb).max()
        ll = daily["day_low"].rolling(lb).min()
        range_val = (hh - lc).combine((hc - ll), max)

        # Shift by 1: today's levels are computed from prior lb days
        daily_upper = (daily["day_open"] + self.k_upper * range_val.shift(1)).to_dict()
        daily_lower = (daily["day_open"] - self.k_lower * range_val.shift(1)).to_dict()

        upper_arr = np.array([daily_upper.get(d, np.nan) for d in dates])
        lower_arr = np.array([daily_lower.get(d, np.nan) for d in dates])

        self._upper = upper_arr
        self._lower = lower_arr
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

        upper = self._upper[i]
        lower = self._lower[i]
        if np.isnan(upper) or np.isnan(lower):
            return None

        upper_prev = self._upper[i - 1]
        lower_prev = self._lower[i - 1]
        if np.isnan(upper_prev) or np.isnan(lower_prev):
            return None

        close = float(data.close_1m[i])
        close_prev = float(data.close_1m[i - 1])
        atr = float(self._atr[i])
        if atr <= 0.0:
            return None

        # Long: price just crossed above upper level
        if close > upper and close_prev <= upper_prev:
            sl = close - self.sl_atr_multiplier * atr
            tp = close + self.rr_ratio * self.sl_atr_multiplier * atr
            return Order(
                direction=1,
                size_value=self.risk_pct,
                size_type=SizeType.PCT_RISK,
                order_type=OrderType.MARKET,
                sl_price=sl,
                tp_price=tp,
                trade_reason="dual_thrust_long",
            )

        # Short: price just crossed below lower level
        if close < lower and close_prev >= lower_prev:
            sl = close + self.sl_atr_multiplier * atr
            tp = close - self.rr_ratio * self.sl_atr_multiplier * atr
            return Order(
                direction=-1,
                size_value=self.risk_pct,
                size_type=SizeType.PCT_RISK,
                order_type=OrderType.MARKET,
                sl_price=sl,
                tp_price=tp,
                trade_reason="dual_thrust_short",
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
