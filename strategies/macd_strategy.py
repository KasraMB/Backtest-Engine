"""
MACD Crossover Strategy (NQ Futures)

Trades bullish/bearish SMA crossovers (fast/slow) during the NY session.
SL/TP are ATR-based (Wilder's ATR), sized by risk_pct of equity.
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


def _wilder_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    n = len(close)
    atr = np.zeros(n)
    if n < period + 1:
        return atr
    seed = np.mean(high[:period] - low[:period])
    atr[period - 1] = seed
    for i in range(period, n):
        tr = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
        atr[i] = atr[i - 1] * (period - 1) / period + tr / period
    return atr


class MACDStrategy(BaseStrategy):
    """
    SMA crossover entry with Wilder ATR-based SL/TP.
    Fast/slow SMA crossover during NY session 09:30–15:59 ET.
    """

    trading_hours = [(_time(9, 30), _time(15, 59))]
    min_lookback = 200

    def __init__(self, params: dict = None) -> None:
        super().__init__(params)
        p = params or {}

        self.ma_fast: int           = int(p.get('ma_fast', 12))
        self.ma_slow: int           = int(p.get('ma_slow', 26))
        self.rr_ratio: float        = float(p.get('rr_ratio', 1.5))
        self.sl_atr_multiplier: float = float(p.get('sl_atr_multiplier', 1.0))
        self.atr_period: int        = int(p.get('atr_period', 14))
        self.risk_pct: float        = float(p.get('risk_pct', 0.01))

        self._ma_fast: Optional[np.ndarray] = None
        self._ma_slow: Optional[np.ndarray] = None
        self._atr: Optional[np.ndarray] = None

    def _setup(self, data: MarketData) -> None:
        if self._atr is not None:
            return
        close = data.close_1m
        n = len(close)

        # Compute fast SMA
        fast = np.zeros(n)
        for i in range(self.ma_fast - 1, n):
            fast[i] = np.mean(close[i - self.ma_fast + 1: i + 1])
        self._ma_fast = fast

        # Compute slow SMA
        slow = np.zeros(n)
        for i in range(self.ma_slow - 1, n):
            slow[i] = np.mean(close[i - self.ma_slow + 1: i + 1])
        self._ma_slow = slow

        self._atr = _wilder_atr(data.high_1m, data.low_1m, close, self.atr_period)

    def generate_signals(self, data: MarketData, i: int) -> Optional[Order]:
        self._setup(data)

        if i < self.min_lookback:
            return None
        if self._ma_slow[i] == 0.0 or self._atr[i] <= 0.0:
            return None

        t = data.df_1m.index[i]
        if not (_time(9, 30) <= t.time() <= _time(15, 59)):
            return None

        fast_cur  = self._ma_fast[i]
        fast_prev = self._ma_fast[i - 1]
        slow_cur  = self._ma_slow[i]
        slow_prev = self._ma_slow[i - 1]

        if fast_cur > slow_cur and fast_prev <= slow_prev:
            direction = 1
        elif fast_cur < slow_cur and fast_prev >= slow_prev:
            direction = -1
        else:
            return None

        atr = self._atr[i]
        sl_dist = self.sl_atr_multiplier * atr
        entry_approx = float(data.close_1m[i])
        if direction == 1:
            sl_price = round(round((entry_approx - sl_dist) / 0.25) * 0.25, 10)
            tp_price = round(round((entry_approx + self.rr_ratio * sl_dist) / 0.25) * 0.25, 10)
        else:
            sl_price = round(round((entry_approx + sl_dist) / 0.25) * 0.25, 10)
            tp_price = round(round((entry_approx - self.rr_ratio * sl_dist) / 0.25) * 0.25, 10)

        return Order(
            direction=direction,
            order_type=OrderType.MARKET,
            size_type=SizeType.PCT_RISK,
            size_value=self.risk_pct,
            sl_price=sl_price,
            tp_price=tp_price,
        )

    def on_fill(self, position: OpenPosition, data: MarketData, i: int) -> None:
        entry = position.entry_price
        atr = self._atr[i] if self._atr is not None else 0.0
        if atr <= 0.0:
            return
        sl_dist = self.sl_atr_multiplier * atr
        if position.is_long():
            sl = round(round((entry - sl_dist) / 0.25) * 0.25, 10)
            tp = round(round((entry + self.rr_ratio * sl_dist) / 0.25) * 0.25, 10)
        else:
            sl = round(round((entry + sl_dist) / 0.25) * 0.25, 10)
            tp = round(round((entry - self.rr_ratio * sl_dist) / 0.25) * 0.25, 10)
        position.set_initial_sl_tp(sl, tp)

    def manage_position(
        self,
        data: MarketData,
        i: int,
        position: OpenPosition,
    ) -> Optional[PositionUpdate]:
        return None
