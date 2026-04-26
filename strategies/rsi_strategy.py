"""
RSI Mean Reversion Strategy (NQ Futures)

Enters long when RSI crosses below oversold threshold (mean-reversion long),
short when RSI crosses above overbought threshold.
ATR-based SL/TP, sized by risk_pct of equity.
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


def _compute_rsi(close: np.ndarray, period: int) -> np.ndarray:
    n = len(close)
    rsi = np.full(n, 50.0)
    if n < period + 1:
        return rsi
    deltas = np.diff(close)
    gains  = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    for i in range(period, n - 1):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        rs = avg_gain / avg_loss if avg_loss > 1e-10 else 100.0
        rsi[i + 1] = 100.0 - 100.0 / (1.0 + rs)
    return rsi


class RSIStrategy(BaseStrategy):
    """
    RSI mean-reversion entry with Wilder ATR-based SL/TP.
    Enters on RSI crossing into oversold/overbought during NY session.
    """

    trading_hours = [(_time(9, 30), _time(15, 59))]
    min_lookback = 50

    def __init__(self, params: dict = None) -> None:
        super().__init__(params)
        p = params or {}

        self.rsi_period: int          = int(p.get('rsi_period', 14))
        self.oversold: float          = float(p.get('oversold', 30.0))
        self.overbought: float        = float(p.get('overbought', 70.0))
        self.rr_ratio: float          = float(p.get('rr_ratio', 1.5))
        self.sl_atr_multiplier: float = float(p.get('sl_atr_multiplier', 1.0))
        self.atr_period: int          = int(p.get('atr_period', 14))
        self.risk_pct: float          = float(p.get('risk_pct', 0.01))

        self._rsi: Optional[np.ndarray] = None
        self._atr: Optional[np.ndarray] = None

    def _setup(self, data: MarketData) -> None:
        if self._atr is not None:
            return
        self._rsi = _compute_rsi(data.close_1m, self.rsi_period)
        self._atr = _wilder_atr(data.high_1m, data.low_1m, data.close_1m, self.atr_period)

    def generate_signals(self, data: MarketData, i: int) -> Optional[Order]:
        self._setup(data)

        if i < self.min_lookback:
            return None
        if self._atr[i] <= 0.0:
            return None

        t = data.df_1m.index[i]
        if not (_time(9, 30) <= t.time() <= _time(15, 59)):
            return None

        rsi_cur  = self._rsi[i]
        rsi_prev = self._rsi[i - 1]

        if rsi_cur < self.oversold and rsi_prev >= self.oversold:
            direction = 1
        elif rsi_cur > self.overbought and rsi_prev <= self.overbought:
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
