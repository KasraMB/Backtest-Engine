"""
Parabolic SAR Strategy (NQ Futures)

Trades SAR trend reversals during the NY session.
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


def _compute_psar(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    initial_af: float,
    step_af: float,
    end_af: float,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(close)
    psar  = np.zeros(n)
    trend = np.zeros(n, dtype=np.int8)  # 1=up, -1=down
    ep    = np.zeros(n)
    af    = np.zeros(n)

    if n < 2:
        return psar, trend

    trend[1] = 1 if close[1] > close[0] else -1
    psar[1]  = low[0] if trend[1] > 0 else high[0]
    ep[1]    = high[1] if trend[1] > 0 else low[1]
    af[1]    = initial_af

    for i in range(2, n):
        new_sar = psar[i - 1] + af[i - 1] * (ep[i - 1] - psar[i - 1])
        if trend[i - 1] > 0:
            new_sar = min(new_sar, low[i - 1], low[i - 2] if i >= 2 else low[i - 1])
            if new_sar > low[i]:  # reversal
                trend[i] = -1
                new_sar  = ep[i - 1]
                ep[i]    = low[i]
                af[i]    = initial_af
            else:
                trend[i] = 1
                ep[i]    = max(ep[i - 1], high[i])
                af[i]    = min(end_af, af[i - 1] + step_af) if ep[i] > ep[i - 1] else af[i - 1]
        else:
            new_sar = max(new_sar, high[i - 1], high[i - 2] if i >= 2 else high[i - 1])
            if new_sar < high[i]:  # reversal
                trend[i] = 1
                new_sar  = ep[i - 1]
                ep[i]    = high[i]
                af[i]    = initial_af
            else:
                trend[i] = -1
                ep[i]    = min(ep[i - 1], low[i])
                af[i]    = min(end_af, af[i - 1] + step_af) if ep[i] < ep[i - 1] else af[i - 1]
        psar[i] = new_sar

    return psar, trend


class ParabolicSARStrategy(BaseStrategy):
    """
    Parabolic SAR trend-reversal entry with Wilder ATR-based SL/TP.
    Enters on SAR flip during NY session 09:30–15:59 ET.
    """

    trading_hours = [(_time(9, 30), _time(15, 59))]
    min_lookback = 10

    def __init__(self, params: dict = None) -> None:
        super().__init__(params)
        p = params or {}

        self.initial_af: float        = float(p.get('initial_af', 0.02))
        self.step_af: float           = float(p.get('step_af', 0.02))
        self.end_af: float            = float(p.get('end_af', 0.2))
        self.rr_ratio: float          = float(p.get('rr_ratio', 1.5))
        self.sl_atr_multiplier: float = float(p.get('sl_atr_multiplier', 1.0))
        self.atr_period: int          = int(p.get('atr_period', 14))
        self.risk_pct: float          = float(p.get('risk_pct', 0.01))

        self._psar: Optional[np.ndarray]  = None
        self._trend: Optional[np.ndarray] = None
        self._atr: Optional[np.ndarray]   = None

    def _setup(self, data: MarketData) -> None:
        if self._atr is not None:
            return
        self._psar, self._trend = _compute_psar(
            data.high_1m, data.low_1m, data.close_1m,
            self.initial_af, self.step_af, self.end_af,
        )
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

        trend_cur  = int(self._trend[i])
        trend_prev = int(self._trend[i - 1])

        if trend_cur == 1 and trend_prev == -1:
            direction = 1
        elif trend_cur == -1 and trend_prev == 1:
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
