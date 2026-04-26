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


def _wilder_atr_full(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    n = len(close)
    out = np.zeros(n)
    if n < period + 1:
        return out
    seed = 0.0
    for k in range(period):
        hl = high[k] - low[k]
        seed += (hl if k == 0 else max(hl, abs(high[k] - close[k - 1]), abs(low[k] - close[k - 1])))
    out[period - 1] = seed / period
    inv, alpha = 1.0 - 1.0 / period, 1.0 / period
    for i in range(period, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        out[i] = out[i - 1] * inv + max(hl, hc, lc) * alpha
    return out


class AwesomeOscillatorStrategy(BaseStrategy):
    trading_hours: list = [(_time(9, 30), _time(15, 59))]
    min_lookback: int = 100

    def __init__(self, params: dict = None) -> None:
        super().__init__(params)
        p = params or {}
        self.ao_fast: int = int(p.get('ao_fast', 5))
        self.ao_slow: int = int(p.get('ao_slow', 34))
        self.signal_type: str = str(p.get('signal_type', 'zero_cross'))
        self.rr_ratio: float = float(p.get('rr_ratio', 1.5))
        self.sl_atr_mult: float = float(p.get('sl_atr_multiplier', 1.0))
        self.atr_period: int = int(p.get('atr_period', 14))
        self.risk_pct: float = float(p.get('risk_pct', 0.01))

        self._ao: Optional[np.ndarray] = None
        self._atr: Optional[np.ndarray] = None

    def _setup(self, data: MarketData) -> None:
        if self._atr is not None:
            return
        h = data.high_1m
        l = data.low_1m
        c = data.close_1m
        mid = (h + l) / 2.0
        s = pd.Series(mid)
        fast_sma = s.rolling(self.ao_fast, min_periods=self.ao_fast).mean().to_numpy()
        slow_sma = s.rolling(self.ao_slow, min_periods=self.ao_slow).mean().to_numpy()
        self._ao = fast_sma - slow_sma
        self._atr = _wilder_atr_full(h, l, c, self.atr_period)

    def generate_signals(self, data: MarketData, i: int) -> Optional[Order]:
        self._setup(data)
        t = data.df_1m.index[i]
        t_time = t.time()
        if not (_time(9, 30) <= t_time <= _time(15, 59)):
            return None
        if i < self.ao_slow + 2:
            return None
        if self._atr[i] <= 0:
            return None

        ao = self._ao
        close_i = data.close_1m[i]
        atr_i = float(self._atr[i])
        sl_dist = self.sl_atr_mult * atr_i

        if np.isnan(ao[i]) or np.isnan(ao[i - 1]):
            return None

        direction: Optional[int] = None

        if self.signal_type == 'zero_cross':
            if ao[i] > 0 and ao[i - 1] <= 0:
                direction = 1
            elif ao[i] < 0 and ao[i - 1] >= 0:
                direction = -1

        elif self.signal_type == 'saucer':
            if np.isnan(ao[i - 2]):
                return None
            # Long saucer: all negative, dip then recovery
            if ao[i] < 0 and ao[i - 1] < 0 and ao[i - 2] < 0:
                if ao[i - 1] < ao[i - 2] and ao[i] > ao[i - 1]:
                    direction = 1
            # Short saucer: all positive, peak then decline
            elif ao[i] > 0 and ao[i - 1] > 0 and ao[i - 2] > 0:
                if ao[i - 1] > ao[i - 2] and ao[i] < ao[i - 1]:
                    direction = -1

        if direction is None:
            return None

        if direction == 1:
            sl = round(round((close_i - sl_dist) / 0.25) * 0.25, 10)
            tp = round(round((close_i + self.rr_ratio * sl_dist) / 0.25) * 0.25, 10)
        else:
            sl = round(round((close_i + sl_dist) / 0.25) * 0.25, 10)
            tp = round(round((close_i - self.rr_ratio * sl_dist) / 0.25) * 0.25, 10)

        return Order(
            direction=direction,
            size_value=self.risk_pct,
            size_type=SizeType.PCT_RISK,
            order_type=OrderType.MARKET,
            sl_price=sl,
            tp_price=tp,
        )

    def on_fill(self, position: OpenPosition, data: MarketData, i: int) -> None:
        atr = float(self._atr[i]) if self._atr[i] > 0 else float(self._atr[max(0, i - 1)])
        sl_dist = self.sl_atr_mult * atr
        if position.is_long():
            sl = round(round((position.entry_price - sl_dist) / 0.25) * 0.25, 10)
            tp = round(round((position.entry_price + self.rr_ratio * sl_dist) / 0.25) * 0.25, 10)
        else:
            sl = round(round((position.entry_price + sl_dist) / 0.25) * 0.25, 10)
            tp = round(round((position.entry_price - self.rr_ratio * sl_dist) / 0.25) * 0.25, 10)
        position.set_initial_sl_tp(sl, tp)

    def manage_position(self, data: MarketData, i: int, position: OpenPosition) -> Optional[PositionUpdate]:
        return None
