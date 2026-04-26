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


class HeikinAshiStrategy(BaseStrategy):
    trading_hours: list = [(_time(9, 30), _time(15, 59))]
    min_lookback: int = 50

    def __init__(self, params: dict = None) -> None:
        super().__init__(params)
        p = params or {}
        self.rr_ratio: float = float(p.get('rr_ratio', 1.5))
        self.sl_atr_mult: float = float(p.get('sl_atr_multiplier', 1.0))
        self.atr_period: int = int(p.get('atr_period', 14))
        self.risk_pct: float = float(p.get('risk_pct', 0.01))
        self.require_no_wick: bool = bool(p.get('require_no_wick', True))

        self._ha_open: Optional[np.ndarray] = None
        self._ha_close: Optional[np.ndarray] = None
        self._ha_high: Optional[np.ndarray] = None
        self._ha_low: Optional[np.ndarray] = None
        self._atr: Optional[np.ndarray] = None

    def _setup(self, data: MarketData) -> None:
        if self._atr is not None:
            return
        o = data.open_1m
        h = data.high_1m
        l = data.low_1m
        c = data.close_1m
        n = len(c)

        ha_close = (o + h + l + c) / 4.0
        ha_open = np.empty(n)
        ha_open[0] = o[0]
        for i in range(1, n):
            ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2.0
        ha_high = np.maximum(h, np.maximum(ha_open, ha_close))
        ha_low = np.minimum(l, np.minimum(ha_open, ha_close))

        self._ha_open = ha_open
        self._ha_close = ha_close
        self._ha_high = ha_high
        self._ha_low = ha_low
        self._atr = _wilder_atr_full(h, l, c, self.atr_period)

    def generate_signals(self, data: MarketData, i: int) -> Optional[Order]:
        self._setup(data)
        t = data.df_1m.index[i]
        t_time = t.time()
        if not (_time(9, 30) <= t_time <= _time(15, 59)):
            return None
        if i < 2:
            return None

        ha_o = self._ha_open
        ha_c = self._ha_close
        ha_h = self._ha_high
        ha_l = self._ha_low
        atr = self._atr

        if atr[i] <= 0:
            return None

        close_i = data.close_1m[i]
        sl_dist = self.sl_atr_mult * float(atr[i])

        bullish_now = ha_c[i] > ha_o[i]
        bullish_prev = ha_c[i - 1] > ha_o[i - 1]
        bearish_now = ha_c[i] < ha_o[i]
        bearish_prev = ha_c[i - 1] < ha_o[i - 1]

        # Long: HA turns bullish
        if bullish_now and not bullish_prev:
            if self.require_no_wick and abs(ha_o[i] - ha_l[i]) >= 0.01:
                pass
            else:
                sl = round(round((close_i - sl_dist) / 0.25) * 0.25, 10)
                tp = round(round((close_i + self.rr_ratio * sl_dist) / 0.25) * 0.25, 10)
                return Order(
                    direction=1,
                    size_value=self.risk_pct,
                    size_type=SizeType.PCT_RISK,
                    order_type=OrderType.MARKET,
                    sl_price=sl,
                    tp_price=tp,
                )

        # Short: HA turns bearish
        if bearish_now and not bearish_prev:
            if self.require_no_wick and abs(ha_o[i] - ha_h[i]) >= 0.01:
                pass
            else:
                sl = round(round((close_i + sl_dist) / 0.25) * 0.25, 10)
                tp = round(round((close_i - self.rr_ratio * sl_dist) / 0.25) * 0.25, 10)
                return Order(
                    direction=-1,
                    size_value=self.risk_pct,
                    size_type=SizeType.PCT_RISK,
                    order_type=OrderType.MARKET,
                    sl_price=sl,
                    tp_price=tp,
                )

        return None

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
