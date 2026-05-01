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


class ShootingStarStrategy(BaseStrategy):
    trading_hours: list = [(_time(9, 30), _time(15, 59))]
    min_lookback: int = 50

    def __init__(self, params: dict = None) -> None:
        super().__init__(params)
        p = params or {}
        self.lower_wick_factor: float = float(p.get('lower_wick_factor', 0.2))
        self.upper_wick_factor: float = float(p.get('upper_wick_factor', 2.0))
        self.body_size_atr_factor: float = float(p.get('body_size_atr_factor', 0.3))
        self.rr_ratio: float = float(p.get('rr_ratio', 1.5))
        self.sl_atr_mult: float = float(p.get('sl_atr_multiplier', 1.0))
        self.atr_period: int = int(p.get('atr_period', 14))
        self.risk_pct: float = float(p.get('risk_pct', 0.01))
        self.detect_hammer: bool = bool(p.get('detect_hammer', True))

        self._atr: Optional[np.ndarray] = None

    def _setup(self, data: MarketData) -> None:
        if self._atr is not None:
            return
        self._atr = _wilder_atr_full(data.high_1m, data.low_1m, data.close_1m, self.atr_period)

    def generate_signals(self, data: MarketData, i: int) -> Optional[Order]:
        self._setup(data)
        if i < 4:
            return None
        t = data.df_1m.index[i]
        t_time = t.time()
        if not (_time(9, 30) <= t_time <= _time(15, 59)):
            return None
        if self._atr[i] <= 0 or self._atr[i - 1] <= 0:
            return None

        o = data.open_1m
        h = data.high_1m
        l = data.low_1m
        c = data.close_1m
        atr = self._atr

        # --- Shooting star (short signal) --- check pattern at bar i-1
        p_o = o[i - 1]; p_h = h[i - 1]; p_l = l[i - 1]; p_c = c[i - 1]
        p_atr = float(atr[i - 1])
        body = abs(p_o - p_c)
        body_denom = body + 0.001

        ss_cond1 = (p_o >= p_c) or (abs(p_o - p_c) < 0.1 * p_atr)
        ss_cond2 = body < self.body_size_atr_factor * p_atr
        ss_cond3 = (min(p_o, p_c) - p_l) < self.lower_wick_factor * body_denom
        ss_cond4 = (p_h - max(p_o, p_c)) >= self.upper_wick_factor * body_denom
        ss_cond5 = c[i - 2] < p_c and c[i - 3] < c[i - 2]
        ss_cond6 = c[i] < p_c

        if ss_cond1 and ss_cond2 and ss_cond3 and ss_cond4 and ss_cond5 and ss_cond6:
            sl_above = round(round((p_h + 0.5 * p_atr) / 0.25) * 0.25, 10)
            sl_dist = sl_above - c[i]
            if sl_dist > 0:
                tp = round(round((c[i] - self.rr_ratio * sl_dist) / 0.25) * 0.25, 10)
                return Order(
                    direction=-1,
                    size_value=self.risk_pct,
                    size_type=SizeType.PCT_RISK,
                    order_type=OrderType.MARKET,
                    sl_price=sl_above,
                    tp_price=tp,
                )

        # --- Hammer (long signal) ---
        if self.detect_hammer:
            h_o = o[i - 1]; h_h = h[i - 1]; h_l = l[i - 1]; h_c = c[i - 1]
            hbody = abs(h_c - h_o)
            hbody_denom = hbody + 0.001

            hm_cond1 = h_c >= h_o
            hm_cond2 = hbody < self.body_size_atr_factor * p_atr
            hm_cond3 = (h_h - max(h_o, h_c)) < self.lower_wick_factor * hbody_denom
            hm_cond4 = (min(h_o, h_c) - h_l) >= self.upper_wick_factor * hbody_denom
            hm_cond5 = c[i - 2] > h_c and c[i - 3] > c[i - 2]
            hm_cond6 = c[i] > h_c

            if hm_cond1 and hm_cond2 and hm_cond3 and hm_cond4 and hm_cond5 and hm_cond6:
                sl_below = round(round((h_l - 0.5 * p_atr) / 0.25) * 0.25, 10)
                sl_dist = c[i] - sl_below
                if sl_dist > 0:
                    tp = round(round((c[i] + self.rr_ratio * sl_dist) / 0.25) * 0.25, 10)
                    return Order(
                        direction=1,
                        size_value=self.risk_pct,
                        size_type=SizeType.PCT_RISK,
                        order_type=OrderType.MARKET,
                        sl_price=sl_below,
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
