"""
Parabolic SAR Strategy (NQ Futures)

Trades SAR trend reversals during the NY session.
ATR-based SL/TP, sized by risk_pct of equity.
"""
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


def _resample_offset(bar_minutes: int):
    """Return pandas Timedelta offset so candle boundaries align correctly.
    <=30m: 9:30 ET is always a boundary (570 % bar_minutes; 0 for 5/10/15/30).
    60m: midnight-aligned hourly (9:00, 10:00...).
    240m: 2h offset → 2am,6am,10am,2pm,6pm,10pm ET.
    """
    if bar_minutes <= 30:
        offset_min = (9 * 60 + 30) % bar_minutes
        return pd.Timedelta(minutes=offset_min) if offset_min else None
    elif bar_minutes == 240:
        return pd.Timedelta(hours=2)
    return None


def _build_resampled(data, bar_minutes: int):
    """Resample 1m MarketData OHLC to bar_minutes bars with correct alignment.
    Returns (df_rs, signal_bar_map) where signal_bar_map is dict {1m_bar_idx: rs_bar_idx}.
    Signals only fire at the 1m bar corresponding to the CLOSE of each resampled bar.
    """
    df = pd.DataFrame({
        'open':  data.open_1m,
        'high':  data.high_1m,
        'low':   data.low_1m,
        'close': data.close_1m,
    }, index=data.df_1m.index)
    offset = _resample_offset(bar_minutes)
    kwargs = dict(closed='left', label='left')
    if offset is not None:
        kwargs['offset'] = offset
    df_rs = df.resample(f'{bar_minutes}min', **kwargs).agg(
        open=('open', 'first'), high=('high', 'max'),
        low=('low', 'min'),    close=('close', 'last'),
    ).dropna()

    one_min_idx = data.df_1m.index
    signal_bar_map = {}
    rs_times = df_rs.index
    for j in range(len(rs_times)):
        close_time = rs_times[j] + pd.Timedelta(minutes=bar_minutes - 1)
        pos = one_min_idx.searchsorted(close_time, side='right') - 1
        if 0 <= pos < len(one_min_idx):
            signal_bar_map[int(pos)] = j
    return df_rs, signal_bar_map


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
        self.bar_minutes: int         = int(p.get('bar_minutes', 5))

        self._psar: Optional[np.ndarray]  = None
        self._trend: Optional[np.ndarray] = None
        self._atr: Optional[np.ndarray]   = None
        self._signal_bar_map: dict | None = None
        self._trend_rs: Optional[np.ndarray] = None

    def _setup(self, data: MarketData) -> None:
        if self._signal_bar_map is not None:
            return
        self._psar, self._trend = _compute_psar(
            data.high_1m, data.low_1m, data.close_1m,
            self.initial_af, self.step_af, self.end_af,
        )
        self._atr = _wilder_atr(data.high_1m, data.low_1m, data.close_1m, self.atr_period)

        df_rs, self._signal_bar_map = _build_resampled(data, self.bar_minutes)
        _, self._trend_rs = _compute_psar(
            df_rs['high'].to_numpy(), df_rs['low'].to_numpy(), df_rs['close'].to_numpy(),
            self.initial_af, self.step_af, self.end_af,
        )

    def generate_signals(self, data: MarketData, i: int) -> Optional[Order]:
        self._setup(data)

        if i < self.min_lookback:
            return None

        t = data.df_1m.index[i]
        if not (_time(9, 30) <= t.time() <= _time(15, 59)):
            return None
        j = self._signal_bar_map.get(i)
        if j is None or j < 2:
            return None
        atr = float(self._atr[i])
        if atr <= 0:
            return None
        if self._trend_rs[j] == 1 and self._trend_rs[j - 1] == -1:  # flipped to uptrend
            sl = round(round((data.close_1m[i] - self.sl_atr_multiplier * atr) / 0.25) * 0.25, 10)
            tp = round(round((data.close_1m[i] + self.rr_ratio * self.sl_atr_multiplier * atr) / 0.25) * 0.25, 10)
            return Order(direction=1, size_value=self.risk_pct, size_type=SizeType.PCT_RISK,
                         order_type=OrderType.MARKET, sl_price=sl, tp_price=tp)
        if self._trend_rs[j] == -1 and self._trend_rs[j - 1] == 1:  # flipped to downtrend
            sl = round(round((data.close_1m[i] + self.sl_atr_multiplier * atr) / 0.25) * 0.25, 10)
            tp = round(round((data.close_1m[i] - self.rr_ratio * self.sl_atr_multiplier * atr) / 0.25) * 0.25, 10)
            return Order(direction=-1, size_value=self.risk_pct, size_type=SizeType.PCT_RISK,
                         order_type=OrderType.MARKET, sl_price=sl, tp_price=tp)
        return None

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
