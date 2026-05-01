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
        self.bar_minutes: int         = int(p.get('bar_minutes', 5))

        self._rsi: Optional[np.ndarray] = None
        self._atr: Optional[np.ndarray] = None
        self._signal_bar_map: dict | None = None
        self._rsi_rs: Optional[np.ndarray] = None

    def _setup(self, data: MarketData) -> None:
        if self._signal_bar_map is not None:
            return
        self._rsi = _compute_rsi(data.close_1m, self.rsi_period)
        self._atr = _wilder_atr(data.high_1m, data.low_1m, data.close_1m, self.atr_period)

        df_rs, self._signal_bar_map = _build_resampled(data, self.bar_minutes)
        self._rsi_rs = _compute_rsi(df_rs['close'].to_numpy(), self.rsi_period)

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
        rsi_j, rsi_prev = self._rsi_rs[j], self._rsi_rs[j - 1]
        # Long: RSI crosses below oversold
        if rsi_j < self.oversold and rsi_prev >= self.oversold:
            sl = round(round((data.close_1m[i] - self.sl_atr_multiplier * atr) / 0.25) * 0.25, 10)
            tp = round(round((data.close_1m[i] + self.rr_ratio * self.sl_atr_multiplier * atr) / 0.25) * 0.25, 10)
            return Order(direction=1, size_value=self.risk_pct, size_type=SizeType.PCT_RISK,
                         order_type=OrderType.MARKET, sl_price=sl, tp_price=tp)
        # Short: RSI crosses above overbought
        if rsi_j > self.overbought and rsi_prev <= self.overbought:
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
