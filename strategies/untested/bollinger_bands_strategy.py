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


class BollingerBandsStrategy(BaseStrategy):
    trading_hours: list = [(_time(9, 30), _time(15, 59))]
    min_lookback: int = 100

    def __init__(self, params: dict = None) -> None:
        super().__init__(params)
        p = params or {}
        self.bb_period: int = int(p.get('bb_period', 20))
        self.bb_std: float = float(p.get('bb_std', 2.0))
        self.signal_type: str = str(p.get('signal_type', 'breakout'))
        self.rr_ratio: float = float(p.get('rr_ratio', 1.5))
        self.sl_atr_mult: float = float(p.get('sl_atr_multiplier', 1.0))
        self.atr_period: int = int(p.get('atr_period', 14))
        self.risk_pct: float = float(p.get('risk_pct', 0.01))
        self.bar_minutes: int = int(p.get('bar_minutes', 5))

        self._upper: Optional[np.ndarray] = None
        self._mid: Optional[np.ndarray] = None
        self._lower: Optional[np.ndarray] = None
        self._atr: Optional[np.ndarray] = None
        self._signal_bar_map: dict | None = None
        self._upper_rs: Optional[np.ndarray] = None
        self._lower_rs: Optional[np.ndarray] = None
        self._close_rs: Optional[np.ndarray] = None

    def _setup(self, data: MarketData) -> None:
        if self._signal_bar_map is not None:
            return
        h = data.high_1m
        l = data.low_1m
        c = data.close_1m
        s = pd.Series(c)
        mid = s.rolling(self.bb_period, min_periods=self.bb_period).mean()
        std = s.rolling(self.bb_period, min_periods=self.bb_period).std()
        self._mid = mid.to_numpy()
        self._upper = (mid + self.bb_std * std).to_numpy()
        self._lower = (mid - self.bb_std * std).to_numpy()
        self._atr = _wilder_atr_full(h, l, c, self.atr_period)

        df_rs, self._signal_bar_map = _build_resampled(data, self.bar_minutes)
        close_rs = pd.Series(df_rs['close'].to_numpy())
        mid_rs = close_rs.rolling(self.bb_period, min_periods=self.bb_period).mean()
        std_rs = close_rs.rolling(self.bb_period, min_periods=self.bb_period).std()
        self._upper_rs = (mid_rs + self.bb_std * std_rs).to_numpy()
        self._lower_rs = (mid_rs - self.bb_std * std_rs).to_numpy()
        self._close_rs = df_rs['close'].to_numpy()

    def generate_signals(self, data: MarketData, i: int) -> Optional[Order]:
        self._setup(data)
        t = data.df_1m.index[i]
        if not (_time(9, 30) <= t.time() <= _time(15, 59)):
            return None
        j = self._signal_bar_map.get(i)
        if j is None or j < 2:
            return None
        atr = float(self._atr[i])
        if atr <= 0 or np.isnan(self._upper_rs[j]) or np.isnan(self._lower_rs[j]):
            return None
        cl = data.close_1m[i]
        up_j = self._upper_rs[j];  up_p = self._upper_rs[j - 1]
        lo_j = self._lower_rs[j];  lo_p = self._lower_rs[j - 1]
        rs_c_j = self._close_rs[j]; rs_c_p = self._close_rs[j - 1]
        if self.signal_type == 'mean_reversion':
            if rs_c_j < lo_j and rs_c_p >= lo_p:
                sl = round(round((cl - self.sl_atr_mult * atr) / 0.25) * 0.25, 10)
                tp = round(round((cl + self.rr_ratio * self.sl_atr_mult * atr) / 0.25) * 0.25, 10)
                return Order(direction=1, size_value=self.risk_pct, size_type=SizeType.PCT_RISK,
                             order_type=OrderType.MARKET, sl_price=sl, tp_price=tp)
            if rs_c_j > up_j and rs_c_p <= up_p:
                sl = round(round((cl + self.sl_atr_mult * atr) / 0.25) * 0.25, 10)
                tp = round(round((cl - self.rr_ratio * self.sl_atr_mult * atr) / 0.25) * 0.25, 10)
                return Order(direction=-1, size_value=self.risk_pct, size_type=SizeType.PCT_RISK,
                             order_type=OrderType.MARKET, sl_price=sl, tp_price=tp)
        else:  # breakout
            if rs_c_j > up_j and rs_c_p <= up_p:
                sl = round(round((cl - self.sl_atr_mult * atr) / 0.25) * 0.25, 10)
                tp = round(round((cl + self.rr_ratio * self.sl_atr_mult * atr) / 0.25) * 0.25, 10)
                return Order(direction=1, size_value=self.risk_pct, size_type=SizeType.PCT_RISK,
                             order_type=OrderType.MARKET, sl_price=sl, tp_price=tp)
            if rs_c_j < lo_j and rs_c_p >= lo_p:
                sl = round(round((cl + self.sl_atr_mult * atr) / 0.25) * 0.25, 10)
                tp = round(round((cl - self.rr_ratio * self.sl_atr_mult * atr) / 0.25) * 0.25, 10)
                return Order(direction=-1, size_value=self.risk_pct, size_type=SizeType.PCT_RISK,
                             order_type=OrderType.MARKET, sl_price=sl, tp_price=tp)
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
