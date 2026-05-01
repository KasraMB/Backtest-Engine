"""
Internal Bar Strength (IBS) Mean Reversion Strategy — NQ Futures

IBS = (D1_close - D1_low) / (D1_high - D1_low)

At the RTH open (9:30 ET) each day, look at the prior completed session's IBS:
  - IBS < ibs_low_thresh  → enter LONG  (close was near the day's low → expect bounce)
  - IBS > ibs_high_thresh → enter SHORT (close was near the day's high → expect fade)

SL  = ATR(14) × sl_atr_mult from entry price
TP  = SL distance × rr_ratio
EOD exit via RunConfig.eod_exit_time.

Optional: D1 SMA(200) trend filter — only trade in direction of trend.

Daily data source: data/NQ_1day_full_data.parquet
  Date = session START date (18:00 ET); corresponds to NEXT calendar day's RTH.
  For RTH date D, we use the daily row with Date < D (last completed session).
"""
from __future__ import annotations

import os
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
TICK        = 0.25

_RTH_OPEN_MIN = 9 * 60 + 30  # 570

_DAILY_DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'NQ_1day_full_data.parquet')


def _tick(price: float) -> float:
    return round(round(price / TICK) * TICK, 10)


def _load_daily_arrays() -> tuple:
    """
    Returns (date_arr, ibs_arr, atr14_arr, sma200_arr, close_arr).
    date_arr is numpy datetime64[D], sorted ascending.
    """
    df = pd.read_parquet(_DAILY_DATA_PATH)
    df = df.rename(columns={'Last': 'Close'})
    df['Date'] = pd.to_datetime(df['Date']).dt.normalize()
    df = df.sort_values('Date').reset_index(drop=True)

    rng = df['High'] - df['Low']
    df['ibs'] = np.where(rng > 0, (df['Close'] - df['Low']) / rng, 0.5)

    prev_close = df['Close'].shift(1)
    tr = pd.concat([
        df['High'] - df['Low'],
        (df['High'] - prev_close).abs(),
        (df['Low']  - prev_close).abs(),
    ], axis=1).max(axis=1)
    df['atr14']  = tr.rolling(14, min_periods=14).mean()
    df['sma200'] = df['Close'].rolling(200, min_periods=200).mean()

    return (
        df['Date'].values.astype('datetime64[D]'),
        df['ibs'].values.astype(np.float64),
        df['atr14'].values.astype(np.float64),
        df['sma200'].values.astype(np.float64),
        df['Close'].values.astype(np.float64),
    )


_DAILY_CACHE: Optional[tuple] = None


def _get_daily_arrays() -> tuple:
    global _DAILY_CACHE
    if _DAILY_CACHE is None:
        _DAILY_CACHE = _load_daily_arrays()
    return _DAILY_CACHE


class IBSStrategy(BaseStrategy):
    """
    IBS mean-reversion on NQ.  Enters once per day at the RTH open (9:30 ET).
    signal_bar_mask restricts generate_signals to exactly those bars.
    """

    trading_hours = [(_time(9, 30), _time(9, 30))]
    min_lookback  = 1

    def __init__(self, params: dict = None) -> None:
        super().__init__(params)
        p = params or {}

        self.ibs_low_thresh  = float(p.get('ibs_low_thresh',  0.25))
        self.ibs_high_thresh = float(p.get('ibs_high_thresh', 0.75))
        self.sl_atr_mult     = float(p.get('sl_atr_mult',     1.5))
        self.rr_ratio        = float(p.get('rr_ratio',        2.0))
        self.use_sma_filter  = bool(p.get('use_200sma_filter', False))
        self.risk_per_trade  = float(p.get('risk_per_trade',  0.01))
        self.starting_equity = float(p.get('starting_equity', 100_000))
        self.equity_mode     = str(p.get('equity_mode',       'dynamic'))

        self._times_min:   Optional[np.ndarray] = None
        self._bar_ibs:     Optional[np.ndarray] = None
        self._bar_atr14:   Optional[np.ndarray] = None
        self._bar_sma200:  Optional[np.ndarray] = None
        self._bar_prev_cls: Optional[np.ndarray] = None
        self._sig_mask:    Optional[np.ndarray] = None

        self._today        = None
        self._traded_today = False

        self._eq_cache   = self.starting_equity
        self._eq_cache_n = 0

    def _setup(self, data: MarketData) -> None:
        if self._times_min is not None:
            return

        idx = data.df_1m.index

        if data.bar_times_1m_min is not None:
            self._times_min = data.bar_times_1m_min
        else:
            self._times_min = (idx.hour * 60 + idx.minute).to_numpy(np.int32)

        if data.bar_day_int_1m is not None:
            self._bar_day = data.bar_day_int_1m
        else:
            self._bar_day = idx.normalize().view(np.int64) // (86_400 * 10**9)

        d_dates, d_ibs, d_atr14, d_sma200, d_close = _get_daily_arrays()

        bar_dates = idx.normalize().values.astype('datetime64[D]')
        pos = np.searchsorted(d_dates, bar_dates, side='left') - 1

        valid    = pos >= 0
        idx_safe = np.clip(pos, 0, len(d_dates) - 1)

        self._bar_ibs     = np.where(valid, d_ibs[idx_safe],    np.nan)
        self._bar_atr14   = np.where(valid, d_atr14[idx_safe],  np.nan)
        self._bar_sma200  = np.where(valid, d_sma200[idx_safe], np.nan)
        self._bar_prev_cls = np.where(valid, d_close[idx_safe], np.nan)

        # signal_bar_mask: only RTH open bars
        self._sig_mask = (self._times_min == _RTH_OPEN_MIN)

    def signal_bar_mask(self, data: MarketData) -> np.ndarray:
        self._setup(data)
        return self._sig_mask

    def _current_equity(self) -> float:
        if self.equity_mode == 'fixed':
            return self.starting_equity
        n = len(self.closed_trades)
        while self._eq_cache_n < n:
            self._eq_cache += self.closed_trades[self._eq_cache_n].net_pnl_dollars
            self._eq_cache_n += 1
        return self._eq_cache

    def generate_signals(self, data: MarketData, i: int) -> Optional[Order]:
        self._setup(data)

        today = int(self._bar_day[i])
        if today != self._today:
            self._today        = today
            self._traded_today = False

        if self._traded_today:
            return None

        ibs    = float(self._bar_ibs[i])
        atr14  = float(self._bar_atr14[i])
        sma200 = float(self._bar_sma200[i])
        prev_cls = float(self._bar_prev_cls[i])

        if np.isnan(ibs) or np.isnan(atr14) or atr14 <= 0:
            return None

        entry = float(data.open_1m[i])

        direction = 0
        if ibs < self.ibs_low_thresh:
            direction = 1   # long — close was near low, expect bounce
        elif ibs > self.ibs_high_thresh:
            direction = -1  # short — close was near high, expect fade

        if direction == 0:
            return None

        # SMA(200) trend filter
        if self.use_sma_filter and not np.isnan(sma200):
            if direction == 1  and prev_cls < sma200:
                return None
            if direction == -1 and prev_cls > sma200:
                return None

        sl_dist = self.sl_atr_mult * atr14
        if sl_dist <= 0:
            return None

        if direction == 1:
            sl_price = _tick(entry - sl_dist)
            tp_price = _tick(entry + self.rr_ratio * sl_dist)
        else:
            sl_price = _tick(entry + sl_dist)
            tp_price = _tick(entry - self.rr_ratio * sl_dist)

        equity    = self._current_equity()
        contracts = max(1, int(equity * self.risk_per_trade / (sl_dist * POINT_VALUE)))

        self._traded_today = True
        return Order(
            direction    = direction,
            order_type   = OrderType.MARKET,
            size_type    = SizeType.CONTRACTS,
            size_value   = float(contracts),
            sl_price     = sl_price,
            tp_price     = tp_price,
            trade_reason = 'ibs_long' if direction == 1 else 'ibs_short',
        )

    def on_fill(self, position: OpenPosition, data: MarketData, bar_index: int) -> None:
        position.set_initial_sl_tp(position.sl_price, position.tp_price)

    def manage_position(
        self, data: MarketData, i: int, position: OpenPosition,
    ) -> Optional[PositionUpdate]:
        return None
