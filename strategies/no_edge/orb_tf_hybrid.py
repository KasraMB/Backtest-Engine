"""
ORB + TrendFollowing Hybrid Strategy

Entry: Standard ORB breakout at 10:00-15:00 (same as orb_atr_breakout).
Exit:
- If trade is in profit at EOD (15:59), HOLD overnight instead of closing.
- Trailing stop activates overnight (same as TF: ATR × mult trail).
- Exit at next RTH if trailing stop hit, or TP hit, or next day EOD.

Additional filter: Only take ORB entries when D1 SMA(fast) > SMA(slow) for longs,
reverse for shorts (TF trend filter applied to intraday entries).

This combines:
- ORB's high-frequency, same-day edge (entry timing, 1+ trades/day)
- TF's overnight momentum capture (holds winners, cuts losers with trail)
"""
from __future__ import annotations

import os
from datetime import time as _time
from typing import Optional

import numpy as np
import pandas as pd

from backtest.data.market_data import MarketData
from backtest.strategy.base import BaseStrategy
from backtest.strategy.enums import OrderType, SizeType, ExitReason
from backtest.strategy.order import Order
from backtest.strategy.update import OpenPosition, PositionUpdate

POINT_VALUE = 20.0
TICK = 0.25

_RTH_OPEN_MIN = 9 * 60 + 30  # 09:30
_ORB_END_MIN = 10 * 60  # 10:00
_WIN_END_MIN = 15 * 60  # 15:00 (standard ORB end)
_EOD_MIN = 15 * 60 + 59  # 15:59 (last bar for EOD decision)

_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'NQ_1day_full_data.parquet')


def _tick(price: float) -> float:
    return round(round(price / TICK) * TICK, 10)


def _load_daily():
    """Load D1 data with ATR5, SMA200, SMA20/50/100, prev close."""
    df = pd.read_parquet(_PATH)
    df = df.rename(columns={'Last': 'Close'})
    df['Date'] = pd.to_datetime(df['Date']).dt.normalize()
    df = df.sort_values('Date').reset_index(drop=True)

    prev_close = df['Close'].shift(1)
    tr = pd.concat([
        df['High'] - df['Low'],
        (df['High'] - prev_close).abs(),
        (df['Low'] - prev_close).abs(),
    ], axis=1).max(axis=1)

    df['atr5'] = tr.rolling(5, min_periods=5).mean()
    df['atr14'] = tr.rolling(14, min_periods=14).mean()
    df['sma200'] = df['Close'].rolling(200, min_periods=200).mean()
    df['sma20'] = df['Close'].rolling(20, min_periods=20).mean()
    df['sma50'] = df['Close'].rolling(50, min_periods=50).mean()
    df['sma100'] = df['Close'].rolling(100, min_periods=100).mean()
    df['prev_close'] = df['Close'].shift(1)

    return (
        df['Date'].values.astype('datetime64[D]'),
        df['atr5'].values.astype(np.float64),
        df['atr14'].values.astype(np.float64),
        df['sma200'].values.astype(np.float64),
        df['sma20'].values.astype(np.float64),
        df['sma50'].values.astype(np.float64),
        df['sma100'].values.astype(np.float64),
        df['Close'].values.astype(np.float64),
        df['prev_close'].values.astype(np.float64),
    )


_CACHE = None


def _get_daily():
    global _CACHE
    if _CACHE is None:
        _CACHE = _load_daily()
    return _CACHE


class ORBTFHybridStrategy(BaseStrategy):
    """
    ORB entries with trend filter, overnight hold on in-profit trades.
    """
    trading_hours = [(_time(9, 30), _time(15, 59))]
    min_lookback = 200

    def __init__(self, params: dict = None) -> None:
        super().__init__(params)
        p = params or {}

        # ORB params
        self.atr_space_mult = float(p.get('atr_space_multiplier', 0.15))
        self.sl_pct = float(p.get('sl_pct', 0.003))
        self.rr_ratio = float(p.get('rr_ratio', 2.0))
        self.max_trades_per_day = int(p.get('max_trades_per_day', 1))

        # Trend filter params
        self.use_trend_filter = bool(p.get('use_trend_filter', True))
        self.sma_fast = int(p.get('sma_fast', 20))
        self.sma_slow = int(p.get('sma_slow', 100))

        # Overnight hold params
        self.hold_overnight = bool(p.get('hold_overnight', True))
        self.trail_atr_mult = float(p.get('trail_atr_mult', 3.0))
        self.atr_period = int(p.get('atr_period', 14))

        # Risk
        self.risk_per_trade = float(p.get('risk_per_trade', 0.01))
        self.starting_equity = float(p.get('starting_equity', 100_000))
        self.equity_mode = str(p.get('equity_mode', 'dynamic'))

        # Daily arrays
        self._dates = None
        self._atr5 = None
        self._atr14 = None
        self._sma200 = None
        self._sma20 = None
        self._sma50 = None
        self._sma100 = None
        self._close = None
        self._prev_close = None

        # Day tracking
        self._today = None
        self._traded_today = False

        # Position state for overnight tracking
        self._pending_overnight = None  # Position info if we're holding overnight
        self._trail_sl = None
        self._entry_price = None
        self._direction = None

        # Equity cache
        self._eq_cache = self.starting_equity
        self._eq_cache_n = 0

    def _setup(self, data: MarketData) -> None:
        if self._dates is not None:
            return
        dates, atr5, atr14, sma200, sma20, sma50, sma100, close, prev_close = _get_daily()
        self._dates = dates
        self._atr5 = atr5
        self._atr14 = atr14
        self._sma200 = sma200
        self._sma20 = sma20
        self._sma50 = sma50
        self._sma100 = sma100
        self._close = close
        self._prev_close = prev_close

        idx = data.df_1m.index
        if data.bar_times_1m_min is not None:
            self._times_min = data.bar_times_1m_min
        else:
            self._times_min = (idx.hour * 60 + idx.minute).to_numpy(np.int32)

        if data.bar_day_int_1m is not None:
            self._bar_day = data.bar_day_int_1m
        else:
            self._bar_day = idx.normalize().view(np.int64) // (86_400 * 10**9)

    def _map_day(self, bar_idx: int) -> int:
        """Return index into daily arrays for bar_idx."""
        day = int(self._bar_day[bar_idx])
        # self._dates is datetime64[D], so just use day directly as index lookup
        # Find the position in self._dates that matches this day integer
        pos = np.searchsorted(self._dates, np.datetime64(day, 'D'), side='left') - 1
        return max(0, min(pos, len(self._dates) - 1))

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
        t_min = int(self._times_min[i])

        today = int(self._bar_day[i])
        if today != self._today:
            self._today = today
            self._traded_today = False

        if self._traded_today:
            return None

        if t_min < _RTH_OPEN_MIN or t_min >= _WIN_END_MIN:
            return None

        day_idx = self._map_day(i)
        if day_idx < 0 or day_idx >= len(self._atr5):
            return None

        atr5 = self._atr5[day_idx]
        if atr5 <= 0:
            return None

        open_price = float(data.open_1m[i])
        close_price = float(data.close_1m[i])

        # ORB levels (10:00 breakout)
        if t_min < _ORB_END_MIN:
            return None

        # Set breakout levels
        entry = float(data.open_1m[i])

        # Long breakout level
        long_level = _tick(float(data.close_1m[i - 1] if i > 0 else open_price) + atr5 * self.atr_space_mult)
        # Short breakout level
        short_level = _tick(float(data.close_1m[i - 1] if i > 0 else open_price) - atr5 * self.atr_space_mult)

        # Trend filter (D1 SMA20 vs SMA100)
        direction = 0
        sl_price = tp_price = None
        entry_atr = self._atr14[day_idx] if self.atr_period == 14 else atr5

        # Long breakout
        if close_price > long_level:
            if self.use_trend_filter:
                sma_f = self._sma20[day_idx]
                sma_s = self._sma100[day_idx]
                if np.isnan(sma_f) or np.isnan(sma_s) or sma_f <= sma_s:
                    return None
            direction = 1
            sl_price = _tick(entry * (1 - self.sl_pct))
            sl_dist = entry - sl_price
            tp_price = _tick(entry + self.rr_ratio * sl_dist)

        # Short breakout
        elif close_price < short_level:
            if self.use_trend_filter:
                sma_f = self._sma20[day_idx]
                sma_s = self._sma100[day_idx]
                if np.isnan(sma_f) or np.isnan(sma_s) or sma_f >= sma_s:
                    return None
            direction = -1
            sl_price = _tick(entry * (1 + self.sl_pct))
            sl_dist = sl_price - entry
            tp_price = _tick(entry - self.rr_ratio * sl_dist)

        if direction == 0:
            return None

        equity = self._current_equity()
        contracts = max(1, int(equity * self.risk_per_trade / (sl_dist * POINT_VALUE)))

        self._traded_today = True
        self._direction = direction
        self._entry_price = entry
        self._trail_sl = sl_price

        # Store for overnight tracking
        self._pending_overnight = {
            'entry': entry,
            'sl': sl_price,
            'trail_sl': sl_price,
            'direction': direction,
        }

        return Order(
            direction=direction,
            order_type=OrderType.MARKET,
            size_type=SizeType.CONTRACTS,
            size_value=float(contracts),
            sl_price=sl_price,
            tp_price=tp_price,
            trade_reason='orb_tf_hybrid_long' if direction == 1 else 'orb_tf_hybrid_short',
        )

    def on_fill(self, position: OpenPosition, data: MarketData, bar_index: int) -> None:
        position.set_initial_sl_tp(position.sl_price, position.tp_price)

    def manage_position(self, data: MarketData, i: int, position: OpenPosition) -> Optional[PositionUpdate]:
        self._setup(data)
        t_min = int(self._times_min[i])
        close = float(data.close_1m[i])

        day_idx = self._map_day(i)
        if day_idx < 0 or day_idx >= len(self._atr14):
            return None

        atr14 = self._atr14[day_idx]
        if atr14 <= 0:
            return None

        dir_mult = self._direction if self._direction else position.direction

        # Update trailing stop
        if dir_mult == 1:
            new_sl = _tick(close - self.trail_atr_mult * atr14)
            if new_sl > self._trail_sl:
                self._trail_sl = new_sl
            # Only update if we're in profit or breakeven
            pnl = (close - self._entry_price) * position.contracts * POINT_VALUE
            if pnl >= 0:
                return PositionUpdate(new_sl_price=new_sl)
        else:
            new_sl = _tick(close + self.trail_atr_mult * atr14)
            if new_sl < self._trail_sl:
                self._trail_sl = new_sl
            pnl = (self._entry_price - close) * position.contracts * POINT_VALUE
            if pnl >= 0:
                return PositionUpdate(new_sl_price=new_sl)

        return None
