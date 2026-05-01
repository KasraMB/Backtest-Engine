"""
Asian Session Range Strategy — NQ Futures

Asia range is computed INLINE at signal time (no pre-built price arrays → no lookahead).
The range spans from asia_start_hour ET through the bar before entry_hour ET.

Multiple attack angles via params:
  entry_mode  : 'breakout'   — stop order at range edge (trend follow)
              : 'fade'       — stop order at range edge, counter-trend
              : 'ny_confirm' — market at 9:30 only if price already outside range

  direction   : 'long' | 'short' | 'momentum' (follow overnight) | 'contrarian'

  sl_type     : 'atr'   — SL = sl_atr_mult × ATR from entry
              : 'range'  — SL = full Asia range width from entry

  min_range_atr, max_range_atr — filter by range size relative to ATR

Asia session (ET, year-round — US and EU both observe DST, so London open stays at 3 AM ET):
  Default: asia_start_hour=18 (6 PM), entry_hour=3 (3 AM London open)
  Tokyo range: 19:00–01:00 ET | London open: 03:00 ET
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
TICK        = 0.25
_MINS_PER_DAY = 1440


def _tick(price: float) -> float:
    return round(round(price / TICK) * TICK, 10)


def _atr_at(high: np.ndarray, low: np.ndarray, close: np.ndarray,
            period: int, i: int) -> float:
    """Wilder ATR at bar i using only bars ≤ i."""
    if i < 1:
        return 0.0
    start = max(1, i - period * 3)
    h = high[start:i + 1]
    l = low[start:i + 1]
    c = close[start - 1:i]
    trs = np.maximum(h - l, np.maximum(np.abs(h - c), np.abs(l - c)))
    atr = float(trs[0])
    for tr in trs[1:]:
        atr = atr * (1.0 - 1.0 / period) + float(tr) / period
    return atr


def _compute_asia_range(times_min: np.ndarray, high: np.ndarray, low: np.ndarray,
                        close: np.ndarray, i: int, asia_start_min: int,
                        entry_min: int) -> tuple[float, float, float, float]:
    """
    Scan backward from bar i to collect the Asia session's H/L/open/close.

    Asia window wraps midnight: bars with t_min >= asia_start_min
    OR bars with t_min < entry_min (still in the pre-entry morning).

    Returns (asia_high, asia_low, asia_open, asia_close) or (nan, nan, nan, nan) if
    no Asia bars found.
    """
    asia_h   = -np.inf
    asia_l   =  np.inf
    asia_open  = np.nan
    asia_close = np.nan

    # Scan back up to 18 hours (1080 bars) — enough to cover any Asia window
    max_lookback = min(i, 1080)
    for j in range(i, i - max_lookback - 1, -1):
        if j < 0:
            break
        t = int(times_min[j])
        in_asia = (t >= asia_start_min) or (t < entry_min)
        if not in_asia:
            # We've passed back through the entry gap (between entry_min and asia_start_min)
            # This means we've exited the Asia window for this day.
            # But we might still be in the Asia window on the previous day's side.
            # Stop if we hit the entry gap and have already collected some bars.
            if asia_h > -np.inf:
                break
            continue

        if high[j] > asia_h:
            asia_h = float(high[j])
        if low[j] < asia_l:
            asia_l = float(low[j])
        # The chronologically FIRST bar in the window sets the open price
        # Since we're scanning backward, the last j we visit (lowest) is the Asia open
        asia_open  = float(close[j])   # updated each backward step → ends at earliest bar
        if np.isnan(asia_close):
            asia_close = float(close[j])   # first bar we visit = chronological Asia close

    if asia_h == -np.inf or asia_l == np.inf:
        return np.nan, np.nan, np.nan, np.nan
    return asia_h, asia_l, asia_open, asia_close


class AsiaBreakoutStrategy(BaseStrategy):
    """
    Multi-angle Asian range breakout/fade/confirm strategy.
    Asia range computed inline — no pre-built price arrays, no lookahead.
    """

    min_lookback = 30

    def __init__(self, params: dict = None) -> None:
        super().__init__(params)
        p = params or {}

        asia_start_h = float(p.get('asia_start_hour', 18))   # 6 PM ET default
        entry_h      = float(p.get('entry_hour',       3))   # 3 AM ET (London open)
        self._asia_start_min = int(asia_start_h * 60)
        self._entry_min      = int(entry_h * 60)

        self.entry_mode     = str(p.get('entry_mode',          'breakout'))
        self.direction      = str(p.get('direction',           'momentum'))
        self.sl_type        = str(p.get('sl_type',             'atr'))
        self.rr_ratio       = float(p.get('rr_ratio',           1.5))
        self.sl_atr_mult    = float(p.get('sl_atr_multiplier',  0.75))
        self.atr_period     = int(p.get('atr_period',           14))
        self.risk_per_trade = float(p.get('risk_per_trade',     0.01))
        self.min_range_atr  = float(p.get('min_range_atr',      0.0))
        self.max_range_atr  = float(p.get('max_range_atr',    200.0))

        # Expiry: stop order expires just before NY open
        # ny_confirm uses a full RTH session expiry
        self._expiry_bars = int(p.get('expiry_bars', 380))

        # Set trading hours: include entry time
        if self.entry_mode == 'ny_confirm':
            self.trading_hours = [(_time(9, 30), _time(16, 0))]
        else:
            entry_hh = int(entry_h)
            entry_mm = int((entry_h - entry_hh) * 60)
            self.trading_hours = [(_time(entry_hh, entry_mm), _time(16, 0))]

        self._times_min: Optional[np.ndarray] = None
        self._cur_date  = None
        self._order_placed = False

    def _setup(self, data: MarketData) -> None:
        """Only builds the time index — no price arrays, no lookahead."""
        if self._times_min is not None:
            return
        idx = data.df_1m.index
        self._times_min = (idx.hour * 60 + idx.minute).to_numpy(np.int32)
        if data.bar_times_1m_min is None:
            data.bar_times_1m_min = self._times_min

    def generate_signals(self, data: MarketData, i: int) -> Optional[Order]:
        self._setup(data)

        t_min    = int(self._times_min[i])
        bar_date = data.df_1m.index[i].date()

        if bar_date != self._cur_date:
            self._cur_date     = bar_date
            self._order_placed = False

        if self._order_placed:
            return None

        # Fire only at the correct entry time
        if self.entry_mode == 'ny_confirm':
            if t_min != 9 * 60 + 30:
                return None
        else:
            if t_min != self._entry_min:
                return None

        # Compute Asia range inline (only bars ≤ i)
        asia_h, asia_l, asia_open, asia_close = _compute_asia_range(
            self._times_min, data.high_1m, data.low_1m, data.close_1m,
            i, self._asia_start_min, self._entry_min,
        )
        if np.isnan(asia_h):
            return None

        atr = _atr_at(data.high_1m, data.low_1m, data.close_1m, self.atr_period, i)
        if atr <= 0:
            return None

        # Range quality filter
        asia_range = asia_h - asia_l
        if atr > 0:
            range_atr = asia_range / atr
            if range_atr < self.min_range_atr or range_atr > self.max_range_atr:
                return None

        # Overnight momentum: direction from Asia session open to close
        if not np.isnan(asia_open) and not np.isnan(asia_close):
            ovn_dir = 1 if asia_close > asia_open else (-1 if asia_close < asia_open else 0)
        else:
            ovn_dir = 0

        # Determine trade direction
        if self.direction == 'long':
            go_long = True
        elif self.direction == 'short':
            go_long = False
        elif self.direction == 'momentum':
            if ovn_dir == 0:
                return None
            go_long = (ovn_dir == 1)
        elif self.direction == 'contrarian':
            if ovn_dir == 0:
                return None
            go_long = (ovn_dir == -1)
        else:
            return None

        # SL/TP distances
        if self.sl_type == 'range':
            sl_dist = _tick(max(asia_range, TICK * 4))
        else:
            sl_dist = _tick(max(self.sl_atr_mult * atr, TICK * 4))
        tp_dist = _tick(sl_dist * self.rr_ratio)

        self._order_placed = True

        # ── Build order ────────────────────────────────────────────────────
        if self.entry_mode == 'breakout':
            if go_long:
                stop_px = _tick(asia_h + TICK)
                return Order(
                    direction=1, order_type=OrderType.STOP,
                    size_type=SizeType.PCT_RISK, size_value=self.risk_per_trade,
                    stop_price=stop_px,
                    sl_price=_tick(stop_px - sl_dist), tp_price=_tick(stop_px + tp_dist),
                    expiry_bars=self._expiry_bars, trade_reason='asia_break_long',
                )
            else:
                stop_px = _tick(asia_l - TICK)
                return Order(
                    direction=-1, order_type=OrderType.STOP,
                    size_type=SizeType.PCT_RISK, size_value=self.risk_per_trade,
                    stop_price=stop_px,
                    sl_price=_tick(stop_px + sl_dist), tp_price=_tick(stop_px - tp_dist),
                    expiry_bars=self._expiry_bars, trade_reason='asia_break_short',
                )

        elif self.entry_mode == 'fade':
            # Fade = counter-direction stop at the range edge
            # If momentum was up, fade by going short when price tags Asia high
            if go_long:
                # go_long after contrarian → we expect price to reach Asia low and rebound
                stop_px = _tick(asia_l)
                return Order(
                    direction=1, order_type=OrderType.STOP,
                    size_type=SizeType.PCT_RISK, size_value=self.risk_per_trade,
                    stop_price=stop_px,
                    sl_price=_tick(stop_px - sl_dist), tp_price=_tick(stop_px + tp_dist),
                    expiry_bars=self._expiry_bars, trade_reason='asia_fade_long',
                )
            else:
                stop_px = _tick(asia_h)
                return Order(
                    direction=-1, order_type=OrderType.STOP,
                    size_type=SizeType.PCT_RISK, size_value=self.risk_per_trade,
                    stop_price=stop_px,
                    sl_price=_tick(stop_px + sl_dist), tp_price=_tick(stop_px - tp_dist),
                    expiry_bars=self._expiry_bars, trade_reason='asia_fade_short',
                )

        elif self.entry_mode == 'ny_confirm':
            curr = float(data.open_1m[i])
            if go_long:
                if curr <= asia_h:
                    return None  # price hasn't broken above Asia range
                return Order(
                    direction=1, order_type=OrderType.MARKET,
                    size_type=SizeType.PCT_RISK, size_value=self.risk_per_trade,
                    stop_price=0.0,
                    sl_price=_tick(curr - sl_dist), tp_price=_tick(curr + tp_dist),
                    expiry_bars=390, trade_reason='asia_ny_long',
                )
            else:
                if curr >= asia_l:
                    return None  # price hasn't broken below Asia range
                return Order(
                    direction=-1, order_type=OrderType.MARKET,
                    size_type=SizeType.PCT_RISK, size_value=self.risk_per_trade,
                    stop_price=0.0,
                    sl_price=_tick(curr + sl_dist), tp_price=_tick(curr - tp_dist),
                    expiry_bars=390, trade_reason='asia_ny_short',
                )

        return None

    def on_fill(self, position: OpenPosition, data: MarketData, i: int) -> None:
        if self.entry_mode != 'ny_confirm':
            # SL/TP already on Order; snapshot as initial values for R-multiple tracking
            position.set_initial_sl_tp(position.sl_price, position.tp_price)
            return
        fill = position.entry_price
        atr  = _atr_at(data.high_1m, data.low_1m, data.close_1m, self.atr_period, i)
        if atr <= 0:
            return
        if self.sl_type == 'range':
            position.set_initial_sl_tp(position.sl_price, position.tp_price)
            return
        sl_dist = _tick(max(self.sl_atr_mult * atr, TICK * 4))
        tp_dist = _tick(sl_dist * self.rr_ratio)
        if position.is_long():
            position.set_initial_sl_tp(_tick(fill - sl_dist), _tick(fill + tp_dist))
        else:
            position.set_initial_sl_tp(_tick(fill + sl_dist), _tick(fill - tp_dist))

    def manage_position(self, data: MarketData, i: int,
                        position: OpenPosition) -> Optional[PositionUpdate]:
        return None
