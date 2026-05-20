"""
ORB-Delta Breakout (long-only).

Port of the IVB.md EasyLanguage strategy. Opening-range breakout with
delta (bid/ask aggressor imbalance) confirmation.

Setup
─────
  ORB_window:   [orb_start_hm, orb_start_hm + orb_dur_mins]  (NY time)
  Trade_window: (orb_end, trade_end_hm]
  Entry:        Close > ORB_high  AND  bar_delta >= delta_threshold
                 (optional) AND cum_delta_since_orb_start >= cum_delta_threshold
  SL:           ORB_low
  TP:           entry + tp_rr_ratio × (entry - ORB_low)
  Exit:         EOD at trade_end_hm  (force exit on close)
  Direction:    Long only
  Trades/day:   1 max

Delta source
────────────
  bar_delta = ask_volume - bid_volume   (from Sierra-style 1-min cache)
              Equivalent to TradeStation's Upticks - Downticks in concept,
              but expressed in CONTRACTS (volume) rather than TICK COUNTS.
              Threshold tuning differs accordingly.

Time convention
───────────────
  The cache index is tz-aware (NY/ET). Hours/minutes pulled directly from
  the DatetimeIndex - no conversion needed.

Params (dict)
─────────────
  orb_start_hm:           int  HHMM, default 830   (8:30 NY)
  orb_dur_mins:           int       default 30
  trade_end_hm:           int  HHMM, default 1400  (14:00 NY)
  tp_rr_ratio:            float     default 1.0
  delta_threshold:        float     default 200    (contracts of net buying)
  use_cumulative_delta:   bool      default False
  cum_delta_threshold:    float     default 500
"""
from __future__ import annotations

from datetime import time
from typing import Optional

import numpy as np
import pandas as pd

from backtest.data.market_data import MarketData
from backtest.engine.base import BaseStrategy
from backtest.strategy.enums import OrderType, SizeType
from backtest.strategy.order import Order
from backtest.strategy.update import OpenPosition, PositionUpdate


class ORBDeltaBreakoutStrategy(BaseStrategy):
    """Long-only Opening Range Breakout with delta-volume confirmation."""

    # Trading hours are dynamically determined by params; declared as None
    # so the runner doesn't pre-filter. We gate entries internally via the
    # session-time arrays below.
    trading_hours = None
    min_lookback = 5   # need ~5 bars to be safe before any logic fires

    def __init__(self, params: Optional[dict] = None):
        super().__init__(params or {})
        p = params or {}
        self.orb_start_hm: int        = int(p.get("orb_start_hm",        830))
        self.orb_dur_mins: int        = int(p.get("orb_dur_mins",        30))
        self.trade_end_hm: int        = int(p.get("trade_end_hm",        1400))
        self.tp_rr_ratio: float       = float(p.get("tp_rr_ratio",       1.0))
        self.delta_threshold: float   = float(p.get("delta_threshold",   200.0))
        self.use_cumulative_delta: bool = bool(p.get("use_cumulative_delta", False))
        self.cum_delta_threshold: float = float(p.get("cum_delta_threshold", 500.0))
        self.num_contracts: float     = float(p.get("num_contracts",     1.0))

        # ── Derived: convert HHMM → minutes-since-midnight ──
        self._orb_start_min = self._hm_to_min(self.orb_start_hm)
        self._orb_end_min   = self._orb_start_min + self.orb_dur_mins
        self._trade_end_min = self._hm_to_min(self.trade_end_hm)

        # ── Per-day state (reset on date change) ──
        self._cur_date = None
        self._orb_high: Optional[float] = None
        self._orb_low:  Optional[float] = None
        self._orb_inited: bool = False
        self._orb_session_ended: bool = False
        self._long_done_today: bool = False
        self._cum_delta: float = 0.0
        self._pending_sl: Optional[float] = None
        self._pending_tp: Optional[float] = None

        # ── Cached per-bar numpy arrays (built lazily on first call) ──
        self._times_min: Optional[np.ndarray] = None
        self._dates_ord: Optional[np.ndarray] = None
        self._bar_delta: Optional[np.ndarray] = None

    # ────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _hm_to_min(hhmm: int) -> int:
        return (hhmm // 100) * 60 + (hhmm % 100)

    def _ensure_arrays(self, data: MarketData) -> None:
        if self._times_min is not None:
            return
        idx = data.df_1m.index
        if data.bar_times_1m_min is not None:
            self._times_min = data.bar_times_1m_min
        else:
            self._times_min = (idx.hour * 60 + idx.minute).to_numpy(np.int32)
        self._dates_ord = pd.DatetimeIndex(idx).normalize().asi8   # ns since epoch; equal-day cells match
        # bar_delta in CONTRACTS: ask_volume - bid_volume per bar
        if 'ask_volume' in data.df_1m.columns and 'bid_volume' in data.df_1m.columns:
            self._bar_delta = (
                data.df_1m['ask_volume'].to_numpy(np.float64)
                - data.df_1m['bid_volume'].to_numpy(np.float64)
            )
        else:
            # Fallback: no delta info → use 0 (filter will never fire)
            self._bar_delta = np.zeros(len(data.df_1m), dtype=np.float64)

    def signal_bar_mask(self, data: MarketData) -> np.ndarray:
        """Restrict generate_signals to bars in the ORB-building or trade window.
        Outside this band the strategy can't do anything anyway, so skipping
        avoids ~80% of bars per day."""
        self._ensure_arrays(data)
        t = self._times_min
        # Cover full ORB+trade window so day-state (ORB high/low, cum_delta)
        # accumulates correctly. start from orb_start_min so the first bar of
        # the ORB window is included.
        mask = (t > self._orb_start_min) & (t <= self._trade_end_min)
        return mask

    def _reset_day(self) -> None:
        self._orb_high = None
        self._orb_low  = None
        self._orb_inited = False
        self._orb_session_ended = False
        self._long_done_today = False
        self._cum_delta = 0.0
        self._pending_sl = None
        self._pending_tp = None

    # ────────────────────────────────────────────────────────────────────────
    def generate_signals(self, data: MarketData, i: int) -> Optional[Order]:
        self._ensure_arrays(data)

        # Day rollover
        day_ord = self._dates_ord[i]
        if day_ord != self._cur_date:
            self._cur_date = day_ord
            self._reset_day()

        t_min = int(self._times_min[i])
        in_orb     = (t_min >  self._orb_start_min) and (t_min <= self._orb_end_min)
        in_trade   = (t_min >  self._orb_end_min)   and (t_min <= self._trade_end_min)

        # Build ORB high/low
        if in_orb:
            h, l = float(data.high_1m[i]), float(data.low_1m[i])
            if not self._orb_inited:
                self._orb_high, self._orb_low = h, l
                self._orb_inited = True
            else:
                if h > self._orb_high:
                    self._orb_high = h
                if l < self._orb_low:
                    self._orb_low = l

        if self._orb_inited and (not self._orb_session_ended) and (t_min > self._orb_end_min):
            self._orb_session_ended = True

        # Cumulative delta tracking — accumulates from ORB start through trade end
        if (t_min > self._orb_start_min) and (t_min <= self._trade_end_min):
            self._cum_delta += float(self._bar_delta[i])

        # Entry logic
        if not (self._orb_session_ended and in_trade and not self._long_done_today):
            return None
        if self._orb_high is None or self._orb_low is None:
            return None

        close_i = float(data.close_1m[i])
        if not (close_i > self._orb_high):
            return None

        sl_price = self._orb_low
        if sl_price >= close_i:
            return None   # invalid SL

        bar_delta_i = float(self._bar_delta[i])
        delta_ok = bar_delta_i >= self.delta_threshold
        if self.use_cumulative_delta:
            delta_ok = delta_ok and (self._cum_delta >= self.cum_delta_threshold)
        if not delta_ok:
            return None

        # Compose order — MARKET buy at next bar open. SL/TP set in on_fill.
        ep = close_i
        tp_price = ep + self.tp_rr_ratio * (ep - sl_price)
        self._pending_sl = sl_price
        self._pending_tp = tp_price
        self._long_done_today = True

        return Order(
            direction  = 1,
            order_type = OrderType.MARKET,
            size_type  = SizeType.CONTRACTS,
            size_value = self.num_contracts,
            trade_reason = (
                f"ORB[{self.orb_start_hm:04d}+{self.orb_dur_mins}m] long: "
                f"close={ep:.2f} > orb_high={self._orb_high:.2f}, "
                f"sl={sl_price:.2f}, tp={tp_price:.2f}, "
                f"bar_delta={bar_delta_i:.0f}, cum_delta={self._cum_delta:.0f}"
            ),
        )

    # ────────────────────────────────────────────────────────────────────────
    def on_fill(self, position: OpenPosition, data: MarketData, bar_index: int) -> None:
        if self._pending_sl is not None and self._pending_tp is not None:
            position.set_initial_sl_tp(sl=self._pending_sl, tp=self._pending_tp)
            self._pending_sl = None
            self._pending_tp = None

    # ────────────────────────────────────────────────────────────────────────
    def manage_position(
        self,
        data: MarketData,
        i: int,
        position: OpenPosition,
    ) -> Optional[PositionUpdate]:
        # EOD: when we pass trade_end_min, tighten SL to current bar close so
        # the engine exits next bar (the runner doesn't support a direct
        # "exit-at-close" from manage_position; the cleanest forced exit is
        # SL=close which triggers same-bar SL fill).
        # NOTE: this is conservative — if SL would cross the original SL the
        # engine takes the better fill. For ORB-style EOD this just locks in
        # current price.
        self._ensure_arrays(data)
        t_min = int(self._times_min[i])
        if t_min >= self._trade_end_min and position.is_long():
            close_i = float(data.close_1m[i])
            # Move SL to current close so the next bar exits at SL.
            # Only update if it would actually move SL up (engine enforces
            # favorable-only updates), otherwise no-op.
            if position.sl_price is None or close_i > position.sl_price:
                return PositionUpdate(new_sl_price=close_i)
        return None
