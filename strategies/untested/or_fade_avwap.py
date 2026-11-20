"""Opening-Range fade to Anchored VWAP — agent-suggested 2nd candidate.

Mechanism (per the strategy-class agent):
  - Opening Range (OR) = high-low of 9:30-10:00 ET (30 minutes).
  - If price BREAKS the OR by >= k * OR_size within the entry window (default
    10:00-11:30), fade the break back toward the anchored VWAP computed from
    the prior session's close onward.
  - SL: just beyond the break extreme + buffer.
  - TP: at the anchored VWAP, OR at RR * sl_dist (whichever is closer/further
    based on tp_mode).

Distinct from existing ORB-ATR breakout (which TRADES the break, not fades it).
"""
from __future__ import annotations

from datetime import time
from typing import Optional

import numpy as np

from backtest.data.market_data import MarketData
from backtest.strategy.base import BaseStrategy
from backtest.strategy.enums import OrderType, SizeType
from backtest.strategy.order import Order
from backtest.strategy.update import OpenPosition, PositionUpdate


class ORFadeAVWAP(BaseStrategy):
    trading_hours = [(time(9, 30), time(15, 30))]
    min_lookback = 0

    def __init__(self, params: dict = None):
        super().__init__(params)
        p = params or {}
        self.or_start_h = int(p.get("or_start_h", 9))
        self.or_start_m = int(p.get("or_start_m", 30))
        self.or_end_h   = int(p.get("or_end_h",  10))
        self.or_end_m   = int(p.get("or_end_m",   0))
        self.entry_end_h = int(p.get("entry_end_h", 11))
        self.entry_end_m = int(p.get("entry_end_m", 30))
        self.break_k = float(p.get("break_k", 0.25))   # break > k * OR_size triggers fade
        self.rr      = float(p.get("rr",        1.5))
        self.sl_buffer = float(p.get("sl_buffer", 1.0))
        self.exit_h  = int(p.get("exit_h",  15))
        self.exit_m  = int(p.get("exit_m",  30))
        self.contracts = int(p.get("contracts", 1))
        self.max_per_day = int(p.get("max_per_day", 2))

        self._or_start = time(self.or_start_h, self.or_start_m)
        self._or_end   = time(self.or_end_h,   self.or_end_m)
        self._entry_end = time(self.entry_end_h, self.entry_end_m)

        # Per-day state
        self._cur_date = None
        self._or_hi = -1e18
        self._or_lo =  1e18
        self._or_built = False
        self._vwap_num = 0.0    # sum of price*vol since anchor
        self._vwap_den = 0.0    # sum of vol since anchor
        self._anchor_bar = -1
        self._traded_up = False
        self._traded_dn = False
        self._trades_today = 0
        self._pending_sl: Optional[float] = None
        self._pending_tp: Optional[float] = None

    def _on_new_day(self, data: MarketData, i: int):
        cur_date = data.df_1m.index[i].date()
        # Anchor VWAP at the PRIOR session close bar (last RTH bar of prior date).
        anchor = -1
        lo = max(0, i - 2000)
        for j in range(i - 1, lo, -1):
            jd = data.df_1m.index[j].date()
            if jd == cur_date:
                continue
            jt = data.df_1m.index[j].time()
            if time(15, 30) <= jt <= time(16, 0):
                anchor = j
                break
            if jd != data.df_1m.index[max(0, j - 1)].date() and anchor < 0:
                anchor = j
                break
        if anchor < 0:
            anchor = max(0, i - 1)
        self._anchor_bar = anchor
        self._vwap_num = 0.0
        self._vwap_den = 0.0
        # Seed VWAP with bars from anchor to i-1
        for j in range(anchor, i):
            price = (float(data.high_1m[j]) + float(data.low_1m[j])
                     + float(data.close_1m[j])) / 3.0
            vol = float(data.volume_1m[j]) if hasattr(data, 'volume_1m') else 1.0
            self._vwap_num += price * vol
            self._vwap_den += vol
        self._cur_date = cur_date
        self._or_hi = -1e18
        self._or_lo =  1e18
        self._or_built = False
        self._traded_up = False
        self._traded_dn = False
        self._trades_today = 0

    def generate_signals(self, data: MarketData, i: int) -> Optional[Order]:
        ts = data.df_1m.index[i]
        if ts.date() != self._cur_date:
            self._on_new_day(data, i)
        bt = ts.time()
        # Update running VWAP
        price = (float(data.high_1m[i]) + float(data.low_1m[i])
                 + float(data.close_1m[i])) / 3.0
        vol = float(data.volume_1m[i]) if hasattr(data, 'volume_1m') else 1.0
        self._vwap_num += price * vol
        self._vwap_den += vol
        # Build OR
        if self._or_start <= bt < self._or_end:
            h = float(data.high_1m[i]); l = float(data.low_1m[i])
            if h > self._or_hi: self._or_hi = h
            if l < self._or_lo: self._or_lo = l
            return None
        if bt < self._or_end:
            return None
        if not self._or_built:
            self._or_built = True
        if bt >= self._entry_end:
            return None
        if self._trades_today >= self.max_per_day:
            return None
        or_size = self._or_hi - self._or_lo
        if or_size <= 0:
            return None
        c = float(data.close_1m[i])
        vwap = self._vwap_num / max(self._vwap_den, 1e-9)

        # Upside break -> fade SHORT (target VWAP)
        if not self._traded_up and c > self._or_hi + self.break_k * or_size and c > vwap:
            self._traded_up = True
            self._trades_today += 1
            sl = self._round_tick(float(data.high_1m[i]) + self.sl_buffer)
            sl_dist = sl - c
            if sl_dist <= 0:
                return None
            tp = self._round_tick(c - self.rr * sl_dist)
            self._pending_sl, self._pending_tp = sl, tp
            return Order(direction=-1, order_type=OrderType.MARKET,
                         size_type=SizeType.CONTRACTS, size_value=self.contracts,
                         sl_price=sl, tp_price=tp, trade_reason="or_fade_short")
        # Downside break -> fade LONG
        if not self._traded_dn and c < self._or_lo - self.break_k * or_size and c < vwap:
            self._traded_dn = True
            self._trades_today += 1
            sl = self._round_tick(float(data.low_1m[i]) - self.sl_buffer)
            sl_dist = c - sl
            if sl_dist <= 0:
                return None
            tp = self._round_tick(c + self.rr * sl_dist)
            self._pending_sl, self._pending_tp = sl, tp
            return Order(direction=1, order_type=OrderType.MARKET,
                         size_type=SizeType.CONTRACTS, size_value=self.contracts,
                         sl_price=sl, tp_price=tp, trade_reason="or_fade_long")
        return None

    def on_fill(self, position: OpenPosition, data: MarketData, bar_index: int) -> None:
        if self._pending_sl is not None and self._pending_tp is not None:
            position.set_initial_sl_tp(self._pending_sl, self._pending_tp)
        self._pending_sl = None
        self._pending_tp = None

    def manage_position(self, data: MarketData, i: int,
                        position: OpenPosition) -> Optional[PositionUpdate]:
        ts = data.df_1m.index[i]
        if ts.time() >= time(self.exit_h, self.exit_m):
            tick = self.tick_size
            if position.direction == 1:
                return PositionUpdate(new_sl_price=self._round_tick(float(data.close_1m[i]) + tick))
            else:
                return PositionUpdate(new_sl_price=self._round_tick(float(data.close_1m[i]) - tick))
        return None
