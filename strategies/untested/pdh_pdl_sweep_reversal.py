"""PDH/PDL liquidity-sweep reversal — Larry Williams "Oops" / true ICT turtle-soup
applied to PRIOR DAY HIGH / PRIOR DAY LOW (not noisy intraday swings).

Mechanism (per the strategy-class agent):
  - Identify Prior-Day High (PDH) and Prior-Day Low (PDL) from prior RTH session.
  - During today's RTH window, look for a SWEEP + REJECTION:
      SHORT: bar's high > PDH AND bar closes back below PDH -> short the rejection
      LONG : bar's low  < PDL AND bar closes back above  PDL -> long  the rejection
  - SL just beyond the sweep extreme + 1-tick buffer; structural, not ATR-based.
  - TP at fill +/- RR * sl_dist (default RR=1.5, the trader's eval-aligned).

Distinct from existing ICT/SMC (which has degenerate sub-tick SLs): the structural
SL just beyond the swept PDH/PDL is meaningful (typically 10-30 NQ pts), not the
0.25-floor proxy that broke the audit.
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


class PDHPDLSweepReversal(BaseStrategy):
    trading_hours = [(time(9, 30), time(15, 30))]
    min_lookback = 0

    def __init__(self, params: dict = None):
        super().__init__(params)
        p = params or {}
        self.rr           = float(p.get("rr",            1.5))
        self.sl_buffer    = float(p.get("sl_buffer",     1.0))  # pts beyond extreme
        self.entry_h      = int  (p.get("entry_h",         9))
        self.entry_m      = int  (p.get("entry_m",        30))
        self.exit_h       = int  (p.get("exit_h",         15))
        self.exit_m       = int  (p.get("exit_m",         30))
        self.contracts    = int  (p.get("contracts",       1))
        self.max_per_day  = int  (p.get("max_per_day",     2))

        # Per-day state
        self._pdh: float = 0.0
        self._pdl: float = 0.0
        self._cur_date = None
        self._prev_date = None
        self._today_high: float = -1e18
        self._today_low:  float =  1e18
        self._traded_long_today  = False
        self._traded_short_today = False
        self._trades_today = 0
        self._pending_sl: Optional[float] = None
        self._pending_tp: Optional[float] = None

    def _on_new_day(self, data: MarketData, i: int):
        """Reset day state; PDH/PDL = prior session's RTH high/low."""
        # Scan backward for the prior trading date's RTH bars (9:30-16:00).
        # Cheap: walk back ~390 bars + buffer to find the prior date.
        cur_date = data.df_1m.index[i].date()
        ph, pl = -1e18, 1e18
        prev_d = None
        # cap lookback to avoid pathological scans at history start
        lo = max(0, i - 2000)
        for j in range(i - 1, lo, -1):
            jd = data.df_1m.index[j].date()
            if jd == cur_date:
                continue
            if prev_d is None:
                prev_d = jd
            if jd != prev_d:
                break
            t = data.df_1m.index[j].time()
            if time(9, 30) <= t <= time(16, 0):
                h = float(data.high_1m[j]); l = float(data.low_1m[j])
                if h > ph: ph = h
                if l < pl: pl = l
        if ph > -1e17 and pl < 1e17:
            self._pdh, self._pdl = ph, pl
        else:
            self._pdh, self._pdl = 0.0, 0.0
        self._cur_date = cur_date
        self._today_high = -1e18
        self._today_low  =  1e18
        self._traded_long_today  = False
        self._traded_short_today = False
        self._trades_today = 0

    def generate_signals(self, data: MarketData, i: int) -> Optional[Order]:
        ts = data.df_1m.index[i]
        if ts.date() != self._cur_date:
            self._on_new_day(data, i)
        if self._pdh == 0.0 or self._pdl == 0.0:
            return None
        h = float(data.high_1m[i]); l = float(data.low_1m[i]); c = float(data.close_1m[i])
        if h > self._today_high: self._today_high = h
        if l < self._today_low:  self._today_low  = l
        if self._trades_today >= self.max_per_day:
            return None

        # SHORT: sweep above PDH, reject (close back below)
        if not self._traded_short_today and h > self._pdh and c < self._pdh:
            self._traded_short_today = True
            self._trades_today += 1
            sl = self._round_tick(h + self.sl_buffer)
            sl_dist = sl - c
            if sl_dist <= 0:
                return None
            tp = self._round_tick(c - self.rr * sl_dist)
            self._pending_sl, self._pending_tp = sl, tp
            return Order(direction=-1, order_type=OrderType.MARKET,
                         size_type=SizeType.CONTRACTS, size_value=self.contracts,
                         sl_price=sl, tp_price=tp, trade_reason="pdh_sweep_short")

        # LONG: sweep below PDL, reject (close back above)
        if not self._traded_long_today and l < self._pdl and c > self._pdl:
            self._traded_long_today = True
            self._trades_today += 1
            sl = self._round_tick(l - self.sl_buffer)
            sl_dist = c - sl
            if sl_dist <= 0:
                return None
            tp = self._round_tick(c + self.rr * sl_dist)
            self._pending_sl, self._pending_tp = sl, tp
            return Order(direction=1, order_type=OrderType.MARKET,
                         size_type=SizeType.CONTRACTS, size_value=self.contracts,
                         sl_price=sl, tp_price=tp, trade_reason="pdl_sweep_long")
        return None

    def on_fill(self, position: OpenPosition, data: MarketData, bar_index: int) -> None:
        # SL/TP were already on the order; reaffirm.
        if self._pending_sl is not None and self._pending_tp is not None:
            position.set_initial_sl_tp(self._pending_sl, self._pending_tp)
        self._pending_sl = None
        self._pending_tp = None

    def manage_position(self, data: MarketData, i: int,
                        position: OpenPosition) -> Optional[PositionUpdate]:
        # Force flat by exit time (so we don't carry into the 4:45 PM rule)
        ts = data.df_1m.index[i]
        if ts.time() >= time(self.exit_h, self.exit_m):
            tick = self.tick_size
            if position.direction == 1:
                return PositionUpdate(new_sl_price=self._round_tick(float(data.close_1m[i]) + tick))
            else:
                return PositionUpdate(new_sl_price=self._round_tick(float(data.close_1m[i]) - tick))
        return None
