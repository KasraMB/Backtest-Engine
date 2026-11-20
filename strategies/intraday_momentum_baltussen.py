"""
Intraday Momentum Strategy — Baltussen, Büsing & Lohre (2021)
"Hedging Demand and Market Intraday Momentum"

Mechanism
─────────
The return from the previous close to entry_time positively predicts the
return during the final window before the close.

Signal:
    r_day = (price_at_entry_time / prev_close) - 1

Trade rule:
    r_day >  threshold → long  at entry_time
    r_day < -threshold → short at entry_time

Entry: MARKET order at entry_time (default 15:00 ET).
Exit:  EOD only — engine closes at eod_exit_time (default 15:30 ET).
       No SL or TP.

Params dict keys:
    entry_time_h    : int   — hour of entry (ET). Default 15
    entry_time_m    : int   — minute of entry (ET). Default 0  → 15:00
    threshold_pct   : float — min |r_day| % to trade. Default 0.0 (always)
    contracts       : int   — fixed contracts. Default 1
"""
from __future__ import annotations

from datetime import time
from typing import Optional

from backtest.data.market_data import MarketData
from backtest.strategy.base import BaseStrategy
from backtest.strategy.enums import OrderType, SizeType
from backtest.strategy.order import Order
from backtest.strategy.update import OpenPosition, PositionUpdate


class IntradayMomentumBaltussen(BaseStrategy):
    """
    Last-window momentum trade based on the day's return to entry_time.
    One trade per day. Entry MARKET at entry_time, exit EOD only.
    """

    trading_hours = [(time(9, 30), time(17, 0))]
    min_lookback  = 2

    def __init__(self, params: dict = None):
        super().__init__(params)
        p = params or {}

        self.entry_time_h:  int   = p.get("entry_time_h",  14)
        self.entry_time_m:  int   = p.get("entry_time_m",   0)
        self.threshold_pct: float = p.get("threshold_pct", 0.0)
        self.contracts:     int   = p.get("contracts",       1)

        self._entry_time = time(self.entry_time_h, self.entry_time_m)

        self._traded_today: bool  = False
        self._prev_close:   float = 0.0
        self._current_date        = None

    def generate_signals(self, data: MarketData, i: int) -> Optional[Order]:
        ts       = data.df_1m.index[i]
        bar_time = ts.time()
        bar_date = ts.date()

        # ── Day rollover ──────────────────────────────────────────────────────
        if bar_date != self._current_date:
            self._traded_today = False
            self._current_date = bar_date
            for j in range(i - 1, max(0, i - 500), -1):
                if data.df_1m.index[j].date() != bar_date:
                    self._prev_close = float(data.close_1m[j])
                    break

        if self._traded_today or bar_time != self._entry_time:
            return None
        if self._prev_close <= 0:
            return None

        current_price = float(data.close_1m[i])
        r_day = (current_price / self._prev_close) - 1.0

        if abs(r_day) < self.threshold_pct / 100.0:
            return None

        self._traded_today = True
        return Order(
            direction  = 1 if r_day > 0 else -1,
            order_type = OrderType.MARKET,
            size_type  = SizeType.CONTRACTS,
            size_value = self.contracts,
        )

    def manage_position(self, data: MarketData, i: int,
                        position: OpenPosition) -> Optional[PositionUpdate]:
        return None