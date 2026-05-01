"""
Intraday Momentum with Overnight Return Signal — Rosa (2022)
"Understanding Intraday Momentum Strategies"

Mechanism
─────────
The overnight return (previous close → today's open) predicts the last-window
return.  Rosa's key finding: a threshold filter substantially improves results
versus always trading.

Signal:
    r_overnight = (today_open / prev_close) - 1

Trade rule:
    r_overnight >  threshold → long  at close_entry_time
    r_overnight < -threshold → short at close_entry_time
    |r_overnight| ≤ threshold → no trade

Entry: MARKET order at close_entry_time (default 15:00 ET).
Exit:  EOD only — engine closes at eod_exit_time. No SL or TP.

Optional dual_confirm: only trade when the prior day's close-to-close return
agrees with the overnight direction (double filter, also from Rosa 2022).

Params dict keys:
    threshold_pct   : float — min |r_overnight| % to trade. Default 0.2
    close_entry_h   : int   — entry hour. Default 15
    close_entry_m   : int   — entry minute. Default 0
    dual_confirm    : bool  — require same-direction prior day return. Default False
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


class OvernightMomentumRosa(BaseStrategy):
    """
    Overnight-return signal trades the last 30 minutes.
    Threshold filter is central to Rosa (2022).
    One trade per day. Entry MARKET at close_entry_time, exit EOD only.
    """

    trading_hours = [(time(8, 30), time(16, 0))]
    min_lookback  = 2

    def __init__(self, params: dict = None):
        super().__init__(params)
        p = params or {}

        self.threshold_pct: float = p.get("threshold_pct", 0.2)
        self.close_entry_h: int   = p.get("close_entry_h", 14)
        self.close_entry_m: int   = p.get("close_entry_m",  0)
        self.dual_confirm:  bool  = p.get("dual_confirm",   False)
        self.contracts:     int   = p.get("contracts",       1)

        self._close_entry = time(self.close_entry_h, self.close_entry_m)

        self._current_date      = None
        self._traded_today:     bool  = False
        self._prev_close:       float = 0.0
        self._prev_prev_close:  float = 0.0
        self._today_open:       float = 0.0
        self._open_captured:    bool  = False

    def _reset_day(self) -> None:
        self._traded_today   = False
        self._today_open     = 0.0
        self._open_captured  = False

    def generate_signals(self, data: MarketData, i: int) -> Optional[Order]:
        ts       = data.df_1m.index[i]
        bar_time = ts.time()
        bar_date = ts.date()

        if bar_date != self._current_date:
            self._reset_day()
            self._current_date = bar_date
            closes_seen = 0
            for j in range(i - 1, max(0, i - 1000), -1):
                jdate = data.df_1m.index[j].date()
                if jdate != bar_date:
                    if closes_seen == 0:
                        self._prev_close = float(data.close_1m[j])
                        closes_seen = 1
                    elif closes_seen == 1 and self.dual_confirm:
                        d2 = jdate
                        for k in range(j - 1, max(0, j - 1000), -1):
                            if data.df_1m.index[k].date() != d2:
                                self._prev_prev_close = float(data.close_1m[k])
                                break
                        break
                    else:
                        break

        if bar_time == time(8, 30) and not self._open_captured:
            self._today_open   = float(data.open_1m[i])
            self._open_captured = True

        if self._traded_today or bar_time != self._close_entry:
            return None
        if self._prev_close <= 0 or self._today_open <= 0:
            return None

        r_overnight = (self._today_open / self._prev_close) - 1.0

        if abs(r_overnight) < self.threshold_pct / 100.0:
            return None

        direction = 1 if r_overnight > 0 else -1

        if self.dual_confirm and self._prev_prev_close > 0:
            r_prior = (self._prev_close / self._prev_prev_close) - 1.0
            if (1 if r_prior > 0 else -1) != direction:
                return None

        self._traded_today = True
        return Order(
            direction  = direction,
            order_type = OrderType.MARKET,
            size_type  = SizeType.CONTRACTS,
            size_value = self.contracts,
        )

    def manage_position(self, data: MarketData, i: int,
                        position: OpenPosition) -> Optional[PositionUpdate]:
        return None