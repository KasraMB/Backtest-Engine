"""
Overnight Drift — equity-index futures.

The overnight return anomaly: equity index returns accrue overnight (positive,
low vol) while intraday is flat/negative with negative skew. Persistent since
French & Roll (1986); re-documented for index futures by the NY Fed
(Boyarchenko, Larsen, Whelan, "The Overnight Drift", SR 917).

LucidFlex compliance: positions auto-close 4:45 PM ET; market reopens 6:00 PM ET.
Entering at the 6:00 PM reopen and exiting next morning is allowed (the overnight
session is tradeable; only holding ACROSS 4:45-6:00 PM is blocked).

Rule (LONG only — the drift is a structural risk premium, no prediction needed):
    Entry: MARKET long at the evening reopen (default 18:00 ET), once per evening.
    Exit:  next-morning RTH open (default 09:30 ET), forced flat.
    Stop:  fill x (1 - sl_pct) to bound overnight-gap adverse moves.
    TP:    optional, fill x (1 + tp_pct); 0 disables (exit on time only).

Params:
    entry_h/entry_m : evening entry time. Default 18:00 ET.
    exit_h/exit_m   : morning exit time.  Default 09:30 ET.
    sl_pct          : stop as % of fill price. Default 0.5 (%).
    tp_pct          : take-profit as % of fill. Default 0.0 (disabled).
    contracts       : fixed contracts. Default 1.
"""
from __future__ import annotations

from datetime import time
from typing import Optional

from backtest.data.market_data import MarketData
from backtest.strategy.base import BaseStrategy
from backtest.strategy.enums import OrderType, SizeType
from backtest.strategy.order import Order
from backtest.strategy.update import OpenPosition, PositionUpdate


class OvernightDriftStrategy(BaseStrategy):
    """Long the overnight session (evening reopen -> next-morning open)."""

    # generate_signals only needs to fire in the evening entry window.
    trading_hours = [(time(18, 0), time(23, 59))]
    min_lookback = 0

    def __init__(self, params: dict = None):
        super().__init__(params)
        p = params or {}
        self.entry_time = time(int(p.get("entry_h", 18)), int(p.get("entry_m", 0)))
        self.exit_time  = time(int(p.get("exit_h",  9)),  int(p.get("exit_m", 30)))
        self.sl_pct   = float(p.get("sl_pct", 0.5)) / 100.0
        self.tp_pct   = float(p.get("tp_pct", 0.0)) / 100.0
        self.contracts = int(p.get("contracts", 1))

        self._entry_eve = None    # calendar date of the evening we last entered
        self._entry_date = None   # calendar date of the fill (for exit gating)

    def generate_signals(self, data: MarketData, i: int) -> Optional[Order]:
        ts = data.df_1m.index[i]
        if ts.time() >= self.entry_time and self._entry_eve != ts.date():
            self._entry_eve = ts.date()
            return Order(
                direction=1,
                order_type=OrderType.MARKET,
                size_type=SizeType.CONTRACTS,
                size_value=self.contracts,
                trade_reason="overnight_drift",
            )
        return None

    def on_fill(self, position: OpenPosition, data: MarketData, bar_index: int) -> None:
        entry = position.entry_price
        sl = self._round_tick(entry * (1.0 - self.sl_pct))
        tp = self._round_tick(entry * (1.0 + self.tp_pct)) if self.tp_pct > 0 else None
        self._entry_date = data.df_1m.index[bar_index].date()
        position.set_initial_sl_tp(sl, tp)

    def manage_position(self, data: MarketData, i: int,
                        position: OpenPosition) -> Optional[PositionUpdate]:
        ts = data.df_1m.index[i]
        # Force flat at the morning exit time on a later calendar date than entry
        # (SL-to-price forced exit, per the established interval-exit pattern).
        if ts.time() >= self.exit_time and ts.date() != self._entry_date:
            return PositionUpdate(
                new_sl_price=self._round_tick(float(data.close_1m[i]) + self.tick_size)
            )
        return None
