"""
DummyLongStrategy
─────────────────
A minimal strategy that:
  - Enters long every N bars (configurable via params["entry_every"])
  - Uses a fixed SL and TP offset (configurable)
  - Calls manage_position but does nothing (lets engine handle exits)

Purpose: prove the runner loop works end-to-end before writing real strategies.
Not intended for actual use.
"""
from __future__ import annotations
from datetime import time
from typing import Optional

from backtest.data.market_data import MarketData
from backtest.strategy.base import BaseStrategy
from backtest.strategy.enums import OrderType, SizeType
from backtest.strategy.order import Order
from backtest.strategy.update import OpenPosition, PositionUpdate


class DummyLongStrategy(BaseStrategy):
    """
    Enters a long market order every `entry_every` bars during trading hours.
    Exits via fixed SL and TP set on the order.
    manage_position does nothing — engine handles all exits.
    """

    trading_hours = [(time(9, 30), time(15, 30))]
    min_lookback = 1

    def __init__(self, params: dict = None):
        params = params or {}
        self.entry_every: int = params.get("entry_every", 30)  # bars between entries
        self.sl_offset: float = params.get("sl_offset", 20.0)  # points below entry
        self.tp_offset: float = params.get("tp_offset", 40.0)  # points above entry
        self.contracts: int = params.get("contracts", 1)
        self._bars_since_entry: int = 0

    def generate_signals(self, data: MarketData, i: int) -> Optional[Order]:
        self._bars_since_entry += 1

        if self._bars_since_entry < self.entry_every:
            return None

        self._bars_since_entry = 0
        close = data.close_1m[i]

        return Order(
            direction=1,
            order_type=OrderType.MARKET,
            size_type=SizeType.CONTRACTS,
            size_value=self.contracts,
            sl_price=close - self.sl_offset,
            tp_price=close + self.tp_offset,
        )

    def manage_position(
        self,
        data: MarketData,
        i: int, # current bar index
        position: OpenPosition,
    ) -> Optional[PositionUpdate]:
        # Let the engine handle everything via SL/TP
        return None