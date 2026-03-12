from __future__ import annotations
from abc import ABC, abstractmethod
from datetime import time
from typing import Optional

from backtest.data.market_data import MarketData
from backtest.strategy.order import Order
from backtest.strategy.update import OpenPosition, PositionUpdate


class BaseStrategy(ABC):
    """
    Abstract base class for all strategies.

    Subclass this and implement generate_signals() and manage_position().
    Everything else is infrastructure — the runner, engine, and risk manager
    handle fills, exits, sizing, and session rules automatically.

    Class attributes to override:
        trading_hours:  list of (start, end) time tuples defining when signals
                        are generated. None = all hours. Position management
                        always runs regardless of trading_hours.
        min_lookback:   Minimum number of bars required before the loop starts
                        calling generate_signals. Set this to your longest
                        indicator lookback period.

    Example:
        class MyStrategy(BaseStrategy):
            trading_hours = [(time(9, 30), time(15, 30))]
            min_lookback = 20

            def __init__(self, params: dict):
                self.lookback = params.get("lookback", 20)

            def generate_signals(self, data, i):
                close = data.close_1m[i]
                ...
                return Order(direction=1, ...)

            def manage_position(self, data, i, position):
                return None  # let engine handle exits
    """

    # Override in subclass — list of (start_time, end_time) tuples, or None for all hours
    trading_hours: Optional[list[tuple[time, time]]] = None

    # Override in subclass — loop won't call generate_signals before this bar index
    min_lookback: int = 0

    def __init__(self, params: dict = None):
        """
        Args:
            params: Strategy parameters dict from RunConfig.params.
                    Override __init__ in your subclass to read these.
        """
        pass

    @abstractmethod
    def generate_signals(self, data: MarketData, i: int) -> Optional[Order]:
        """
        Called on every bar where:
          - No position is currently open
          - Bar i falls within declared trading_hours (or trading_hours is None)
          - i >= min_lookback

        Args:
            data: MarketData object with df_1m, df_5m, numpy arrays, bar_map
            i:    Current bar index into data.df_1m (and the numpy arrays)

        Returns:
            Order to submit, or None to do nothing this bar.

        Contract:
            - Only read data at index <= i. Never access data.close_1m[i+1] etc.
            - Do not manage exits here — use sl_price/tp_price on the Order.
            - This method may be called on consecutive bars if no fill occurs
              (e.g. a limit order that hasn't filled yet won't block new signals
              — but the runner enforces one pending order at a time).
        """
        ...

    @abstractmethod
    def manage_position(
        self,
        data: MarketData,
        i: int,
        position: OpenPosition,
    ) -> Optional[PositionUpdate]:
        """
        Called on every bar while a position is open, regardless of trading_hours.

        Args:
            data:     MarketData object
            i:        Current bar index
            position: The currently open position (read-only — mutate via PositionUpdate)

        Returns:
            PositionUpdate to modify SL/TP, or None to leave them unchanged.

        Contract:
            - The engine enforces that SL only moves in the favorable direction.
              Unfavorable updates are silently ignored — safe to emit every bar.
            - Setting new_sl beyond current price triggers a forced exit.
            - Setting new_tp at/inside current price triggers a forced exit.
        """
        ...