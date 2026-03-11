from __future__ import annotations
from dataclasses import dataclass

from backtest.strategy.enums import SizeType
from backtest.strategy.order import Order

POINT_VALUE = 20.0  # NQ: $20 per point per contract


@dataclass
class Account:
    balance: float

    def __post_init__(self) -> None:
        if self.balance <= 0:
            raise ValueError(f"Account balance must be positive, got {self.balance}")


class RiskManager:
    """
    Converts an Order's size specification into a concrete contract count.

    Rules:
      CONTRACTS  -> use size_value directly
      DOLLARS    -> size_value / (sl_distance_points * point_value)
      PCT_RISK   -> (balance * size_value) / (sl_distance_points * point_value)

    For trail-based sizing (no fixed SL), sl_distance is derived from trail_points.
    Result is always rounded DOWN to nearest whole contract, minimum 1.
    """

    def resolve_contracts(
        self,
        order: Order,
        account: Account,
        fill_price: float,
    ) -> int:
        """
        Returns the number of whole contracts to trade.

        Args:
            order:      The Order object (already validated)
            account:    Current account state (balance)
            fill_price: The price at which the order will fill

        Returns:
            Integer contract count, minimum 1
        """
        if order.size_type == SizeType.CONTRACTS:
            return int(order.size_value)

        sl_distance = self._sl_distance(order, fill_price)
        if sl_distance <= 0:
            raise ValueError(
                f"Cannot resolve contracts: SL distance is {sl_distance}. "
                f"fill_price={fill_price}, sl_price={order.sl_price}, "
                f"trail_points={order.trail_points}"
            )

        if order.size_type == SizeType.DOLLARS:
            raw = order.size_value / (sl_distance * POINT_VALUE)
        else:  # PCT_RISK
            raw = (account.balance * order.size_value) / (sl_distance * POINT_VALUE)

        contracts = max(1, int(raw))  # floor, minimum 1
        return contracts

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _sl_distance(self, order: Order, fill_price: float) -> float:
        """
        Distance in points between fill price and the effective stop.
        Always returns a positive number (or 0 if something is wrong).
        """
        if order.sl_price is not None:
            return abs(fill_price - order.sl_price)
        if order.trail_points is not None:
            return order.trail_points
        return 0.0