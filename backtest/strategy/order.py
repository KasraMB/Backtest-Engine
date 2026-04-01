from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

from .enums import OrderType, SizeType


@dataclass
class Order:
    direction: int                          # 1 = long, -1 = short
    order_type: OrderType
    size_type: SizeType
    size_value: float                       # contracts | dollars | pct (0.01 = 1%)

    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    expiry_bars: Optional[int] = None       # None = GTC

    sl_price: Optional[float] = None
    tp_price: Optional[float] = None
    trail_points: Optional[float] = None
    trail_activation_points: Optional[float] = None

    trade_reason: str = ''  # human-readable entry rationale set by the strategy

    # §9.3: cancel pending limit order if TP is reached before fill.
    # cancel_above: long order cancelled when bar.high >= cancel_above AND bar.low > limit_price
    # cancel_below: short order cancelled when bar.low <= cancel_below AND bar.high < limit_price
    cancel_above: Optional[float] = None
    cancel_below: Optional[float] = None

    def __post_init__(self) -> None:
        self._validate()

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def is_long(self) -> bool:
        return self.direction == 1

    def is_short(self) -> bool:
        return self.direction == -1

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate(self) -> None:
        self._validate_direction()
        self._validate_size()
        self._validate_sl_trail_exclusive()
        self._validate_sizing_requires_risk_anchor()
        self._validate_limit_price_side()
        self._validate_stop_price_side()
        self._validate_trail_points()
        self._validate_size_value()

    def _validate_direction(self) -> None:
        if self.direction not in (1, -1):
            raise ValueError(f"direction must be 1 or -1, got {self.direction}")

    def _validate_size(self) -> None:
        if self.size_value <= 0:
            raise ValueError(f"size_value must be positive, got {self.size_value}")

    def _validate_sl_trail_exclusive(self) -> None:
        if self.sl_price is not None and self.trail_points is not None:
            raise ValueError("sl_price and trail_points are mutually exclusive — use one or the other")

    def _validate_sizing_requires_risk_anchor(self) -> None:
        if self.size_type in (SizeType.DOLLARS, SizeType.PCT_RISK):
            has_anchor = self.sl_price is not None or self.trail_points is not None
            if not has_anchor:
                raise ValueError(
                    f"size_type={self.size_type.name} requires either sl_price or trail_points "
                    f"to compute position size"
                )

    def _validate_limit_price_side(self) -> None:
        """
        For LIMIT and STOP_LIMIT orders the limit price must be on the correct
        side relative to the expected fill direction:
          - Long  LIMIT : buy below market  -> limit_price must be set (enforced by caller at fill time)
          - Short LIMIT : sell above market -> limit_price must be set
        We only validate that the price exists here; directional sanity vs entry
        price is checked at fill time when we know the actual market price.
        """
        if self.order_type in (OrderType.LIMIT, OrderType.STOP_LIMIT):
            if self.limit_price is None:
                raise ValueError(
                    f"order_type={self.order_type.name} requires limit_price"
                )

    def _validate_stop_price_side(self) -> None:
        if self.order_type in (OrderType.STOP, OrderType.STOP_LIMIT):
            if self.stop_price is None:
                raise ValueError(
                    f"order_type={self.order_type.name} requires stop_price"
                )

    def _validate_trail_points(self) -> None:
        if self.trail_points is not None and self.trail_points <= 0:
            raise ValueError(f"trail_points must be positive, got {self.trail_points}")
        if self.trail_activation_points is not None:
            if self.trail_points is None:
                raise ValueError("trail_activation_points requires trail_points to be set")
            if self.trail_activation_points < 0:
                raise ValueError(
                    f"trail_activation_points must be >= 0, got {self.trail_activation_points}"
                )

    def _validate_size_value(self) -> None:
        if self.size_type == SizeType.PCT_RISK:
            if not (0 < self.size_value <= 1.0):
                raise ValueError(
                    f"PCT_RISK size_value must be in (0, 1], got {self.size_value}. "
                    f"Use 0.01 for 1%, not 1."
                )
        if self.size_type == SizeType.CONTRACTS:
            if self.size_value < 1 or self.size_value != int(self.size_value):
                raise ValueError(
                    f"CONTRACTS size_value must be a positive integer, got {self.size_value}"
                )