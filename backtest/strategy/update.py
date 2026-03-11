from __future__ import annotations
from dataclasses import dataclass, field
from datetime import time
from typing import Optional

from .enums import ExitReason


# ---------------------------------------------------------------------------
# PositionUpdate — returned by manage_position()
# ---------------------------------------------------------------------------

@dataclass
class PositionUpdate:
    new_sl_price: Optional[float] = None
    new_tp_price: Optional[float] = None

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        if self.new_sl_price is None and self.new_tp_price is None:
            raise ValueError(
                "PositionUpdate must set at least one of new_sl_price or new_tp_price"
            )


# ---------------------------------------------------------------------------
# OpenPosition — live position state held by the engine
# ---------------------------------------------------------------------------

@dataclass
class OpenPosition:
    direction: int                          # 1 = long, -1 = short
    entry_price: float
    entry_bar: int
    contracts: float

    sl_price: Optional[float] = None
    tp_price: Optional[float] = None

    trail_points: Optional[float] = None
    trail_activation_points: Optional[float] = None
    trail_sl_price: Optional[float] = None      # current live trail level
    trail_watermark: Optional[float] = None     # best price seen since entry

    def is_long(self) -> bool:
        return self.direction == 1

    def is_short(self) -> bool:
        return self.direction == -1

    def effective_sl(self) -> Optional[float]:
        """
        Returns the most protective SL currently active.
        If both a fixed SL and a trail SL exist, returns the one
        that is closest to the current price (most protective).
        """
        candidates = [p for p in (self.sl_price, self.trail_sl_price) if p is not None]
        if not candidates:
            return None
        if self.direction == 1:
            return max(candidates)      # long: higher SL = more protective
        else:
            return min(candidates)      # short: lower SL = more protective


# ---------------------------------------------------------------------------
# RunConfig — parameters for a single backtest run
# ---------------------------------------------------------------------------

@dataclass
class RunConfig:
    starting_capital: float
    slippage_points: float
    commission_per_contract: float
    params: dict = field(default_factory=dict)
    eod_exit_time: time = time(15, 30)      # 15:30 ET default

    def __post_init__(self) -> None:
        if self.starting_capital <= 0:
            raise ValueError(f"starting_capital must be positive, got {self.starting_capital}")
        if self.slippage_points < 0:
            raise ValueError(f"slippage_points must be >= 0, got {self.slippage_points}")
        if self.commission_per_contract < 0:
            raise ValueError(
                f"commission_per_contract must be >= 0, got {self.commission_per_contract}"
            )


# ---------------------------------------------------------------------------
# Trade — completed trade record
# ---------------------------------------------------------------------------

POINT_VALUE = 20.0  # NQ: $20 per point per contract


@dataclass
class Trade:
    entry_bar: int
    exit_bar: int
    entry_price: float
    exit_price: float
    direction: int                      # 1 = long, -1 = short
    contracts: float
    slippage_points: float              # per-side slippage in points (from RunConfig)
    commission_per_contract: float      # per-side commission in dollars (from RunConfig)
    exit_reason: ExitReason

    @property
    def pnl_points(self) -> float:
        return (self.exit_price - self.entry_price) * self.direction

    @property
    def pnl_dollars(self) -> float:
        return self.pnl_points * self.contracts * POINT_VALUE

    @property
    def slippage_paid(self) -> float:
        # Entry + exit slippage
        return self.slippage_points * 2 * self.contracts * POINT_VALUE

    @property
    def commission_paid(self) -> float:
        # Entry + exit commission
        return self.commission_per_contract * 2 * self.contracts

    @property
    def net_pnl_dollars(self) -> float:
        return self.pnl_dollars - self.slippage_paid - self.commission_paid

    @property
    def is_winner(self) -> bool:
        return self.net_pnl_dollars > 0

    @property
    def bars_held(self) -> int:
        return self.exit_bar - self.entry_bar