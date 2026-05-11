from __future__ import annotations
from dataclasses import dataclass, field
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

    initial_sl_price: Optional[float] = None  # SL set at entry (before any trailing/updates)
    initial_tp_price: Optional[float] = None  # TP set at entry (before any updates)

    trail_points: Optional[float] = None
    trail_activation_points: Optional[float] = None
    trail_sl_price: Optional[float] = None      # current live trail level
    trail_watermark: Optional[float] = None     # best price seen since entry

    mae_points:   float = 0.0  # max adverse excursion from entry (points, always >= 0)
    mfe_points:   float = 0.0  # max favorable excursion from entry (points, always >= 0)
    trade_reason: str   = ''   # human-readable entry rationale set by the strategy
    order_placed_bar: Optional[int] = None  # bar index when the limit/stop order was registered
    fib_levels:      list = field(default_factory=list)  # [{p, t, d, v}] captured at signal time
    signal_features: dict = field(default_factory=dict)  # ML feature snapshot at signal time

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

    def set_initial_sl_tp(self, sl: Optional[float], tp: Optional[float]) -> None:
        """
        Set SL and TP directly on a freshly opened position, bypassing the
        apply_position_update enforcement rules (which check against current
        bar close and would trigger false forced exits if bar closed past TP).
        Only call this immediately after a fill, before check_exits runs.
        """
        self.sl_price = sl
        self.tp_price = tp
        self.initial_sl_price = sl
        self.initial_tp_price = tp


# ---------------------------------------------------------------------------
# RunConfig — re-exported from runner.config (canonical definition lives there)
# ---------------------------------------------------------------------------
from backtest.runner.config import RunConfig  # noqa: F401



# ---------------------------------------------------------------------------
# Trade — completed trade record
# ---------------------------------------------------------------------------

POINT_VALUE = 20.0  # NQ: $20 per point per contract
TICK_SIZE   = 0.25  # NQ: minimum price increment


def round_to_tick(price: float, tick_size: float = TICK_SIZE) -> float:
    """Round a price to the nearest valid tick. e.g. 18311.13 → 18311.25."""
    return round(round(price / tick_size) * tick_size, 10)


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
    sl_price: Optional[float] = None          # SL at exit time (may be trailed from initial)
    tp_price: Optional[float] = None          # TP at exit time
    initial_sl_price: Optional[float] = None  # SL set at entry (before any trailing/updates)
    initial_tp_price: Optional[float] = None  # TP set at entry (before any updates)
    had_trailing: bool = False                # True if a trailing stop was active at any point
    mae_points:   float = 0.0  # max adverse excursion from entry (points)
    mfe_points:   float = 0.0  # max favorable excursion from entry (points)
    trade_reason: str   = ''   # human-readable entry rationale set by the strategy
    order_placed_bar: Optional[int] = None  # bar index when the limit/stop order was registered
    fib_levels:      list = field(default_factory=list)  # [{p, t, d, v}] captured at signal time
    signal_features: dict = field(default_factory=dict)  # ML feature snapshot at signal time

    @property
    def _initial_risk_points(self) -> Optional[float]:
        if self.initial_sl_price is None:
            return None
        r = abs(self.entry_price - self.initial_sl_price)
        return r if r > 1e-9 else None

    @property
    def r_multiple(self) -> Optional[float]:
        risk = self._initial_risk_points
        return self.pnl_points / risk if risk is not None else None

    @property
    def mae_r(self) -> Optional[float]:
        risk = self._initial_risk_points
        return self.mae_points / risk if risk is not None else None

    @property
    def mfe_r(self) -> Optional[float]:
        risk = self._initial_risk_points
        return self.mfe_points / risk if risk is not None else None

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