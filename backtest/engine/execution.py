from __future__ import annotations
import logging
from dataclasses import dataclass, field
from datetime import time
from typing import Optional

from backtest.strategy.enums import ExitReason, OrderType
from backtest.strategy.order import Order
from backtest.strategy.update import OpenPosition, PositionUpdate, Trade, POINT_VALUE
from backtest.engine.trail import update_trail

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# A single OHLC bar — engine-internal representation
# ---------------------------------------------------------------------------

@dataclass
class Bar:
    index: int
    open: float
    high: float
    low: float
    close: float
    bar_time: time  # time component of the bar's timestamp


# ---------------------------------------------------------------------------
# A pending order waiting to be filled
# ---------------------------------------------------------------------------

@dataclass
class PendingOrder:
    order: Order
    registered_bar: int         # bar index when the order was registered
    bars_remaining: Optional[int]  # None = GTC; counts down each bar


# ---------------------------------------------------------------------------
# Fill result — returned when a pending order fills
# ---------------------------------------------------------------------------

@dataclass
class FillResult:
    fill_price: float
    contracts: int


# ---------------------------------------------------------------------------
# Exit result — returned when a position exits
# ---------------------------------------------------------------------------

@dataclass
class ExitResult:
    exit_price: float
    exit_reason: ExitReason


# ---------------------------------------------------------------------------
# ExecutionEngine
# ---------------------------------------------------------------------------

class ExecutionEngine:
    """
    Handles all fill and exit logic for the backtesting engine.

    Responsibilities:
      - Attempt fills for pending orders (MARKET, LIMIT, STOP, STOP_LIMIT)
      - Check SL/TP/EOD exits for open positions
      - Apply PositionUpdate with enforcement rules
      - Handle same-bar fill + exit
      - Handle gap opens
      - Trail SL updates (delegated to trail.py)

    SL priority is a global invariant — never configurable.
    """

    def __init__(self, slippage_points: float, commission_per_contract: float, eod_exit_time: time):
        self.slippage_points = slippage_points
        self.commission_per_contract = commission_per_contract
        self.eod_exit_time = eod_exit_time

    # ------------------------------------------------------------------
    # Fill logic
    # ------------------------------------------------------------------

    def attempt_fill(self, pending: PendingOrder, bar: Bar) -> Optional[FillResult]:
        """
        Check whether a pending order fills on the given bar.
        Returns FillResult if filled, None if not.

        Also handles expiry — the caller must check pending.bars_remaining
        and cancel if expired before calling this.
        """
        order = pending.order
        ot = order.order_type

        if ot == OrderType.MARKET:
            # Market orders fill on the signal bar's close — handled by caller
            # This path won't normally be reached (market orders fill immediately)
            return FillResult(fill_price=bar.close, contracts=0)

        elif ot == OrderType.LIMIT:
            return self._try_limit_fill(order, bar)

        elif ot == OrderType.STOP:
            return self._try_stop_fill(order, bar)

        elif ot == OrderType.STOP_LIMIT:
            return self._try_stop_limit_fill(order, bar)

        return None

    def fill_market_order(self, order: Order, bar: Bar) -> FillResult:
        """Fill a market order at bar close."""
        return FillResult(fill_price=bar.close, contracts=0)

    def _try_limit_fill(self, order: Order, bar: Bar) -> Optional[FillResult]:
        lp = order.limit_price
        if order.is_long():
            # Long limit: fill if bar.low <= limit_price
            if bar.low <= lp:
                return FillResult(fill_price=lp, contracts=0)
        else:
            # Short limit: fill if bar.high >= limit_price
            if bar.high >= lp:
                return FillResult(fill_price=lp, contracts=0)
        return None

    def _try_stop_fill(self, order: Order, bar: Bar) -> Optional[FillResult]:
        sp = order.stop_price
        if order.is_long():
            # Long stop: fill if bar.high >= stop_price
            if bar.high >= sp:
                # Gap through stop: fill at open if open is already above stop
                fill_price = max(bar.open, sp)
                return FillResult(fill_price=fill_price, contracts=0)
        else:
            # Short stop: fill if bar.low <= stop_price
            if bar.low <= sp:
                fill_price = min(bar.open, sp)
                return FillResult(fill_price=fill_price, contracts=0)
        return None

    def _try_stop_limit_fill(self, order: Order, bar: Bar) -> Optional[FillResult]:
        """Stop triggers first, then limit fills using limit rules."""
        sp = order.stop_price
        lp = order.limit_price

        if order.is_long():
            if bar.high >= sp:
                # Stop triggered — now check limit
                if bar.low <= lp:
                    return FillResult(fill_price=lp, contracts=0)
        else:
            if bar.low <= sp:
                if bar.high >= lp:
                    return FillResult(fill_price=lp, contracts=0)
        return None

    # ------------------------------------------------------------------
    # Same-bar fill + exit
    # ------------------------------------------------------------------

    def check_same_bar_exit(
        self,
        position: OpenPosition,
        bar: Bar,
        fill_price: float,
    ) -> Optional[ExitResult]:
        """
        After a limit/stop fill within a bar, check whether SL or TP is
        also hit in the remaining range of the same bar.

        SL is always checked first.

        For a long filled at fill_price:
          - Remaining high = bar.high (price could still go up)
          - Remaining low  = fill_price (can't use anything below fill)
        """
        sl = position.effective_sl()
        tp = position.tp_price

        if position.is_long():
            # Use full bar range — SL priority means we assume adverse move happened first
            if sl is not None and bar.low <= sl:
                exit_price = sl  # filled at SL level (no gap, fill happened before this bar)
                return ExitResult(exit_price=exit_price, exit_reason=ExitReason.SAME_BAR_SL)

            if tp is not None and bar.high >= tp:
                return ExitResult(exit_price=tp, exit_reason=ExitReason.SAME_BAR_TP)

        else:  # short
            if sl is not None and bar.high >= sl:
                exit_price = sl
                return ExitResult(exit_price=exit_price, exit_reason=ExitReason.SAME_BAR_SL)

            if tp is not None and bar.low <= tp:
                return ExitResult(exit_price=tp, exit_reason=ExitReason.SAME_BAR_TP)

        return None

    # ------------------------------------------------------------------
    # Exit checking (normal bar)
    # ------------------------------------------------------------------

    def check_exits(
        self,
        position: OpenPosition,
        bar: Bar,
        is_last_bar_of_session: bool,
    ) -> Optional[ExitResult]:
        """
        Check all exit conditions for an open position on a normal bar.
        Priority: SL -> TP -> EOD -> (SIGNAL handled by runner)
        """
        # --- Trail SL update ---
        update_trail(position, bar.high, bar.low)

        sl = position.effective_sl()
        tp = position.tp_price

        if position.is_long():
            # Gap-open below SL
            if sl is not None and bar.open <= sl:
                return ExitResult(exit_price=bar.open, exit_reason=ExitReason.SL)

            # Gap-open above TP (SL checked first — if both gapped, SL wins)
            if tp is not None and bar.open >= tp:
                if sl is not None and bar.open <= sl:
                    return ExitResult(exit_price=bar.open, exit_reason=ExitReason.SL)
                return ExitResult(exit_price=bar.open, exit_reason=ExitReason.TP)

            # Normal intrabar SL check
            if sl is not None and bar.low <= sl:
                # Gap protection: if bar opened below SL, fill at open (worse).
                # Otherwise fill at SL level.
                # Long SL is below entry, so adverse move is downward.
                # Worse price = lower price = min(open, sl).
                exit_price = min(bar.open, sl)
                return ExitResult(exit_price=exit_price, exit_reason=ExitReason.SL)

            # Normal intrabar TP check (only if SL not hit)
            if tp is not None and bar.high >= tp:
                return ExitResult(exit_price=tp, exit_reason=ExitReason.TP)

        else:  # short
            # Gap-open above SL
            if sl is not None and bar.open >= sl:
                return ExitResult(exit_price=bar.open, exit_reason=ExitReason.SL)

            # Gap-open below TP
            if tp is not None and bar.open <= tp:
                if sl is not None and bar.open >= sl:
                    return ExitResult(exit_price=bar.open, exit_reason=ExitReason.SL)
                return ExitResult(exit_price=bar.open, exit_reason=ExitReason.TP)

            # Normal intrabar SL check
            if sl is not None and bar.high >= sl:
                # Gap protection: if bar opened above SL, fill at open (worse).
                # Short SL is above entry, so adverse move is upward.
                # Worse price = higher price = max(open, sl).
                exit_price = max(bar.open, sl)
                return ExitResult(exit_price=exit_price, exit_reason=ExitReason.SL)

            # Normal intrabar TP check
            if tp is not None and bar.low <= tp:
                return ExitResult(exit_price=tp, exit_reason=ExitReason.TP)

        # EOD exit
        if is_last_bar_of_session:
            return ExitResult(exit_price=bar.close, exit_reason=ExitReason.EOD)

        return None

    # ------------------------------------------------------------------
    # PositionUpdate enforcement
    # ------------------------------------------------------------------

    def apply_position_update(
        self,
        position: OpenPosition,
        update: PositionUpdate,
        current_price: float,
    ) -> Optional[ExitResult]:
        """
        Apply a PositionUpdate to an open position with full enforcement rules.

        Rules:
          - SL only moves in favorable direction (unfavorable moves silently ignored)
          - new_sl beyond current price -> forced exit at current price
          - new_tp at/inside current price -> forced exit at current price
          - new_sl and new_tp crossing -> logged as error, rejected entirely

        Returns ExitResult if the update triggers a forced exit, None otherwise.
        """
        new_sl = update.new_sl_price
        new_tp = update.new_tp_price

        # Cross-validation: SL and TP must not cross
        if new_sl is not None and new_tp is not None:
            if position.is_long() and new_sl >= new_tp:
                logger.error(
                    f"PositionUpdate rejected: new_sl {new_sl} >= new_tp {new_tp} on long position. "
                    f"Update discarded entirely."
                )
                return None
            if position.is_short() and new_sl <= new_tp:
                logger.error(
                    f"PositionUpdate rejected: new_sl {new_sl} <= new_tp {new_tp} on short position. "
                    f"Update discarded entirely."
                )
                return None

        # Apply SL update
        if new_sl is not None:
            forced_exit = self._apply_sl_update(position, new_sl, current_price)
            if forced_exit:
                return forced_exit

        # Apply TP update
        if new_tp is not None:
            forced_exit = self._apply_tp_update(position, new_tp, current_price)
            if forced_exit:
                return forced_exit

        return None

    def _apply_sl_update(
        self,
        position: OpenPosition,
        new_sl: float,
        current_price: float,
    ) -> Optional[ExitResult]:
        """
        Apply new SL with enforcement.
        Returns ExitResult if new_sl triggers a forced exit.
        """
        if position.is_long():
            # Check: new_sl beyond current price -> forced exit
            if new_sl >= current_price:
                return ExitResult(exit_price=current_price, exit_reason=ExitReason.FORCED_EXIT)
            # Favorable-only: long SL only moves up
            if position.sl_price is None or new_sl > position.sl_price:
                position.sl_price = new_sl
            # else: unfavorable move, silently ignored

        else:  # short
            # Check: new_sl beyond current price -> forced exit
            if new_sl <= current_price:
                return ExitResult(exit_price=current_price, exit_reason=ExitReason.FORCED_EXIT)
            # Favorable-only: short SL only moves down
            if position.sl_price is None or new_sl < position.sl_price:
                position.sl_price = new_sl

        return None

    def _apply_tp_update(
        self,
        position: OpenPosition,
        new_tp: float,
        current_price: float,
    ) -> Optional[ExitResult]:
        """
        Apply new TP with enforcement.
        Returns ExitResult if new_tp triggers a forced exit.
        """
        if position.is_long():
            # TP at or inside current price -> forced exit
            if new_tp <= current_price:
                return ExitResult(exit_price=current_price, exit_reason=ExitReason.FORCED_EXIT)
            position.tp_price = new_tp

        else:  # short
            if new_tp >= current_price:
                return ExitResult(exit_price=current_price, exit_reason=ExitReason.FORCED_EXIT)
            position.tp_price = new_tp

        return None

    # ------------------------------------------------------------------
    # Multi-contract delta resolution
    # ------------------------------------------------------------------

    def resolve_delta(
        self,
        pending_order: Order,
        resolved_contracts: int,
        position: Optional[OpenPosition],
    ) -> tuple[int, int]:
        """
        Resolve the delta between a new signal and an existing position.

        Returns (contracts_to_close, contracts_to_open) where:
          - contracts_to_close: how many of the current position to close (0 if none)
          - contracts_to_open:  how many contracts to open in the new direction (0 if none)

        Examples:
          No position:                   (0, resolved_contracts)
          Same direction, add:           (0, delta)
          Same direction, reduce:        (delta, 0)
          Opposite direction, flip:      (all_current, remainder_in_new_dir)
          Opposite direction, exact:     (all_current, 0)
        """
        if position is None:
            return (0, resolved_contracts)

        current_signed = position.contracts * position.direction
        new_signed = resolved_contracts * pending_order.direction
        delta = new_signed - current_signed

        if delta == 0:
            return (0, 0)

        if pending_order.direction == position.direction:
            # Same direction
            if delta > 0:
                # Adding to position
                return (0, int(abs(delta)))
            else:
                # Reducing position
                return (int(abs(delta)), 0)
        else:
            # Opposite direction — close all current, open remainder in new direction
            # remainder = abs(new) - abs(current), floored at 0
            contracts_to_open = max(0, int(abs(new_signed)) - int(position.contracts))
            return (int(position.contracts), contracts_to_open)

    # ------------------------------------------------------------------
    # Trade construction
    # ------------------------------------------------------------------

    def build_trade(
        self,
        position: OpenPosition,
        exit_bar: int,
        exit_price: float,
        exit_reason: ExitReason,
    ) -> Trade:
        """Construct a completed Trade record from an open position and exit info."""
        return Trade(
            entry_bar=position.entry_bar,
            exit_bar=exit_bar,
            entry_price=position.entry_price,
            exit_price=exit_price,
            direction=position.direction,
            contracts=position.contracts,
            slippage_points=self.slippage_points,
            commission_per_contract=self.commission_per_contract,
            exit_reason=exit_reason,
        )

    # ------------------------------------------------------------------
    # Pending order expiry
    # ------------------------------------------------------------------

    def tick_expiry(self, pending: PendingOrder) -> bool:
        """
        Decrement expiry counter.
        Returns True if the order has expired and should be cancelled.
        """
        if pending.bars_remaining is None:
            return False  # GTC, never expires
        pending.bars_remaining -= 1
        return pending.bars_remaining <= 0