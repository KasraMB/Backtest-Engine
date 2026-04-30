from __future__ import annotations
from abc import ABC, abstractmethod
from datetime import time
from typing import Optional

from backtest.data.market_data import MarketData
from backtest.strategy.order import Order
from backtest.strategy.enums import OrderType
from backtest.strategy.update import OpenPosition, PositionUpdate


class BaseStrategy(ABC):
    """
    Abstract base class for all strategies.

    Subclass this and implement generate_signals() and manage_position().
    Everything else is infrastructure — the runner, engine, and risk manager
    handle fills, exits, sizing, and session rules automatically.

    Reversal mode
    ─────────────
    When reverse_mode=True (set by the runner from RunConfig.reverse_signals),
    every signal is transparently flipped at the base-class level:

      • generate_signals: direction flipped; SL/TP mirrored through the
        order's anchor price (stop_price or limit_price).
        For orders with no SL/TP on the order itself (e.g. ORB uses on_fill),
        the order direction is flipped but fill trigger (stop_price) is kept
        identical to guarantee the same trade count.

      • on_fill: after the subclass sets SL/TP via set_initial_sl_tp(),
        the base class mirrors them through entry_price and flips direction.

    Subclasses never need to know about reverse_mode — zero changes required.

    Class attributes to override:
        trading_hours:  list of (start, end) time tuples. None = all hours.
        min_lookback:   Minimum bars before generate_signals is called.
    """

    trading_hours: Optional[list[tuple[time, time]]] = None
    min_lookback: int = 0

    def __init__(self, params: dict = None):
        self.closed_trades: list = []
        # Set by the runner from RunConfig.reverse_signals before the bar loop.
        self._reverse_mode: bool = False

    def signal_bar_mask(self, data: 'MarketData') -> Optional['np.ndarray']:
        """Return a boolean array (length = n_bars) marking bars where
        generate_signals should be called.  The runner also uses this as the
        flat-bar skip mask so it can skip irrelevant bars when no position is
        open — potentially giving a large speedup for sparse strategies.

        Return None to use the default trading_hours-based behaviour."""
        return None

    # ── Reversal helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _mirror_order(order: Order) -> Order:
        """Flip direction and mirror SL/TP through anchor (stop or limit price)."""
        anchor = (order.stop_price if order.stop_price is not None
                  else order.limit_price if order.limit_price is not None
                  else None)
        new_sl: Optional[float] = None
        new_tp: Optional[float] = None
        if anchor is not None:
            if order.sl_price is not None:
                new_sl = anchor - (order.sl_price - anchor)
            if order.tp_price is not None:
                new_tp = anchor - (order.tp_price - anchor)
        return Order(
            direction   = -order.direction,
            order_type  = order.order_type,
            size_type   = order.size_type,
            size_value  = order.size_value,
            limit_price = order.limit_price,
            stop_price  = order.stop_price,
            expiry_bars = order.expiry_bars,
            sl_price    = new_sl,
            tp_price    = new_tp,
            trail_points            = order.trail_points,
            trail_activation_points = order.trail_activation_points,
        )

    @staticmethod
    def _mirror_position(position: OpenPosition) -> None:
        """Flip direction and mirror SL/TP through entry price, in-place."""
        entry = position.entry_price
        old_sl, old_tp = position.sl_price, position.tp_price
        position.direction = -position.direction
        position.sl_price  = (2 * entry - old_sl) if old_sl is not None else None
        position.tp_price  = (2 * entry - old_tp) if old_tp is not None else None

    # ── Internal wrappers — runner calls these, not the public methods ────────

    def _generate_signals(self, data: MarketData, i: int) -> Optional[Order]:
        order = self.generate_signals(data, i)
        if order is None or not self._reverse_mode:
            return order
        has_sl_tp = order.sl_price is not None or order.tp_price is not None
        if has_sl_tp or order.order_type == OrderType.MARKET:
            # SL/TP are on the order — mirror them now
            return self._mirror_order(order)
        else:
            # SL/TP set post-fill in on_fill — keep fill trigger identical,
            # just flip direction. _on_fill will mirror after subclass runs.
            return Order(
                direction   = -order.direction,
                order_type  = order.order_type,
                size_type   = order.size_type,
                size_value  = order.size_value,
                limit_price = order.limit_price,
                stop_price  = order.stop_price,
                expiry_bars = order.expiry_bars,
                sl_price    = None,
                tp_price    = None,
                trail_points            = order.trail_points,
                trail_activation_points = order.trail_activation_points,
            )

    def _on_fill(self, position: OpenPosition, data: MarketData, bar_index: int) -> None:
        self.on_fill(position, data, bar_index)
        if self._reverse_mode and (position.sl_price is not None or position.tp_price is not None):
            self._mirror_position(position)

    # ── Public interface for subclasses ──────────────────────────────────────

    @abstractmethod
    def generate_signals(self, data: MarketData, i: int) -> Optional[Order]:
        """Return an Order to submit, or None. Called when flat and in trading_hours."""
        ...

    def on_fill(self, position: OpenPosition, data: MarketData, bar_index: int) -> None:
        """
        Optional. Called immediately after any order fills.
        Use to set initial SL/TP from actual fill price via position.set_initial_sl_tp(sl, tp).
        In reverse_mode the base class mirrors them automatically after this returns.
        """
        pass

    @abstractmethod
    def manage_position(
        self,
        data: MarketData,
        i: int,
        position: OpenPosition,
    ) -> Optional[PositionUpdate]:
        """
        Called every bar while in a position. Return PositionUpdate or None.
        The engine enforces SL only moves favorably. Unfavorable updates ignored.
        """
        ...