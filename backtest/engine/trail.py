from __future__ import annotations

from backtest.strategy.update import OpenPosition


def update_trail(position: OpenPosition, bar_high: float, bar_low: float) -> None:
    """
    Update the trail SL state on an open position for the current bar.
    Mutates position in-place.

    For longs:
      1. Update watermark = max(watermark, bar.high)
      2. Check activation threshold
      3. Recalculate trail_sl = watermark - trail_points
      4. Enforce favorable-only: trail_sl = max(trail_sl, previous_trail_sl)

    For shorts, mirror all logic.

    This function only updates position.trail_sl_price and position.trail_watermark.
    The caller (ExecutionEngine) is responsible for resolving effective_sl() and
    checking for exits afterward.
    """
    if position.trail_points is None:
        return  # no trail configured

    if position.is_long():
        _update_trail_long(position, bar_high)
    else:
        _update_trail_short(position, bar_low)


def _update_trail_long(position: OpenPosition, bar_high: float) -> None:
    # Step 1: Update watermark
    if position.trail_watermark is None:
        position.trail_watermark = position.entry_price
    position.trail_watermark = max(position.trail_watermark, bar_high)

    # Step 2: Check activation threshold
    if position.trail_activation_points is not None:
        move = position.trail_watermark - position.entry_price
        if move < position.trail_activation_points:
            return  # not yet activated — don't move trail

    # Step 3: Recalculate trail SL
    new_trail_sl = position.trail_watermark - position.trail_points

    # Step 4: Enforce favorable-only (trail only moves up for longs)
    if position.trail_sl_price is None:
        position.trail_sl_price = new_trail_sl
    else:
        position.trail_sl_price = max(position.trail_sl_price, new_trail_sl)


def _update_trail_short(position: OpenPosition, bar_low: float) -> None:
    # Step 1: Update watermark (tracks lowest price for shorts)
    if position.trail_watermark is None:
        position.trail_watermark = position.entry_price
    position.trail_watermark = min(position.trail_watermark, bar_low)

    # Step 2: Check activation threshold
    if position.trail_activation_points is not None:
        move = position.entry_price - position.trail_watermark
        if move < position.trail_activation_points:
            return  # not yet activated

    # Step 3: Recalculate trail SL
    new_trail_sl = position.trail_watermark + position.trail_points

    # Step 4: Enforce favorable-only (trail only moves down for shorts)
    if position.trail_sl_price is None:
        position.trail_sl_price = new_trail_sl
    else:
        position.trail_sl_price = min(position.trail_sl_price, new_trail_sl)