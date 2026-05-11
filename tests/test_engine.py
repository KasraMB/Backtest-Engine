"""
Phase 3 unit tests — Execution Engine
Run with: python -m pytest tests/test_engine.py -v
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from datetime import time

from backtest.strategy.enums import OrderType, SizeType, ExitReason
from backtest.strategy.order import Order
from backtest.strategy.update import OpenPosition, PositionUpdate, POINT_VALUE
from backtest.engine.risk import RiskManager, Account
from backtest.engine.trail import update_trail
from backtest.engine.execution import ExecutionEngine, Bar, PendingOrder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_bar(index=0, open_=19000.0, high=19100.0, low=18900.0, close=19050.0,
             bar_time=time(10, 0)):
    return Bar(index=index, open=open_, high=high, low=low, close=close, bar_time=bar_time)

def make_engine(slippage=0.25, commission=4.50, eod_time=time(15, 30)):
    return ExecutionEngine(
        slippage_points=slippage,
        commission_per_contract=commission,
        eod_exit_time=eod_time,
    )

def make_long_position(entry=19000.0, bar=0, contracts=1,
                       sl=None, tp=None, trail=None, trail_act=None):
    return OpenPosition(
        direction=1, entry_price=entry, entry_bar=bar, contracts=contracts,
        sl_price=sl, tp_price=tp, trail_points=trail,
        trail_activation_points=trail_act,
    )

def make_short_position(entry=19000.0, bar=0, contracts=1,
                        sl=None, tp=None, trail=None, trail_act=None):
    return OpenPosition(
        direction=-1, entry_price=entry, entry_bar=bar, contracts=contracts,
        sl_price=sl, tp_price=tp, trail_points=trail,
        trail_activation_points=trail_act,
    )


# ===========================================================================
# RiskManager
# ===========================================================================

class TestRiskManager:

    def setup_method(self):
        self.rm = RiskManager()
        self.account = Account(balance=100_000)

    def test_contracts_sizing(self):
        order = Order(direction=1, order_type=OrderType.MARKET,
                      size_type=SizeType.CONTRACTS, size_value=3)
        assert self.rm.resolve_contracts(order, self.account, fill_price=19000) == 3

    def test_dollars_sizing(self):
        # Risk $1000, SL 25 pts away -> $1000 / (25 * $20) = 2 contracts
        order = Order(direction=1, order_type=OrderType.MARKET,
                      size_type=SizeType.DOLLARS, size_value=1000.0,
                      sl_price=18975.0)
        result = self.rm.resolve_contracts(order, self.account, fill_price=19000.0)
        assert result == 2

    def test_dollars_sizing_floors_down(self):
        # Risk $500, SL 25 pts -> $500 / (25 * $20) = 1.0 -> floor to 1
        order = Order(direction=1, order_type=OrderType.MARKET,
                      size_type=SizeType.DOLLARS, size_value=500.0,
                      sl_price=18975.0)
        result = self.rm.resolve_contracts(order, self.account, fill_price=19000.0)
        assert result == 1

    def test_dollars_minimum_one_contract(self):
        # Very tight risk amount -> should still get at least 1 contract
        order = Order(direction=1, order_type=OrderType.MARKET,
                      size_type=SizeType.DOLLARS, size_value=1.0,
                      sl_price=18975.0)
        result = self.rm.resolve_contracts(order, self.account, fill_price=19000.0)
        assert result == 1

    def test_pct_risk_sizing(self):
        # 1% of $100k = $1000, SL 25 pts -> 2 contracts
        order = Order(direction=1, order_type=OrderType.MARKET,
                      size_type=SizeType.PCT_RISK, size_value=0.01,
                      sl_price=18975.0)
        result = self.rm.resolve_contracts(order, self.account, fill_price=19000.0)
        assert result == 2

    def test_pct_risk_scales_with_balance(self):
        large_account = Account(balance=200_000)
        order = Order(direction=1, order_type=OrderType.MARKET,
                      size_type=SizeType.PCT_RISK, size_value=0.01,
                      sl_price=18975.0)
        result = self.rm.resolve_contracts(order, large_account, fill_price=19000.0)
        assert result == 4  # 2% of 200k = $2000 -> 4 contracts

    def test_trail_based_sizing(self):
        # Trail 50 pts -> $1000 / (50 * $20) = 1 contract
        order = Order(direction=1, order_type=OrderType.MARKET,
                      size_type=SizeType.DOLLARS, size_value=1000.0,
                      trail_points=50.0)
        result = self.rm.resolve_contracts(order, self.account, fill_price=19000.0)
        assert result == 1

    def test_short_sl_distance(self):
        # Short, entry 19000, SL 19025 -> 25 pts distance
        order = Order(direction=-1, order_type=OrderType.MARKET,
                      size_type=SizeType.DOLLARS, size_value=1000.0,
                      sl_price=19025.0)
        result = self.rm.resolve_contracts(order, self.account, fill_price=19000.0)
        assert result == 2


# ===========================================================================
# Trail SL
# ===========================================================================

class TestTrailSL:

    def test_trail_initializes_on_first_bar_long(self):
        pos = make_long_position(entry=19000, trail=50.0)
        update_trail(pos, bar_high=19020, bar_low=18980)
        assert pos.trail_watermark == 19020
        assert pos.trail_sl_price == 19020 - 50  # 18970

    def test_trail_only_moves_favorably_long(self):
        pos = make_long_position(entry=19000, trail=50.0)
        update_trail(pos, bar_high=19100, bar_low=19000)
        assert pos.trail_sl_price == 19050  # watermark 19100 - 50

        # Price pulls back — watermark stays, trail stays
        update_trail(pos, bar_high=19050, bar_low=19000)
        assert pos.trail_sl_price == 19050  # unchanged

    def test_trail_watermark_advances_long(self):
        pos = make_long_position(entry=19000, trail=50.0)
        update_trail(pos, bar_high=19100, bar_low=19000)
        update_trail(pos, bar_high=19200, bar_low=19100)
        assert pos.trail_watermark == 19200
        assert pos.trail_sl_price == 19150

    def test_trail_initializes_short(self):
        pos = make_short_position(entry=19000, trail=50.0)
        update_trail(pos, bar_high=19010, bar_low=18980)
        assert pos.trail_watermark == 18980
        assert pos.trail_sl_price == 18980 + 50  # 19030

    def test_trail_only_moves_favorably_short(self):
        pos = make_short_position(entry=19000, trail=50.0)
        update_trail(pos, bar_high=19000, bar_low=18900)
        assert pos.trail_sl_price == 18950  # 18900 + 50

        # Price bounces — trail stays
        update_trail(pos, bar_high=18960, bar_low=18940)
        assert pos.trail_sl_price == 18950  # unchanged

    def test_trail_activation_not_triggered(self):
        # Activation threshold 30 pts, price only moved 10 pts
        pos = make_long_position(entry=19000, trail=50.0, trail_act=30.0)
        update_trail(pos, bar_high=19010, bar_low=19000)
        assert pos.trail_sl_price is None  # not activated yet

    def test_trail_activation_triggered(self):
        pos = make_long_position(entry=19000, trail=50.0, trail_act=30.0)
        # Move 31 pts to trigger activation
        update_trail(pos, bar_high=19031, bar_low=19000)
        assert pos.trail_sl_price is not None
        assert pos.trail_sl_price == 19031 - 50  # 18981

    def test_no_trail_configured(self):
        pos = make_long_position(entry=19000)  # no trail
        update_trail(pos, bar_high=19100, bar_low=19000)
        assert pos.trail_sl_price is None
        assert pos.trail_watermark is None


# ===========================================================================
# Fill logic
# ===========================================================================

class TestFills:

    def setup_method(self):
        self.engine = make_engine()

    def _make_pending(self, order, registered_bar=0):
        bars = order.expiry_bars
        return PendingOrder(order=order, registered_bar=registered_bar,
                            bars_remaining=bars)

    # --- LIMIT fills ---

    def test_limit_long_fills_when_low_touches(self):
        order = Order(direction=1, order_type=OrderType.LIMIT,
                      size_type=SizeType.CONTRACTS, size_value=1,
                      limit_price=18950.0)
        bar = make_bar(low=18940.0, high=19100.0)  # low below limit
        pending = self._make_pending(order)
        result = self.engine.attempt_fill(pending, bar)
        assert result is not None
        assert result.fill_price == 18950.0

    def test_limit_long_no_fill_when_low_above_limit(self):
        order = Order(direction=1, order_type=OrderType.LIMIT,
                      size_type=SizeType.CONTRACTS, size_value=1,
                      limit_price=18900.0)
        bar = make_bar(low=18950.0, high=19100.0)  # low above limit
        pending = self._make_pending(order)
        result = self.engine.attempt_fill(pending, bar)
        assert result is None

    def test_limit_short_fills_when_high_touches(self):
        order = Order(direction=-1, order_type=OrderType.LIMIT,
                      size_type=SizeType.CONTRACTS, size_value=1,
                      limit_price=19100.0)
        bar = make_bar(low=18900.0, high=19150.0)  # high above limit
        pending = self._make_pending(order)
        result = self.engine.attempt_fill(pending, bar)
        assert result is not None
        assert result.fill_price == 19100.0

    def test_limit_short_no_fill_when_high_below_limit(self):
        order = Order(direction=-1, order_type=OrderType.LIMIT,
                      size_type=SizeType.CONTRACTS, size_value=1,
                      limit_price=19200.0)
        bar = make_bar(low=18900.0, high=19100.0)
        pending = self._make_pending(order)
        result = self.engine.attempt_fill(pending, bar)
        assert result is None

    # --- STOP fills ---

    def test_stop_long_fills_when_high_touches(self):
        order = Order(direction=1, order_type=OrderType.STOP,
                      size_type=SizeType.CONTRACTS, size_value=1,
                      stop_price=19100.0)
        bar = make_bar(open_=19000, high=19150.0, low=18900.0)
        pending = self._make_pending(order)
        result = self.engine.attempt_fill(pending, bar)
        assert result is not None
        assert result.fill_price == 19100.0  # normal fill at stop price

    def test_stop_long_gap_through(self):
        # Bar opens above stop price -> fill at open
        order = Order(direction=1, order_type=OrderType.STOP,
                      size_type=SizeType.CONTRACTS, size_value=1,
                      stop_price=19100.0)
        bar = make_bar(open_=19200.0, high=19300.0, low=19150.0)  # opened above stop
        pending = self._make_pending(order)
        result = self.engine.attempt_fill(pending, bar)
        assert result is not None
        assert result.fill_price == 19200.0  # gap fill at open

    def test_stop_short_fills(self):
        order = Order(direction=-1, order_type=OrderType.STOP,
                      size_type=SizeType.CONTRACTS, size_value=1,
                      stop_price=18900.0)
        bar = make_bar(open_=19000, high=19100.0, low=18850.0)
        pending = self._make_pending(order)
        result = self.engine.attempt_fill(pending, bar)
        assert result is not None
        assert result.fill_price == 18900.0

    # --- STOP_LIMIT fills ---

    def test_stop_limit_long_fills(self):
        order = Order(direction=1, order_type=OrderType.STOP_LIMIT,
                      size_type=SizeType.CONTRACTS, size_value=1,
                      stop_price=19050.0, limit_price=19100.0)
        # Stop triggered (high >= 19050), limit fills (low <= 19100)
        bar = make_bar(open_=19000, high=19150.0, low=18950.0)
        pending = self._make_pending(order)
        result = self.engine.attempt_fill(pending, bar)
        assert result is not None
        assert result.fill_price == 19100.0

    def test_stop_limit_no_fill_stop_not_triggered(self):
        order = Order(direction=1, order_type=OrderType.STOP_LIMIT,
                      size_type=SizeType.CONTRACTS, size_value=1,
                      stop_price=19200.0, limit_price=19250.0)
        bar = make_bar(high=19100.0)  # stop not triggered
        pending = self._make_pending(order)
        result = self.engine.attempt_fill(pending, bar)
        assert result is None

    # --- Market order ---

    def test_market_order_fills_at_close(self):
        order = Order(direction=1, order_type=OrderType.MARKET,
                      size_type=SizeType.CONTRACTS, size_value=1)
        bar = make_bar(close=19050.0)
        result = self.engine.fill_market_order(order, bar)
        assert result.fill_price == 19050.0


# ===========================================================================
# Same-bar fill + exit
# ===========================================================================

class TestSameBarExit:

    def setup_method(self):
        self.engine = make_engine()

    def test_same_bar_sl_hit_long(self):
        pos = make_long_position(entry=19000, sl=18950.0)
        bar = make_bar(open_=19000, high=19100, low=18900)
        result = self.engine.check_same_bar_exit(pos, bar, fill_price=19000.0)
        assert result is not None
        assert result.exit_reason == ExitReason.SAME_BAR_SL

    def test_same_bar_tp_hit_long(self):
        pos = make_long_position(entry=19000, tp=19080.0)
        bar = make_bar(open_=19000, high=19100, low=18990)
        result = self.engine.check_same_bar_exit(pos, bar, fill_price=19000.0)
        assert result is not None
        assert result.exit_reason == ExitReason.SAME_BAR_TP

    def test_same_bar_sl_wins_over_tp(self):
        # Both SL and TP in range — SL wins
        pos = make_long_position(entry=19000, sl=18950.0, tp=19080.0)
        bar = make_bar(open_=19000, high=19100, low=18900)
        result = self.engine.check_same_bar_exit(pos, bar, fill_price=19000.0)
        assert result.exit_reason == ExitReason.SAME_BAR_SL

    def test_same_bar_no_exit(self):
        pos = make_long_position(entry=19000, sl=18900.0, tp=19200.0)
        bar = make_bar(open_=19000, high=19100, low=18950)
        result = self.engine.check_same_bar_exit(pos, bar, fill_price=19000.0)
        assert result is None

    def test_same_bar_sl_short(self):
        pos = make_short_position(entry=19000, sl=19050.0)
        bar = make_bar(open_=19000, high=19100, low=18900)
        result = self.engine.check_same_bar_exit(pos, bar, fill_price=19000.0)
        assert result.exit_reason == ExitReason.SAME_BAR_SL

    def test_same_bar_tp_short(self):
        pos = make_short_position(entry=19000, tp=18900.0)
        bar = make_bar(open_=19000, high=19050, low=18850)
        result = self.engine.check_same_bar_exit(pos, bar, fill_price=19000.0)
        assert result is not None
        assert result.exit_reason == ExitReason.SAME_BAR_TP
        assert result.exit_price == 18900.0

    def test_same_bar_sl_beats_tp_short(self):
        # Both in range for short — SL wins
        pos = make_short_position(entry=19000, sl=19050.0, tp=18900.0)
        bar = make_bar(open_=19000, high=19100, low=18850)
        result = self.engine.check_same_bar_exit(pos, bar, fill_price=19000.0)
        assert result.exit_reason == ExitReason.SAME_BAR_SL


# ===========================================================================
# Normal bar exit checking
# ===========================================================================

class TestCheckExits:

    def setup_method(self):
        self.engine = make_engine()

    def test_sl_hit_long(self):
        # open below sl -> gap fill at SL; open above SL -> fill at SL
        pos = make_long_position(entry=19000, sl=18950.0)
        # Bar opens at 18940 (below SL=18950) -> gap protection: fill at open
        bar = make_bar(open_=18940, high=18970, low=18900)
        result = self.engine.check_exits(pos, bar, is_last_bar_of_session=False)
        assert result.exit_reason == ExitReason.SL
        assert result.exit_price == 18940.0  # gap: opened below SL, fill at open

    def test_tp_hit_long(self):
        pos = make_long_position(entry=19000, tp=19100.0)
        bar = make_bar(open_=19000, high=19150, low=18980)
        result = self.engine.check_exits(pos, bar, is_last_bar_of_session=False)
        assert result.exit_reason == ExitReason.TP
        assert result.exit_price == 19100.0

    def test_sl_beats_tp_same_bar_long(self):
        # Both SL and TP in range — SL always wins
        pos = make_long_position(entry=19000, sl=18950.0, tp=19100.0)
        bar = make_bar(open_=19000, high=19150, low=18900)
        result = self.engine.check_exits(pos, bar, is_last_bar_of_session=False)
        assert result.exit_reason == ExitReason.SL

    def test_eod_exit(self):
        pos = make_long_position(entry=19000)
        bar = make_bar(open_=19000, high=19050, low=18980, close=19020)
        result = self.engine.check_exits(pos, bar, is_last_bar_of_session=True)
        assert result.exit_reason == ExitReason.EOD
        assert result.exit_price == 19020.0

    def test_gap_open_below_sl_long(self):
        # Bar opens below SL -> fill at open (worse than SL)
        pos = make_long_position(entry=19000, sl=18950.0)
        bar = make_bar(open_=18900.0, high=18950, low=18880)  # gapped down through SL
        result = self.engine.check_exits(pos, bar, is_last_bar_of_session=False)
        assert result.exit_reason == ExitReason.SL
        assert result.exit_price == 18900.0  # gap fill at open

    def test_sl_hit_short(self):
        pos = make_short_position(entry=19000, sl=19050.0)
        bar = make_bar(open_=19000, high=19100, low=18950)
        result = self.engine.check_exits(pos, bar, is_last_bar_of_session=False)
        assert result.exit_reason == ExitReason.SL

    def test_tp_hit_short(self):
        pos = make_short_position(entry=19000, tp=18900.0)
        bar = make_bar(open_=19000, high=19050, low=18850)
        result = self.engine.check_exits(pos, bar, is_last_bar_of_session=False)
        assert result.exit_reason == ExitReason.TP

    def test_no_exit(self):
        pos = make_long_position(entry=19000, sl=18900.0, tp=19200.0)
        bar = make_bar(open_=19000, high=19100, low=18950)
        result = self.engine.check_exits(pos, bar, is_last_bar_of_session=False)
        assert result is None

    def test_trail_sl_exit_long(self):
        pos = make_long_position(entry=19000, trail=50.0)
        # First bar: price goes to 19100, trail moves to 19050
        update_trail(pos, bar_high=19100, bar_low=19000)
        assert pos.trail_sl_price == 19050

        # Second bar: price drops below trail SL
        bar = make_bar(open_=19060, high=19070, low=19000)
        result = self.engine.check_exits(pos, bar, is_last_bar_of_session=False)
        assert result.exit_reason == ExitReason.SL

    def test_gap_open_above_tp_long(self):
        # Bar opens above TP — TP exit at bar.open, not TP level
        pos = make_long_position(entry=19000, sl=18900.0, tp=19100.0)
        bar = make_bar(open_=19200.0, high=19250, low=19150)
        result = self.engine.check_exits(pos, bar, is_last_bar_of_session=False)
        assert result is not None
        assert result.exit_reason == ExitReason.TP
        assert result.exit_price == 19200.0  # fills at open, not TP level

    def test_gap_open_below_tp_short(self):
        # Bar opens below TP — TP exit at bar.open
        pos = make_short_position(entry=19000, sl=19100.0, tp=18900.0)
        bar = make_bar(open_=18800.0, high=18850, low=18750)
        result = self.engine.check_exits(pos, bar, is_last_bar_of_session=False)
        assert result is not None
        assert result.exit_reason == ExitReason.TP
        assert result.exit_price == 18800.0

    def test_gap_open_sl_beats_tp_long(self):
        # Bar gaps so far that it's simultaneously through both — SL wins
        pos = make_long_position(entry=19000, sl=18900.0, tp=18800.0)
        # This shouldn't happen in practice but the code guards it:
        # open <= sl AND open >= tp on a long → SL wins
        bar = make_bar(open_=18850.0, high=18900, low=18800)
        result = self.engine.check_exits(pos, bar, is_last_bar_of_session=False)
        assert result.exit_reason == ExitReason.SL

    def test_effective_sl_prefers_trail_over_fixed_long(self):
        # Trail SL (18950) is closer to current price than fixed SL (18900).
        # A bar whose low crosses trail SL but NOT fixed SL must trigger an exit,
        # proving effective_sl() returns the more protective (higher) value.
        pos = make_long_position(entry=19000, sl=18900.0)
        pos.trail_sl_price = 18950.0

        # Bar low = 18910: below trail SL (18950) but above fixed SL (18900)
        bar = make_bar(open_=18960, high=18970, low=18910)
        result = self.engine.check_exits(pos, bar, is_last_bar_of_session=False)
        assert result is not None, "Should exit — trail SL (18950) was crossed"
        assert result.exit_reason == ExitReason.SL
        assert result.exit_price == pytest.approx(18950.0)  # filled at trail SL

    def test_trail_sl_timing_uses_previous_bar_state(self):
        """
        Regression test for the trail SL timing fix.

        Before the fix, update_trail ran inside check_exits — the current bar's
        low would drag the trail SL down, and then the gap-open check would
        fire against the newly-lowered SL, producing a false SL exit on a bar
        that should have hit TP.

        After the fix, the runner calls tick_trail() AFTER check_exits returns
        None, so check_exits always uses the trail SL set at the END of the
        previous bar.

        Scenario (short):
          trail_sl at start of bar = 19772  (from previous bar)
          bar: open=19736.50, high=19742, low=19632, tp=19632.25
          Wrong (old) behaviour: trail moves to 19716 from bar.low=19632,
            gap-open 19736.50 >= 19716 → SL exit.
          Correct (new) behaviour: trail stays at 19772,
            gap-open 19736.50 >= 19772 → no,
            intrabar SL 19742 >= 19772 → no,
            TP 19632 <= 19632.25 → TP exit.
        """
        pos = make_short_position(entry=19842.50, sl=19940.0, tp=19632.25, trail=84.0)
        pos.trail_sl_price = 19772.0   # set by previous bar's tick_trail
        pos.trail_watermark = 19726.5  # lowest seen before this bar

        bar = make_bar(open_=19736.50, high=19742.0, low=19632.0, close=19640.0)
        result = self.engine.check_exits(pos, bar, is_last_bar_of_session=False)

        assert result is not None
        assert result.exit_reason == ExitReason.TP, (
            f"Expected TP exit, got {result.exit_reason} at {result.exit_price} — "
            "trail SL was likely updated from current bar's low before gap-open check ran"
        )
        assert result.exit_price == pytest.approx(19632.25)


# ===========================================================================
# PositionUpdate enforcement
# ===========================================================================

class TestPositionUpdateEnforcement:

    def setup_method(self):
        self.engine = make_engine()

    def test_sl_moves_favorably_long(self):
        pos = make_long_position(entry=19000, sl=18900.0)
        update = PositionUpdate(new_sl_price=18950.0)  # moving up -> favorable
        result = self.engine.apply_position_update(pos, update, current_price=19050.0)
        assert result is None  # no forced exit
        assert pos.sl_price == 18950.0  # updated

    def test_sl_unfavorable_move_ignored_long(self):
        pos = make_long_position(entry=19000, sl=18950.0)
        update = PositionUpdate(new_sl_price=18900.0)  # moving down -> unfavorable
        result = self.engine.apply_position_update(pos, update, current_price=19050.0)
        assert result is None
        assert pos.sl_price == 18950.0  # unchanged

    def test_sl_beyond_current_price_forces_exit_long(self):
        pos = make_long_position(entry=19000, sl=18900.0)
        update = PositionUpdate(new_sl_price=19100.0)  # above current price
        result = self.engine.apply_position_update(pos, update, current_price=19050.0)
        assert result is not None
        assert result.exit_reason == ExitReason.FORCED_EXIT
        assert result.exit_price == 19050.0

    def test_tp_update_long(self):
        pos = make_long_position(entry=19000, tp=19100.0)
        update = PositionUpdate(new_tp_price=19200.0)  # extending TP
        result = self.engine.apply_position_update(pos, update, current_price=19050.0)
        assert result is None
        assert pos.tp_price == 19200.0

    def test_tp_at_current_price_forces_exit_long(self):
        pos = make_long_position(entry=19000, tp=19200.0)
        update = PositionUpdate(new_tp_price=19040.0)  # below current price
        result = self.engine.apply_position_update(pos, update, current_price=19050.0)
        assert result is not None
        assert result.exit_reason == ExitReason.FORCED_EXIT

    def test_crossing_sl_tp_rejected(self):
        pos = make_long_position(entry=19000, sl=18900.0, tp=19200.0)
        # SL >= TP on long -> invalid
        update = PositionUpdate(new_sl_price=19300.0, new_tp_price=19100.0)
        result = self.engine.apply_position_update(pos, update, current_price=19050.0)
        # Both sl and tp cross -> entire update rejected
        assert result is None
        assert pos.sl_price == 18900.0  # unchanged
        assert pos.tp_price == 19200.0  # unchanged

    def test_short_sl_moves_favorably(self):
        pos = make_short_position(entry=19000, sl=19100.0)
        update = PositionUpdate(new_sl_price=19050.0)  # moving down -> favorable
        result = self.engine.apply_position_update(pos, update, current_price=18950.0)
        assert result is None
        assert pos.sl_price == 19050.0

    def test_short_sl_unfavorable_move_ignored(self):
        pos = make_short_position(entry=19000, sl=19100.0)
        update = PositionUpdate(new_sl_price=19150.0)  # moving up -> unfavorable
        result = self.engine.apply_position_update(pos, update, current_price=18950.0)
        assert result is None
        assert pos.sl_price == 19100.0  # unchanged

    def test_short_sl_beyond_current_price_forces_exit(self):
        pos = make_short_position(entry=19000, sl=19100.0)
        # new_sl below current price — moves SL into profit zone past current price
        update = PositionUpdate(new_sl_price=18900.0)
        result = self.engine.apply_position_update(pos, update, current_price=18950.0)
        assert result is not None
        assert result.exit_reason == ExitReason.FORCED_EXIT
        assert result.exit_price == 18950.0

    def test_tp_only_update_long(self):
        # PositionUpdate with only new_tp (no sl) is valid and should apply
        pos = make_long_position(entry=19000, sl=18900.0, tp=19100.0)
        update = PositionUpdate(new_tp_price=19300.0)
        result = self.engine.apply_position_update(pos, update, current_price=19050.0)
        assert result is None
        assert pos.tp_price == 19300.0
        assert pos.sl_price == 18900.0  # unchanged


# ===========================================================================
# Delta resolution
# ===========================================================================

class TestDeltaResolution:

    def setup_method(self):
        self.engine = make_engine()

    def _make_order(self, direction, contracts):
        return Order(direction=direction, order_type=OrderType.MARKET,
                     size_type=SizeType.CONTRACTS, size_value=contracts)

    def test_no_position(self):
        order = self._make_order(1, 3)
        close, open_ = self.engine.resolve_delta(order, 3, position=None)
        assert close == 0
        assert open_ == 3

    def test_same_direction_add(self):
        pos = make_long_position(contracts=2)
        order = self._make_order(1, 5)
        close, open_ = self.engine.resolve_delta(order, 5, pos)
        assert close == 0
        assert open_ == 3  # add 3 more to get to 5

    def test_same_direction_reduce(self):
        pos = make_long_position(contracts=5)
        order = self._make_order(1, 2)
        close, open_ = self.engine.resolve_delta(order, 2, pos)
        assert close == 3  # close 3 to get to 2
        assert open_ == 0

    def test_opposite_direction_full_flip(self):
        # Long 5, new signal is short 6
        # -> close all 5 longs, open net overshoot = 6-5 = 1 short
        # (spec: "close the 5 open long contracts and enter 1 contract short")
        pos = make_long_position(contracts=5)
        order = self._make_order(-1, 6)
        close, open_ = self.engine.resolve_delta(order, 6, pos)
        assert close == 5
        assert open_ == 1

    def test_opposite_direction_exact_close(self):
        # Long 3, short signal 3 -> exactly close, no new position
        pos = make_long_position(contracts=3)
        order = self._make_order(-1, 3)
        close, open_ = self.engine.resolve_delta(order, 3, pos)
        assert close == 3
        assert open_ == 0

    def test_no_change(self):
        pos = make_long_position(contracts=2)
        order = self._make_order(1, 2)
        close, open_ = self.engine.resolve_delta(order, 2, pos)
        assert close == 0
        assert open_ == 0


# ===========================================================================
# Trade construction
# ===========================================================================

class TestBuildTrade:

    def test_winning_long_trade(self):
        engine = make_engine(slippage=0.25, commission=4.50)
        pos = make_long_position(entry=19000.0, bar=10, contracts=2)
        trade = engine.build_trade(pos, exit_bar=20, exit_price=19050.0,
                                   exit_reason=ExitReason.TP)
        assert trade.pnl_points == 50.0
        assert trade.pnl_dollars == 50.0 * 2 * POINT_VALUE  # 2000
        assert trade.is_winner
        assert trade.bars_held == 10

    def test_losing_short_trade(self):
        engine = make_engine()
        pos = make_short_position(entry=19000.0, bar=5, contracts=1)
        trade = engine.build_trade(pos, exit_bar=10, exit_price=19050.0,
                                   exit_reason=ExitReason.SL)
        assert trade.pnl_points == -50.0
        assert not trade.is_winner
        assert trade.exit_reason == ExitReason.SL

    def test_costs_computed(self):
        engine = make_engine(slippage=0.25, commission=4.50)
        pos = make_long_position(entry=19000.0, contracts=2)
        trade = engine.build_trade(pos, exit_bar=1, exit_price=19050.0,
                                   exit_reason=ExitReason.TP)
        # Slippage: 0.25 pts * 2 sides * 2 contracts * $20 = $20
        assert trade.slippage_paid == pytest.approx(20.0)
        # Commission: $4.50 * 2 sides * 2 contracts = $18
        assert trade.commission_paid == pytest.approx(18.0)


# ===========================================================================
# Expiry
# ===========================================================================

class TestExpiry:

    def test_gtc_never_expires(self):
        engine = make_engine()
        order = Order(direction=1, order_type=OrderType.LIMIT,
                      size_type=SizeType.CONTRACTS, size_value=1,
                      limit_price=18900.0)
        pending = PendingOrder(order=order, registered_bar=0, bars_remaining=None)
        for _ in range(100):
            assert not engine.tick_expiry(pending)

    def test_expiry_counts_down(self):
        engine = make_engine()
        order = Order(direction=1, order_type=OrderType.LIMIT,
                      size_type=SizeType.CONTRACTS, size_value=1,
                      limit_price=18900.0, expiry_bars=3)
        pending = PendingOrder(order=order, registered_bar=0, bars_remaining=3)
        assert not engine.tick_expiry(pending)  # 2 remaining
        assert not engine.tick_expiry(pending)  # 1 remaining
        assert engine.tick_expiry(pending)      # 0 remaining -> expired


if __name__ == "__main__":
    pytest.main([__file__, "-v"])