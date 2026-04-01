"""
Phase 2 unit tests — Core Objects
Run with: python -m pytest tests/test_core_objects.py -v
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from datetime import time

from backtest.strategy.enums import OrderType, SizeType, ExitReason
from backtest.strategy.order import Order
from backtest.strategy.update import PositionUpdate, OpenPosition, RunConfig, Trade, POINT_VALUE


# ===========================================================================
# Order — valid construction
# ===========================================================================

class TestOrderValidConstruction:

    def test_market_long_contracts(self):
        o = Order(direction=1, order_type=OrderType.MARKET,
                  size_type=SizeType.CONTRACTS, size_value=2)
        assert o.direction == 1
        assert o.is_long()
        assert not o.is_short()

    def test_market_short_contracts(self):
        o = Order(direction=-1, order_type=OrderType.MARKET,
                  size_type=SizeType.CONTRACTS, size_value=1)
        assert o.is_short()

    def test_limit_long_with_sl(self):
        o = Order(direction=1, order_type=OrderType.LIMIT,
                  size_type=SizeType.CONTRACTS, size_value=1,
                  limit_price=19000.0, sl_price=18900.0)
        assert o.limit_price == 19000.0
        assert o.sl_price == 18900.0

    def test_stop_short(self):
        o = Order(direction=-1, order_type=OrderType.STOP,
                  size_type=SizeType.CONTRACTS, size_value=1,
                  stop_price=19500.0, sl_price=19600.0)
        assert o.stop_price == 19500.0

    def test_stop_limit(self):
        o = Order(direction=1, order_type=OrderType.STOP_LIMIT,
                  size_type=SizeType.CONTRACTS, size_value=1,
                  stop_price=19000.0, limit_price=19010.0)
        assert o.stop_price == 19000.0
        assert o.limit_price == 19010.0

    def test_dollars_sizing_with_sl(self):
        o = Order(direction=1, order_type=OrderType.MARKET,
                  size_type=SizeType.DOLLARS, size_value=1000.0,
                  sl_price=18900.0)
        assert o.size_type == SizeType.DOLLARS

    def test_pct_risk_sizing_with_trail(self):
        o = Order(direction=1, order_type=OrderType.MARKET,
                  size_type=SizeType.PCT_RISK, size_value=0.01,
                  trail_points=50.0)
        assert o.size_value == 0.01

    def test_trail_with_activation(self):
        o = Order(direction=1, order_type=OrderType.MARKET,
                  size_type=SizeType.CONTRACTS, size_value=1,
                  trail_points=40.0, trail_activation_points=20.0)
        assert o.trail_activation_points == 20.0

    def test_gtc_order(self):
        o = Order(direction=1, order_type=OrderType.LIMIT,
                  size_type=SizeType.CONTRACTS, size_value=1,
                  limit_price=19000.0)
        assert o.expiry_bars is None

    def test_expiry_bars_set(self):
        o = Order(direction=1, order_type=OrderType.LIMIT,
                  size_type=SizeType.CONTRACTS, size_value=1,
                  limit_price=19000.0, expiry_bars=5)
        assert o.expiry_bars == 5


# ===========================================================================
# Order — invalid construction
# ===========================================================================

class TestOrderInvalidConstruction:

    def test_invalid_direction(self):
        with pytest.raises(ValueError, match="direction must be"):
            Order(direction=0, order_type=OrderType.MARKET,
                  size_type=SizeType.CONTRACTS, size_value=1)

    def test_sl_and_trail_mutually_exclusive(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            Order(direction=1, order_type=OrderType.MARKET,
                  size_type=SizeType.CONTRACTS, size_value=1,
                  sl_price=18900.0, trail_points=50.0)

    def test_dollars_sizing_requires_sl_or_trail(self):
        with pytest.raises(ValueError, match="requires either sl_price or trail_points"):
            Order(direction=1, order_type=OrderType.MARKET,
                  size_type=SizeType.DOLLARS, size_value=500.0)

    def test_pct_risk_sizing_requires_sl_or_trail(self):
        with pytest.raises(ValueError, match="requires either sl_price or trail_points"):
            Order(direction=-1, order_type=OrderType.MARKET,
                  size_type=SizeType.PCT_RISK, size_value=0.02)

    def test_limit_order_missing_limit_price(self):
        with pytest.raises(ValueError, match="requires limit_price"):
            Order(direction=1, order_type=OrderType.LIMIT,
                  size_type=SizeType.CONTRACTS, size_value=1)

    def test_stop_order_missing_stop_price(self):
        with pytest.raises(ValueError, match="requires stop_price"):
            Order(direction=1, order_type=OrderType.STOP,
                  size_type=SizeType.CONTRACTS, size_value=1)

    def test_stop_limit_missing_both(self):
        with pytest.raises(ValueError):
            Order(direction=1, order_type=OrderType.STOP_LIMIT,
                  size_type=SizeType.CONTRACTS, size_value=1)

    def test_negative_trail_points(self):
        with pytest.raises(ValueError, match="trail_points must be positive"):
            Order(direction=1, order_type=OrderType.MARKET,
                  size_type=SizeType.CONTRACTS, size_value=1,
                  trail_points=-10.0)

    def test_trail_activation_without_trail_points(self):
        with pytest.raises(ValueError, match="trail_activation_points requires trail_points"):
            Order(direction=1, order_type=OrderType.MARKET,
                  size_type=SizeType.CONTRACTS, size_value=1,
                  trail_activation_points=20.0)

    def test_pct_risk_value_over_100_percent(self):
        with pytest.raises(ValueError, match="PCT_RISK size_value must be in"):
            Order(direction=1, order_type=OrderType.MARKET,
                  size_type=SizeType.PCT_RISK, size_value=1.5,
                  sl_price=18900.0)

    def test_pct_risk_value_as_percentage_not_decimal(self):
        with pytest.raises(ValueError, match="PCT_RISK size_value must be in"):
            Order(direction=1, order_type=OrderType.MARKET,
                  size_type=SizeType.PCT_RISK, size_value=2.0,
                  sl_price=18900.0)

    def test_contracts_fractional(self):
        with pytest.raises(ValueError, match="CONTRACTS size_value must be a positive integer"):
            Order(direction=1, order_type=OrderType.MARKET,
                  size_type=SizeType.CONTRACTS, size_value=1.5)

    def test_zero_size_value(self):
        with pytest.raises(ValueError, match="size_value must be positive"):
            Order(direction=1, order_type=OrderType.MARKET,
                  size_type=SizeType.CONTRACTS, size_value=0)


# ===========================================================================
# PositionUpdate
# ===========================================================================

class TestPositionUpdate:

    def test_sl_only(self):
        pu = PositionUpdate(new_sl_price=18900.0)
        assert pu.new_sl_price == 18900.0
        assert pu.new_tp_price is None

    def test_tp_only(self):
        pu = PositionUpdate(new_tp_price=19500.0)
        assert pu.new_tp_price == 19500.0

    def test_both(self):
        pu = PositionUpdate(new_sl_price=18900.0, new_tp_price=19500.0)
        assert pu.new_sl_price == 18900.0
        assert pu.new_tp_price == 19500.0

    def test_neither_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            PositionUpdate()


# ===========================================================================
# OpenPosition — effective_sl
# ===========================================================================

class TestOpenPositionEffectiveSl:

    def test_fixed_sl_only_long(self):
        pos = OpenPosition(direction=1, entry_price=19000, entry_bar=0,
                           contracts=1, sl_price=18900.0)
        assert pos.effective_sl() == 18900.0

    def test_trail_sl_only_long(self):
        pos = OpenPosition(direction=1, entry_price=19000, entry_bar=0,
                           contracts=1, trail_sl_price=18950.0)
        assert pos.effective_sl() == 18950.0

    def test_trail_wins_over_fixed_long(self):
        pos = OpenPosition(direction=1, entry_price=19000, entry_bar=0,
                           contracts=1, sl_price=18900.0, trail_sl_price=18970.0)
        assert pos.effective_sl() == 18970.0

    def test_fixed_wins_over_trail_long(self):
        pos = OpenPosition(direction=1, entry_price=19000, entry_bar=0,
                           contracts=1, sl_price=18960.0, trail_sl_price=18920.0)
        assert pos.effective_sl() == 18960.0

    def test_no_sl_returns_none(self):
        pos = OpenPosition(direction=1, entry_price=19000, entry_bar=0, contracts=1)
        assert pos.effective_sl() is None

    def test_effective_sl_short(self):
        pos = OpenPosition(direction=-1, entry_price=19000, entry_bar=0,
                           contracts=1, sl_price=19100.0, trail_sl_price=19050.0)
        assert pos.effective_sl() == 19050.0


# ===========================================================================
# RunConfig
# ===========================================================================

class TestRunConfig:

    def test_valid_config(self):
        cfg = RunConfig(starting_capital=100_000, slippage_points=0.25,
                        commission_per_contract=4.50)
        assert cfg.eod_exit_time == time(17, 0)
        assert cfg.params == {}

    def test_custom_eod_time(self):
        cfg = RunConfig(starting_capital=50_000, slippage_points=0.5,
                        commission_per_contract=2.0, eod_exit_time=time(15, 0))
        assert cfg.eod_exit_time == time(15, 0)

    def test_params_passed_through(self):
        cfg = RunConfig(starting_capital=100_000, slippage_points=0.25,
                        commission_per_contract=4.50,
                        params={"atr_mult": 2.0, "lookback": 20})
        assert cfg.params["atr_mult"] == 2.0

    def test_negative_capital_raises(self):
        with pytest.raises(ValueError, match="starting_capital"):
            RunConfig(starting_capital=-1000, slippage_points=0.25,
                      commission_per_contract=4.50)

    def test_negative_slippage_raises(self):
        with pytest.raises(ValueError, match="slippage_points"):
            RunConfig(starting_capital=100_000, slippage_points=-0.5,
                      commission_per_contract=4.50)

    def test_negative_commission_raises(self):
        with pytest.raises(ValueError, match="commission_per_contract"):
            RunConfig(starting_capital=100_000, slippage_points=0.25,
                      commission_per_contract=-1.0)


# ===========================================================================
# Trade — computed properties
# ===========================================================================

class TestTrade:

    def _make_trade(self, direction=1, entry=19000.0, exit_=19050.0,
                    contracts=2.0, slippage_points=0.25, commission_per_contract=4.50):
        return Trade(
            entry_bar=10, exit_bar=20,
            entry_price=entry, exit_price=exit_,
            direction=direction, contracts=contracts,
            slippage_points=slippage_points,
            commission_per_contract=commission_per_contract,
            exit_reason=ExitReason.TP,
        )

    def test_pnl_points_long(self):
        t = self._make_trade(direction=1, entry=19000, exit_=19050)
        assert t.pnl_points == 50.0

    def test_pnl_points_short_winner(self):
        t = self._make_trade(direction=-1, entry=19000, exit_=18950)
        assert t.pnl_points == 50.0

    def test_pnl_points_short_loser(self):
        t = self._make_trade(direction=-1, entry=19000, exit_=19050)
        assert t.pnl_points == -50.0

    def test_pnl_dollars_long(self):
        # 50 pts * 2 contracts * $20 = $2000
        t = self._make_trade(direction=1, entry=19000, exit_=19050, contracts=2)
        assert t.pnl_dollars == 2000.0

    def test_slippage_paid(self):
        # 0.25 pts * 2 sides * 2 contracts * $20 = $20
        t = self._make_trade(contracts=2, slippage_points=0.25)
        assert t.slippage_paid == 20.0

    def test_commission_paid(self):
        # $4.50 * 2 sides * 2 contracts = $18
        t = self._make_trade(contracts=2, commission_per_contract=4.50)
        assert t.commission_paid == 18.0

    def test_net_pnl_dollars(self):
        # gross $2000 - slippage $20 - commission $18 = $1962
        t = self._make_trade(direction=1, entry=19000, exit_=19050,
                             contracts=2, slippage_points=0.25,
                             commission_per_contract=4.50)
        assert t.net_pnl_dollars == 2000.0 - 20.0 - 18.0

    def test_is_winner_true(self):
        t = self._make_trade(direction=1, entry=19000, exit_=19050)
        assert t.is_winner

    def test_is_winner_false(self):
        t = self._make_trade(direction=1, entry=19000, exit_=18950)
        assert not t.is_winner

    def test_bars_held(self):
        t = self._make_trade()
        assert t.bars_held == 10

    def test_exit_reason_stored(self):
        t = self._make_trade()
        assert t.exit_reason == ExitReason.TP


# ===========================================================================
# Enums
# ===========================================================================

class TestEnums:

    def test_all_order_types_exist(self):
        assert OrderType.MARKET
        assert OrderType.LIMIT
        assert OrderType.STOP
        assert OrderType.STOP_LIMIT

    def test_all_size_types_exist(self):
        assert SizeType.CONTRACTS
        assert SizeType.DOLLARS
        assert SizeType.PCT_RISK

    def test_all_exit_reasons_exist(self):
        reasons = {ExitReason.SL, ExitReason.TP, ExitReason.EOD,
                   ExitReason.SIGNAL, ExitReason.SAME_BAR_SL,
                   ExitReason.SAME_BAR_TP, ExitReason.FORCED_EXIT}
        assert len(reasons) == 7


if __name__ == "__main__":
    pytest.main([__file__, "-v"])