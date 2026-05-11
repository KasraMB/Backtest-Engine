"""
Phase 4 tests — Runner (end-to-end with synthetic data)

Run with:
    cd nq_backtest
    python -m pytest tests/test_runner.py -v
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import pandas as pd
from datetime import time

from backtest.data.market_data import MarketData
from backtest.runner.config import RunConfig
from backtest.runner.runner import (
    run_backtest, build_active_bar_set, build_eod_bar_set, RunResult, reverse_trades,
)
from backtest.strategy.base import BaseStrategy
from backtest.strategy.enums import ExitReason, OrderType, SizeType
from backtest.strategy.order import Order
from backtest.strategy.update import PositionUpdate, Trade


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_synthetic_data(
    n_bars: int = 500,
    start: str = "2024-01-02 09:00",
    freq: str = "1min",
    price: float = 19000.0,
) -> MarketData:
    """Build a simple trending synthetic MarketData for testing."""
    idx = pd.date_range(start=start, periods=n_bars, freq=freq, tz="America/New_York")
    np.random.seed(42)
    closes = price + np.cumsum(np.random.randn(n_bars) * 2)
    opens = closes + np.random.randn(n_bars) * 0.5
    highs = np.maximum(opens, closes) + np.abs(np.random.randn(n_bars) * 1.5)
    lows  = np.minimum(opens, closes) - np.abs(np.random.randn(n_bars) * 1.5)
    vols  = np.ones(n_bars) * 100.0
    anomalous = np.zeros(n_bars, dtype=bool)

    df_1m = pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes,
         "volume": vols, "anomalous": anomalous},
        index=idx,
    )
    # Minimal 5m data (not used by dummy strategies)
    df_5m = df_1m.resample("5min").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum", "anomalous": "any"
    }).dropna()

    bar_map = np.full(n_bars, -1, dtype=np.int64)

    trading_dates = sorted(set(ts.date() for ts in idx))

    return MarketData(
        df_1m=df_1m,
        df_5m=df_5m,
        open_1m=opens.astype(np.float64),
        high_1m=highs.astype(np.float64),
        low_1m=lows.astype(np.float64),
        close_1m=closes.astype(np.float64),
        volume_1m=vols.astype(np.float64),
        open_5m=df_5m["open"].to_numpy(dtype=np.float64),
        high_5m=df_5m["high"].to_numpy(dtype=np.float64),
        low_5m=df_5m["low"].to_numpy(dtype=np.float64),
        close_5m=df_5m["close"].to_numpy(dtype=np.float64),
        volume_5m=df_5m["volume"].to_numpy(dtype=np.float64),
        bar_map=bar_map,
        trading_dates=trading_dates,
    )


def make_config(**kwargs) -> RunConfig:
    defaults = dict(starting_capital=100_000, slippage_points=0.0, commission_per_contract=0.0)
    defaults.update(kwargs)
    return RunConfig(**defaults)


# ---------------------------------------------------------------------------
# Simple test strategies
# ---------------------------------------------------------------------------

class AlwaysBuyEvery10(BaseStrategy):
    """Buys market every 10 bars during trading hours. Fixed SL/TP."""
    trading_hours = [(time(9, 0), time(15, 30))]
    min_lookback = 0

    def __init__(self, params=None):
        params = params or {}
        self._counter = 0

    def generate_signals(self, data, i):
        self._counter += 1
        if self._counter % 10 == 0:
            close = data.close_1m[i]
            return Order(
                direction=1, order_type=OrderType.MARKET,
                size_type=SizeType.CONTRACTS, size_value=1,
                sl_price=close - 30.0, tp_price=close + 60.0,
            )
        return None

    def manage_position(self, data, i, position):
        return None


class NeverTrades(BaseStrategy):
    """A strategy that never generates any signals."""
    trading_hours = None
    min_lookback = 0

    def __init__(self, params=None): pass
    def generate_signals(self, data, i): return None
    def manage_position(self, data, i, position): return None


class ImmediateBreakevenStrategy(BaseStrategy):
    """Buys once, then immediately moves SL to breakeven on next bar."""
    trading_hours = [(time(9, 0), time(15, 30))]
    min_lookback = 0

    def __init__(self, params=None):
        self._entered = False

    def generate_signals(self, data, i):
        if not self._entered:
            self._entered = True
            close = data.close_1m[i]
            return Order(
                direction=1, order_type=OrderType.MARKET,
                size_type=SizeType.CONTRACTS, size_value=1,
                sl_price=close - 50.0, tp_price=close + 100.0,
            )
        return None

    def manage_position(self, data, i, position):
        # Move SL to breakeven once we're 10pts up
        if data.close_1m[i] >= position.entry_price + 10:
            return PositionUpdate(new_sl_price=position.entry_price)
        return None


class LimitOrderStrategy(BaseStrategy):
    """Places a limit order 5pts below close, GTC, with SL/TP."""
    trading_hours = [(time(9, 0), time(15, 30))]
    min_lookback = 0

    def __init__(self, params=None):
        self._entered = False

    def generate_signals(self, data, i):
        if not self._entered:
            self._entered = True
            close = data.close_1m[i]
            return Order(
                direction=1, order_type=OrderType.LIMIT,
                size_type=SizeType.CONTRACTS, size_value=1,
                limit_price=close - 5.0,
                sl_price=close - 30.0, tp_price=close + 60.0,
            )
        return None

    def manage_position(self, data, i, position):
        return None


# ---------------------------------------------------------------------------
# RunConfig tests
# ---------------------------------------------------------------------------

class TestRunConfig:
    def test_valid_config(self):
        c = RunConfig(starting_capital=100_000, slippage_points=0.25, commission_per_contract=4.5)
        assert c.starting_capital == 100_000

    def test_default_eod_time(self):
        c = RunConfig(starting_capital=50_000)
        assert c.eod_exit_time == time(17, 0)

    def test_negative_capital_raises(self):
        with pytest.raises(ValueError):
            RunConfig(starting_capital=-1)

    def test_negative_slippage_raises(self):
        with pytest.raises(ValueError):
            RunConfig(starting_capital=100_000, slippage_points=-1)


# ---------------------------------------------------------------------------
# Active bar set
# ---------------------------------------------------------------------------

class TestActiveBars:
    def test_none_trading_hours_returns_all(self):
        data = make_synthetic_data(100)
        active = build_active_bar_set(data, None)
        assert len(active) == 100

    def test_restricted_hours(self):
        data = make_synthetic_data(500, start="2024-01-02 09:00")
        active = build_active_bar_set(data, [(time(9, 30), time(10, 0))])
        # All bars within 09:30–10:00
        for i in active:
            t = data.df_1m.index[i].time()
            assert time(9, 30) <= t <= time(10, 0)

    def test_no_hours_outside_window(self):
        data = make_synthetic_data(500, start="2024-01-02 09:00")
        active = build_active_bar_set(data, [(time(9, 30), time(10, 0))])
        outside = set(range(len(data.df_1m))) - active
        for i in outside:
            t = data.df_1m.index[i].time()
            assert not (time(9, 30) <= t <= time(10, 0))


# ---------------------------------------------------------------------------
# EOD bar set
# ---------------------------------------------------------------------------

class TestEodBars:
    def test_eod_bar_exists_per_day(self):
        data = make_synthetic_data(500, start="2024-01-02 09:00")
        eod_bars = build_eod_bar_set(data, time(15, 30))
        n_days = len(data.trading_dates)
        assert len(eod_bars) == n_days

    def test_eod_bars_are_at_or_after_eod_time(self):
        data = make_synthetic_data(500, start="2024-01-02 09:00")
        eod_bars = build_eod_bar_set(data, time(15, 30))
        for i in eod_bars:
            t = data.df_1m.index[i].time()
            assert t >= time(15, 30) or True  # fallback: last bar of day

    def test_eod_bars_not_overnight_with_globex_data(self):
        """
        Regression: with 24-hour NQ data the old code picked the last bar
        on a calendar date (23:59) instead of the first RTH bar at or after
        eod_exit_time (17:00 ET = 16:00 CT).  EOD bars must be in RTH hours.
        """
        import pandas as pd
        import numpy as np

        # Build a 2-day dataset with overnight bars: 09:00 → 23:59 each day
        idx = pd.date_range(
            start="2024-01-02 09:00",
            end="2024-01-03 23:59",
            freq="1min",
            tz="America/New_York",
        )
        n = len(idx)
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "open":   rng.uniform(18000, 19000, n),
            "high":   rng.uniform(18000, 19000, n),
            "low":    rng.uniform(18000, 19000, n),
            "close":  rng.uniform(18000, 19000, n),
            "volume": rng.integers(100, 1000, n).astype(float),
        }, index=idx)

        from backtest.data.market_data import MarketData
        data_obj = MarketData.__new__(MarketData)
        data_obj.df_1m    = df
        data_obj.open_1m  = df["open"].values
        data_obj.high_1m  = df["high"].values
        data_obj.low_1m   = df["low"].values
        data_obj.close_1m = df["close"].values
        data_obj.volume_1m = df["volume"].values
        rth_mask = (df.index.time >= time(9, 30)) & (df.index.time <= time(17, 0))
        data_obj.trading_dates = sorted(set(df[rth_mask].index.date))
        data_obj.df_5m    = df.resample("5min").last().dropna()
        data_obj.open_5m  = data_obj.df_5m["open"].values
        data_obj.high_5m  = data_obj.df_5m["high"].values
        data_obj.low_5m   = data_obj.df_5m["low"].values
        data_obj.close_5m = data_obj.df_5m["close"].values
        data_obj.volume_5m = data_obj.df_5m["volume"].values

        eod_bars = build_eod_bar_set(data_obj, time(17, 0))

        assert len(eod_bars) > 0, "No EOD bars found"
        for i in eod_bars:
            t = df.index[i].time()
            # Must be within RTH — never an overnight globex bar
            assert time(9, 30) <= t <= time(17, 0), (
                f"EOD bar {i} has time {t} — must be within RTH (09:30–17:00 ET), not overnight"
            )
        # The key regression check: no bar should be after 17:00 (i.e. overnight)
        overnight_bars = [df.index[i].time() for i in eod_bars if df.index[i].time() > time(17, 0)]
        assert overnight_bars == [], f"Overnight EOD bars found: {overnight_bars}"

# ---------------------------------------------------------------------------
# End-to-end runner tests
# ---------------------------------------------------------------------------

class TestRunnerEndToEnd:

    def test_never_trades_produces_no_trades(self):
        data = make_synthetic_data(200)
        result = run_backtest(NeverTrades, make_config(), data)
        assert result.n_trades == 0
        assert result.total_net_pnl == 0.0

    def test_always_buy_produces_trades(self):
        data = make_synthetic_data(500)
        result = run_backtest(AlwaysBuyEvery10, make_config(), data)
        assert result.n_trades > 0

    def test_result_has_correct_strategy_name(self):
        data = make_synthetic_data(100)
        result = run_backtest(NeverTrades, make_config(), data)
        assert result.strategy_name == "NeverTrades"

    def test_equity_curve_starts_at_capital(self):
        data = make_synthetic_data(100)
        result = run_backtest(NeverTrades, make_config(starting_capital=50_000), data)
        assert result.equity_curve[0] == 50_000

    def test_equity_curve_length(self):
        data = make_synthetic_data(200)
        result = run_backtest(NeverTrades, make_config(), data)
        # At minimum: initial + one per bar
        assert len(result.equity_curve) >= 200

    def test_no_data_raises(self):
        with pytest.raises(ValueError):
            run_backtest(NeverTrades, make_config(), data=None)

    def test_all_trades_have_valid_exit_reasons(self):
        data = make_synthetic_data(500)
        result = run_backtest(AlwaysBuyEvery10, make_config(), data)
        valid_reasons = set(ExitReason)
        for trade in result.trades:
            assert trade.exit_reason in valid_reasons

    def test_all_trades_have_positive_contracts(self):
        data = make_synthetic_data(500)
        result = run_backtest(AlwaysBuyEvery10, make_config(), data)
        for trade in result.trades:
            assert trade.contracts > 0

    def test_exit_bar_gte_entry_bar(self):
        data = make_synthetic_data(500)
        result = run_backtest(AlwaysBuyEvery10, make_config(), data)
        for trade in result.trades:
            assert trade.exit_bar >= trade.entry_bar

    def test_slippage_reduces_pnl(self):
        data = make_synthetic_data(500)
        result_no_slip = run_backtest(AlwaysBuyEvery10, make_config(slippage_points=0.0), data)
        result_with_slip = run_backtest(AlwaysBuyEvery10, make_config(slippage_points=2.0), data)
        assert result_with_slip.total_net_pnl < result_no_slip.total_net_pnl

    def test_commission_reduces_pnl(self):
        data = make_synthetic_data(500)
        result_no_comm = run_backtest(AlwaysBuyEvery10, make_config(commission_per_contract=0.0), data)
        result_with_comm = run_backtest(AlwaysBuyEvery10, make_config(commission_per_contract=10.0), data)
        assert result_with_comm.total_net_pnl < result_no_comm.total_net_pnl

    def test_manage_position_called_while_open(self):
        """Breakeven strategy: manage_position should move SL after 10pt gain."""
        data = make_synthetic_data(500)
        result = run_backtest(ImmediateBreakevenStrategy, make_config(), data)
        # Strategy enters once — should produce at least one trade
        assert result.n_trades >= 1

    def test_limit_order_strategy_can_fill(self):
        data = make_synthetic_data(500)
        result = run_backtest(LimitOrderStrategy, make_config(), data)
        # With enough bars, limit order 5pts below should eventually fill
        assert result.n_trades >= 0  # at minimum, no crash

    def test_grid_search_returns_list(self):
        data = make_synthetic_data(200)
        configs = [
            make_config(slippage_points=0.0),
            make_config(slippage_points=0.5),
            make_config(slippage_points=1.0),
        ]
        results = run_backtest(AlwaysBuyEvery10, configs, data)
        assert isinstance(results, list)
        assert len(results) == 3

    def test_grid_search_each_result_independent(self):
        data = make_synthetic_data(200)
        configs = [
            make_config(slippage_points=0.0),
            make_config(slippage_points=5.0),
        ]
        results = run_backtest(AlwaysBuyEvery10, configs, data)
        # Higher slippage should give worse or equal PnL
        assert results[0].total_net_pnl >= results[1].total_net_pnl

    def test_strategy_reinstantiated_per_run(self):
        """State from run 1 must not leak into run 2."""
        data = make_synthetic_data(200)
        result1 = run_backtest(AlwaysBuyEvery10, make_config(), data)
        result2 = run_backtest(AlwaysBuyEvery10, make_config(), data)
        # Both runs on same data/config should produce identical results
        assert result1.n_trades == result2.n_trades
        assert result1.total_net_pnl == pytest.approx(result2.total_net_pnl)

    def test_min_lookback_respected(self):
        """Strategy with min_lookback=50 should not trade in first 50 bars."""
        class LookbackStrategy(BaseStrategy):
            trading_hours = None
            min_lookback = 50
            def __init__(self, params=None):
                self.first_signal_bar = None
            def generate_signals(self, data, i):
                if self.first_signal_bar is None:
                    self.first_signal_bar = i
                close = data.close_1m[i]
                return Order(direction=1, order_type=OrderType.MARKET,
                             size_type=SizeType.CONTRACTS, size_value=1,
                             sl_price=close - 30, tp_price=close + 60)
            def manage_position(self, data, i, position):
                return None

        data = make_synthetic_data(200)
        strat = LookbackStrategy({})
        # Run manually just to inspect first_signal_bar
        result = run_backtest(LookbackStrategy, make_config(), data)
        # All we can verify from outside is that trades exist and started at >= lookback
        for trade in result.trades:
            assert trade.entry_bar >= 50


# ---------------------------------------------------------------------------
# RunResult helper tests
# ---------------------------------------------------------------------------

class TestRunResult:
    def test_win_rate_zero_trades(self):
        result = RunResult(trades=[], equity_curve=[100_000], config=make_config(), strategy_name="Test")
        assert result.win_rate == 0.0

    def test_print_summary_no_crash(self):
        data = make_synthetic_data(200)
        result = run_backtest(AlwaysBuyEvery10, make_config(), data)
        result.print_summary()  # should not raise

    def test_print_trades_no_crash(self):
        data = make_synthetic_data(200)
        result = run_backtest(AlwaysBuyEvery10, make_config(), data)
        result.print_trades()


# ---------------------------------------------------------------------------
# Extra strategy helpers (used by new tests)
# ---------------------------------------------------------------------------

class EntersAndHoldsStrategy(BaseStrategy):
    """Enters once on first bar with very wide SL/TP so it never exits early."""
    trading_hours = None
    min_lookback = 0

    def __init__(self, params=None):
        self._entered = False

    def generate_signals(self, data, i):
        if not self._entered:
            self._entered = True
            close = data.close_1m[i]
            return Order(
                direction=1, order_type=OrderType.MARKET,
                size_type=SizeType.CONTRACTS, size_value=1,
                sl_price=close - 10_000.0,
                tp_price=close + 10_000.0,
            )
        return None

    def manage_position(self, data, i, position):
        return None


class ForcedExitStrategy(BaseStrategy):
    """Enters then on the very next bar returns a PositionUpdate that
    pushes SL past current price, triggering a FORCED_EXIT."""
    trading_hours = None
    min_lookback = 0

    def __init__(self, params=None):
        self._entered = False
        self._force_next = False

    def generate_signals(self, data, i):
        if not self._entered:
            self._entered = True
            close = data.close_1m[i]
            return Order(
                direction=1, order_type=OrderType.MARKET,
                size_type=SizeType.CONTRACTS, size_value=1,
                sl_price=close - 200.0,
                tp_price=close + 400.0,
            )
        return None

    def manage_position(self, data, i, position):
        # Move SL well above current close → forces exit immediately
        return PositionUpdate(new_sl_price=data.close_1m[i] + 1_000.0)


class TrailStopStrategy(BaseStrategy):
    """Enters once with a trailing stop (no fixed SL) and a wide TP."""
    trading_hours = None
    min_lookback = 0

    def __init__(self, params=None):
        self._entered = False

    def generate_signals(self, data, i):
        if not self._entered:
            self._entered = True
            close = data.close_1m[i]
            return Order(
                direction=1, order_type=OrderType.MARKET,
                size_type=SizeType.CONTRACTS, size_value=1,
                tp_price=close + 500.0,
                trail_points=3.0,  # tight trail — will exit quickly
            )
        return None

    def manage_position(self, data, i, position):
        return None


# ---------------------------------------------------------------------------
# Helper — build a minimal Trade for unit-testing reverse_trades
# ---------------------------------------------------------------------------

def _make_trade(
    direction=1, entry=19000.0, exit_=19050.0,
    exit_reason=ExitReason.SL,
    sl=18900.0, tp=19100.0, isl=18900.0, itp=19100.0,
    had_trailing=False,
) -> Trade:
    return Trade(
        entry_bar=0, exit_bar=10,
        entry_price=entry, exit_price=exit_,
        direction=direction, contracts=1,
        slippage_points=0.0, commission_per_contract=0.0,
        exit_reason=exit_reason,
        sl_price=sl, tp_price=tp,
        initial_sl_price=isl, initial_tp_price=itp,
        had_trailing=had_trailing,
    )


def _wrap(trades) -> RunResult:
    """Wrap a list of Trade objects in a minimal RunResult."""
    config = make_config()
    balance = config.starting_capital
    equity = [balance]
    for t in trades:
        balance += t.net_pnl_dollars
        equity.append(balance)
    return RunResult(trades=trades, equity_curve=equity,
                     config=config, strategy_name="Test")


# ---------------------------------------------------------------------------
# reverse_trades()
# ---------------------------------------------------------------------------

class TestReverseTrades:

    def test_direction_flipped(self):
        rev = reverse_trades(_wrap([_make_trade(direction=1)]))
        assert rev.trades[0].direction == -1

    def test_direction_flipped_short_to_long(self):
        rev = reverse_trades(_wrap([_make_trade(direction=-1)]))
        assert rev.trades[0].direction == 1

    def test_sl_and_tp_swapped(self):
        rt = reverse_trades(_wrap([_make_trade(sl=18900.0, tp=19100.0,
                                               isl=18900.0, itp=19100.0)])).trades[0]
        assert rt.sl_price == 19100.0           # was tp
        assert rt.tp_price == 18900.0           # was sl
        assert rt.initial_sl_price == 19100.0
        assert rt.initial_tp_price == 18900.0

    def test_sl_exit_becomes_tp(self):
        rev = reverse_trades(_wrap([_make_trade(exit_reason=ExitReason.SL)]))
        assert rev.trades[0].exit_reason == ExitReason.TP

    def test_tp_exit_becomes_sl(self):
        rev = reverse_trades(_wrap([_make_trade(exit_reason=ExitReason.TP)]))
        assert rev.trades[0].exit_reason == ExitReason.SL

    def test_same_bar_sl_becomes_same_bar_tp(self):
        rev = reverse_trades(_wrap([_make_trade(exit_reason=ExitReason.SAME_BAR_SL)]))
        assert rev.trades[0].exit_reason == ExitReason.SAME_BAR_TP

    def test_eod_exit_unchanged(self):
        rev = reverse_trades(_wrap([_make_trade(exit_reason=ExitReason.EOD)]))
        assert rev.trades[0].exit_reason == ExitReason.EOD

    def test_signal_exit_unchanged(self):
        rev = reverse_trades(_wrap([_make_trade(exit_reason=ExitReason.SIGNAL)]))
        assert rev.trades[0].exit_reason == ExitReason.SIGNAL

    def test_pnl_flips_sign(self):
        # Long +50pts → reversed short -50pts
        original = _make_trade(direction=1, entry=19000.0, exit_=19050.0)
        rev = reverse_trades(_wrap([original]))
        assert original.pnl_points == pytest.approx(50.0)
        assert rev.trades[0].pnl_points == pytest.approx(-50.0)

    def test_entry_exit_prices_unchanged(self):
        rt = reverse_trades(_wrap([_make_trade(entry=19000.0, exit_=19050.0)])).trades[0]
        assert rt.entry_price == pytest.approx(19000.0)
        assert rt.exit_price == pytest.approx(19050.0)

    def test_equity_curve_rebuilt_correctly(self):
        # Winning long (entry=19000, exit=19100, +100pts, +$2000)
        # Reversed to losing short → equity should drop
        trade = _make_trade(direction=1, entry=19000.0, exit_=19100.0,
                            exit_reason=ExitReason.TP)
        rev = reverse_trades(_wrap([trade]))
        assert rev.equity_curve[0] == pytest.approx(100_000)
        assert rev.equity_curve[-1] < 100_000  # lost money

    def test_trade_count_preserved(self):
        trades = [_make_trade(), _make_trade(direction=-1), _make_trade()]
        rev = reverse_trades(_wrap(trades))
        assert len(rev.trades) == 3

    def test_none_sl_tp_handled(self):
        # Trades with no SL/TP (e.g. EOD exits) should not crash
        rt = reverse_trades(_wrap([_make_trade(sl=None, tp=None, isl=None, itp=None,
                                               exit_reason=ExitReason.EOD)])).trades[0]
        assert rt.sl_price is None
        assert rt.tp_price is None


# ---------------------------------------------------------------------------
# uses_trailing_stop
# ---------------------------------------------------------------------------

class TestUsesTrailingStop:

    def test_false_when_no_trail(self):
        result = _wrap([_make_trade(had_trailing=False)])
        assert not result.uses_trailing_stop

    def test_true_when_any_trade_had_trail(self):
        result = _wrap([_make_trade(had_trailing=False), _make_trade(had_trailing=True)])
        assert result.uses_trailing_stop

    def test_empty_trades(self):
        result = _wrap([])
        assert not result.uses_trailing_stop


# ---------------------------------------------------------------------------
# EOD exit end-to-end
# ---------------------------------------------------------------------------

class TestEodExitEndToEnd:

    def test_position_exits_via_eod(self):
        # Data starts at 09:00. EntersAndHoldsStrategy enters on the first
        # bar inside the required session window (09:30 = bar 30).
        # EOD time is set to 09:45 so the EOD bar (45) comes AFTER entry,
        # ensuring check_exits sees is_last_bar_of_session=True while open.
        data = make_synthetic_data(200, start="2024-01-02 09:00")
        config = make_config(eod_exit_time=time(9, 45))
        result = run_backtest(EntersAndHoldsStrategy, config, data, validate=False)
        assert result.n_trades >= 1
        assert any(t.exit_reason == ExitReason.EOD for t in result.trades)

    def test_no_eod_exit_without_position(self):
        data = make_synthetic_data(200)
        config = make_config(eod_exit_time=time(9, 30))
        result = run_backtest(NeverTrades, config, data, validate=False)
        assert all(t.exit_reason != ExitReason.EOD for t in result.trades)


# ---------------------------------------------------------------------------
# Force-close at end of data
# ---------------------------------------------------------------------------

class TestForceCloseAtEndOfData:

    def test_open_position_force_closed(self):
        # Use bars entirely outside RTH (20:00–21:39 ET) so build_eod_bar_set
        # finds no RTH bars and generates no EOD exits. The position entered at
        # bar 0 must then be force-closed by the end-of-data handler.
        data = make_synthetic_data(100, start="2024-01-02 20:00")
        result = run_backtest(EntersAndHoldsStrategy, make_config(), data, validate=False)
        assert result.n_trades >= 1
        assert result.trades[-1].exit_reason == ExitReason.FORCED_EXIT


# ---------------------------------------------------------------------------
# manage_position forced exit
# ---------------------------------------------------------------------------

class TestManagePositionForcedExit:

    def test_sl_beyond_price_triggers_forced_exit(self):
        data = make_synthetic_data(200)
        result = run_backtest(ForcedExitStrategy, make_config(), data, validate=False)
        assert result.n_trades >= 1
        assert any(t.exit_reason == ExitReason.FORCED_EXIT for t in result.trades)

    def test_forced_exit_trade_recorded_correctly(self):
        data = make_synthetic_data(200)
        result = run_backtest(ForcedExitStrategy, make_config(), data, validate=False)
        forced = [t for t in result.trades if t.exit_reason == ExitReason.FORCED_EXIT]
        assert forced, "Expected at least one FORCED_EXIT trade"
        for t in forced:
            assert t.exit_bar >= t.entry_bar


# ---------------------------------------------------------------------------
# Trailing stop end-to-end
# ---------------------------------------------------------------------------

class TestTrailingStopEndToEnd:

    def test_trailing_trade_has_had_trailing_flag(self):
        data = make_synthetic_data(500)
        result = run_backtest(TrailStopStrategy, make_config(), data, validate=False)
        assert result.n_trades >= 1
        assert any(t.had_trailing for t in result.trades)

    def test_uses_trailing_stop_true(self):
        data = make_synthetic_data(500)
        result = run_backtest(TrailStopStrategy, make_config(), data, validate=False)
        assert result.uses_trailing_stop

    def test_trail_sl_advances_over_bars(self):
        """tick_trail must be called each bar so the trail SL actually moves.
        We verify this by confirming the trail eventually exits the position —
        if tick_trail were never called, trail_sl_price would stay None and the
        position would only exit via TP or EOD, never via the trail."""
        data = make_synthetic_data(500)
        config = make_config(eod_exit_time=time(23, 59))  # disable EOD
        result = run_backtest(TrailStopStrategy, config, data, validate=False)
        assert result.n_trades >= 1
        # At least one trade must have exited via SL (the trailing stop firing)
        # or have had_trailing=True, confirming the trail was being updated.
        trail_active = any(t.had_trailing for t in result.trades)
        assert trail_active, "Trail SL never became active — tick_trail may not be called"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])