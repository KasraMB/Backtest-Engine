"""
Phase 6 tests — Performance Engine

Run with:
    python -m pytest tests/test_performance.py -v

What to expect:
    - 35 tests covering all metric computations
    - No real data required — uses synthetic trades
    - All 35 should pass
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import numpy as np
import pandas as pd
from datetime import time
from typing import Optional

from backtest.performance.engine import PerformanceEngine
from backtest.performance.results import Results
from backtest.performance.tearsheet import TearsheetRenderer
from backtest.data.market_data import MarketData
from backtest.runner.config import RunConfig
from backtest.strategy.enums import ExitReason
from backtest.strategy.update import Trade, POINT_VALUE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_trade_counter = 0

def make_trade(
    pnl_dollars: float,
    entry_bar: int = 0,
    exit_bar: int = None,
    direction: int = 1,
    contracts: int = 1,
    exit_reason: ExitReason = ExitReason.TP,
    entry_price: float = 19000.0,
    slippage_points: float = 0.0,
    commission_per_contract: float = 0.0,
) -> Trade:
    pnl_points = pnl_dollars / (contracts * POINT_VALUE)
    global _trade_counter
    if exit_bar is None:
        # Spread trades across different bars/days so daily aggregation has variance
        exit_bar = entry_bar + 10 + (_trade_counter * 50)
        _trade_counter += 1
    exit_price = entry_price + pnl_points * direction
    return Trade(
        entry_bar=entry_bar,
        exit_bar=exit_bar,
        entry_price=entry_price,
        exit_price=exit_price,
        direction=direction,
        contracts=contracts,
        exit_reason=exit_reason,
        slippage_points=slippage_points,
        commission_per_contract=commission_per_contract,
    )


def make_data(n_bars: int = 500) -> MarketData:
    # Use enough bars to span multiple days (390 bars/session)
    n_bars = max(n_bars, 390 * 5)
    np.random.seed(7)
    price = 19000.0
    closes = price + np.cumsum(np.random.randn(n_bars) * 2)
    opens  = closes + np.random.randn(n_bars) * 0.3
    highs  = np.maximum(opens, closes) + np.abs(np.random.randn(n_bars) * 1.5)
    lows   = np.minimum(opens, closes) - np.abs(np.random.randn(n_bars) * 1.5)
    vols   = np.ones(n_bars) * 100.0

    idx = pd.date_range("2024-01-02 09:30", periods=n_bars, freq="1min", tz="America/New_York")
    df_1m = pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes,
         "volume": vols, "anomalous": np.zeros(n_bars, dtype=bool)},
        index=idx,
    )
    df_5m = df_1m.resample("5min").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum", "anomalous": "any",
    }).dropna()
    bar_map = np.full(n_bars, -1, dtype=np.int64)
    trading_dates = sorted(set(ts.date() for ts in idx))
    return MarketData(
        df_1m=df_1m, df_5m=df_5m,
        open_1m=opens.astype(np.float64), high_1m=highs.astype(np.float64),
        low_1m=lows.astype(np.float64), close_1m=closes.astype(np.float64),
        volume_1m=vols.astype(np.float64),
        open_5m=df_5m["open"].to_numpy(np.float64), high_5m=df_5m["high"].to_numpy(np.float64),
        low_5m=df_5m["low"].to_numpy(np.float64), close_5m=df_5m["close"].to_numpy(np.float64),
        volume_5m=df_5m["volume"].to_numpy(np.float64),
        bar_map=bar_map, trading_dates=trading_dates,
    )


class _FakeResult:
    def __init__(self, trades, capital=100_000):
        self.trades = trades
        self.equity_curve = self._build_equity(trades, capital)
        self.config = RunConfig(starting_capital=capital)
        self.strategy_name = "TestStrategy"

    def _build_equity(self, trades, capital):
        """
        Build a bar-length equity curve so _daily_equity_returns can resample it.
        make_data() generates 1950 bars (5 sessions of 390).
        We scatter each trade's PnL to its exit_bar and hold it flat otherwise.
        equity[0] = starting capital; equity[i+1] = account balance after bar i.
        """
        n_bars = 390 * 5   # must match make_data()
        eq = np.full(n_bars + 1, float(capital))
        bal = float(capital)
        # Map exit_bar -> cumulative balance at that point
        trade_by_exit = {}
        for t in trades:
            bal += t.net_pnl_dollars
            bar = min(t.exit_bar, n_bars - 1)
            trade_by_exit[bar] = bal
        # Fill forward: equity[i+1] = balance after bar i
        cur = float(capital)
        for i in range(n_bars):
            if i in trade_by_exit:
                cur = trade_by_exit[i]
            eq[i + 1] = cur
        return eq.tolist()


def make_results(trades, capital=100_000) -> Results:
    data = make_data()
    result = _FakeResult(trades, capital)
    return PerformanceEngine().compute(result, data, n_mc_sims=100)


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

class TestCoreMetrics:

    def test_win_rate(self):
        trades = [make_trade(100), make_trade(100), make_trade(-50)]
        r = make_results(trades)
        assert abs(r.win_rate - 2/3) < 1e-9

    def test_profit_factor(self):
        trades = [make_trade(200), make_trade(200), make_trade(-100)]
        r = make_results(trades)
        assert abs(r.profit_factor - 4.0) < 1e-6

    def test_profit_factor_no_losses(self):
        trades = [make_trade(100), make_trade(200)]
        r = make_results(trades)
        assert r.profit_factor == float("inf") or r.profit_factor > 1000

    def test_total_net_pnl(self):
        trades = [make_trade(300), make_trade(-100), make_trade(200)]
        r = make_results(trades)
        assert abs(r.total_net_pnl - 400.0) < 1e-6

    def test_avg_win_avg_loss(self):
        trades = [make_trade(200), make_trade(100), make_trade(-50), make_trade(-150)]
        r = make_results(trades)
        assert abs(r.avg_win_dollars - 150.0) < 1e-6
        assert abs(r.avg_loss_dollars - (-100.0)) < 1e-6

    def test_largest_win_loss(self):
        trades = [make_trade(500), make_trade(100), make_trade(-300), make_trade(-50)]
        r = make_results(trades)
        assert abs(r.largest_win - 500.0) < 1e-6
        assert abs(r.largest_loss - (-300.0)) < 1e-6

    def test_n_trades(self):
        trades = [make_trade(100)] * 7
        r = make_results(trades)
        assert r.n_trades == 7

    def test_zero_trades_returns_results(self):
        r = make_results([])
        assert r.n_trades == 0
        assert r.total_net_pnl == 0.0


# ---------------------------------------------------------------------------
# Drawdown
# ---------------------------------------------------------------------------

class TestDrawdown:

    def test_no_drawdown_on_all_winners(self):
        trades = [make_trade(100)] * 10
        r = make_results(trades)
        assert r.drawdown.max_dd_pct >= -1e-9  # essentially 0

    def test_max_drawdown_detected(self):
        # Lose 1000 then recover
        trades = [make_trade(-500), make_trade(-500), make_trade(2000)]
        r = make_results(trades)
        assert r.drawdown.max_dd_pct < 0.0

    def test_drawdown_curve_length_matches_equity(self):
        trades = [make_trade(100), make_trade(-50), make_trade(200)]
        r = make_results(trades)
        assert len(r.drawdown_curve_pct) == len(r.equity_curve)

    def test_drawdown_curve_nonpositive(self):
        trades = [make_trade(100), make_trade(-50), make_trade(200)]
        r = make_results(trades)
        assert (r.drawdown_curve_pct <= 1e-9).all()


# ---------------------------------------------------------------------------
# Risk-adjusted
# ---------------------------------------------------------------------------

class TestRiskAdjusted:

    def test_sharpe_positive_on_consistent_winners(self):
        # Vary PnL slightly so std dev > 0
        np.random.seed(1)
        trades = [make_trade(100 + np.random.randn() * 10) for _ in range(50)]
        r = make_results(trades)
        assert r.sharpe > 0

    def test_sharpe_negative_on_consistent_losers(self):
        np.random.seed(2)
        trades = [make_trade(-100 + np.random.randn() * 10) for _ in range(50)]
        r = make_results(trades)
        assert r.sharpe < 0

    def test_sortino_defined(self):
        trades = [make_trade(100), make_trade(-50)] * 20
        r = make_results(trades)
        assert isinstance(r.sortino, float)

    def test_calmar_positive_when_profitable(self):
        trades = [make_trade(100)] * 20 + [make_trade(-50)] * 5
        r = make_results(trades)
        assert r.calmar > 0 or r.calmar == float("inf")


# ---------------------------------------------------------------------------
# Streaks
# ---------------------------------------------------------------------------

class TestStreaks:

    def test_win_streak(self):
        trades = [make_trade(100)] * 5 + [make_trade(-50)] + [make_trade(100)] * 3
        r = make_results(trades)
        assert r.longest_win_streak == 5

    def test_loss_streak(self):
        trades = [make_trade(-50)] * 4 + [make_trade(100)] + [make_trade(-50)] * 2
        r = make_results(trades)
        assert r.longest_loss_streak == 4


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------

class TestMonteCarlo:

    def test_mc_has_all_percentiles(self):
        trades = [make_trade(100), make_trade(-50)] * 20
        r = make_results(trades)
        for p in [5, 25, 50, 75, 95]:
            assert p in r.monte_carlo.shuffle_percentiles
            assert p in r.monte_carlo.bootstrap_percentiles

    def test_mc_p95_gte_p5_shuffle(self):
        trades = [make_trade(100), make_trade(-50)] * 20
        r = make_results(trades)
        assert r.monte_carlo.shuffle_p95 >= r.monte_carlo.shuffle_p5

    def test_mc_p95_gte_p5_bootstrap(self):
        trades = [make_trade(100), make_trade(-50)] * 20
        r = make_results(trades)
        assert r.monte_carlo.bootstrap_p95 >= r.monte_carlo.bootstrap_p5

    def test_mc_final_equity_shape(self):
        trades = [make_trade(100), make_trade(-50)] * 10
        r = make_results(trades)
        assert len(r.monte_carlo.shuffle_final_equity) == 100  # n_mc_sims=100
        assert len(r.monte_carlo.bootstrap_final_equity) == 100


# ---------------------------------------------------------------------------
# Bootstrap p-value
# ---------------------------------------------------------------------------

class TestBootstrapPValue:

    def test_pvalue_low_on_strong_edge(self):
        trades = [make_trade(200)] * 50  # always wins
        r = make_results(trades)
        assert r.bootstrap_pvalue < 0.05

    def test_pvalue_high_on_losers(self):
        trades = [make_trade(-100)] * 50  # always loses
        r = make_results(trades)
        assert r.bootstrap_pvalue > 0.95

    def test_pvalue_between_0_and_1(self):
        trades = [make_trade(100), make_trade(-80)] * 20
        r = make_results(trades)
        assert 0.0 <= r.bootstrap_pvalue <= 1.0


# ---------------------------------------------------------------------------
# Confidence intervals
# ---------------------------------------------------------------------------

class TestConfidenceIntervals:

    def test_ci_sharpe_ordered(self):
        trades = [make_trade(100), make_trade(-50)] * 20
        r = make_results(trades)
        lo, hi = r.confidence_intervals.sharpe
        assert lo <= hi

    def test_ci_win_rate_in_0_1(self):
        trades = [make_trade(100), make_trade(-50)] * 20
        r = make_results(trades)
        lo, hi = r.confidence_intervals.win_rate
        assert 0.0 <= lo <= hi <= 1.0

    def test_ci_profit_factor_ordered(self):
        trades = [make_trade(200), make_trade(-50)] * 10
        r = make_results(trades)
        lo, hi = r.confidence_intervals.profit_factor
        assert lo <= hi


# ---------------------------------------------------------------------------
# Exit breakdown
# ---------------------------------------------------------------------------

class TestExitBreakdown:

    def test_exit_breakdown_counts_correct(self):
        trades = (
            [make_trade(100, exit_reason=ExitReason.TP)] * 3 +
            [make_trade(-50, exit_reason=ExitReason.SL)] * 2
        )
        r = make_results(trades)
        by_reason = {eb.reason: eb for eb in r.exit_breakdown}
        assert by_reason["TP"].count == 3
        assert by_reason["SL"].count == 2

    def test_exit_breakdown_win_rates(self):
        trades = (
            [make_trade(100, exit_reason=ExitReason.TP)] * 4 +
            [make_trade(-50, exit_reason=ExitReason.SL)] * 4
        )
        r = make_results(trades)
        by_reason = {eb.reason: eb for eb in r.exit_breakdown}
        assert by_reason["TP"].win_rate == 1.0
        assert by_reason["SL"].win_rate == 0.0


# ---------------------------------------------------------------------------
# Tearsheet rendering
# ---------------------------------------------------------------------------

class TestTearsheet:

    def test_render_produces_file(self, tmp_path):
        trades = [make_trade(100), make_trade(-50)] * 10
        r = make_results(trades)
        path = str(tmp_path / "test_tearsheet.html")
        TearsheetRenderer().render(r, output_path=path, auto_open=False)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 10_000  # sanity: not empty

    def test_render_contains_plotly(self, tmp_path):
        trades = [make_trade(100), make_trade(-50)] * 10
        r = make_results(trades)
        path = str(tmp_path / "test_tearsheet2.html")
        TearsheetRenderer().render(r, output_path=path, auto_open=False)
        with open(path, encoding="utf-8") as f:
            content = f.read()
        assert "plotly" in content.lower()
        assert r.strategy_name in content

    def test_render_zero_trades_no_crash(self, tmp_path):
        r = make_results([])
        path = str(tmp_path / "empty_tearsheet.html")
        TearsheetRenderer().render(r, output_path=path, auto_open=False)
        assert os.path.exists(path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])