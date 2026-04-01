"""
Results — Phase 6
─────────────────
Dataclass holding all computed performance metrics for a single backtest run.
Produced by PerformanceEngine.compute(). Consumed by TearsheetRenderer.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class DrawdownStats:
    max_dd_dollars: float
    max_dd_pct: float
    avg_dd_pct: float
    max_dd_duration_bars: int
    max_dd_duration_days: float


@dataclass
class MonteCarloResults:
    """Results from both MC simulation methods."""
    # Percentile equity curves — shape (n_sims, n_trades+1)
    shuffle_percentiles: dict[int, np.ndarray]      # {5: curve, 25: curve, ...}
    bootstrap_percentiles: dict[int, np.ndarray]

    # Final equity distribution
    shuffle_final_equity: np.ndarray                # shape (n_sims,)
    bootstrap_final_equity: np.ndarray

    # Max drawdown distribution
    shuffle_max_dd_pct: np.ndarray
    bootstrap_max_dd_pct: np.ndarray

    # Percentile stats on final equity
    shuffle_p5: float
    shuffle_p50: float
    shuffle_p95: float
    bootstrap_p5: float
    bootstrap_p50: float
    bootstrap_p95: float


@dataclass
class ConfidenceIntervals:
    """95% bootstrap confidence intervals on key metrics."""
    sharpe:         tuple[float, float]
    win_rate:       tuple[float, float]
    expectancy:     tuple[float, float]
    profit_factor:  tuple[float, float]
    max_dd_pct:     tuple[float, float]


@dataclass
class ExitBreakdown:
    reason: str
    count: int
    win_rate: float
    avg_pnl_dollars: float
    total_pnl_dollars: float


@dataclass
class HourlyBreakdown:
    hour: int
    count: int
    avg_pnl_dollars: float
    win_rate: float
    total_pnl_dollars: float


@dataclass
class TradeRow:
    """One row in the trade log table — all display-ready strings plus raw values for sorting."""
    trade_num:    int
    entry_time:   str      # "2024-01-02 09:31"
    exit_time:    str
    direction:    str      # "LONG" / "SHORT"
    contracts:    float
    entry_price:  float
    exit_price:   float
    pnl_points:   float
    gross_pnl:    float
    net_pnl:      float
    exit_reason:  str
    duration_bars: int


@dataclass
class BenchmarkResult:
    """
    Buy-and-hold benchmark — two variants.

    Fixed (1 contract, entire period):
        Same position size as the strategy. Honest dollar comparison.
        Pays entry slippage + commission once. Never exits (no exit costs).

    Compounding (scales contracts daily as equity grows):
        Truly apples-to-apples against a compounding strategy.
        Starts at the same capital, buys as many contracts as equity allows
        each session open, adds contracts when equity grows enough.
        Pays slippage + commission on each newly purchased contract.
    """
    # Fixed variant (primary — used for equity chart overlay)
    total_pnl_dollars: float
    cagr: float
    sharpe: float
    max_dd_pct: float
    equity_curve: np.ndarray

    # Compounding variant
    total_pnl_dollars_compounding: float
    cagr_compounding: float
    sharpe_compounding: float
    max_dd_pct_compounding: float
    equity_curve_compounding: np.ndarray


@dataclass
class Results:
    # ── Identity ────────────────────────────────────────────────────────────
    strategy_name: str
    starting_capital: float
    n_trading_days: int

    # ── PnL ─────────────────────────────────────────────────────────────────
    total_net_pnl: float
    cagr: float
    final_equity: float
    total_commission: float        # total commission paid across all trades ($)
    total_slippage: float          # total slippage paid across all trades ($)

    # ── Trade stats ─────────────────────────────────────────────────────────
    n_trades: int
    win_rate: float
    avg_win_dollars: float
    avg_loss_dollars: float
    avg_trade_pnl_dollars: float        # expectancy in dollars
    payoff_ratio: float                 # avg_win / abs(avg_loss)
    profit_factor: float
    expectancy_r: float                 # avg PnL in R (multiples of avg loss)
    largest_win: float
    largest_loss: float
    longest_win_streak: int
    longest_loss_streak: int
    avg_trade_duration_bars: float

    # ── Risk-adjusted ────────────────────────────────────────────────────────
    sharpe: float
    sortino: float
    calmar: float

    # ── Drawdown ─────────────────────────────────────────────────────────────
    drawdown: DrawdownStats

    # ── MAE / MFE ────────────────────────────────────────────────────────────
    avg_mae_pct: float          # avg max adverse excursion as % of entry
    avg_mfe_pct: float          # avg max favorable excursion as % of entry
    mae_per_trade: np.ndarray   # shape (n_trades,)
    mfe_per_trade: np.ndarray   # shape (n_trades,)

    # ── Curves ───────────────────────────────────────────────────────────────
    equity_curve: np.ndarray
    drawdown_curve_pct: np.ndarray
    equity_timestamps: object          # pd.DatetimeIndex — same length as equity_curve

    # ── Breakdowns ───────────────────────────────────────────────────────────
    exit_breakdown: list[ExitBreakdown]
    hourly_breakdown: list[HourlyBreakdown]

    # ── Distributions ────────────────────────────────────────────────────────
    trade_durations_bars: np.ndarray
    trade_pnls: np.ndarray

    # ── Monte Carlo ──────────────────────────────────────────────────────────
    monte_carlo: MonteCarloResults

    # ── Bootstrap stats ──────────────────────────────────────────────────────
    bootstrap_pvalue: float             # H0: E[PnL] <= 0
    confidence_intervals: ConfidenceIntervals

    # ── Benchmark ────────────────────────────────────────────────────────────
    benchmark: BenchmarkResult

    # ── Trade log ────────────────────────────────────────────────────────────
    trade_log: list  # list[TradeRow]