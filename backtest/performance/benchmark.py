"""
Benchmark — two buy-and-hold variants for fair strategy comparison.

B&H Fixed (1 contract)
──────────────────────
Hold exactly 1 NQ contract from first bar open to last bar close.
Never add or reduce. Pays same entry slippage + commission as strategy.
equity_t = capital + (close_t - entry_price) * POINT_VALUE

This is the apples-to-apples dollar comparison: same position size as the
strategy (which also trades 1 contract), different in that the strategy
exits and re-enters while B&H never does.

B&H Compounding
───────────────
Simulates holding floor(equity / (price × POINT_VALUE)) contracts at the
start of each trading day, adding contracts as equity grows.  This is the
truly fair comparison against a strategy that also compounds capital —
both start at $100k, both scale exposure proportionally.

equity_t = simulated bar-by-bar with daily contract rebalancing.

Costs
─────
Both variants pay the same per-contract entry slippage and commission as
the strategy (passed from RunConfig).  B&H Compounding pays per rebalance
(once/day when contract count changes).  B&H Fixed pays once at entry.
"""
from __future__ import annotations
import numpy as np

from backtest.data.market_data import MarketData
from backtest.performance.results import BenchmarkResult

POINT_VALUE = 20.0


def compute_benchmark(
    data: MarketData,
    starting_capital: float,
    slippage_points: float = 0.0,
    commission_per_contract: float = 0.0,
) -> BenchmarkResult:
    """
    Compute both B&H benchmarks and return the richer BenchmarkResult.

    The primary equity_curve (used in tearsheet overlay and Sharpe) is
    B&H Fixed — same position size as strategy, most honest dollar comparison.
    B&H Compounding equity curve is stored separately for the comparison table.
    """
    opens  = data.open_1m
    closes = data.close_1m
    n_bars = len(closes)

    if n_bars == 0:
        empty = np.array([starting_capital])
        return BenchmarkResult(
            total_pnl_dollars=0.0, cagr=0.0, sharpe=0.0,
            max_dd_pct=0.0, equity_curve=empty,
            total_pnl_dollars_compounding=0.0,
            cagr_compounding=0.0,
            sharpe_compounding=0.0,
            max_dd_pct_compounding=0.0,
            equity_curve_compounding=empty,
        )

    n_days = len(data.trading_dates)
    years  = n_days / 252.0

    # ── Shared: entry fill ────────────────────────────────────────────────
    raw_entry   = float(opens[0])
    entry_price = raw_entry + slippage_points   # long: pay ask (add slippage)
    entry_cost  = commission_per_contract       # one-side commission

    # ── B&H Fixed (1 contract) ───────────────────────────────────────────
    capital_fixed = starting_capital - entry_cost
    floating_pnl  = (closes - entry_price) * POINT_VALUE   # (n_bars,)
    eq_fixed_bars = capital_fixed + floating_pnl

    eq_fixed = np.empty(n_bars + 1)
    eq_fixed[0]  = starting_capital
    eq_fixed[1:] = eq_fixed_bars

    pnl_fixed   = float(eq_fixed[-1] - starting_capital)
    cagr_fixed  = _cagr(eq_fixed[-1], starting_capital, years)
    sharpe_fixed = _daily_sharpe(eq_fixed, data)
    mdd_fixed   = _max_dd(eq_fixed)

    # ── B&H Compounding ──────────────────────────────────────────────────
    # Vectorised: build day boundaries once, loop only over days (~3760),
    # fill bar ranges with numpy slices instead of per-bar Python loops.
    timestamps = data.df_1m.index
    date_codes = timestamps.normalize().asi8          # int64 per bar
    change      = np.concatenate([[True], date_codes[1:] != date_codes[:-1]])
    first_bars  = np.where(change)[0]                 # first bar of each day
    last_bars   = np.concatenate([first_bars[1:] - 1, [n_bars - 1]])

    eq_compound      = np.empty(n_bars + 1)
    eq_compound[0]   = starting_capital
    balance          = starting_capital
    contracts        = 0
    avg_cost         = 0.0

    for first_bar, last_bar in zip(first_bars, last_bars):
        day_open = float(opens[first_bar])
        target   = max(0, int(balance / (day_open * POINT_VALUE)))

        if target > contracts:
            added    = target - contracts
            fill     = day_open + slippage_points
            balance -= added * commission_per_contract
            avg_cost = ((avg_cost * contracts + fill * added) / target
                        if target > 0 else fill)
            contracts = target

        # Vectorised fill: one numpy op instead of per-bar loop
        unrealized = (closes[first_bar:last_bar + 1] - avg_cost) * contracts * POINT_VALUE
        eq_compound[first_bar + 1:last_bar + 2] = balance + unrealized

    pnl_compound   = float(eq_compound[-1] - starting_capital)
    cagr_compound  = _cagr(eq_compound[-1], starting_capital, years)
    sharpe_compound = _daily_sharpe(eq_compound, data)
    mdd_compound   = _max_dd(eq_compound)

    return BenchmarkResult(
        # Fixed (primary)
        total_pnl_dollars = pnl_fixed,
        cagr              = cagr_fixed,
        sharpe            = sharpe_fixed,
        max_dd_pct        = mdd_fixed,
        equity_curve      = eq_fixed,
        # Compounding
        total_pnl_dollars_compounding = pnl_compound,
        cagr_compounding              = cagr_compound,
        sharpe_compounding            = sharpe_compound,
        max_dd_pct_compounding        = mdd_compound,
        equity_curve_compounding      = eq_compound,
    )


# ── Helpers ───────────────────────────────────────────────────────────────

def _cagr(final: float, start: float, years: float) -> float:
    if years <= 0 or start <= 0:
        return 0.0
    if final <= 0:
        return -1.0
    return float((final / start) ** (1.0 / years) - 1.0)


def _max_dd(equity: np.ndarray) -> float:
    peak   = np.maximum.accumulate(equity)
    dd_pct = (equity - peak) / np.where(peak == 0, 1.0, peak)
    return float(dd_pct.min())


def _daily_sharpe(equity: np.ndarray, data: MarketData) -> float:
    """Sharpe from daily dollar P&L — vectorised."""
    ts         = data.df_1m.index
    date_codes = ts.normalize().asi8
    change     = np.concatenate([[True], date_codes[1:] != date_codes[:-1]])
    first_bars = np.where(change)[0]
    last_bars  = np.concatenate([first_bars[1:] - 1, [len(ts) - 1]])

    daily_eq      = equity[last_bars + 1]
    eq_with_start = np.concatenate([[equity[0]], daily_eq])
    dollar_pnl    = np.diff(eq_with_start)

    if len(dollar_pnl) < 2 or dollar_pnl.std() == 0:
        return 0.0
    return float((dollar_pnl.mean() / dollar_pnl.std()) * np.sqrt(252))