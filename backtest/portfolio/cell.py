"""Per-cell (strategy × instrument) backtest + inclusion filter.

A 'cell' is one (strategy_class, instrument_symbol, params) triple. Each cell
is backtested ONCE with realistic execution and the resulting trades+stats are
the unit of portfolio composition.
"""
from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Optional, Type

import numpy as np
import pandas as pd

from backtest.data.market_data import MarketData
from backtest.portfolio.instruments import spec as instrument_spec
from backtest.propfirm.lucidflex import extract_normalised_trades
from backtest.runner.config import RunConfig
from backtest.runner.runner import run_backtest
from backtest.strategy.base import BaseStrategy

# Realistic execution defaults (locked by portfolio pre-reg)
DEFAULT_SLIPPAGE_PT     = 0.5
DEFAULT_COMMISSION_RT_2 = 3.50    # per side; runner doubles for RT
DEFAULT_ORDER_CANCEL    = datetime.time(23, 59)
DEFAULT_FILL_MODE       = 'next_open'


@dataclass
class Cell:
    """One strategy × instrument backtest spec."""
    strategy_class: Type[BaseStrategy]
    instrument:     str                       # 'NQ' | 'ES' | 'YM'
    params:         dict = field(default_factory=dict)
    name:           Optional[str] = None      # human label, default = ClassName_INST

    def __post_init__(self):
        if self.name is None:
            self.name = f"{self.strategy_class.__name__}_{self.instrument}"


@dataclass
class CellResult:
    """Backtested cell — per-trade data + summary stats."""
    cell:      Cell
    trades:    pd.DataFrame           # cols: date, pnl_pts, sl_dist, R, MAE_R, mae_pts
    n_trades:  int
    mean_R:    float
    wr:        float
    sum_R:     float
    by_year:   dict                   # {year: {'n': N, 'mean_R': float, 'sumR': float}}
    proxy_sl_fraction: float = 0.0    # fraction of trades with NO real stop (R is the
                                      # 0.25-floor proxy from extract_normalised_trades →
                                      # not a valid risk multiple; cell must be excluded)
    min_sl_dist:       float = 0.0    # smallest stop distance (pts) across trades

    @property
    def years_positive(self) -> int:
        return sum(1 for y in self.by_year.values() if y.get('mean_R', 0) > 0)

    @property
    def bootstrap_p5_mean_R(self, n_boot: int = 1000, seed: int = 2026) -> float:
        """5th-percentile of bootstrap mean_R distribution.
        Cell passes filter if this > 0."""
        rng = np.random.default_rng(seed)
        R = self.trades['R'].to_numpy(np.float64)
        if len(R) < 30:
            return float('-inf')
        idx = rng.integers(0, len(R), size=(n_boot, len(R)))
        boot_means = R[idx].mean(axis=1)
        return float(np.percentile(boot_means, 5))


def backtest_cell(
    cell: Cell,
    md: MarketData,
    slippage_pt: Optional[float] = None,
    commission_side: float = DEFAULT_COMMISSION_RT_2,
    fill_mode: str = DEFAULT_FILL_MODE,
    order_cancel: datetime.time = DEFAULT_ORDER_CANCEL,
    starting_capital: float = 100_000,
) -> CellResult:
    """Run a single cell with realistic execution. Returns CellResult.

    Per-instrument economics (tick size, slippage, point value) come from the
    InstrumentSpec for cell.instrument and are injected into both the engine
    (cfg.tick_size) and the strategy (params), so ES/YM trade on their true grid.
    """
    ispec = instrument_spec(cell.instrument)
    slip  = ispec.slippage_points if slippage_pt is None else slippage_pt
    params = {
        **cell.params,
        'tick_size':         ispec.tick_size,
        'point_value':       ispec.mini_point_value,
        'instrument_symbol': cell.instrument,
    }
    cfg = RunConfig(
        starting_capital=starting_capital,
        commission_per_contract=commission_side,
        slippage_points=slip,
        tick_size=ispec.tick_size,
        fill_mode=fill_mode,
        order_cancel_time=order_cancel,
        params=params,
    )
    res = run_backtest(cell.strategy_class, cfg, data=md, validate=False)
    pnl, sl = extract_normalised_trades(res.trades)
    if len(pnl) == 0:
        return CellResult(cell=cell, trades=pd.DataFrame(), n_trades=0,
                          mean_R=0.0, wr=0.0, sum_R=0.0, by_year={})

    # Round-trip slippage. extract_normalised_trades returns GROSS points
    # (exit_price - entry_price). For market next-open entries the entry side
    # of slippage is already baked into entry_price by the runner, so the gross
    # points already carry one side; for limit entries the runner applies no
    # entry slippage. Subtracting one slippage_pt completes a realistic round
    # trip in both cases (market: entry+exit; limit: exit only). Commission is
    # deliberately NOT subtracted here — the propfirm sim charges micro/mini
    # commission itself at the actually-traded contract size.
    pnl = pnl - slip
    # No-SL (time/EOD-exit) trades have no stop distance; extract_normalised_trades
    # falls back to a degenerate 0.25-floor proxy that makes R price-scale-inflated.
    # For those trades use the cell's MEAN ABSOLUTE net P&L (its typical trade
    # magnitude) as the risk unit, so R is a comparable, instrument-agnostic
    # quantity. Real-SL trades keep their true stop distance (unchanged).
    no_sl = np.array([t.sl_price is None for t in res.trades], dtype=bool)
    if no_sl.any():
        risk_unit = max(float(np.mean(np.abs(pnl))), 1e-6)
        sl = np.where(no_sl, risk_unit, sl)
    R = pnl / np.maximum(sl, 1e-6)
    mae = np.array([float(getattr(t, 'mae_points', 0.0) or 0.0) for t in res.trades],
                   dtype=np.float64)
    dates = pd.to_datetime([md.df_1m.index[t.entry_bar].date() for t in res.trades])
    trades = pd.DataFrame({
        'date': dates, 'pnl_pts': pnl, 'sl_dist': sl,
        'R': R, 'mae_pts': mae, 'MAE_R': mae / np.maximum(sl, 1e-6),
    })

    # Per-year stats
    by_year = {}
    for y, grp in trades.groupby(trades['date'].dt.year):
        by_year[int(y)] = {
            'n': len(grp),
            'mean_R': float(grp['R'].mean()),
            'sumR': float(grp['R'].sum()),
            'wr': float((grp['R'] > 0).mean()),
        }
    proxy_sl_fraction = float(np.mean([t.sl_price is None for t in res.trades]))
    return CellResult(
        cell=cell, trades=trades, n_trades=len(trades),
        mean_R=float(R.mean()), wr=float((R > 0).mean()),
        sum_R=float(R.sum()), by_year=by_year,
        proxy_sl_fraction=proxy_sl_fraction,
        min_sl_dist=float(sl.min()),
    )


def inclusion_filter(
    result: CellResult,
    min_trades: int = 100,
    min_mean_R: float = 0.05,
    min_years_positive: int = 4,
    min_years_total: int = 5,
    require_bootstrap_p5_positive: bool = True,
) -> tuple[bool, str]:
    """Apply pre-registered inclusion criteria. Returns (passed, reason).

    No-SL (time/EOD-exit) cells are admissible: their R is normalised by mean
    absolute P&L in backtest_cell (proxy_sl_fraction is retained for reporting,
    not as a gate)."""
    if result.n_trades < min_trades:
        return False, f"n={result.n_trades} < {min_trades}"
    if result.mean_R < min_mean_R:
        return False, f"mean_R={result.mean_R:+.4f} < {min_mean_R}"
    if result.years_positive < min_years_positive:
        return False, (f"only {result.years_positive}/{min_years_total} years positive "
                       f"< {min_years_positive}")
    if require_bootstrap_p5_positive and result.bootstrap_p5_mean_R <= 0:
        return False, f"bootstrap p5 mean_R={result.bootstrap_p5_mean_R:+.4f} <= 0"
    return True, "PASS"


def summarize_cells(results: list[CellResult]) -> pd.DataFrame:
    """Tabular summary of cell results."""
    rows = []
    for r in results:
        rows.append({
            'cell': r.cell.name,
            'strategy': r.cell.strategy_class.__name__,
            'instrument': r.cell.instrument,
            'n_trades': r.n_trades,
            'mean_R': r.mean_R,
            'wr': r.wr,
            'sum_R': r.sum_R,
            'years_pos': r.years_positive,
            'years_total': len(r.by_year),
        })
    return pd.DataFrame(rows)
