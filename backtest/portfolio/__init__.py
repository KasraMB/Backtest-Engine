"""Equal-weight multi-strategy portfolio research framework.

Public API:
  - load_instrument(symbol)  -> MarketData (NQ/ES/YM)
  - Cell, CellResult         -> per-(strategy, instrument) backtest
  - backtest_cell(cell, md)  -> CellResult with realistic execution
  - inclusion_filter(...)    -> apply pre-registered thresholds
  - portfolio_daily_R(...)   -> equal-weight combined daily R
  - propfirm_p_profit(...)   -> bootstrap propfirm sim on daily R
"""
from backtest.portfolio.data import load_instrument
from backtest.portfolio.cell import (
    Cell, CellResult, backtest_cell, inclusion_filter, summarize_cells,
)
from backtest.portfolio.sim import (
    build_per_day_arrays, portfolio_daily_R, propfirm_p_profit,
)

__all__ = [
    'load_instrument', 'Cell', 'CellResult', 'backtest_cell',
    'inclusion_filter', 'summarize_cells', 'build_per_day_arrays',
    'portfolio_daily_R', 'propfirm_p_profit',
]
