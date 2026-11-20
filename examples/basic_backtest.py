"""Run a complete backtest using generated prices.

From the repository root, after ``python -m pip install -e .``::

    python -m examples.basic_backtest

The deterministic prices and simple strategy demonstrate the API only; the
output is not evidence of a trading edge. No files are read or written.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.data.loader import DataLoader
from backtest.data.market_data import MarketData
from backtest.runner.config import RunConfig
from backtest.runner.runner import RunResult, run_backtest
from backtest.strategy.base import BaseStrategy
from backtest.strategy.enums import OrderType, SizeType
from backtest.strategy.order import Order
from backtest.strategy.update import OpenPosition, PositionUpdate


class ExampleStrategy(BaseStrategy):
    """Submit one market order every 30 bars, with fixed stop and target."""

    def generate_signals(self, data: MarketData, i: int) -> Order | None:
        if i % 30 != 0:
            return None
        return Order(
            direction=1,
            order_type=OrderType.MARKET,
            size_type=SizeType.CONTRACTS,
            size_value=1,
        )

    def on_fill(self, position: OpenPosition, data: MarketData, bar_index: int) -> None:
        position.set_initial_sl_tp(
            sl=position.entry_price - 5.0,
            tp=position.entry_price + 8.0,
        )

    def manage_position(
        self, data: MarketData, i: int, position: OpenPosition
    ) -> PositionUpdate | None:
        return None


def make_example_data() -> MarketData:
    """Build one synthetic session, including completed five-minute bar mapping."""
    index = pd.date_range(
        "2024-01-02 09:30", periods=390, freq="1min", tz="America/New_York"
    )
    rng = np.random.default_rng(42)
    closes = 19_000.0 + np.cumsum(rng.normal(0, 1.25, len(index)))
    opens = np.concatenate(([closes[0]], closes[:-1]))
    one_minute = pd.DataFrame(
        {
            "open": opens,
            "high": np.maximum(opens, closes) + 0.75,
            "low": np.minimum(opens, closes) - 0.75,
            "close": closes,
            "volume": np.full(len(index), 100.0),
            "anomalous": False,
        },
        index=index,
    )
    five_minute = one_minute.resample("5min").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
            "anomalous": "any",
        }
    )
    return DataLoader(verbose=False).build_market_data(one_minute, five_minute)


def run_example() -> RunResult:
    config = RunConfig(
        starting_capital=100_000,
        slippage_points=0.25,
        commission_per_contract=1.75,
        fill_mode="next_open",
    )
    return run_backtest(ExampleStrategy, config, data=make_example_data())


def main() -> None:
    print("Synthetic data example (illustrates the API, not trading performance)")
    run_example().print_summary()


if __name__ == "__main__":
    main()
