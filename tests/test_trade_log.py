"""
Tests for backtest/performance/trade_log.py — save_trade_log().

Run with:
    python -m pytest tests/test_trade_log.py -v
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

from backtest.performance.trade_log import save_trade_log
from backtest.runner.config import RunConfig
from backtest.strategy.enums import ExitReason
from backtest.strategy.update import Trade
from backtest.data.market_data import MarketData


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_market_data(n: int = 200) -> MarketData:
    idx = pd.date_range("2024-01-02 09:30", periods=n, freq="1min",
                        tz="America/New_York")
    arr = np.full(n, 19000.0)
    df  = pd.DataFrame({"open": arr, "high": arr + 10, "low": arr - 10,
                         "close": arr, "volume": arr,
                         "anomalous": np.zeros(n, dtype=bool)}, index=idx)
    df_5m = df.resample("5min").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum", "anomalous": "any",
    }).dropna()
    return MarketData(
        df_1m=df, df_5m=df_5m,
        open_1m=arr, high_1m=arr + 10, low_1m=arr - 10,
        close_1m=arr, volume_1m=arr,
        open_5m=df_5m["open"].values, high_5m=df_5m["high"].values,
        low_5m=df_5m["low"].values, close_5m=df_5m["close"].values,
        volume_5m=df_5m["volume"].values,
        bar_map=np.full(n, -1, dtype=np.int64),
        trading_dates=sorted(set(ts.date() for ts in idx)),
    )


def make_trade(
    entry_bar=10,
    exit_bar=20,
    direction=1,
    entry_price=19000.0,
    exit_price=19100.0,
    contracts=1,
    slippage=0.25,
    commission=4.50,
    exit_reason=ExitReason.TP,
    initial_sl=18900.0,
    initial_tp=19200.0,
    trade_reason="test",
) -> Trade:
    return Trade(
        entry_bar=entry_bar,
        exit_bar=exit_bar,
        entry_price=entry_price,
        exit_price=exit_price,
        direction=direction,
        contracts=contracts,
        slippage_points=slippage,
        commission_per_contract=commission,
        exit_reason=exit_reason,
        initial_sl_price=initial_sl,
        initial_tp_price=initial_tp,
        trade_reason=trade_reason,
    )


class _FakeResult:
    def __init__(self, trades, strategy_name="TestStrategy"):
        self.trades        = trades
        self.strategy_name = strategy_name
        self.config        = RunConfig(starting_capital=100_000)

        bal = 100_000.0
        self.equity_curve = [bal]
        for t in trades:
            bal += t.net_pnl_dollars
            self.equity_curve.append(bal)


def load_csv(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSaveTradeLog:

    def test_creates_file(self, tmp_path):
        trades = [make_trade()]
        result = _FakeResult(trades)
        data   = make_market_data()
        path   = str(tmp_path / "trades.csv")
        save_trade_log(result, data, path)
        assert Path(path).exists()

    def test_creates_parent_directory(self, tmp_path):
        trades = [make_trade()]
        result = _FakeResult(trades)
        data   = make_market_data()
        path   = str(tmp_path / "subdir" / "nested" / "trades.csv")
        save_trade_log(result, data, path)
        assert Path(path).exists()

    def test_header_row_contains_expected_columns(self, tmp_path):
        result = _FakeResult([make_trade()])
        data   = make_market_data()
        path   = str(tmp_path / "trades.csv")
        save_trade_log(result, data, path)
        rows = load_csv(path)
        assert len(rows) > 0
        expected_cols = {
            "strategy", "entry_time", "exit_time", "direction",
            "entry_price", "exit_price", "initial_sl", "initial_tp",
            "pnl_points", "pnl_dollars", "net_pnl_dollars",
            "r_multiple", "mae_points", "mfe_points",
            "exit_reason", "trade_reason", "bars_held",
            "had_trailing", "slippage_paid", "commission_paid",
        }
        assert expected_cols.issubset(rows[0].keys())

    def test_one_row_per_trade(self, tmp_path):
        trades = [make_trade(entry_bar=i * 10, exit_bar=i * 10 + 5) for i in range(5)]
        result = _FakeResult(trades)
        data   = make_market_data()
        path   = str(tmp_path / "trades.csv")
        save_trade_log(result, data, path)
        rows = load_csv(path)
        assert len(rows) == 5

    def test_strategy_name_written(self, tmp_path):
        result = _FakeResult([make_trade()], strategy_name="MyStrategy")
        data   = make_market_data()
        path   = str(tmp_path / "trades.csv")
        save_trade_log(result, data, path)
        rows = load_csv(path)
        assert rows[0]["strategy"] == "MyStrategy"

    def test_direction_long_written(self, tmp_path):
        result = _FakeResult([make_trade(direction=1)])
        data   = make_market_data()
        path   = str(tmp_path / "trades.csv")
        save_trade_log(result, data, path)
        rows = load_csv(path)
        assert rows[0]["direction"] == "long"

    def test_direction_short_written(self, tmp_path):
        result = _FakeResult([make_trade(direction=-1, initial_sl=19100.0, initial_tp=18800.0)])
        data   = make_market_data()
        path   = str(tmp_path / "trades.csv")
        save_trade_log(result, data, path)
        rows = load_csv(path)
        assert rows[0]["direction"] == "short"

    def test_entry_exit_price_written(self, tmp_path):
        result = _FakeResult([make_trade(entry_price=19000.0, exit_price=19100.0)])
        data   = make_market_data()
        path   = str(tmp_path / "trades.csv")
        save_trade_log(result, data, path)
        rows = load_csv(path)
        assert float(rows[0]["entry_price"]) == pytest.approx(19000.0, abs=0.01)
        assert float(rows[0]["exit_price"])  == pytest.approx(19100.0, abs=0.01)

    def test_exit_reason_written(self, tmp_path):
        result = _FakeResult([make_trade(exit_reason=ExitReason.SL)])
        data   = make_market_data()
        path   = str(tmp_path / "trades.csv")
        save_trade_log(result, data, path)
        rows = load_csv(path)
        assert rows[0]["exit_reason"] == "SL"

    def test_trade_reason_written(self, tmp_path):
        result = _FakeResult([make_trade(trade_reason="OTE_Long_PDH")])
        data   = make_market_data()
        path   = str(tmp_path / "trades.csv")
        save_trade_log(result, data, path)
        rows = load_csv(path)
        assert rows[0]["trade_reason"] == "OTE_Long_PDH"

    def test_pnl_points_correct(self, tmp_path):
        # Long: entry 19000, exit 19100 → +100 pts
        result = _FakeResult([make_trade(entry_price=19000.0, exit_price=19100.0, direction=1)])
        data   = make_market_data()
        path   = str(tmp_path / "trades.csv")
        save_trade_log(result, data, path)
        rows = load_csv(path)
        assert float(rows[0]["pnl_points"]) == pytest.approx(100.0, abs=0.01)

    def test_pnl_dollars_correct(self, tmp_path):
        # Long: +100 pts * 1 contract * $20 = $2000
        result = _FakeResult([make_trade(entry_price=19000.0, exit_price=19100.0,
                                          contracts=1, direction=1)])
        data   = make_market_data()
        path   = str(tmp_path / "trades.csv")
        save_trade_log(result, data, path)
        rows = load_csv(path)
        assert float(rows[0]["pnl_dollars"]) == pytest.approx(2000.0, abs=0.01)

    def test_bars_held_correct(self, tmp_path):
        result = _FakeResult([make_trade(entry_bar=10, exit_bar=25)])
        data   = make_market_data()
        path   = str(tmp_path / "trades.csv")
        save_trade_log(result, data, path)
        rows = load_csv(path)
        assert int(rows[0]["bars_held"]) == 15

    def test_empty_trades_produces_header_only(self, tmp_path):
        result = _FakeResult([])
        data   = make_market_data()
        path   = str(tmp_path / "trades.csv")
        save_trade_log(result, data, path)
        rows = load_csv(path)
        assert len(rows) == 0
        # File still exists with a header
        assert Path(path).exists()

    def test_overwrite_existing_file(self, tmp_path):
        path = str(tmp_path / "trades.csv")
        # Write once with 3 trades
        result_3 = _FakeResult([make_trade(entry_bar=i * 10, exit_bar=i * 10 + 5)
                                 for i in range(3)])
        save_trade_log(result_3, make_market_data(), path)
        assert len(load_csv(path)) == 3

        # Overwrite with 1 trade
        result_1 = _FakeResult([make_trade()])
        save_trade_log(result_1, make_market_data(), path)
        assert len(load_csv(path)) == 1

    def test_slippage_and_commission_written(self, tmp_path):
        # 0.25 pts slip * 2 sides * 1 contract * $20 = $10
        # $4.50 comm * 2 sides * 1 contract = $9
        result = _FakeResult([make_trade(slippage=0.25, commission=4.50, contracts=1)])
        data   = make_market_data()
        path   = str(tmp_path / "trades.csv")
        save_trade_log(result, data, path)
        rows = load_csv(path)
        assert float(rows[0]["slippage_paid"])   == pytest.approx(10.0, abs=0.01)
        assert float(rows[0]["commission_paid"]) == pytest.approx(9.0,  abs=0.01)

    def test_entry_time_from_market_data_index(self, tmp_path):
        """entry_time should come from df_1m.index[entry_bar]."""
        result = _FakeResult([make_trade(entry_bar=0, exit_bar=5)])
        data   = make_market_data()
        path   = str(tmp_path / "trades.csv")
        save_trade_log(result, data, path)
        rows = load_csv(path)
        # entry_time should be non-empty and parseable
        entry_time_str = rows[0]["entry_time"]
        assert entry_time_str != ""
        # Should contain the expected date
        assert "2024-01-02" in str(entry_time_str)

    def test_entry_bar_beyond_index_handled(self, tmp_path):
        """Trade with entry_bar >= len(data) should not crash."""
        result = _FakeResult([make_trade(entry_bar=9999, exit_bar=10000)])
        data   = make_market_data(n=100)   # only 100 bars
        path   = str(tmp_path / "trades.csv")
        save_trade_log(result, data, path)  # must not raise
        rows = load_csv(path)
        assert len(rows) == 1
        assert rows[0]["entry_time"] == ""   # out-of-range → empty string


if __name__ == "__main__":
    import pytest as _pytest
    _pytest.main([__file__, "-v"])
