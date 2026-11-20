"""Check that the public example executes the engine without local data files."""
import math

from examples.basic_backtest import make_example_data, run_example


def test_synthetic_example_completes_with_next_open_fills():
    data = make_example_data()
    result = run_example()

    assert result.trades
    assert math.isfinite(result.total_net_pnl)
    assert result.config.fill_mode == "next_open"
    assert len(result.equity_curve) == len(data.df_1m) + 1
    assert result.equity_curve[0] == result.config.starting_capital
    for trade in result.trades:
        assert trade.entry_bar % 30 == 1
        assert trade.commission_per_contract == 1.75
    assert (data.bar_map[:5] == -1).all()
    assert data.bar_map[5] == 0
