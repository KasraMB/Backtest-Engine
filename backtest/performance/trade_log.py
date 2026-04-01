from __future__ import annotations

import csv
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backtest.runner.runner import RunResult
    from backtest.data.market_data import MarketData


def save_trade_log(
    result: "RunResult",
    data: "MarketData",
    filepath: str,
) -> None:
    """
    Save a CSV trade log for the given RunResult.

    Columns per trade:
        strategy, entry_time, exit_time, direction,
        entry_price, exit_price, initial_sl, initial_tp,
        pnl_points, pnl_dollars, net_pnl_dollars,
        r_multiple, mae_points, mae_r, mfe_points, mfe_r,
        exit_reason, bars_held, had_trailing,
        slippage_paid, commission_paid
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

    idx = data.df_1m.index

    fieldnames = [
        "strategy",
        "entry_time",
        "exit_time",
        "direction",
        "entry_price",
        "exit_price",
        "initial_sl",
        "initial_tp",
        "pnl_points",
        "pnl_dollars",
        "net_pnl_dollars",
        "r_multiple",
        "mae_points",
        "mae_r",
        "mfe_points",
        "mfe_r",
        "exit_reason",
        "trade_reason",
        "bars_held",
        "had_trailing",
        "slippage_paid",
        "commission_paid",
    ]

    def _fmt(v, decimals=4):
        if v is None:
            return ""
        if isinstance(v, float):
            return f"{v:.{decimals}f}"
        return str(v)

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for t in result.trades:
            entry_time = idx[t.entry_bar] if t.entry_bar < len(idx) else ""
            exit_time  = idx[t.exit_bar]  if t.exit_bar  < len(idx) else ""

            writer.writerow({
                "strategy":          result.strategy_name,
                "entry_time":        entry_time,
                "exit_time":         exit_time,
                "direction":         "long" if t.direction == 1 else "short",
                "entry_price":       _fmt(t.entry_price, 2),
                "exit_price":        _fmt(t.exit_price, 2),
                "initial_sl":        _fmt(t.initial_sl_price, 2),
                "initial_tp":        _fmt(t.initial_tp_price, 2),
                "pnl_points":        _fmt(t.pnl_points, 2),
                "pnl_dollars":       _fmt(t.pnl_dollars, 2),
                "net_pnl_dollars":   _fmt(t.net_pnl_dollars, 2),
                "r_multiple":        _fmt(t.r_multiple, 4),
                "mae_points":        _fmt(t.mae_points, 2),
                "mae_r":             _fmt(t.mae_r, 4),
                "mfe_points":        _fmt(t.mfe_points, 2),
                "mfe_r":             _fmt(t.mfe_r, 4),
                "exit_reason":       t.exit_reason.name,
                "trade_reason":      t.trade_reason,
                "bars_held":         t.bars_held,
                "had_trailing":      t.had_trailing,
                "slippage_paid":     _fmt(t.slippage_paid, 2),
                "commission_paid":   _fmt(t.commission_paid, 2),
            })
