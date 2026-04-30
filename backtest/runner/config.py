from __future__ import annotations
from dataclasses import dataclass, field
from datetime import time


@dataclass
class RunConfig:
    """
    Configuration for a single backtest run.

    Pass to run_backtest() directly, or pass a list for grid search.

    Args:
        starting_capital:        Account balance at start of run ($).
        slippage_points:         Points of slippage applied to every fill,
                                 both entry and exit. e.g. 0.25 = 1 tick each way.
        commission_per_contract: Flat dollar commission per contract per side.
                                 e.g. 4.50 = $4.50 entry + $4.50 exit per contract.
        eod_exit_time:           Force-flat time each session. Default 17:00 ET (= 16:00 CT, NQ futures daily close).
        params:                  Strategy parameters dict, passed to strategy __init__.
                                 Use this for any strategy-specific tuning values.

    Example:
        config = RunConfig(
            starting_capital=100_000,
            slippage_points=0.25,
            commission_per_contract=4.50,
            params={"atr_mult": 2.0, "lookback": 20},
        )
    """

    starting_capital: float
    slippage_points: float = 0.0
    commission_per_contract: float = 0.0
    eod_exit_time: time = field(default_factory=lambda: time(17, 0))
    order_cancel_time: time = field(default_factory=lambda: time(11, 0))  # cancel pending orders at this time
    reverse_signals: bool = False   # flip every signal: long→short, short→long (SL/TP offsets preserved)
    track_equity_curve: bool = True  # set False in sweeps to skip 2M per-bar appends
    params: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.starting_capital <= 0:
            raise ValueError(f"starting_capital must be positive, got {self.starting_capital}")
        if self.slippage_points < 0:
            raise ValueError(f"slippage_points must be >= 0, got {self.slippage_points}")
        if self.commission_per_contract < 0:
            raise ValueError(f"commission_per_contract must be >= 0, got {self.commission_per_contract}")