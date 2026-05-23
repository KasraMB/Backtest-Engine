"""Pre-registered portfolio cells (strategy × instrument).

Auto-discovers strategy classes from `strategies/{profitable,no_edge,untested}/`.
LOCKED at first import: cell list cannot change after Phase 2 starts.
"""
from __future__ import annotations

import importlib
import inspect
import pkgutil
from pathlib import Path
from typing import Type

from backtest.portfolio.cell import Cell
from backtest.strategy.base import BaseStrategy as _BaseStrategyA
# Some strategies (orb_delta_breakout) inherit from a legacy BaseStrategy.
# Accept both bases during discovery.
from backtest.engine.base import BaseStrategy as _BaseStrategyB

_STRATEGY_BASES = (_BaseStrategyA, _BaseStrategyB)
BaseStrategy = _BaseStrategyA

INSTRUMENTS = ['NQ', 'ES', 'YM']

# Strategies whose default config is NQ-specific (NQ sessions, bid/ask volume
# delta thresholds tuned for NQ). Tested only on NQ.
_NQ_ONLY_NAMES = {
    'AnchoredMeanReversionStrategy',
    'ICTSMCStrategy',
    'SessionMeanRevStrategy',
    'ORBDeltaBreakoutStrategy',
}

# Strategies to skip entirely (older versions / placeholders).
_SKIP_NAMES = {
    'AnchoredMeanReversionStrategyV1',
    'DummyStrategy',
    'DummyLongStrategy',
}


def _discover_strategies() -> list[Type[BaseStrategy]]:
    """Walk strategies/{profitable,no_edge,untested} and return all
    BaseStrategy subclasses, deduplicated by class name."""
    out: dict[str, Type[BaseStrategy]] = {}
    pkg_root = Path(__file__).resolve().parents[2] / 'strategies'
    for sub in ('profitable', 'no_edge', 'untested'):
        sub_pkg = importlib.import_module(f'strategies.{sub}')
        for finder, name, ispkg in pkgutil.iter_modules(sub_pkg.__path__):
            if ispkg or name.startswith('_'):
                continue
            mod = importlib.import_module(f'strategies.{sub}.{name}')
            for obj_name, obj in inspect.getmembers(mod, inspect.isclass):
                if obj in _STRATEGY_BASES:
                    continue
                if not any(issubclass(obj, b) for b in _STRATEGY_BASES):
                    continue
                if obj.__module__ != f'strategies.{sub}.{name}':
                    continue                           # imported elsewhere
                if obj_name in _SKIP_NAMES:
                    continue
                if obj_name in out:
                    continue                           # dedup (e.g. AMR v1)
                out[obj_name] = obj
    return list(out.values())


def build_pre_registered_cells() -> list[Cell]:
    """Locked cell list. NQ-tuned strategies → NQ only; portable → NQ+ES+YM."""
    cells = []
    for cls in _discover_strategies():
        if cls.__name__ in _NQ_ONLY_NAMES:
            cells.append(Cell(strategy_class=cls, instrument='NQ'))
        else:
            for inst in INSTRUMENTS:
                cells.append(Cell(strategy_class=cls, instrument=inst))
    return cells


CELLS = build_pre_registered_cells()
