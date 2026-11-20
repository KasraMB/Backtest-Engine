"""Compatibility and portfolio ordering checks for the strategy layout migration."""

import importlib
import pickle
from pathlib import Path
import subprocess
import sys

import pytest


def _fresh_python(source):
    subprocess.run(
        [sys.executable, "-c", source],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )


def test_importing_strategy_package_does_not_load_strategy_modules():
    _fresh_python("""
import sys
import strategies
assert not any('backtest' == name or name.startswith('backtest.') for name in sys.modules)
assert not any(f'strategies.{name}' in sys.modules for name in strategies.DISCOVERY_ORDER)
""")


@pytest.mark.parametrize("legacy_first", [False, True])
def test_alias_identity_and_reload_in_both_import_orders(legacy_first):
    # Isolate reload so registered portfolio classes in other tests retain their
    # identity. The sentinel proves reload actually executes the source loader.
    _fresh_python(f"""
import importlib
canonical = 'strategies.amr_strategy'
legacy = 'strategies.profitable.amr_strategy'
names = [legacy, canonical] if {legacy_first!r} else [canonical, legacy]
first, second = [importlib.import_module(name) for name in names]
assert first is second
assert first.__spec__.name == canonical
assert first.__package__ == 'strategies'
original = first.AnchoredMeanReversionStrategy
first.AnchoredMeanReversionStrategy = None
assert importlib.reload(first) is first
assert first.AnchoredMeanReversionStrategy is not None
assert first.AnchoredMeanReversionStrategy is not original
assert importlib.import_module(legacy) is first
assert first.__spec__.name == canonical
""")


@pytest.mark.parametrize("legacy,canonical,class_name", [
    ("profitable.amr_strategy", "amr_strategy", "AnchoredMeanReversionStrategy"),
    ("no_edge.asia_breakout_strategy", "asia_breakout_strategy", "AsiaBreakoutStrategy"),
    ("untested.dual_thrust_strategy", "dual_thrust_strategy", "DualThrustStrategy"),
])
def test_historical_pickle_class_references_resolve(legacy, canonical, class_name):
    # Protocol 0 GLOBAL reproduces the reference stored in older class/instance
    # pickles without committing opaque serialized fixtures.
    serialized = f"cstrategies.{legacy}\n{class_name}\n.".encode("ascii")
    expected = getattr(importlib.import_module(f"strategies.{canonical}"), class_name)
    assert pickle.loads(serialized) is expected


def test_portfolio_discovery_retains_historical_order(monkeypatch):
    from backtest.portfolio import registry

    # Snapshot the pre-migration tracked class order: saved portfolio selections
    # must keep their interpretation even if filesystem iteration changes.
    expected = [
        "AnchoredMeanReversionStrategy", "ICTSMCStrategy", "MACDStrategy",
        "ORBATRBreakoutStrategy", "ORBDeltaBreakoutStrategy",
        "ParabolicSARStrategy", "SessionMeanRevStrategy", "TrendFollowingStrategy",
        "AsiaBreakoutStrategy", "IBSStrategy", "ORBTFHybridStrategy", "RSIStrategy",
        "TurtleSoupStrategy", "VWAPMeanRevStrategy", "AwesomeOscillatorStrategy",
        "BollingerBandsStrategy", "DoubleSessionSweep", "DualThrustStrategy",
        "EnhancedORBStrategy", "GapFillStrategy", "HeikinAshiStrategy",
        "IBSMeanReversion", "IntradayIntervalMomentumHuang",
        "IntradayMomentumBaltussen", "IntradayMomentumJin", "LondonBreakoutStrategy",
        "NoiseBoundaryMomentum", "ORBStopStrategy", "OvernightDriftStrategy",
        "OvernightMomentumRosa", "OvernightReversalCOOC", "PDHPDLSweepReversal",
        "ShootingStarStrategy", "VWAPBandMeanReversion",
    ]
    modules = list(registry.pkgutil.iter_modules(registry.strategies.__path__))
    monkeypatch.setattr(registry.pkgutil, "iter_modules", lambda path: iter(reversed(modules)))
    discovered = registry._discover_strategies()
    assert [cls.__name__ for cls in discovered] == expected
    nq_only = {
        "AnchoredMeanReversionStrategy", "ICTSMCStrategy",
        "ORBDeltaBreakoutStrategy", "SessionMeanRevStrategy",
    }
    cells = registry.build_pre_registered_cells()
    assert [(cell.strategy_class.__name__, cell.instrument) for cell in cells] == [
        (name, instrument)
        for name in expected
        for instrument in (["NQ"] if name in nq_only else ["NQ", "ES", "YM"])
    ]
