"""Strategy implementations and stable discovery order for research portfolios."""

# Preserve the module order used before the strategy directories were consolidated.
# Newly added modules are discovered after these entries in alphabetical order.
DISCOVERY_ORDER = (
    "amr_strategy",
    "amr_strategy_v1",
    "ict_smc",
    "macd_strategy",
    "orb_atr_breakout_strategy",
    "orb_delta_breakout",
    "parabolic_sar_strategy",
    "session_mean_rev",
    "trend_following_strategy",
    "asia_breakout_strategy",
    "ibs_strategy",
    "orb_tf_hybrid",
    "rsi_strategy",
    "turtle_soup_strategy",
    "vwap_mean_rev_strategy",
    "awesome_oscillator_strategy",
    "bollinger_bands_strategy",
    "double_session_sweep",
    "dual_thrust_strategy",
    "dummy",
    "enhanced_orb",
    "gap_fill_strategy",
    "heikin_ashi_strategy",
    "ibs_mean_reversion",
    "intraday_interval_momentum_huang",
    "intraday_momentum_baltussen",
    "intraday_momentum_jin",
    "london_breakout_strategy",
    "noise_boundary_momentum",
    "orb_stop_strategy",
    "overnight_drift",
    "overnight_momentum_rosa",
    "overnight_reversal_cooc",
    "pdh_pdl_sweep_reversal",
    "shooting_star_strategy",
    "vwap_band_reversion",
)

# Keep historical imports and serialized class paths working without importing
# every strategy when this package is loaded.
_LEGACY_MODULES = {
    "profitable": ('amr_strategy', 'amr_strategy_v1', 'ict_smc', 'macd_strategy', 'orb_atr_breakout_strategy', 'orb_delta_breakout', 'parabolic_sar_strategy', 'session_mean_rev', 'trend_following_strategy'),
    "no_edge": ('asia_breakout_strategy', 'ibs_strategy', 'orb_tf_hybrid', 'rsi_strategy', 'turtle_soup_strategy', 'vwap_mean_rev_strategy'),
    "untested": ('awesome_oscillator_strategy', 'bollinger_bands_strategy', 'double_session_sweep', 'dual_thrust_strategy', 'dummy', 'enhanced_orb', 'gap_fill_strategy', 'heikin_ashi_strategy', 'ibs_mean_reversion', 'intraday_interval_momentum_huang', 'intraday_momentum_baltussen', 'intraday_momentum_jin', 'london_breakout_strategy', 'noise_boundary_momentum', 'orb_stop_strategy', 'overnight_drift', 'overnight_momentum_rosa', 'overnight_reversal_cooc', 'pdh_pdl_sweep_reversal', 'shooting_star_strategy', 'vwap_band_reversion'),
}

def _install_legacy_aliases():
    import importlib
    import importlib.abc
    import importlib.util
    from pathlib import Path
    import sys
    import types

    package_dir = Path(__file__).resolve().parent

    class _AliasLoader(importlib.abc.Loader):
        def __init__(self, canonical_name):
            self.canonical_name = canonical_name

        def create_module(self, spec):
            module = importlib.import_module(self.canonical_name)
            self.canonical_spec = module.__spec__
            return module

        def exec_module(self, module):
            # Import machinery sets __spec__ to the alias spec after
            # create_module. Restore canonical metadata so reload() executes
            # the real source loader rather than this compatibility loader.
            module.__spec__ = self.canonical_spec
            module.__loader__ = self.canonical_spec.loader
            module.__package__ = self.canonical_name.rpartition(".")[0]

    class _AliasFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            parts = fullname.split(".")
            if (len(parts) == 3 and parts[0] == __name__
                    and parts[2] in _LEGACY_MODULES.get(parts[1], ())):
                return importlib.util.spec_from_loader(
                    fullname, _AliasLoader(f"{__name__}.{parts[2]}"))
            return None

    for group in _LEGACY_MODULES:
        fullname = f"{__name__}.{group}"
        if fullname not in sys.modules:
            package = types.ModuleType(fullname)
            package.__package__ = fullname
            # An existing local directory may still contain untracked research.
            package.__path__ = [str(package_dir / group)]
            package.__spec__ = importlib.util.spec_from_loader(
                fullname, loader=None, is_package=True)
            sys.modules[fullname] = package
            setattr(sys.modules[__name__], group, package)
    sys.meta_path.insert(0, _AliasFinder())


_install_legacy_aliases()
