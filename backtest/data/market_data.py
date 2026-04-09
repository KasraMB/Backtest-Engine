from dataclasses import dataclass, field
from datetime import date
import pandas as pd
import numpy as np


@dataclass
class MarketData:
    """
    Container for all market data used by the backtest engine.
    Holds cleaned DataFrames and pre-extracted numpy arrays for fast loop access.
    """

    # --- DataFrames (used for indicator access and strategy logic) ---
    df_1m: pd.DataFrame
    df_5m: pd.DataFrame

    # --- Pre-extracted numpy arrays (used in the hot execution loop) ---
    # 1m arrays
    open_1m: np.ndarray
    high_1m: np.ndarray
    low_1m: np.ndarray
    close_1m: np.ndarray
    volume_1m: np.ndarray

    # 5m arrays
    open_5m: np.ndarray
    high_5m: np.ndarray
    low_5m: np.ndarray
    close_5m: np.ndarray
    volume_5m: np.ndarray

    # --- Bar map: for each 1m bar index i, the index of the last COMPLETED 5m bar ---
    # Value is -1 if no completed 5m bar exists yet
    bar_map: np.ndarray

    # --- Session dates present in the data ---
    trading_dates: list[date] = field(default_factory=list)

    # --- Pre-computed bar date/time arrays (cached here so strategy instances
    #     share them rather than each computing from the pandas index) ---
    # Populated lazily by ICTSMCStrategy._ensure_bar_metadata on first call.
    bar_dates_1m_ord: np.ndarray = field(default=None)   # int32 Gregorian ordinals
    bar_times_1m_min: np.ndarray = field(default=None)   # int32 minutes-since-midnight
    bar_dates_5m_ord: np.ndarray = field(default=None)
    bar_times_5m_min: np.ndarray = field(default=None)
    # Pre-computed date → (first_bar_idx, one_past_last_bar_idx) for 1m bars.
    # Dict[int, Tuple[int, int]] — allows O(1) _date_slice lookups instead of
    # two np.searchsorted calls per session level lookup.
    date_to_slice_1m: dict = field(default=None)

    # ---------------------------------------------------------------------------
    # Runner-level caches (config-independent, safe to reuse across backtest calls
    # on the same MarketData object — e.g. within a single ML-collect worker).
    # ---------------------------------------------------------------------------
    # build_eod_bar_set cache: {eod_exit_minute: set[int]}
    _eod_bar_cache: dict = field(default=None)
    # build_required_bar_set cache (trading_hours=None strategies only)
    _required_bar_result: object = field(default=None)
    _required_bar_ready: bool = field(default=False)

    def __post_init__(self):
        self._validate()

    def _validate(self):
        assert len(self.df_1m) > 0, "df_1m is empty"
        assert len(self.df_5m) > 0, "df_5m is empty"
        assert len(self.bar_map) == len(self.df_1m), (
            f"bar_map length {len(self.bar_map)} != df_1m length {len(self.df_1m)}"
        )
        assert isinstance(self.df_1m.index, pd.DatetimeIndex), "df_1m must have DatetimeIndex"
        assert isinstance(self.df_5m.index, pd.DatetimeIndex), "df_5m must have DatetimeIndex"

    @property
    def n_bars_1m(self) -> int:
        return len(self.df_1m)

    @property
    def n_bars_5m(self) -> int:
        return len(self.df_5m)