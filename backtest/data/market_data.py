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