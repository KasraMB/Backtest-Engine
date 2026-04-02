"""
Single source of truth for train / validation / test split boundaries.

Edit the constants below to match your data coverage.
All dates are inclusive end-points.

Splits
------
train       2019-01-01 → 2022-12-31   walk-forward lives here
validation  2023-01-01 → 2023-12-31   threshold tuning, model selection
test1       2024-01-01 → 2024-12-31   one-shot evaluation (use sparingly)
test2       2025-01-01 → present      reserve for a second opinion after iteration
"""
from __future__ import annotations

from datetime import time as dtime
from typing import Optional, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# Boundaries — edit here
# ---------------------------------------------------------------------------

TRAIN_START      = "2019-01-01"
TRAIN_END        = "2022-12-31"
VALIDATION_START = "2023-01-01"
VALIDATION_END   = "2023-12-31"
TEST1_START      = "2024-01-01"
TEST1_END        = "2024-12-31"
TEST2_START      = "2025-01-01"
TEST2_END: Optional[str] = None   # None = up to latest available bar

SPLITS: dict[str, Tuple[str, Optional[str]]] = {
    "train":      (TRAIN_START,      TRAIN_END),
    "validation": (VALIDATION_START, VALIDATION_END),
    "test1":      (TEST1_START,      TEST1_END),
    "test2":      (TEST2_START,      TEST2_END),
}

TZ = "America/New_York"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def split_bounds(split: str) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Return (start_ts, end_ts) as tz-aware Timestamps for the named split."""
    if split not in SPLITS:
        raise ValueError(f"Unknown split '{split}'. Choose from: {list(SPLITS)}")
    start_str, end_str = SPLITS[split]
    start = pd.Timestamp(start_str, tz=TZ)
    end   = (
        pd.Timestamp(end_str, tz=TZ) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        if end_str
        else pd.Timestamp("2200-01-01", tz=TZ)
    )
    return start, end


def filter_df(df: pd.DataFrame, split: str) -> pd.DataFrame:
    """
    Filter a dataset DataFrame (produced by build_dataset) to the given split.
    Expects a 'date' column (Python date or Timestamp).
    """
    start, end = split_bounds(split)
    dates = pd.to_datetime(df["date"]).dt.tz_localize(None)
    s = start.tz_localize(None)
    e = end.tz_localize(None)
    return df[(dates >= s) & (dates <= e)].copy()


def filter_market_data(data, split: str, loader):
    """
    Return a new MarketData filtered to the named split.

    Parameters
    ----------
    data   : MarketData  — full dataset
    split  : str         — one of SPLITS keys
    loader : DataLoader  — used for _build_bar_map

    Returns
    -------
    MarketData
    """
    import numpy as np
    from backtest.data.market_data import MarketData

    start_ts, end_ts = split_bounds(split)

    mask_1m = (data.df_1m.index >= start_ts) & (data.df_1m.index <= end_ts)
    mask_5m = (data.df_5m.index >= start_ts) & (data.df_5m.index <= end_ts)
    df_1m_f = data.df_1m[mask_1m]
    df_5m_f = data.df_5m[mask_5m]

    rth = (df_1m_f.index.time >= dtime(9, 30)) & (df_1m_f.index.time <= dtime(16, 0))
    a1  = {c: df_1m_f[c].to_numpy(dtype="float64") for c in ["open","high","low","close","volume"]}
    a5  = {c: df_5m_f[c].to_numpy(dtype="float64") for c in ["open","high","low","close","volume"]}

    return MarketData(
        df_1m=df_1m_f, df_5m=df_5m_f,
        open_1m=a1["open"],  high_1m=a1["high"],  low_1m=a1["low"],
        close_1m=a1["close"], volume_1m=a1["volume"],
        open_5m=a5["open"],  high_5m=a5["high"],  low_5m=a5["low"],
        close_5m=a5["close"], volume_5m=a5["volume"],
        bar_map=loader._build_bar_map(df_1m_f, df_5m_f),
        trading_dates=sorted(set(df_1m_f[rth].index.date)),
    )
