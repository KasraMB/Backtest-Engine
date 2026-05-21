"""
VXN (Nasdaq-100 volatility index) loader and prior-day z-score feature.

Used by IVB v9-final as the 3rd HMM feature: [returns, log_vol_z, vxn_z_prior_day].
Inclusion gated by (i) ΔBIC > 0 on train-fit AND (ii) regime-conditioned EV
separation widens on holdoutA (stage 3 test in proxy_validation.py).

VXN data source: FRED VXNCLS (CBOE Nasdaq-100 Volatility Index, daily close).
File schema (data/VXN.csv): observation_date, VXNCLS.

Look-ahead safety: we always use the PRIOR DAY's close. Intraday bars on day D
get the z-score computed from VXN closes through D-1. No same-day leak.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_VXN_PATH = Path(__file__).resolve().parents[2] / 'data' / 'VXN.csv'


def load_vxn(path: Path = DEFAULT_VXN_PATH) -> pd.Series:
    """Load FRED VXNCLS as a daily Series indexed by date (no tz).

    Drops the header garbage, parses dates, sorts ascending. Drops NaN closes
    (FRED uses '.' for missing observations).
    """
    df = pd.read_csv(path)
    if not {'observation_date', 'VXNCLS'}.issubset(df.columns):
        raise ValueError(f"Expected columns observation_date, VXNCLS in {path}, got {list(df.columns)}")
    # FRED uses '.' for missing values → coerce to NaN
    vxn = pd.to_numeric(df['VXNCLS'], errors='coerce')
    dates = pd.to_datetime(df['observation_date'])
    s = pd.Series(vxn.to_numpy(), index=pd.DatetimeIndex(dates)).dropna().sort_index()
    s.name = 'vxn'
    return s


def vxn_z_prior_day_daily(vxn: pd.Series, window: int = 60) -> pd.Series:
    """Rolling z-score of log(VXN) over `window` days, SHIFTED by 1 day.

    A given calendar date D gets the z-score computed from VXN closes through D-1.
    Days where VXN doesn't trade (weekends/holidays) get NaN; broadcast layer
    forward-fills to align with NQ trading days.

    Vectorised: pandas rolling C-level mean/std + a shift.
    """
    log_vxn = np.log(vxn.astype(np.float64))
    mu = log_vxn.rolling(window=window, min_periods=window).mean()
    sd = log_vxn.rolling(window=window, min_periods=window).std(ddof=1)
    z = (log_vxn - mu) / sd
    # Shift by 1 trading day in the VXN calendar so today's intraday uses yesterday's close.
    return z.shift(1).rename('vxn_z_prior_day')


def broadcast_vxn_z_to_1m(
    vxn_z_daily: pd.Series,
    bar_index: pd.DatetimeIndex,
) -> np.ndarray:
    """Map daily vxn_z to per-1m-bar array using each bar's session date.

    Forward-fills across weekends/holidays so any NQ trading day picks up the
    last available VXN value. Returns np.ndarray length len(bar_index); leading
    bars before window+1 days of VXN history get NaN.

    Vectorised: pandas reindex + ffill (hash-based, C-level).
    """
    daily_idx = pd.DatetimeIndex(vxn_z_daily.index).normalize()
    src = pd.Series(vxn_z_daily.to_numpy(), index=daily_idx)
    bar_day = pd.DatetimeIndex(bar_index).tz_localize(None).normalize() \
        if bar_index.tz is not None else pd.DatetimeIndex(bar_index).normalize()
    # Reindex to bar dates, ffill across non-VXN days
    aligned = src.reindex(bar_day.unique().sort_values()).ffill()
    return aligned.reindex(bar_day).to_numpy(np.float64)
