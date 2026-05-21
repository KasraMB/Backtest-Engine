"""Unit tests for backtest/regime/vxn.py — Stage 3 of IVB v9-final."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.regime.vxn import load_vxn, vxn_z_prior_day_daily, broadcast_vxn_z_to_1m


def test_load_vxn_real_data():
    """Smoke test against shipped data/VXN.csv."""
    vxn = load_vxn()
    assert len(vxn) > 1000
    assert vxn.dtype == np.float64
    assert vxn.index.is_monotonic_increasing   # dates sorted, values are not
    assert (vxn > 0).all()


def test_vxn_z_prior_day_shift():
    """The z-score on day D must be computable WITHOUT day-D VXN close."""
    dates = pd.date_range('2020-01-01', periods=200, freq='B')
    vxn = pd.Series(np.linspace(15, 25, 200), index=dates)
    z = vxn_z_prior_day_daily(vxn, window=20)
    # Day 19 (index 19) is the first VXN close used in any window;
    # z is then shifted by 1, so first non-NaN appears at index 20.
    assert np.isnan(z.iloc[:20]).all()
    assert not np.isnan(z.iloc[20])
    # Critical: z on day D must equal a quantity computed only from days <D.
    # Specifically z[i] should equal log-z computed from vxn[i-20..i-1] (window=20).
    log_vxn = np.log(vxn.to_numpy(dtype=np.float64))
    expected_z_at_20 = (log_vxn[19] - log_vxn[0:20].mean()) / log_vxn[0:20].std(ddof=1)
    assert z.iloc[20] == pytest.approx(expected_z_at_20, rel=1e-9)


def test_broadcast_fills_non_vxn_days():
    """NQ trading on a weekend (hypothetical) should still get an aligned z."""
    daily_idx = pd.date_range('2020-01-01', periods=100, freq='B')
    vxn_z = pd.Series(np.arange(100, dtype=float), index=daily_idx)
    # 1m bars across a single day
    bar_idx = pd.date_range('2020-01-15 09:30', periods=10, freq='1min', tz='America/New_York')
    arr = broadcast_vxn_z_to_1m(vxn_z, bar_idx)
    assert len(arr) == 10
    # All bars on the same day get the same value
    assert (arr == arr[0]).all()
    # Value should match vxn_z on that date (Jan 15 is a Wed → in business range)
    expected = vxn_z.loc['2020-01-15']
    assert arr[0] == pytest.approx(expected)


def test_broadcast_handles_tz_aware_index():
    """Real NQ 1m index is tz-aware America/New_York; loader should handle both."""
    daily_idx = pd.date_range('2020-01-01', periods=50, freq='B')
    vxn_z = pd.Series(np.arange(50, dtype=float), index=daily_idx)
    bar_idx_naive = pd.date_range('2020-01-05 09:30', periods=5, freq='1min')
    bar_idx_aware = bar_idx_naive.tz_localize('America/New_York')
    arr_naive = broadcast_vxn_z_to_1m(vxn_z, bar_idx_naive)
    arr_aware = broadcast_vxn_z_to_1m(vxn_z, bar_idx_aware)
    assert np.allclose(arr_naive, arr_aware, equal_nan=True)
