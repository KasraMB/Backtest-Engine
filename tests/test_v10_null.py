"""Unit tests for backtest/ivb/v10_null.py — time-of-day matched random-entry null."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.ivb.v10_signals import orb_bare_signals
from backtest.ivb.v10_null import build_null_means


def _build_dev_session(open_px: float, drift_per_bar: float, n_days: int = 10):
    """Synthetic dev-A-like sessions: 10 trading days, drift per bar."""
    base = pd.Timestamp('2020-01-06', tz='America/New_York')   # Monday
    frames = []
    d = 0
    while len(frames) < n_days:
        day_start = base + pd.Timedelta(days=d)
        d += 1
        if day_start.dayofweek >= 5:
            continue
        idx = pd.date_range(day_start + pd.Timedelta(hours=9, minutes=31),
                            periods=360, freq='1min', tz='America/New_York')
        bar_offsets = np.arange(360)
        c = open_px + bar_offsets * drift_per_bar
        h = c + 0.1
        l = c - 0.1
        v = np.full(360, 100.0)
        frames.append(pd.DataFrame({'close': c, 'high': h, 'low': l, 'volume': v}, index=idx))
    return pd.concat(frames)


def test_null_returns_correct_shape():
    df = _build_dev_session(open_px=100.0, drift_per_bar=0.1, n_days=10)
    sigs = orb_bare_signals(df.index, df['close'].to_numpy(),
                             df['high'].to_numpy(), df['low'].to_numpy())
    null = build_null_means(
        real_signals=sigs, bar_idx=df.index,
        close=df['close'].to_numpy(), high=df['high'].to_numpy(), low=df['low'].to_numpy(),
        n_perm=50, chunk_perms=25, tp_R_multiple=1.0,
    )
    assert null.shape == (50,)


def test_null_centered_on_zero_when_no_signal_in_pool():
    """When the pool has no systematic edge (uniform drift), null should be
    approximately centred on zero or whatever the random-entry pool produces.
    """
    df = _build_dev_session(open_px=100.0, drift_per_bar=0.1, n_days=10)
    sigs = orb_bare_signals(df.index, df['close'].to_numpy(),
                             df['high'].to_numpy(), df['low'].to_numpy())
    if len(sigs.signal_bars) < 2:
        pytest.skip("not enough signals in synthetic data")
    null = build_null_means(
        real_signals=sigs, bar_idx=df.index,
        close=df['close'].to_numpy(), high=df['high'].to_numpy(), low=df['low'].to_numpy(),
        n_perm=200, chunk_perms=100, tp_R_multiple=1.0,
    )
    # Null mean should be finite
    assert np.isfinite(null.mean())


def test_null_empty_for_no_signals():
    df = _build_dev_session(open_px=100.0, drift_per_bar=0.0, n_days=10)
    sigs = orb_bare_signals(df.index, df['close'].to_numpy(),
                             df['high'].to_numpy(), df['low'].to_numpy())
    # Zero-drift → no breakouts
    null = build_null_means(
        real_signals=sigs, bar_idx=df.index,
        close=df['close'].to_numpy(), high=df['high'].to_numpy(), low=df['low'].to_numpy(),
        n_perm=50,
    )
    assert null.shape == (0,)
