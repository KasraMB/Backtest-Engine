"""Unit tests for backtest/ivb/v10_signals.py — IVB v10 entry generators."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.ivb.v10_signals import (
    orb_bare_signals, orb_poc_retest_signals, prep_day_views,
    ORB_START_MIN, ORB_END_MIN, TRADE_END_MIN,
)


def _build_session_bars(open_px: float, drift_per_bar: float, n_days: int = 3, with_volume: bool = True):
    """Build a synthetic NQ-style 1m session for n_days starting 2020-01-06 (Mon).

    Each session: 9:31 ET → 15:30 ET = 360 bars (6 hours × 60 min).
    Drift is per-bar linear; close == open + i*drift per bar within day.
    """
    rows = []
    base_date = pd.Timestamp('2020-01-06', tz='America/New_York')   # Monday
    for d in range(n_days):
        day_start = base_date + pd.Timedelta(days=d)
        if day_start.dayofweek >= 5:
            continue
        for m in range(31, 31 + 360):   # 9:31 .. 15:30 = 360 bars
            ts = day_start + pd.Timedelta(minutes=m + 540 - 31)
            # Above: 540 = 9:00 ET in minutes; +m offset; we want 9:31 first → m=31 → 540+31-31=540 → 9:00... rebuild:
        # Use simpler approach: build with pd.date_range
        idx = pd.date_range(day_start + pd.Timedelta(hours=9, minutes=31),
                            periods=360, freq='1min', tz='America/New_York')
        bar_offsets = np.arange(360)
        c = open_px + bar_offsets * drift_per_bar
        h = c + 0.05
        l = c - 0.05
        v = np.full(360, 100.0)
        for ts, ci, hi, li, vi in zip(idx, c, h, l, v):
            rows.append((ts, ci, hi, li, vi))

    df = pd.DataFrame(rows, columns=['ts', 'close', 'high', 'low', 'volume']).set_index('ts')
    return df


def test_prep_day_views_basic():
    df = _build_session_bars(open_px=100.0, drift_per_bar=0.0, n_days=2)
    prep = prep_day_views(df.index, df['high'].to_numpy(), df['low'].to_numpy())
    assert prep.valid_day.all()           # both days valid
    assert (prep.orb_n_bars == 30).all()  # each day has 30 ORB bars
    # ORB high = open + 0.05 (since c = constant=100, h=100.05)
    np.testing.assert_allclose(prep.orb_high_per_day, 100.05)
    np.testing.assert_allclose(prep.orb_low_per_day,  99.95)


def test_orb_bare_signals_no_breakout_drift_zero():
    df = _build_session_bars(open_px=100.0, drift_per_bar=0.0, n_days=2)
    sigs = orb_bare_signals(df.index, df['close'].to_numpy(),
                             df['high'].to_numpy(), df['low'].to_numpy())
    # No drift, close stays at 100 — never breaks ORB extremes
    assert len(sigs.signal_bars) == 0


def test_orb_bare_signals_long_breakout_uptrend():
    # Strong uptrend: c[i] = 100 + i*0.5
    df = _build_session_bars(open_px=100.0, drift_per_bar=0.5, n_days=2)
    sigs = orb_bare_signals(df.index, df['close'].to_numpy(),
                             df['high'].to_numpy(), df['low'].to_numpy())
    # Should detect long breakout on each day
    assert len(sigs.signal_bars) == 2
    assert (sigs.direction == 1).all()
    # SL distance > 0
    assert (sigs.sl_dist > 0).all()


def test_orb_bare_signals_short_breakout_downtrend():
    df = _build_session_bars(open_px=100.0, drift_per_bar=-0.5, n_days=2)
    sigs = orb_bare_signals(df.index, df['close'].to_numpy(),
                             df['high'].to_numpy(), df['low'].to_numpy())
    assert len(sigs.signal_bars) == 2
    assert (sigs.direction == -1).all()


def test_orb_bare_one_signal_per_day():
    """ORB bare emits at most ONE breakout per day (long OR short, whichever first)."""
    df = _build_session_bars(open_px=100.0, drift_per_bar=0.5, n_days=5)
    sigs = orb_bare_signals(df.index, df['close'].to_numpy(),
                             df['high'].to_numpy(), df['low'].to_numpy())
    # Check unique day_ords
    assert len(sigs.signal_bars) == len(set(sigs.day_ord.tolist()))


def test_orb_poc_retest_basic_no_retest_on_pure_trend():
    """Pure trend leaves POC at start, no retest occurs after breakout."""
    df = _build_session_bars(open_px=100.0, drift_per_bar=0.5, n_days=2)
    sigs = orb_poc_retest_signals(df.index, df['close'].to_numpy(),
                                   df['high'].to_numpy(), df['low'].to_numpy(),
                                   df['volume'].to_numpy())
    # Pure trend → price never retraces to POC → no signals expected
    assert len(sigs.signal_bars) == 0


def test_orb_poc_retest_eod_set():
    """Smoke: eod_bar field is populated correctly for any signals emitted."""
    df = _build_session_bars(open_px=100.0, drift_per_bar=0.5, n_days=2)
    sigs = orb_bare_signals(df.index, df['close'].to_numpy(),
                             df['high'].to_numpy(), df['low'].to_numpy())
    if len(sigs.signal_bars) == 0:
        pytest.skip("no signals generated, can't test EOD")
    # EOD bar must be >= signal bar
    assert (sigs.eod_bar >= sigs.signal_bars).all()
