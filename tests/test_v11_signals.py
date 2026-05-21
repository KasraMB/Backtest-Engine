"""Unit tests for backtest/ivb/v11_signals.py — IVB v11 fade signal generator."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.ivb.v11_signals import fade_signals, _compute_orb_vp, V11Signals


def _build_session(open_px: float, orb_pattern: str = 'tight', n_days: int = 25,
                   intraday_pattern: str = 'flat'):
    """Synthetic NQ-style sessions with controllable ORB + intraday behaviour."""
    base = pd.Timestamp('2021-01-04', tz='America/New_York')   # Monday Jan 4 2021
    frames = []
    d = 0
    while len(frames) < n_days:
        day_start = base + pd.Timedelta(days=d)
        d += 1
        if day_start.dayofweek >= 5:
            continue
        idx = pd.date_range(day_start + pd.Timedelta(hours=9, minutes=31),
                            periods=360, freq='1min', tz='America/New_York')

        # ORB pattern: 'tight' (consolidation) or 'wide' (trend day)
        if orb_pattern == 'tight':
            # Tight ORB: 1-pt range, oscillating around open
            orb_pattern_arr = np.tile([0, 0.5, 0, -0.5], 8)[:30]
        else:
            # Wide ORB: 5-pt range, strongly trending
            orb_pattern_arr = np.linspace(0, 5, 30)

        # Intraday pattern AFTER ORB (bars 30..359)
        if intraday_pattern == 'fade_long':
            # Drop to test VAL, then rejection bar (low at VAL, close > open)
            # Then drift up toward POC
            post_orb = np.concatenate([
                np.linspace(0, -1.5, 30),     # drop below ORB low to VAL area
                np.array([-1.0]),              # rejection bar (we'll set close > open)
                np.linspace(-1.0, 0.0, 100),   # drift toward POC (open price)
                np.full(199, 0.0),
            ])
        elif intraday_pattern == 'fade_short':
            post_orb = np.concatenate([
                np.linspace(0, 1.5, 30),
                np.array([1.0]),
                np.linspace(1.0, 0.0, 100),
                np.full(199, 0.0),
            ])
        else:  # flat
            post_orb = np.full(330, 0.0)

        full = np.concatenate([orb_pattern_arr, post_orb])
        c = open_px + full
        # Construct open: bar i open = bar (i-1) close; first bar open == ORB[0]
        o = np.empty(360)
        o[0] = open_px
        o[1:] = c[:-1]
        # If intraday says "fade_long", make bar 30 (first post-ORB) a rejection candle
        if intraday_pattern == 'fade_long' and 30 < len(c):
            # Find the deepest bar (lowest close) in bars 30-60 and make next bar's open lower than close
            # Easier: explicit rejection at bar 60 (after the drop)
            o[60] = c[60] - 0.5
            c[60] = c[60] + 0.5   # close > open → green rejection candle at low
            # Also set low to match the deep level
        h = np.maximum(o, c) + 0.05
        l = np.minimum(o, c) - 0.05
        # Boost low at fade trigger bar to match VAL touch
        if intraday_pattern == 'fade_long' and 60 < len(l):
            l[60] = min(l[60], open_px - 1.5)   # below VAL
        if intraday_pattern == 'fade_short' and 60 < len(h):
            h[60] = max(h[60], open_px + 1.5)
            o[60] = c[60] + 0.5
            c[60] = c[60] - 0.5   # close < open → red rejection candle at high
        v = np.full(360, 100.0)
        frames.append(pd.DataFrame({'open': o, 'close': c, 'high': h, 'low': l, 'volume': v},
                                    index=idx))
    return pd.concat(frames)


def test_compute_orb_vp_basic():
    closes = np.array([100.0, 100.25, 100.50, 100.25, 100.0, 100.25,
                       100.50, 100.75, 101.0, 100.5] * 3)
    volumes = np.ones_like(closes) * 100
    res = _compute_orb_vp(closes, volumes)
    assert res is not None
    poc, vah, val = res
    assert val <= poc <= vah
    # POC should be around 100.5 (most touched)
    assert 100.25 <= poc <= 100.75


def test_compute_orb_vp_degenerate_returns_none():
    closes = np.full(30, 100.0)
    volumes = np.ones(30)
    res = _compute_orb_vp(closes, volumes)
    assert res is None   # all same price → single bin


def test_fade_signals_no_signal_on_trending_days():
    df = _build_session(open_px=15000.0, orb_pattern='wide', n_days=10,
                         intraday_pattern='flat')
    sigs = fade_signals(df.index,
                         df['open'].to_numpy(), df['close'].to_numpy(),
                         df['high'].to_numpy(), df['low'].to_numpy(),
                         df['volume'].to_numpy())
    # Wide ORB → not consolidation → no fades
    assert len(sigs.signal_bars) == 0


def test_fade_signals_empty_when_atr_unavailable():
    """First 20 days have no ATR — signals shouldn't fire."""
    df = _build_session(open_px=15000.0, orb_pattern='tight', n_days=15,
                         intraday_pattern='fade_long')
    sigs = fade_signals(df.index,
                         df['open'].to_numpy(), df['close'].to_numpy(),
                         df['high'].to_numpy(), df['low'].to_numpy(),
                         df['volume'].to_numpy())
    # With only 15 days and ATR_20 needs 20 days, all sigs should fail
    assert len(sigs.signal_bars) == 0


def test_fade_signals_smoke():
    """Smoke test: 30 days, tight ORB pattern. Just verify function runs."""
    df = _build_session(open_px=15000.0, orb_pattern='tight', n_days=30,
                         intraday_pattern='flat')
    sigs = fade_signals(df.index,
                         df['open'].to_numpy(), df['close'].to_numpy(),
                         df['high'].to_numpy(), df['low'].to_numpy(),
                         df['volume'].to_numpy())
    # Smoke: just runs without error; signal count may be 0 or small
    assert isinstance(sigs, V11Signals)
    # All output arrays same length
    n = len(sigs.signal_bars)
    assert len(sigs.direction) == n
    assert len(sigs.sl_price) == n
    assert len(sigs.tp_price) == n
