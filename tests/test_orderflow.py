"""Unit tests for backtest/indicators/orderflow.py — Stage 3 of IVB v9-final."""
from __future__ import annotations

import numpy as np
import pytest

from backtest.indicators.orderflow import bar_delta, momentum_signal, absorption_signal


def test_bar_delta_basic():
    bv = np.array([10, 20, 30], dtype=np.float64)
    av = np.array([15, 18, 50], dtype=np.float64)
    d = bar_delta(bv, av)
    assert np.allclose(d, [5, -2, 20])


def test_bar_delta_handles_int_inputs():
    bv = np.array([10, 20, 30])
    av = np.array([15, 18, 50])
    d = bar_delta(bv, av)
    assert d.dtype == np.float64


def test_momentum_signal_long_direction():
    delta = np.array([100, -50, 200, 150, -300], dtype=np.float64)
    labels = np.array([0, 0, 1, 1, 2])
    thresh = {0: 80.0, 1: 175.0, 2: 250.0}
    mask = momentum_signal(delta, labels, thresh, direction=1, n_states=3)
    # Long fires when delta > thresh[state]
    # bar 0: delta=100 > 80  → True
    # bar 1: delta=-50 (state 0) < 80 → False
    # bar 2: delta=200 > 175 → True
    # bar 3: delta=150 < 175 → False
    # bar 4: delta=-300 (state 2) → -300 < 250 → False
    assert np.array_equal(mask, [True, False, True, False, False])


def test_momentum_signal_short_direction():
    delta = np.array([100, -200, -300, 50, -400], dtype=np.float64)
    labels = np.array([0, 0, 1, 1, 2])
    thresh = {0: 80.0, 1: 175.0, 2: 250.0}
    mask = momentum_signal(delta, labels, thresh, direction=-1, n_states=3)
    # Short fires when -delta > thresh[state]
    # bar 0: -100 < 80 → False
    # bar 1: 200 > 80 → True
    # bar 2: 300 > 175 → True
    # bar 3: -50 < 175 → False
    # bar 4: 400 > 250 → True
    assert np.array_equal(mask, [False, True, True, False, True])


def test_momentum_nan_threshold_suppresses():
    delta = np.array([1000, 1000], dtype=np.float64)
    labels = np.array([0, 1])
    thresh = {0: float('nan'), 1: 50.0}
    mask = momentum_signal(delta, labels, thresh, direction=1, n_states=2)
    # State 0 NaN → +inf threshold → never fires
    # State 1 → 1000 > 50 → True
    assert mask[0] == False
    assert mask[1] == True


def test_momentum_invalid_direction_raises():
    with pytest.raises(ValueError):
        momentum_signal(np.array([1.0]), np.array([0]), {0: 0.5}, direction=0, n_states=1)


def test_absorption_signal_basic():
    vol = np.array([100, 1000, 500, 2000], dtype=np.float64)
    pxchg = np.array([0.5, 0.1, 0.4, 5.0], dtype=np.float64)
    labels = np.array([0, 0, 1, 1])
    vol_thresh = {0: 800.0, 1: 1500.0}
    px_thresh  = {0: 0.3, 1: 0.5}
    mask = absorption_signal(vol, pxchg, labels, vol_thresh, px_thresh, n_states=2)
    # Absorption: vol > vol_thresh AND |Δp| < px_thresh
    # bar 0: vol=100  (<800) → False
    # bar 1: vol=1000 (>800) AND pxchg=0.1 (<0.3) → True
    # bar 2: vol=500  (<1500) → False
    # bar 3: vol=2000 (>1500) AND pxchg=5.0 (>0.5) → False
    assert np.array_equal(mask, [False, True, False, False])


def test_absorption_nan_pxchg_thresh_suppresses():
    vol = np.array([1000, 1000], dtype=np.float64)
    pxchg = np.array([0.0, 0.0], dtype=np.float64)
    labels = np.array([0, 1])
    vol_thresh = {0: 500.0, 1: 500.0}
    px_thresh  = {0: float('nan'), 1: 1.0}
    mask = absorption_signal(vol, pxchg, labels, vol_thresh, px_thresh, n_states=2)
    # State 0: pxchg NaN → -inf threshold → 0.0 < -inf is False → suppressed
    # State 1: 1000>500 AND 0.0<1.0 → True
    assert mask[0] == False
    assert mask[1] == True


def test_momentum_perf_1m_bars():
    """Sanity perf bench: 1M bars under 100 ms."""
    import time
    rng = np.random.default_rng(0)
    N = 1_000_000
    delta = rng.normal(0, 100, N)
    labels = rng.integers(0, 3, N)
    thresh = {0: 100.0, 1: 150.0, 2: 200.0}
    t0 = time.perf_counter()
    mask = momentum_signal(delta, labels, thresh, direction=1, n_states=3)
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.1, f"momentum_signal too slow: {elapsed*1000:.1f} ms"
    assert mask.shape == (N,)
