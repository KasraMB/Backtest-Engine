"""Unit tests for backtest/indicators/* — Stage 2 of IVB v9-final plan."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.indicators.atr import atr_daily, daily_ohlc_from_1m, broadcast_daily_to_1m
from backtest.indicators.value_area import volume_profile, value_area, orb_value_area, _fd_bin_count
from backtest.indicators.quantile import per_state_quantile, bootstrap_quantile_ci, bootstrap_mean_ci, ci_width_pct_of_threshold


# ── ATR ──────────────────────────────────────────────────────────────────────

def _make_1m_fixture(n_days: int = 25, bars_per_day: int = 390, base_price: float = 100.0, daily_range: float = 2.0):
    """Synthetic 1-min bars with deterministic daily True Range = daily_range."""
    rows = []
    start = pd.Timestamp('2020-01-02 09:30:00', tz='America/New_York')
    for d in range(n_days):
        day_start = start + pd.Timedelta(days=d)
        # Drift base price each day so prev_close is set
        day_open  = base_price + d * 0.5
        day_high  = day_open + daily_range
        day_low   = day_open
        day_close = day_open + daily_range * 0.5
        for b in range(bars_per_day):
            ts = day_start + pd.Timedelta(minutes=b)
            if b == 0:
                o, h, l, c = day_open, day_open, day_open, day_open
            elif b == 100:
                o, h, l, c = day_open + 0.5, day_high, day_open, day_high
            elif b == 200:
                o, h, l, c = day_high, day_high, day_low, day_low
            else:
                o = h = l = c = day_close
            rows.append((ts, o, h, l, c, 100))
    df = pd.DataFrame(rows, columns=['ts','open','high','low','close','volume']).set_index('ts')
    return df


def test_daily_ohlc_aggregation():
    df = _make_1m_fixture(n_days=3)
    daily = daily_ohlc_from_1m(df)
    assert len(daily) == 3
    # Each fixture day: open=100+d*0.5, high = open+2.0, low = open, close = open+1.0
    assert daily['high'].iloc[0] == pytest.approx(102.0)
    assert daily['low'].iloc[0] == pytest.approx(100.0)


def test_atr_daily_smoothing():
    df = _make_1m_fixture(n_days=25, daily_range=2.0)
    atr = atr_daily(df, window=20)
    # Day 0 TR = NaN (no prev_close); need 20 valid TR → first valid ATR is iloc[20]
    assert atr.iloc[:20].isna().all()
    # From iloc[20] onward, ATR should reflect 2.0 daily TR
    assert atr.iloc[20] == pytest.approx(2.0, abs=1e-9)


def test_broadcast_no_lookahead():
    df = _make_1m_fixture(n_days=5)
    atr = atr_daily(df, window=3)
    arr = broadcast_daily_to_1m(atr, df.index)
    # First day: no prior daily value → NaN
    assert np.isnan(arr[0])
    # Day 2 bars should equal day 1's ATR (prior-day shift)
    day_bounds = pd.DatetimeIndex(df.index).normalize()
    unique_days = day_bounds.unique().sort_values()
    day2_mask = day_bounds == unique_days[1]
    day1_atr = atr.loc[unique_days[0]]
    if not np.isnan(day1_atr):
        assert np.allclose(arr[day2_mask], day1_atr, equal_nan=True)


# ── Volume Profile + Value Area ──────────────────────────────────────────────

def test_fd_bin_count_reasonable():
    # 1000 normally-distributed prices, tick=0.25
    rng = np.random.default_rng(42)
    prices = rng.normal(100.0, 2.0, 1000)
    n_bins = _fd_bin_count(prices, tick_size=0.25)
    # Should pick something like 30-100 bins for this width
    assert 10 <= n_bins <= 200


def test_volume_profile_basic():
    prices = np.array([100, 101, 102, 101, 101, 100], dtype=np.float64)
    volumes = np.array([10, 50, 5, 50, 50, 10], dtype=np.float64)
    prof = volume_profile(prices, volumes, tick_size=1.0, bins=3)
    # POC should be the 101 bin (volume = 150)
    assert prof.bin_volumes.max() == 150.0
    assert prof.poc == pytest.approx(101.0, abs=0.51)


def test_value_area_70pct_uniform():
    # Uniform distribution: VA should cover ~70% of the range symmetrically
    prices = np.linspace(100, 110, 1000)
    volumes = np.ones_like(prices)
    prof = volume_profile(prices, volumes, tick_size=0.1, bins=100)
    poc, vah, val = value_area(prof, va_pct=0.70)
    width = vah - val
    full = 110 - 100
    # Should be ~70% of full range
    assert 0.65 * full <= width <= 0.75 * full


def test_value_area_total_accumulates():
    rng = np.random.default_rng(7)
    prices = rng.normal(100.0, 1.0, 500)
    volumes = rng.integers(1, 100, 500).astype(np.float64)
    prof = volume_profile(prices, volumes, tick_size=0.25)
    _, vah, val = value_area(prof, va_pct=0.70)
    assert val <= prof.poc <= vah


def test_orb_value_area_wrapper():
    rng = np.random.default_rng(11)
    close = 100.0 + rng.normal(0, 1, 1000).cumsum() * 0.01
    vol = rng.integers(1, 100, 1000).astype(np.float64)
    poc, vah, val = orb_value_area(close, vol, 0, 100, tick_size=0.25, va_pct=0.70)
    assert val <= poc <= vah


# ── Per-state quantile + bootstrap CI ────────────────────────────────────────

def test_per_state_quantile_separates_states():
    rng = np.random.default_rng(13)
    # 3 states: state 0 has small values, state 1 medium, state 2 large
    # All states have ≥30 obs (MIN_OBS threshold)
    vals = np.concatenate([rng.normal(0, 1, 200), rng.normal(5, 1, 200), rng.normal(10, 1, 200)])
    labels = np.concatenate([np.zeros(200), np.ones(200), 2 * np.ones(200)]).astype(int)
    q = per_state_quantile(vals, labels, q=0.90, n_states=3)
    assert q[0] < q[1] < q[2]
    # state 0 ~90th pctile of N(0,1) ≈ 1.28
    assert 0.8 < q[0] < 1.8


def test_per_state_quantile_small_state_nan():
    # MIN_OBS=30: state 0 has 40 obs (valid), state 1 has 25 obs (below threshold)
    vals = np.concatenate([np.arange(40, dtype=float), np.arange(100, 125, dtype=float)])
    labels = np.concatenate([np.zeros(40, dtype=int), np.ones(25, dtype=int)])
    q = per_state_quantile(vals, labels, q=0.50, n_states=2)
    assert np.isnan(q[1])
    assert not np.isnan(q[0])


def test_bootstrap_ci_brackets_point():
    rng = np.random.default_rng(17)
    vals = rng.normal(0, 1, 500)
    point = np.quantile(vals, 0.90)
    lo, hi = bootstrap_quantile_ci(vals, q=0.90, n_boot=500, seed=2026)
    assert lo < point < hi


def test_bootstrap_mean_ci_brackets_true_mean():
    rng = np.random.default_rng(7)
    vals = rng.normal(0.5, 1.0, 500)
    lo, hi = bootstrap_mean_ci(vals, n_boot=1000, ci=0.95)
    assert lo < vals.mean() < hi
    # SE of mean should be ~ 1/sqrt(500) ≈ 0.045; CI width ≈ 1.96 × 2 × 0.045 ≈ 0.176
    assert 0.10 < (hi - lo) < 0.30


def test_bootstrap_mean_ci_one_sided_lower():
    rng = np.random.default_rng(8)
    vals = rng.normal(0.0, 1.0, 500)
    lo, hi = bootstrap_mean_ci(vals, n_boot=1000, ci=0.95, one_sided=True)
    assert hi == float('inf')
    # Lower bound is 5th percentile of bootstrap means — should be < 0 for centred data
    assert lo < 0


def test_ci_width_pct_decreases_with_n():
    rng = np.random.default_rng(19)
    small = rng.normal(5, 1, 50)
    large = rng.normal(5, 1, 5000)
    w_small = ci_width_pct_of_threshold(small, q=0.90, n_boot=200)
    w_large = ci_width_pct_of_threshold(large, q=0.90, n_boot=200)
    assert w_large < w_small
