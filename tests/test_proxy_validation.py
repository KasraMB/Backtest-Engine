"""Unit tests for backtest/ivb/proxy_validation.py — Stage 3 of IVB v9-final."""
from __future__ import annotations

import numpy as np
import pytest

from backtest.ivb.proxy_validation import (
    build_proxy_trades_vec,
    random_entry_null_mean_R,
    evaluate_proxy,
    vxn_ev_separation_test,
)


# ─── Trade builder ───────────────────────────────────────────────────────────

def test_build_trades_long_tp_no_sl_hit():
    # 5 signals; price drifts up linearly; SL never hit; exit at end-of-horizon
    N = 200
    close = np.linspace(100.0, 110.0, N)
    high  = close + 0.1
    low   = close - 0.1
    sl    = np.full(N, 1.0)  # 1pt SL
    sig   = np.array([10, 20, 30, 40, 50], dtype=np.int64)

    trades = build_proxy_trades_vec(sig, close, high, low, sl, direction=1, horizon_bars=20)
    assert len(trades.R) == 5
    assert not trades.hit_sl.any()
    # R > 0 since price rises and SL=1.0pt: 20-bar move on a 100→110 line over 200 bars ≈ 1pt
    assert (trades.R > 0).all()
    assert (trades.mae_R >= 0).all()


def test_build_trades_long_sl_hit():
    # Price drops sharply after entry → SL hit immediately
    N = 100
    close = 100 - np.arange(N) * 0.5     # closes drop 0.5/bar
    high = close + 0.1
    low = close - 0.5
    sl = np.full(N, 1.0)  # SL = 1pt below entry
    sig = np.array([10], dtype=np.int64)

    trades = build_proxy_trades_vec(sig, close, high, low, sl, direction=1, horizon_bars=20)
    assert trades.hit_sl[0]
    # Entry at close[10]=95; sl_price=94; low at bar 11 = close[11]-0.5 = 94.0
    # bar 11 low=94.0 <= sl_price=94.0 → True. So exit_bar=11.
    assert trades.exit_bar[0] == 11
    # R ≈ -1.0 (full SL loss)
    assert trades.R[0] == pytest.approx(-1.0, abs=0.05)


def test_build_trades_short_direction():
    # Price drops monotonically → short should profit
    N = 100
    close = 100 - np.arange(N) * 0.1
    high  = close + 0.05
    low   = close - 0.05
    sl    = np.full(N, 2.0)
    sig   = np.array([10], dtype=np.int64)

    trades = build_proxy_trades_vec(sig, close, high, low, sl, direction=-1, horizon_bars=20)
    assert trades.R[0] > 0  # short profited from price drop


def test_build_trades_drops_late_signals():
    """Signals too close to end-of-array must be dropped."""
    N = 100
    close = np.linspace(100, 101, N)
    high = close + 0.1; low = close - 0.1
    sl = np.full(N, 1.0)
    sig = np.array([10, 80, 95], dtype=np.int64)  # 95 + 20 horizon > 99
    trades = build_proxy_trades_vec(sig, close, high, low, sl, direction=1, horizon_bars=20)
    # Signal at 95: 95+20=115 > N-1=99 → dropped
    # Signal at 80: 80+20=100 == N → also dropped (need <= N-1-horizon = 79)
    assert len(trades.R) == 1
    assert trades.entry_bar[0] == 10


def test_build_trades_invalid_sl_raises_nan():
    N = 100
    close = np.full(N, 100.0); high = close + 0.1; low = close - 0.1
    sl = np.zeros(N)  # all zero → invalid
    sig = np.array([10], dtype=np.int64)
    trades = build_proxy_trades_vec(sig, close, high, low, sl, direction=1, horizon_bars=20)
    assert np.isnan(trades.R[0])


# ─── TP path + EOD cap (v10 extensions) ───────────────────────────────────────

def test_build_trades_tp_hits_before_sl_long():
    # Price drifts up to +1.5R before any SL test → 1R TP fires at intermediate bar
    N = 50
    close = 100 + np.arange(N) * 0.1   # rises 0.1/bar
    high  = close + 0.1
    low   = close - 0.1
    sl    = np.full(N, 1.0)            # SL=1pt, TP=1pt at 1R
    tp    = np.full(N, 1.0)
    sig   = np.array([10], dtype=np.int64)
    trades = build_proxy_trades_vec(sig, close, high, low, sl, direction=1, horizon_bars=20,
                                    tp_dist_per_bar=tp)
    # Entry at close[10]=101; TP price=102; high crosses 102 when close>=101.9 → bar 19
    assert trades.hit_tp[0]
    assert not trades.hit_sl[0]
    assert trades.R[0] == pytest.approx(1.0, abs=0.05)


def test_build_trades_sl_wins_tie_same_bar():
    # Construct a bar where BOTH SL and TP are touched on the same bar.
    # SL must win per v10 micro-spec 9.
    N = 50
    close = np.full(N, 100.0)
    high  = np.full(N, 100.1)
    low   = np.full(N, 99.9)
    # Bar 15 has wide range: high=102, low=98 (both SL=99 and TP=101 within range)
    high[15] = 102.0
    low[15]  = 98.0
    sl = np.full(N, 1.0)   # SL at 99
    tp = np.full(N, 1.0)   # TP at 101
    sig = np.array([10], dtype=np.int64)
    trades = build_proxy_trades_vec(sig, close, high, low, sl, direction=1, horizon_bars=20,
                                    tp_dist_per_bar=tp)
    # SL must win the tie
    assert trades.hit_sl[0]
    assert not trades.hit_tp[0]
    assert trades.R[0] == pytest.approx(-1.0, abs=0.05)


def test_build_trades_eod_cap_forces_exit():
    # Price drifts up slowly, never hits SL or TP within EOD window
    N = 100
    close = 100 + np.arange(N) * 0.01   # tiny drift
    high  = close + 0.05
    low   = close - 0.05
    sl    = np.full(N, 5.0)             # very loose SL
    tp    = np.full(N, 5.0)             # very loose TP
    sig   = np.array([10], dtype=np.int64)
    # Force EOD exit at bar 15 (5 bars after signal)
    max_exit = np.array([15], dtype=np.int64)
    trades = build_proxy_trades_vec(sig, close, high, low, sl, direction=1, horizon_bars=80,
                                    tp_dist_per_bar=tp, max_exit_bar_per_signal=max_exit)
    assert trades.hit_eod[0]
    assert not trades.hit_sl[0]
    assert not trades.hit_tp[0]
    # Exit close ≈ close[15], entry close=100.1 → R = (close[15] - 100.1)/5 ≈ 0.01
    expected_R = (close[15] - 100.1) / 5.0
    assert trades.R[0] == pytest.approx(expected_R, abs=0.01)


def test_build_trades_eod_cap_excludes_post_cap_sl():
    """An SL touch after EOD cap must NOT count — EOD forces earlier exit."""
    N = 50
    close = np.full(N, 100.0)
    high  = np.full(N, 100.05)
    low   = np.full(N, 99.95)
    # SL would hit at bar 25 (far after EOD cap of 15)
    low[25] = 98.0
    sl = np.full(N, 1.0)
    sig = np.array([10], dtype=np.int64)
    max_exit = np.array([15], dtype=np.int64)
    trades = build_proxy_trades_vec(sig, close, high, low, sl, direction=1, horizon_bars=30,
                                    max_exit_bar_per_signal=max_exit)
    assert trades.hit_eod[0]
    assert not trades.hit_sl[0]


def test_build_trades_short_tp_long_short_symmetry():
    # Price drifts DOWN → short TP fires
    N = 50
    close = 100 - np.arange(N) * 0.1
    high  = close + 0.1
    low   = close - 0.1
    sl    = np.full(N, 1.0)
    tp    = np.full(N, 1.0)
    sig   = np.array([10], dtype=np.int64)
    trades = build_proxy_trades_vec(sig, close, high, low, sl, direction=-1, horizon_bars=20,
                                    tp_dist_per_bar=tp)
    assert trades.hit_tp[0]
    assert trades.R[0] == pytest.approx(1.0, abs=0.05)


# ─── Permutation null ────────────────────────────────────────────────────────

# ─── evaluate_proxy ──────────────────────────────────────────────────────────

def test_random_entry_null_centered_on_pool_mean():
    """Random-entry null should be centred on the FULL POOL mean, with
    std ≈ pool_std / sqrt(k)."""
    rng = np.random.default_rng(0)
    pool = rng.normal(0.05, 1.0, 5000)
    null = random_entry_null_mean_R(pool, k_signals=200, n_perm=1000, seed=2026)
    # Centre = pool mean
    assert null.mean() == pytest.approx(pool.mean(), abs=0.05)
    # Std ≈ pool_std / sqrt(k)
    expected_se = pool.std() / np.sqrt(200)
    assert null.std() == pytest.approx(expected_se, rel=0.15)


def test_random_entry_null_small_pool_returns_nan():
    """Pools < 50 obs → NaN signal."""
    pool = np.arange(30, dtype=float)
    null = random_entry_null_mean_R(pool, k_signals=20, n_perm=100)
    assert np.isnan(null[0])


def test_evaluate_proxy_no_edge_vs_pool():
    """When proxy R is sampled from the same distribution as the pool,
    should not pass null."""
    rng = np.random.default_rng(1)
    pool = rng.normal(0.05, 1, 10000)
    proxy = rng.choice(pool, size=200, replace=False)
    res = evaluate_proxy(proxy, pool, direction=1,
                         cost_per_RT_dollars=13.50, avg_sl_dist_pts=25.0)
    assert not res['passes_null']
    assert res['n_trades'] == 200
    assert res['pool_n'] == 10000
    assert res['cost_R'] == pytest.approx(0.027, abs=1e-4)


def test_evaluate_proxy_strong_edge_vs_pool():
    """Proxy R systematically higher than pool → should pass null AND economic."""
    rng = np.random.default_rng(1)
    pool = rng.normal(0.05, 1, 10000)
    proxy_R = rng.normal(0.30, 1, 500)   # strong +0.30R/trade edge
    res = evaluate_proxy(proxy_R, pool, direction=1,
                         cost_per_RT_dollars=13.50, avg_sl_dist_pts=25.0,
                         economic_threshold_R=0.15)
    assert res['passes_null']
    assert res['passes_economic']
    assert res['passes_both']


def test_evaluate_proxy_insufficient_trades():
    pool = np.random.default_rng(2).normal(0, 1, 1000)
    res = evaluate_proxy(np.array([0.5, 1.0, -0.2]), full_R_pool=pool, direction=1,
                         cost_per_RT_dollars=13.50, avg_sl_dist_pts=25.0)
    assert not res['passes_both']
    assert 'insufficient' in res['reason']


def test_evaluate_proxy_invalid_direction_raises():
    pool = np.random.default_rng(3).normal(0, 1, 1000)
    R = np.random.default_rng(4).normal(0.1, 1, 100)
    with pytest.raises(ValueError):
        evaluate_proxy(R, full_R_pool=pool, direction=0,
                       cost_per_RT_dollars=13.50, avg_sl_dist_pts=25.0)


# ─── VXN EV-separation ───────────────────────────────────────────────────────

def test_vxn_ev_separation_detects_real_effect():
    rng = np.random.default_rng(2)
    z = rng.normal(0, 1, 500)
    # Make forward-R systematically higher when z > 0
    R = np.where(z > 0, rng.normal(0.3, 0.5, 500), rng.normal(-0.3, 0.5, 500))
    res = vxn_ev_separation_test(R, z, effect_size_threshold=0.10)
    assert res['passes']
    assert abs(res['delta_mu_R']) > 0.10


def test_vxn_ev_separation_rejects_noise():
    rng = np.random.default_rng(3)
    z = rng.normal(0, 1, 500)
    R = rng.normal(0, 1, 500)   # no relationship to z
    res = vxn_ev_separation_test(R, z, effect_size_threshold=0.10)
    assert not res['passes']


def test_vxn_ev_separation_small_n():
    R = np.array([0.5, 0.3, 0.1]); z = np.array([0.0, 1.0, -1.0])
    res = vxn_ev_separation_test(R, z)
    assert not res['passes']
    assert 'insufficient' in res['reason']
