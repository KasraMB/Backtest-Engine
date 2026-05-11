"""Tests for backtest/propfirm/three_phase_mc.py — AnchoredMeanReversion 3-phase simulator."""
from __future__ import annotations
import numpy as np
import pytest

from backtest.propfirm.lucidflex import LUCIDFLEX_ACCOUNTS
from backtest.propfirm.three_phase_mc import (
    ThreePhaseConfig,
    three_phase_reinvestment_mc,
)


def _strong_config(name='strong', n=200, seed=0):
    """Edge: +R/0.5 winrate, large gains, used to reliably pass evals."""
    rng = np.random.default_rng(seed)
    pnl = rng.normal(loc=4.0, scale=8.0, size=n)
    sl  = np.abs(rng.normal(5.0, 1.0, n)).clip(2.0)
    return ThreePhaseConfig(
        name=name, pnl_pts=pnl.astype(np.float32), sl_dists=sl.astype(np.float32),
        tpd=1.0,
    )


def _high_wr_config(name='hi_wr', n=200, seed=1):
    """Higher WR but smaller wins: better for winning-day farming."""
    rng = np.random.default_rng(seed)
    # 75% wins of ~+3pt, 25% losses of ~-3pt → avg ~+1.5pt
    is_win = rng.random(n) < 0.75
    pnl = np.where(is_win, np.abs(rng.normal(3.0, 0.5, n)), -np.abs(rng.normal(3.0, 0.5, n)))
    sl  = np.full(n, 3.0)
    return ThreePhaseConfig(
        name=name, pnl_pts=pnl.astype(np.float32), sl_dists=sl.astype(np.float32),
        tpd=1.0,
    )


class TestBasic:
    def test_returns_expected_keys(self):
        cfg_a = _strong_config(seed=0)
        cfg_b = _high_wr_config(seed=1)
        out = three_phase_reinvestment_mc(
            eval_grind_config=cfg_a, eval_grind_risk=0.30,
            payout_config=cfg_b, account=LUCIDFLEX_ACCOUNTS['50K'],
            budget=3_000, horizon=30, max_concurrent=10, n_sims=200, seed=7,
        )
        for k in ('cash', 'n_payouts', 'n_passed_eval', 'n_hit_target',
                  'withdrawn', 'eval_fees_paid'):
            assert k in out
            assert len(out[k]) == 200

    def test_deterministic(self):
        cfg_a = _strong_config(seed=0)
        cfg_b = _high_wr_config(seed=1)
        kwargs = dict(
            eval_grind_config=cfg_a, eval_grind_risk=0.30,
            payout_config=cfg_b, account=LUCIDFLEX_ACCOUNTS['50K'],
            budget=3_000, horizon=30, max_concurrent=5, n_sims=100, seed=42,
        )
        r1 = three_phase_reinvestment_mc(**kwargs)
        r2 = three_phase_reinvestment_mc(**kwargs)
        np.testing.assert_array_equal(r1['cash'], r2['cash'])
        np.testing.assert_array_equal(r1['n_payouts'], r2['n_payouts'])


class TestEdgeCases:
    def test_zero_edge_no_payouts(self):
        """Break-even strategy with 50% WR should rarely pass evals,
        and even rarer get any payouts."""
        rng = np.random.default_rng(0)
        n = 200
        is_win = rng.random(n) < 0.5
        pnl    = np.where(is_win, 1.0, -1.0)
        cfg = ThreePhaseConfig(
            name='breakeven', pnl_pts=pnl.astype(np.float32),
            sl_dists=np.ones(n, dtype=np.float32), tpd=1.0,
        )
        out = three_phase_reinvestment_mc(
            eval_grind_config=cfg, eval_grind_risk=0.30,
            payout_config=cfg, account=LUCIDFLEX_ACCOUNTS['50K'],
            budget=3_000, horizon=30, max_concurrent=10, n_sims=200, seed=11,
        )
        # Allow a tiny non-zero (random luck) but should be very low
        assert out['n_payouts'].mean() < 0.5, \
            f"Expected near-zero payouts with no edge, got {out['n_payouts'].mean()}"

    def test_strong_edge_some_pass(self):
        cfg = _strong_config(seed=0)
        out = three_phase_reinvestment_mc(
            eval_grind_config=cfg, eval_grind_risk=0.40,
            payout_config=cfg, account=LUCIDFLEX_ACCOUNTS['50K'],
            budget=3_000, horizon=60, max_concurrent=10, n_sims=300, seed=11,
        )
        assert out['n_passed_eval'].sum() > 0, "Strong edge should pass some evals"

    def test_budget_constrains_initial_opens(self):
        """$200 budget on $98/eval account → can only open 2 evals upfront."""
        cfg = _strong_config(seed=0)
        out = three_phase_reinvestment_mc(
            eval_grind_config=cfg, eval_grind_risk=0.30,
            payout_config=cfg, account=LUCIDFLEX_ACCOUNTS['50K'],
            budget=200, horizon=5, max_concurrent=10, n_sims=50, seed=11,
        )
        # 2 evals × $98 = $196 spent on day 0; budget left $4 < $98 so no reinvest
        # All sims should have spent exactly $196 (assuming no eval finished in 5 days)
        # Allow some tolerance — eval could pass fast and refund effectively reinvested... no, eval doesn't refund
        # So total_eval_fees should be 196 minimum (could be more if passed+reinvest)
        assert out['eval_fees_paid'].min() >= 196 - 1e-6


class TestSizing:
    def test_payout_phase_uses_min_winning_day_sizing(self):
        """50K threshold $150. With pnl_pts avg ~+3 (from _high_wr_config),
        sizing should be derived from min_contracts_for_winning_day."""
        cfg_a = _strong_config(seed=0)
        cfg_b = _high_wr_config(seed=1)
        out = three_phase_reinvestment_mc(
            eval_grind_config=cfg_a, eval_grind_risk=0.30,
            payout_config=cfg_b, account=LUCIDFLEX_ACCOUNTS['50K'],
            budget=3_000, horizon=30, max_concurrent=5, n_sims=50, seed=7,
        )
        # Should be sized to clear $150 net
        n_min, n_mic = out['_p3_n_minis'], out['_p3_n_micros']
        assert (n_min > 0) ^ (n_mic > 0), "Must be minis xor micros"
        # Risk should be reasonable for 50K (well below MLL $2K)
        assert 0 < out['_p3_risk'] < 1000, \
            f"Phase 3 risk {out['_p3_risk']} unreasonable"

    def test_eval_grind_sizing_matches_risk_pct(self):
        cfg = _strong_config(seed=0)
        out = three_phase_reinvestment_mc(
            eval_grind_config=cfg, eval_grind_risk=0.30,
            payout_config=cfg, account=LUCIDFLEX_ACCOUNTS['50K'],
            budget=3_000, horizon=10, max_concurrent=5, n_sims=20, seed=7,
        )
        # eg_risk should be ~ 0.30 × 2000 = $600
        assert abs(out['_eg_risk'] - 600.0) < 1e-6
