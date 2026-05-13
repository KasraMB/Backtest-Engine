"""Tests for backtest/propfirm/three_phase_mc.py — AnchoredMeanReversion 3-phase simulator."""
from __future__ import annotations
from datetime import date, timedelta

import numpy as np
import pytest

from backtest.propfirm.lucidflex import LUCIDFLEX_ACCOUNTS
from backtest.propfirm.three_phase_mc import (
    ThreePhaseConfig,
    three_phase_reinvestment_mc,
)
from backtest.regime.hmm import RegimeResult


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


class TestMultiConfig:
    """Multi-session combos: list of configs for P12 and/or P3."""

    def test_list_input_accepted(self):
        cfg_a = _strong_config(name='a', seed=0)
        cfg_b = _strong_config(name='b', seed=1)
        out = three_phase_reinvestment_mc(
            eval_grind_config=[cfg_a, cfg_b], eval_grind_risk=0.30,
            payout_config=[cfg_a, cfg_b],
            account=LUCIDFLEX_ACCOUNTS['50K'],
            budget=3_000, horizon=20, max_concurrent=5, n_sims=100, seed=7,
        )
        assert 'cash' in out
        assert len(out['cash']) == 100

    def test_deterministic_multi_config(self):
        cfg_a = _strong_config(name='a', seed=0)
        cfg_b = _high_wr_config(name='b', seed=1)
        kwargs = dict(
            eval_grind_config=[cfg_a, cfg_b], eval_grind_risk=0.30,
            payout_config=cfg_b, account=LUCIDFLEX_ACCOUNTS['50K'],
            budget=3_000, horizon=20, max_concurrent=5, n_sims=100, seed=42,
        )
        r1 = three_phase_reinvestment_mc(**kwargs)
        r2 = three_phase_reinvestment_mc(**kwargs)
        np.testing.assert_array_equal(r1['cash'], r2['cash'])

    def test_risk_scaling_on_vs_off_changes_results(self):
        """With scale_risk_by_n=True, each session risks 1/N of total.
        With False, each session risks the full amount → much higher daily risk."""
        cfg_a = _strong_config(name='a', seed=0)
        cfg_b = _strong_config(name='b', seed=1)
        kwargs = dict(
            eval_grind_config=[cfg_a, cfg_b], eval_grind_risk=0.30,
            payout_config=cfg_a, account=LUCIDFLEX_ACCOUNTS['50K'],
            budget=3_000, horizon=20, max_concurrent=5, n_sims=200, seed=42,
        )
        r_scaled = three_phase_reinvestment_mc(**kwargs, scale_risk_by_n=True)
        r_full   = three_phase_reinvestment_mc(**kwargs, scale_risk_by_n=False)
        # With full risk per session, each trade is bigger → results differ
        assert not np.allclose(r_scaled['cash'], r_full['cash'])
        # The per-config sizing should be larger when not scaled
        # (eg_risk reflects total daily risk capacity)
        assert r_full['_eg_risk'] > r_scaled['_eg_risk']

    def test_single_config_unchanged_by_refactor(self):
        """Passing a single config (not list) should give same result as before."""
        cfg = _strong_config(seed=0)
        # This was working in pre-refactor tests; ensure still works
        out = three_phase_reinvestment_mc(
            eval_grind_config=cfg, eval_grind_risk=0.30,
            payout_config=cfg, account=LUCIDFLEX_ACCOUNTS['50K'],
            budget=3_000, horizon=20, max_concurrent=5, n_sims=100, seed=7,
        )
        # Should also accept list-of-1 with same result
        out_list = three_phase_reinvestment_mc(
            eval_grind_config=[cfg], eval_grind_risk=0.30,
            payout_config=[cfg], account=LUCIDFLEX_ACCOUNTS['50K'],
            budget=3_000, horizon=20, max_concurrent=5, n_sims=100, seed=7,
        )
        # Same single underlying config + same seed → same output
        np.testing.assert_array_equal(out['cash'], out_list['cash'])


# ── Helpers for regime-aware tests ────────────────────────────────────────

def _make_regime_result(n_days=200, n_states=3, transition=None):
    """Build a minimal RegimeResult with explicit transition matrix."""
    if transition is None:
        transition = np.array([
            [0.7, 0.2, 0.1],
            [0.2, 0.6, 0.2],
            [0.1, 0.2, 0.7],
        ])
    labels = {
        date(2020, 1, 1) + timedelta(days=i): i % n_states
        for i in range(n_days)
    }
    return RegimeResult(
        labels=labels, label_names={0: 'bear', 1: 'neutral', 2: 'bull'},
        train_end_date=None, in_sample_dates=[], out_of_sample_dates=[],
        transition_matrix=transition,
        state_means=np.array([-0.01, 0.0, 0.01]),
        state_stds=np.array([0.02, 0.01, 0.015]),
        avg_duration_days={'bear': 3.3, 'neutral': 2.5, 'bull': 3.3},
        n_states=n_states,
    )


def _regime_config(name='regime_cfg', seed=0):
    """Config with distinct per-regime pools — bull regime wins, bear loses."""
    rng = np.random.default_rng(seed)
    return ThreePhaseConfig(
        name=name,
        # flat fallback (used when regime_result is None)
        pnl_pts=rng.normal(0.5, 5.0, 100).astype(np.float32),
        sl_dists=np.full(100, 5.0, dtype=np.float32),
        tpd=1.0,
        # Distinct regime pools:
        pnl_pts_by_regime={
            0: np.full(50, -5.0, dtype=np.float32),   # bear: lose 5 pts always
            1: np.full(50,  0.0, dtype=np.float32),   # neutral: zero
            2: np.full(50, +5.0, dtype=np.float32),   # bull: win 5 pts always
        },
        sl_dists_by_regime={
            0: np.full(50, 5.0, dtype=np.float32),
            1: np.full(50, 5.0, dtype=np.float32),
            2: np.full(50, 5.0, dtype=np.float32),
        },
    )


class TestRegimeAware:
    """Regime-aware draws via the transition matrix."""

    def test_regime_result_changes_results(self):
        """Passing regime_result should change outputs vs no regime."""
        cfg = _regime_config(seed=0)
        rr  = _make_regime_result()
        kwargs = dict(
            eval_grind_config=cfg, eval_grind_risk=0.30,
            payout_config=cfg, account=LUCIDFLEX_ACCOUNTS['50K'],
            budget=3_000, horizon=30, max_concurrent=5, n_sims=200, seed=11,
        )
        r_no_regime  = three_phase_reinvestment_mc(**kwargs)
        r_w_regime   = three_phase_reinvestment_mc(**kwargs, regime_result=rr)
        assert not np.allclose(r_no_regime['cash'], r_w_regime['cash']), \
            "Regime-aware draws should change outputs vs flat draws"

    def test_regime_aware_deterministic(self):
        """Same seed + same regime_result → same outputs."""
        cfg = _regime_config(seed=0)
        rr  = _make_regime_result()
        kwargs = dict(
            eval_grind_config=cfg, eval_grind_risk=0.30,
            payout_config=cfg, account=LUCIDFLEX_ACCOUNTS['50K'],
            budget=3_000, horizon=30, max_concurrent=5, n_sims=100, seed=42,
            regime_result=rr,
        )
        r1 = three_phase_reinvestment_mc(**kwargs)
        r2 = three_phase_reinvestment_mc(**kwargs)
        np.testing.assert_array_equal(r1['cash'], r2['cash'])

    def test_bear_only_regime_loses(self):
        """If transition_matrix forces bear forever, account loses (since cfg
        has bear=-5 pool). Tests that regime sequence actually influences draws."""
        bear_only = np.eye(3)  # absorbing — start state stays forever
        bear_only[0, 0] = 1.0  # bear → bear
        bear_only[1, 1] = 1.0
        bear_only[2, 2] = 1.0
        rr = _make_regime_result(transition=bear_only)
        # Force initial distribution to be all-bear:
        rr.labels = {d: 0 for d in rr.labels}   # historical labels all bear
        cfg = _regime_config(seed=0)
        out = three_phase_reinvestment_mc(
            eval_grind_config=cfg, eval_grind_risk=0.30,
            payout_config=cfg, account=LUCIDFLEX_ACCOUNTS['50K'],
            budget=3_000, horizon=30, max_concurrent=5, n_sims=200, seed=11,
            regime_result=rr,
        )
        # All trades come from bear pool (-5 pts each) → never pass eval
        assert out['n_passed_eval'].sum() == 0, \
            f"Bear-only regime should fail all evals, got {out['n_passed_eval'].sum()} passes"
        assert out['n_payouts'].sum() == 0

    def test_bull_only_regime_thrives(self):
        """All-bull regime → cfg's bull pool is +5 → evals pass easily."""
        bull_only = np.eye(3)
        rr = _make_regime_result(transition=bull_only)
        rr.labels = {d: 2 for d in rr.labels}   # historical labels all bull
        cfg = _regime_config(seed=0)
        out = three_phase_reinvestment_mc(
            eval_grind_config=cfg, eval_grind_risk=0.30,
            payout_config=cfg, account=LUCIDFLEX_ACCOUNTS['50K'],
            budget=3_000, horizon=30, max_concurrent=5, n_sims=200, seed=11,
            regime_result=rr,
        )
        # Bull pool = +5 pts/trade → should pass evals quickly
        assert out['n_passed_eval'].sum() > 0, "Bull-only regime should pass some evals"

    def _regime_config_b(self, name='cfg_b'):
        """Second config so two-slot tests have a distinct second session."""
        return ThreePhaseConfig(
            name=name,
            pnl_pts=np.array([2.0, -2.0, 3.0, -1.0] * 25, dtype=np.float32),
            sl_dists=np.full(100, 3.0, dtype=np.float32),
            tpd=1.0,
        )

    def test_fallback_when_config_lacks_regime_pools(self):
        """Config without per-regime pools + regime_result should fall back
        to flat-pool draws and produce some output (no crash)."""
        from backtest.propfirm.three_phase_mc import ThreePhaseConfig
        cfg = ThreePhaseConfig(
            name='flat', pnl_pts=np.array([1.0, -1.0, 2.0, -0.5] * 10, dtype=np.float32),
            sl_dists=np.full(40, 5.0, dtype=np.float32), tpd=1.0,
            # No regime pools
        )
        rr = _make_regime_result()
        out = three_phase_reinvestment_mc(
            eval_grind_config=cfg, eval_grind_risk=0.30,
            payout_config=cfg, account=LUCIDFLEX_ACCOUNTS['50K'],
            budget=3_000, horizon=30, max_concurrent=5, n_sims=50, seed=7,
            regime_result=rr,
        )
        assert 'cash' in out
        assert len(out['cash']) == 50


class TestPerSlotConfigs:
    """Per-slot mode: each account slot has its own dedicated (P12, P3) configs."""

    def _strong_p12(self):
        rng = np.random.default_rng(0)
        return ThreePhaseConfig(
            name='strong', pnl_pts=rng.normal(4.0, 8.0, 200).astype(np.float32),
            sl_dists=np.abs(rng.normal(5.0, 1.0, 200)).clip(2.0).astype(np.float32),
            tpd=1.0,
        )

    def _strong_p3(self):
        rng = np.random.default_rng(1)
        return ThreePhaseConfig(
            name='strong_p3', pnl_pts=rng.normal(3.0, 5.0, 200).astype(np.float32),
            sl_dists=np.full(200, 3.0, dtype=np.float32), tpd=1.0,
        )

    def test_per_slot_accepted_and_runs(self):
        p12 = self._strong_p12()
        p3  = self._strong_p3()
        slots = [(p12, p3)] * 5   # max_c = 5 slots, all same
        out = three_phase_reinvestment_mc(
            slot_assignments=slots,
            account=LUCIDFLEX_ACCOUNTS['50K'],
            budget=3_000, horizon=20, max_concurrent=5,
            n_sims=100, seed=7, eval_grind_risk=0.30,
        )
        assert 'cash' in out
        assert len(out['cash']) == 100

    def test_per_slot_deterministic(self):
        p12 = self._strong_p12()
        p3  = self._strong_p3()
        slots = [(p12, p3)] * 5
        kwargs = dict(
            slot_assignments=slots, account=LUCIDFLEX_ACCOUNTS['50K'],
            budget=3_000, horizon=20, max_concurrent=5,
            n_sims=100, seed=42, eval_grind_risk=0.30,
        )
        r1 = three_phase_reinvestment_mc(**kwargs)
        r2 = three_phase_reinvestment_mc(**kwargs)
        np.testing.assert_array_equal(r1['cash'], r2['cash'])

    def test_per_slot_length_mismatch_raises(self):
        p12 = self._strong_p12()
        p3  = self._strong_p3()
        slots = [(p12, p3)] * 3   # only 3 slots
        with pytest.raises(ValueError, match="slot_assignments length"):
            three_phase_reinvestment_mc(
                slot_assignments=slots, account=LUCIDFLEX_ACCOUNTS['50K'],
                budget=3_000, horizon=20, max_concurrent=5,   # max_c=5
                n_sims=50, seed=7, eval_grind_risk=0.30,
            )

    def test_per_slot_missing_inputs_raises(self):
        """No slot_assignments AND no legacy configs → error."""
        with pytest.raises(ValueError, match="slot_assignments OR"):
            three_phase_reinvestment_mc(
                account=LUCIDFLEX_ACCOUNTS['50K'],
                budget=3_000, horizon=20, max_concurrent=5,
                n_sims=50, seed=7, eval_grind_risk=0.30,
            )

    def test_per_slot_diverges_from_uniform(self):
        """Heterogeneous slots (mix of strong + weak configs) should give a
        different result than all-strong slots."""
        rng = np.random.default_rng(2)
        weak = ThreePhaseConfig(
            name='weak', pnl_pts=rng.normal(-2.0, 5.0, 200).astype(np.float32),
            sl_dists=np.full(200, 5.0, dtype=np.float32), tpd=1.0,
        )
        strong = self._strong_p12()
        p3 = self._strong_p3()

        slots_uniform = [(strong, p3)] * 5
        slots_mixed   = [(strong, p3), (strong, p3), (weak, p3), (weak, p3), (weak, p3)]

        out_u = three_phase_reinvestment_mc(
            slot_assignments=slots_uniform, account=LUCIDFLEX_ACCOUNTS['50K'],
            budget=3_000, horizon=30, max_concurrent=5,
            n_sims=200, seed=11, eval_grind_risk=0.30,
        )
        out_m = three_phase_reinvestment_mc(
            slot_assignments=slots_mixed, account=LUCIDFLEX_ACCOUNTS['50K'],
            budget=3_000, horizon=30, max_concurrent=5,
            n_sims=200, seed=11, eval_grind_risk=0.30,
        )
        assert out_u['cash'].mean() > out_m['cash'].mean(), \
            "Uniform-strong should outperform mixed-with-weak"

    def test_per_slot_refund_preserves_session(self):
        """When an account closes and gets refunded into the same slot,
        the new account uses the SAME (P12, P3) configs.  We test this
        indirectly: blow up accounts repeatedly with a guaranteed-bad config
        and verify that the slot's config never 'morphs' — refund just keeps
        eating fees."""
        # Guaranteed-loser config
        bad = ThreePhaseConfig(
            name='bad', pnl_pts=np.full(100, -10.0, dtype=np.float32),
            sl_dists=np.full(100, 1.0, dtype=np.float32), tpd=1.0,
        )
        good = self._strong_p12()
        p3   = self._strong_p3()
        # Slot 0 has the bad config; others have good ones
        slots = [(bad, p3)] + [(good, p3)] * 4

        out = three_phase_reinvestment_mc(
            slot_assignments=slots, account=LUCIDFLEX_ACCOUNTS['50K'],
            budget=3_000, horizon=20, max_concurrent=5,
            n_sims=200, seed=7, eval_grind_risk=0.50,
        )
        # The simulator should run to completion; slot 0 keeps blowing up so
        # avg passed eval should be lower than if all slots were 'good'
        out_good = three_phase_reinvestment_mc(
            slot_assignments=[(good, p3)] * 5, account=LUCIDFLEX_ACCOUNTS['50K'],
            budget=3_000, horizon=20, max_concurrent=5,
            n_sims=200, seed=7, eval_grind_risk=0.50,
        )
        # With 1 bad slot, fewer passes overall
        assert out['n_passed_eval'].mean() < out_good['n_passed_eval'].mean()
