from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from backtest.propfirm.correlated_mc import (
    StrategyConfig,
    AccountSlot,
    AccountManagementStrategy,
    _sample_regime_sequences,
)
from backtest.regime.hmm import RegimeResult


def _make_cfg(name="A", entry_time_min=30, eval_risk=0.2, fund_risk=0.4):
    return StrategyConfig(
        name=name,
        pnl_pts_by_regime={0: np.array([5.0, -3.0], dtype=np.float32)},
        sl_dists_by_regime={0: np.array([4.0, 4.0], dtype=np.float32)},
        tpd_by_regime={0: 0.5},
        entry_time_min=entry_time_min,
        eval_risk=eval_risk,
        fund_risk=fund_risk,
    )


def _make_regime_result(trans: np.ndarray, labels: dict[date, int] | None = None) -> RegimeResult:
    """Build a minimal RegimeResult for testing."""
    n = trans.shape[0]
    if labels is None:
        # Uniform empirical distribution
        labels = {date(2020, 1, i + 1): i % n for i in range(n * 10)}
    return RegimeResult(
        labels=labels,
        label_names={i: str(i) for i in range(n)},
        train_end_date=None,
        in_sample_dates=[],
        out_of_sample_dates=[],
        transition_matrix=trans,
        state_means=np.zeros(n),
        state_stds=np.ones(n),
        avg_duration_days={str(i): 5.0 for i in range(n)},
        n_states=n,
    )


class TestDataclasses:
    def test_strategy_config_regime_keys(self):
        cfg = _make_cfg()
        assert 0 in cfg.pnl_pts_by_regime
        assert cfg.eval_risk == 0.2

    def test_account_slot_priority_order(self):
        cfg_a = _make_cfg("A")
        cfg_b = _make_cfg("B")
        slot = AccountSlot([cfg_a, cfg_b])
        assert slot.configs[0].name == "A"
        assert slot.configs[1].name == "B"

    def test_account_management_strategy_fields(self):
        s = AccountManagementStrategy(
            trigger="greedy", max_concurrent=3, reserve_n_evals=1, stagger_days=0
        )
        assert s.trigger == "greedy"
        assert s.max_concurrent == 3


class TestRegimeSequences:
    def _rr(self, trans=None):
        if trans is None:
            trans = np.array([[0.7, 0.2, 0.1],
                               [0.1, 0.8, 0.1],
                               [0.1, 0.2, 0.7]])
        return _make_regime_result(trans)

    def test_shape(self):
        seqs = _sample_regime_sequences(self._rr(), 100, 84, np.random.default_rng(0))
        assert seqs.shape == (100, 84)

    def test_valid_states(self):
        seqs = _sample_regime_sequences(self._rr(), 200, 84, np.random.default_rng(1))
        assert int(seqs.min()) >= 0
        assert int(seqs.max()) <= 2

    def test_absorbing_state(self):
        # All labels are regime 0 → initial_dist = [1, 0, 0]
        labels = {date(2020, 1, i + 1): 0 for i in range(30)}
        rr = _make_regime_result(np.eye(3), labels=labels)
        seqs = _sample_regime_sequences(rr, 10, 10, np.random.default_rng(0))
        assert np.all(seqs == 0), "Absorbing state 0 should persist"

    def test_two_regime_transitions(self):
        # Always transition: 0→1, 1→0
        trans = np.array([[0.0, 1.0], [1.0, 0.0]])
        # All labels are regime 0 → starts at 0
        labels = {date(2020, 1, i + 1): 0 for i in range(20)}
        rr = _make_regime_result(trans, labels=labels)
        seqs = _sample_regime_sequences(rr, 5, 6, np.random.default_rng(0))
        expected = np.tile([0, 1, 0, 1, 0, 1], (5, 1))
        np.testing.assert_array_equal(seqs, expected)

    def test_initial_dist_from_labels(self):
        # 20 days of regime 0, 80 days of regime 1 → initial_dist ≈ [0.2, 0.8]
        labels = {}
        for i in range(20):
            labels[date(2020, 1, i + 1)] = 0
        base = date(2020, 4, 1)
        from datetime import timedelta
        for i in range(80):
            labels[base + timedelta(days=i)] = 1
        trans = np.array([[0.5, 0.5], [0.5, 0.5]])
        rr = _make_regime_result(trans, labels=labels)
        seqs = _sample_regime_sequences(rr, 10_000, 1, np.random.default_rng(0))
        regime_0_frac = (seqs[:, 0] == 0).mean()
        assert abs(regime_0_frac - 0.2) < 0.03, f"Expected ~0.2, got {regime_0_frac}"
