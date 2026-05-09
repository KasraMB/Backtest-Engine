from __future__ import annotations

import numpy as np
import pytest

from backtest.propfirm.correlated_mc import (
    RegimeModel,
    StrategyConfig,
    AccountSlot,
    AccountManagementStrategy,
    _sample_regime_sequences,
)


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


class TestDataclasses:
    def test_regime_model_fields(self):
        rm = RegimeModel(n_regimes=3, transition=np.eye(3), initial_dist=np.full(3, 1/3))
        assert rm.n_regimes == 3
        assert rm.transition.shape == (3, 3)
        assert rm.initial_dist.shape == (3,)

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
    def _rm(self, trans=None):
        if trans is None:
            trans = np.array([[0.7, 0.2, 0.1],
                               [0.1, 0.8, 0.1],
                               [0.1, 0.2, 0.7]])
        return RegimeModel(3, trans, np.full(3, 1/3))

    def test_shape(self):
        seqs = _sample_regime_sequences(self._rm(), 100, 84, np.random.default_rng(0))
        assert seqs.shape == (100, 84)

    def test_valid_states(self):
        seqs = _sample_regime_sequences(self._rm(), 200, 84, np.random.default_rng(1))
        assert int(seqs.min()) >= 0
        assert int(seqs.max()) <= 2

    def test_absorbing_state(self):
        rm = RegimeModel(3, np.eye(3), np.array([1.0, 0.0, 0.0]))
        seqs = _sample_regime_sequences(rm, 10, 10, np.random.default_rng(0))
        assert np.all(seqs == 0), "Absorbing state 0 should persist"

    def test_two_regime_transitions(self):
        # Always transition: 0→1, 1→0
        trans = np.array([[0.0, 1.0], [1.0, 0.0]])
        rm = RegimeModel(2, trans, np.array([1.0, 0.0]))
        seqs = _sample_regime_sequences(rm, 5, 6, np.random.default_rng(0))
        expected = np.tile([0, 1, 0, 1, 0, 1], (5, 1))
        np.testing.assert_array_equal(seqs, expected)
