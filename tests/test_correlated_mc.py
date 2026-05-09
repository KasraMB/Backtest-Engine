from __future__ import annotations

import numpy as np
import pytest

from backtest.propfirm.correlated_mc import (
    RegimeModel,
    StrategyConfig,
    AccountSlot,
    AccountManagementStrategy,
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
