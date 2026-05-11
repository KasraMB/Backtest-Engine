"""Tests for backtest/propfirm/lucidflex.py."""
from __future__ import annotations
import numpy as np
from backtest.propfirm.lucidflex import LUCIDFLEX_ACCOUNTS, simulate_eval_batch, run_propfirm_grid

RNG_SEED = 7

def _trade_pool(n=500, seed=0):
    rng = np.random.default_rng(seed)
    pnl = rng.normal(0.0, 5.0, n)
    sl  = np.abs(rng.normal(4.0, 1.0, n)).clip(0.25)
    return pnl.astype(np.float32), sl.astype(np.float32)


class TestSimulateEvalBatch:

    def test_pass_rate_in_range(self):
        pnl, sl = _trade_pool()
        acc = LUCIDFLEX_ACCOUNTS['25K']
        rng = np.random.default_rng(RNG_SEED)
        passed, _, _ = simulate_eval_batch(
            pnl, sl, acc, 0.20, 'fixed_dollar', 'micros', rng, 1.5, 500,
        )
        rate = passed.mean()
        assert 0.0 <= rate <= 1.0, f"Pass rate {rate} out of [0,1]"

    def test_all_schemes_run(self):
        from backtest.propfirm.lucidflex import RISK_GEOMETRIES
        pnl, sl = _trade_pool()
        acc = LUCIDFLEX_ACCOUNTS['25K']
        for scheme in RISK_GEOMETRIES:
            rng = np.random.default_rng(RNG_SEED)
            passed, _, _ = simulate_eval_batch(
                pnl, sl, acc, 0.30, scheme, 'micros', rng, 1.5, 200,
            )
            assert len(passed) == 200


class TestRunPropfirmGrid:

    def test_grid_returns_all_schemes(self):
        from backtest.propfirm.lucidflex import RISK_GEOMETRIES

        class _FakeTrade:
            def __init__(self):
                self.entry_price = 19000.0
                self.exit_price  = 19050.0
                self.direction   = 1
                self.sl_price    = 18950.0
                self.entry_bar   = 0

        trades = [_FakeTrade() for _ in range(100)]
        acc    = LUCIDFLEX_ACCOUNTS['25K']
        result = run_propfirm_grid(trades, acc, n_sims=100, n_workers=1)
        for scheme in RISK_GEOMETRIES:
            assert scheme in result
