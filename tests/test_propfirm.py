"""Tests for backtest/propfirm/lucidflex.py."""
from __future__ import annotations
import numpy as np
from backtest.propfirm.lucidflex import (
    LUCIDFLEX_ACCOUNTS,
    MICRO_COMM_RT,
    MINI_COMM_RT,
    MICRO_POINT_VALUE,
    MINI_POINT_VALUE,
    min_contracts_for_winning_day,
    run_propfirm_grid,
    simulate_eval_batch,
    simulate_funded_batch,
)

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


class TestWinningDayMin:
    """Lucid: a day counts toward the 5-day payout requirement only if
    daily PnL ≥ winning_day_min (account-specific)."""

    def test_account_fields_populated(self):
        assert LUCIDFLEX_ACCOUNTS['25K'].winning_day_min  == 100
        assert LUCIDFLEX_ACCOUNTS['50K'].winning_day_min  == 150
        assert LUCIDFLEX_ACCOUNTS['100K'].winning_day_min == 200
        assert LUCIDFLEX_ACCOUNTS['150K'].winning_day_min == 250

    def test_funded_under_threshold_does_not_count(self):
        """Pool of tiny wins (always +1 pt, sl=2pt) on 50K micros: per-trade
        gross win = $2/contract, net = $1. At 1 micro/trade with tpd=1, no day
        can clear $150 — so no payouts should ever fire."""
        # Construct a degenerate pool: every trade is +1pt with sl=2pt
        pnl = np.full(50, 1.0,  dtype=np.float32)
        sl  = np.full(50, 2.0,  dtype=np.float32)
        acc = LUCIDFLEX_ACCOUNTS['50K']
        rng = np.random.default_rng(RNG_SEED)
        sb  = np.full(200, float(acc.starting_balance), dtype=np.float32)

        # Force micros-only with risk_pct=0.005 → $10 risk → 1 micro
        out = simulate_funded_batch(
            pnl, sl, acc,
            risk_pct=0.005, scheme='fixed_dollar', sizing_mode='micros',
            rng=rng, trades_per_day=1.0, starting_balances=sb,
        )
        _, _, n_payouts, _, _, _ = out
        assert n_payouts.sum() == 0, \
            f"Expected no payouts (wins don't clear $150), got {n_payouts.sum()}"


class TestMinContractsForWinningDay:
    """Sizing helper: minimum contracts to clear net winning-day threshold."""

    def test_returns_zero_for_zero_edge(self):
        acc = LUCIDFLEX_ACCOUNTS['50K']
        n_min, n_mic, risk = min_contracts_for_winning_day(0.0, 5.0, acc)
        assert (n_min, n_mic, risk) == (0, 0, 0.0)

        n_min, n_mic, risk = min_contracts_for_winning_day(10.0, 0.0, acc)
        assert (n_min, n_mic, risk) == (0, 0, 0.0)

    def test_micro_preferred_for_low_threshold(self):
        """50K needs $150 net. With avg_win=10pt:
          - 1 micro gross = $20, net = $19 → need ceil(150/19)=8 micros
          - 1 mini gross = $200, net = $196.5 → need 1 mini
          - micro risk ≈ 8 * 5 * 2 = $80 ; mini risk ≈ 1 * 5 * 20 = $100
          → micros wins on lower $ risk."""
        acc = LUCIDFLEX_ACCOUNTS['50K']
        n_min, n_mic, risk = min_contracts_for_winning_day(
            avg_win_pts=10.0, avg_sl_pts=5.0, account=acc,
        )
        assert n_min == 0
        assert n_mic == 8
        # Risk = 8 micros × 5 pts × $2/pt = $80
        assert abs(risk - 80.0) < 1e-6

    def test_mini_when_threshold_high_relative_to_avg_win(self):
        """150K needs $250 net. With small avg_win (e.g. 3pt):
          - 1 micro gross = $6, net = $5 → 50 micros, risk = 50*3*2 = $300
          - 1 mini gross = $60, net = $56.5 → 5 minis, risk = 5*3*20 = $300
          Roughly equal — but contract caps may force one or the other."""
        acc = LUCIDFLEX_ACCOUNTS['150K']
        n_min, n_mic, risk = min_contracts_for_winning_day(
            avg_win_pts=3.0, avg_sl_pts=3.0, account=acc,
        )
        # Either is acceptable; verify the chosen sizing actually clears threshold
        if n_min > 0:
            net = n_min * (3.0 * MINI_POINT_VALUE - MINI_COMM_RT)
        else:
            net = n_mic * (3.0 * MICRO_POINT_VALUE - MICRO_COMM_RT)
        assert net >= acc.winning_day_min, f"Sizing failed: net={net} < {acc.winning_day_min}"

    def test_at_least_one_contract_when_threshold_clearable(self):
        """If 1 contract net win >> threshold, returns exactly 1 of that contract."""
        acc = LUCIDFLEX_ACCOUNTS['25K']  # $100 threshold
        # avg_win = 100pt → 1 mini = $2000 net, 1 micro = $200 net
        n_min, n_mic, risk = min_contracts_for_winning_day(100.0, 50.0, acc)
        # Picks micros for lower $ risk: 1 micro, risk = 50 * 2 = $100
        assert n_mic == 1
        assert n_min == 0
        assert abs(risk - 100.0) < 1e-6
