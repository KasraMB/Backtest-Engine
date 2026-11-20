"""
Targeted tests for the LucidFlex eval-only consistency rule in
three_phase_reinvestment_mc(enforce_consistency_rule=True).

LucidFlex rule (eval phase only):
  Eval pass requires
    (largest_single_day_pnl <= 0.5 × total_eval_profit) AND
    (n_profitable_days >= 2)

Funded phase (grind, payout) has no consistency requirement.

These tests use synthetic per-slot bootstrap arrays to drive deterministic
sequences and assert exact pass/fail behaviour.
"""
from __future__ import annotations

import numpy as np
import pytest

from backtest.propfirm.lucidflex import LUCIDFLEX_ACCOUNTS
from backtest.propfirm.three_phase_mc import three_phase_reinvestment_mc


ACCOUNT = LUCIDFLEX_ACCOUNTS['50K']
SB      = ACCOUNT.starting_balance       # 50_000
PT      = ACCOUNT.profit_target          # 3_000
MLL     = ACCOUNT.mll_amount             # 2_000


def _build_arrays_from_pnl_sequence(pnl_dollars_seq, risk_dollars=1_800.0,
                                     sl_pts=10.0, n_oos_days=None):
    """Construct (1, n_days) arrays where bootstrapping in cyclic order replays
    the supplied dollar PnL sequence (NET of commission) with one trade/day.

    The MC formula is $PnL_day = R × risk_$ − n_trades × comm_per_trade.
    To make the realized $PnL match the target seq, we pre-inflate R to cancel
    the commission: R = (target_$ + comm_per_trade) / risk_$.
    """
    if n_oos_days is None:
        n_oos_days = len(pnl_dollars_seq)
    # Sizing → commission per trade (mini): n_minis × MINI_COMM_RT
    from backtest.propfirm.lucidflex import MINI_COMM_RT, MINI_POINT_VALUE
    mini_risk_1c = sl_pts * MINI_POINT_VALUE
    n_minis = int(max(np.floor(risk_dollars / mini_risk_1c), 1))
    comm = n_minis * MINI_COMM_RT
    # R inflated so net day PnL == requested $ exactly
    R = np.array([(p + comm) / risk_dollars for p in pnl_dollars_seq],
                 dtype=np.float32).reshape(1, -1)
    sl = np.full((1, n_oos_days), sl_pts, dtype=np.float32)
    fired = np.ones((1, n_oos_days), dtype=bool)
    nt = np.ones((1, n_oos_days), dtype=np.int16)
    return R, sl, fired, nt


def _run_eval_only(pnl_dollars_seq, enforce_consistency, eg_risk=0.9,
                   horizon=None, n_sims=1):
    """Run a tiny single-slot deterministic sim (block_len=1 → cyclic replay)."""
    if horizon is None:
        horizon = len(pnl_dollars_seq)
    R, sl, fired, nt = _build_arrays_from_pnl_sequence(
        pnl_dollars_seq, risk_dollars=eg_risk * MLL,
    )
    return three_phase_reinvestment_mc(
        account=ACCOUNT, budget=2 * ACCOUNT.eval_fee, horizon=horizon,
        max_concurrent=1, n_sims=n_sims, seed=0,
        eval_grind_risk=eg_risk, p3_risk_pct=0.0,
        bootstrap_block_len=1.0,            # block-of-1 = cyclic deterministic replay
        bootstrap_oos_R_per_slot=R,
        bootstrap_oos_sl_dists_per_slot=sl,
        bootstrap_oos_fired_per_slot=fired,
        bootstrap_oos_n_trades_per_slot=nt,
        bootstrap_cap_daily_risk=False,
        enforce_consistency_rule=enforce_consistency,
    )


class TestConsistencyRule:

    def test_single_big_day_blocked_when_enforced(self):
        """One +$3,000 day clears PT but max=$3000, total=$3000 → ratio 100% > 50% → must NOT pass."""
        out = _run_eval_only([3_000.0, 0.0, 0.0, 0.0], enforce_consistency=True)
        assert out['n_passed_eval'][0] == 0, \
            "Single-big-day eval must NOT pass when consistency enforced"

    def test_single_big_day_passes_when_disabled(self):
        """Same input but rule off → eval passes immediately on day 0."""
        out = _run_eval_only([3_000.0, 0.0, 0.0, 0.0], enforce_consistency=False)
        assert out['n_passed_eval'][0] == 1, \
            "Single-big-day eval should pass when consistency NOT enforced"

    def test_two_balanced_days_pass(self):
        """+$1,700 + +$1,700 → max=$1700, total=$3400, ratio=50% (≤ 50%) AND 2 prof days → passes.
        Slight margin over PT to dodge float32 boundary precision."""
        out = _run_eval_only([1_700.0, 1_700.0, 0.0, 0.0], enforce_consistency=True)
        assert out['n_passed_eval'][0] == 1, \
            "Two equal $1.7K days should pass consistency"

    def test_dilution_until_pass(self):
        """+$3,000 day → blocked. Then add small profits to dilute.
        Need total ≥ $6,000 for max=$3000 to be ≤ 50% of total.
        Sequence: +3000, +500, +500, +500, +500, +500, +500 → total $6000, max=$3000, 50% — passes.
        """
        out = _run_eval_only(
            [3_000.0] + [500.0] * 6, enforce_consistency=True, horizon=10,
        )
        assert out['n_passed_eval'][0] == 1, \
            "Eval should pass after enough small profitable days dilute the big day"

    def test_one_profitable_day_insufficient(self):
        """+$3,500 (just one profitable day, rule requires ≥2 prof days)
        Even ignoring the 50% rule, only 1 profitable day → must fail."""
        out = _run_eval_only([3_500.0, 0.0, 0.0, 0.0], enforce_consistency=True)
        assert out['n_passed_eval'][0] == 0

    def test_funded_phase_unaffected(self):
        """After eval passes (with consistency satisfied), funded phase grind
        should have NO consistency check — a single big funded day should
        trigger the grind→payout transition normally.
        Sequence:
          eval (2 days):  +1700, +1700        → eval passes (max=$1700 = 50% of $3400)
          funded (1 day): +3500                → fu_total_profit=$3500 → hit_target=True
        """
        out = _run_eval_only(
            [1_700.0, 1_700.0, 0.0, 3_500.0], enforce_consistency=True, horizon=10,
        )
        assert out['n_passed_eval'][0] == 1, "eval should pass"
        assert out['n_hit_target'][0] >= 1, \
            "funded grind should hit target on the +$3K day; consistency must NOT apply post-eval"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
