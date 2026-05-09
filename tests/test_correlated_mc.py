from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from backtest.propfirm.correlated_mc import (
    StrategyConfig,
    AccountSlot,
    AccountManagementStrategy,
    _sample_regime_sequences,
    _draw_day_trades,
    _resolve_slot_trade,
    _trigger_fires,
    _should_open,
    _EvalState,
    _eval_step,
    _FundedState,
    _funded_step,
)
from backtest.propfirm.lucidflex import LUCIDFLEX_ACCOUNTS
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


class TestDrawDayTrades:
    def test_no_trade_when_tpd_zero(self):
        cfg = _make_cfg()
        cfg.tpd_by_regime[0] = 0.0
        result = _draw_day_trades([cfg], regime=0, rng=np.random.default_rng(0))
        assert result["A"] is None

    def test_trade_fires_when_tpd_high(self):
        cfg = StrategyConfig(
            "A",
            {0: np.array([5.0], dtype=np.float32)},
            {0: np.array([4.0], dtype=np.float32)},
            {0: 100.0}, 30, 0.2, 0.4,
        )
        result = _draw_day_trades([cfg], regime=0, rng=np.random.default_rng(0))
        assert result["A"] is not None
        pnl, sl = result["A"]
        assert pnl == pytest.approx(5.0)
        assert sl == pytest.approx(4.0)

    def test_same_config_appears_once(self):
        cfg = _make_cfg()
        # Pass the same config twice — should deduplicate by name
        result = _draw_day_trades([cfg, cfg], regime=0, rng=np.random.default_rng(0))
        assert list(result.keys()) == ["A"]

    def test_different_configs_independent(self):
        cfg_a = _make_cfg("A")
        cfg_b = _make_cfg("B")
        cfg_a.tpd_by_regime[0] = 100.0
        cfg_b.tpd_by_regime[0] = 100.0
        result = _draw_day_trades([cfg_a, cfg_b], regime=0, rng=np.random.default_rng(0))
        assert "A" in result and "B" in result

    def test_regime_selects_correct_pool(self):
        cfg = StrategyConfig(
            "A",
            {0: np.array([1.0], dtype=np.float32), 1: np.array([99.0], dtype=np.float32)},
            {0: np.array([4.0], dtype=np.float32), 1: np.array([4.0], dtype=np.float32)},
            {0: 100.0, 1: 100.0}, 30, 0.2, 0.4,
        )
        result = _draw_day_trades([cfg], regime=1, rng=np.random.default_rng(0))
        assert result["A"] is not None
        assert result["A"][0] == pytest.approx(99.0)


class TestResolveSlotTrade:
    def test_no_trade_when_none(self):
        cfg = _make_cfg()
        slot = AccountSlot([cfg])
        assert _resolve_slot_trade(slot, {"A": None}) is None

    def test_single_config_fires(self):
        cfg = _make_cfg()
        slot = AccountSlot([cfg])
        result = _resolve_slot_trade(slot, {"A": (5.0, 4.0)})
        assert result is not None
        winner, pnl, sl = result
        assert winner.name == "A"
        assert pnl == 5.0

    def test_earliest_entry_wins(self):
        cfg_a = _make_cfg("A", entry_time_min=60)
        cfg_b = _make_cfg("B", entry_time_min=30)
        slot = AccountSlot([cfg_a, cfg_b])
        trades = {"A": (1.0, 4.0), "B": (2.0, 4.0)}
        winner, pnl, _ = _resolve_slot_trade(slot, trades)
        assert winner.name == "B"
        assert pnl == 2.0

    def test_tie_uses_list_priority(self):
        cfg_a = _make_cfg("A", entry_time_min=30)
        cfg_b = _make_cfg("B", entry_time_min=30)
        slot = AccountSlot([cfg_a, cfg_b])  # A at index 0 = priority
        trades = {"A": (1.0, 4.0), "B": (2.0, 4.0)}
        winner, _, _ = _resolve_slot_trade(slot, trades)
        assert winner.name == "A"

    def test_only_fired_configs_eligible(self):
        cfg_a = _make_cfg("A", entry_time_min=30)
        cfg_b = _make_cfg("B", entry_time_min=10)  # earlier, but didn't fire
        slot = AccountSlot([cfg_a, cfg_b])
        trades = {"A": (1.0, 4.0), "B": None}
        winner, _, _ = _resolve_slot_trade(slot, trades)
        assert winner.name == "A"


ACC = LUCIDFLEX_ACCOUNTS['25K']
# 25K account: starting_balance=25000, profit_target=1250, mll_amount=1000,
# max_micros=20, split=0.90, payout_cap=1500


class TestEvalStep:
    def _fresh_state(self):
        return _EvalState(
            balance=ACC.starting_balance,
            mll=ACC.starting_balance - ACC.mll_amount,
            peak_eod=ACC.starting_balance,
        )

    def test_running_on_small_loss(self):
        state = self._fresh_state()
        new_state, event = _eval_step(state, pnl_pts=-1.0, sl_dist=4.0,
                                       risk_dollars=200.0, account=ACC)
        assert event == "running"
        assert new_state.balance < state.balance

    def test_fail_on_mll_breach(self):
        # Start just above MLL, take big loss
        state = _EvalState(
            balance=ACC.starting_balance - ACC.mll_amount + 10,
            mll=ACC.starting_balance - ACC.mll_amount,
            peak_eod=ACC.starting_balance,
        )
        _, event = _eval_step(state, pnl_pts=-50.0, sl_dist=4.0,
                               risk_dollars=200.0, account=ACC)
        assert event == "failed"

    def test_mll_trails_up_with_balance(self):
        state = self._fresh_state()
        new_state, _ = _eval_step(state, pnl_pts=50.0, sl_dist=4.0,
                                   risk_dollars=200.0, account=ACC)
        assert new_state.peak_eod > state.peak_eod
        assert new_state.mll > state.mll

    def test_pass_when_consistent(self):
        # 2 profitable days, max_day <= 50% of total — consistent pass
        state = _EvalState(
            balance=ACC.starting_balance + 900,
            mll=ACC.starting_balance - ACC.mll_amount + 900,
            peak_eod=ACC.starting_balance + 900,
            total_profit=900.0,
            max_day_profit=300.0,
            n_prof_days=2,
        )
        # risk_dollars=80 → exactly 1 mini (sl_safe×20=80 ≤ 80)
        # dollar = 25 × 1 × 20 − 3.50 = $496.50 → total ~$1396, max=496 ≤ 698 ✓
        new_state, event = _eval_step(state, pnl_pts=25.0, sl_dist=4.0,
                                       risk_dollars=80.0, account=ACC)
        assert event == "passed"


class TestFundedStep:
    def _fresh_funded(self):
        return _FundedState(
            balance=ACC.starting_balance,
            mll=ACC.starting_balance - ACC.mll_amount,
            peak_eod=ACC.starting_balance,
        )

    def test_running_on_small_win(self):
        state = self._fresh_funded()
        new_state, event, payout = _funded_step(state, 5.0, 4.0, 400.0, ACC)
        assert event == "running"
        assert payout == 0.0
        assert new_state.cycle_prof_days == 1

    def test_blown_on_mll_breach(self):
        state = _FundedState(
            balance=ACC.starting_balance - ACC.mll_amount + 10,
            mll=ACC.starting_balance - ACC.mll_amount,
            peak_eod=ACC.starting_balance,
        )
        _, event, payout = _funded_step(state, -50.0, 4.0, 400.0, ACC)
        assert event == "blown"
        assert payout == 0.0

    def test_mll_locks_when_balance_reaches_start(self):
        state = _FundedState(
            balance=ACC.starting_balance - 50,
            mll=ACC.starting_balance - ACC.mll_amount,
            peak_eod=ACC.starting_balance,
            mll_locked=False,
        )
        new_state, _, _ = _funded_step(state, 20.0, 4.0, 400.0, ACC)
        assert new_state.mll_locked
        assert new_state.mll == pytest.approx(ACC.starting_balance - 100.0)

    def test_payout_after_five_prof_days(self):
        state = _FundedState(
            balance=ACC.starting_balance + 5000,
            mll=ACC.starting_balance - 100,
            peak_eod=ACC.starting_balance + 5000,
            mll_locked=True,
            cycle_prof_days=4,
            payout_count=0,
        )
        new_state, event, payout = _funded_step(state, 10.0, 4.0, 400.0, ACC)
        assert event == "payout"
        assert payout > 0
        assert new_state.cycle_prof_days == 0
        assert new_state.payout_count == 1

    def test_sixth_payout_closes_account(self):
        state = _FundedState(
            balance=ACC.starting_balance + 5000,
            mll=ACC.starting_balance - 100,
            peak_eod=ACC.starting_balance + 5000,
            mll_locked=True,
            cycle_prof_days=4,
            payout_count=5,  # one more → closes
        )
        _, event, payout = _funded_step(state, 10.0, 4.0, 400.0, ACC)
        assert event == "closed"
        assert payout > 0


class TestTriggerFires:
    def _s(self, trigger, stagger=0):
        return AccountManagementStrategy(trigger, 3, 0, stagger)

    def test_greedy_always_fires(self):
        assert _trigger_fires(self._s("greedy"), None, 0, -1) is True

    def test_on_fail_fires_on_fail(self):
        assert _trigger_fires(self._s("on_fail"), "fail", 0, -1) is True
        assert _trigger_fires(self._s("on_fail"), "payout", 0, -1) is False
        assert _trigger_fires(self._s("on_fail"), None, 0, -1) is False

    def test_on_pass_fires_on_pass(self):
        assert _trigger_fires(self._s("on_pass"), "pass", 0, -1) is True
        assert _trigger_fires(self._s("on_pass"), "fail", 0, -1) is False

    def test_on_close_fires_on_close_events(self):
        for ev in ("fail", "closed", "blown"):
            assert _trigger_fires(self._s("on_close"), ev, 0, -1) is True
        assert _trigger_fires(self._s("on_close"), "pass", 0, -1) is False

    def test_on_payout_fires_on_payout(self):
        assert _trigger_fires(self._s("on_payout"), "payout", 0, -1) is True
        assert _trigger_fires(self._s("on_payout"), "fail", 0, -1) is False

    def test_staggered_respects_interval(self):
        s = self._s("staggered", stagger=7)
        assert _trigger_fires(s, None, day=7,  last_open_day=0) is True
        assert _trigger_fires(s, None, day=6,  last_open_day=0) is False
        assert _trigger_fires(s, None, day=14, last_open_day=7) is True


class TestShouldOpen:
    def _s(self, trigger="greedy", max_c=3, reserve=0, stagger=0):
        return AccountManagementStrategy(trigger, max_c, reserve, stagger)

    def test_greedy_opens_when_slots_and_cash(self):
        open_now, _ = _should_open(self._s(), None, 0, 500, 70, 0, -999, 0)
        assert open_now

    def test_does_not_open_when_at_max(self):
        open_now, queue = _should_open(self._s("greedy", max_c=2), "fail", 2, 500, 70, 0, -999, 0)
        assert not open_now

    def test_queues_when_full_and_trigger_fires(self):
        _, add_q = _should_open(self._s("on_fail", max_c=2), "fail", 2, 500, 70, 0, -999, 0)
        assert add_q

    def test_reserve_blocks_open(self):
        # reserve=2: keep 2×70=140, cash=140 → after open cash=70 < 140 → blocked
        open_now, _ = _should_open(self._s("greedy", reserve=2), None, 1, 140, 70, 0, -999, 0)
        assert not open_now

    def test_fallback_when_no_accounts(self):
        # on_payout trigger, no payout event — but n_active=0 and pending=0 → fallback fires
        open_now, _ = _should_open(self._s("on_payout", reserve=2), None, 0, 300, 70, 0, -999, 0)
        assert open_now

    def test_fallback_not_when_pending_in_queue(self):
        # n_active=0 but pending_in_queue=1 → no fallback
        open_now, _ = _should_open(self._s("on_payout"), None, 0, 300, 70, 0, -999, pending_in_queue=1)
        assert not open_now

    def test_not_enough_cash(self):
        open_now, _ = _should_open(self._s(), None, 0, 50, 70, 0, -999, 0)
        assert not open_now
