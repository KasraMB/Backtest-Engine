"""
backtest/propfirm/correlated_mc.py
────────────────────────────────────
Regime-aware correlated reinvestment Monte Carlo.

Simulates N investors over `horizon` trading days. Accounts within one sim
share the same daily market draw (correlated). Different sims are independent.

Public API: correlated_reinvestment_mc(...)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from backtest.propfirm.lucidflex import (
    LucidFlexAccount,
    MICRO_POINT_VALUE,
    MINI_POINT_VALUE,
    MICRO_COMM_RT,
    MINI_COMM_RT,
)
from backtest.regime.hmm import RegimeResult


@dataclass
class StrategyConfig:
    name:               str
    pnl_pts_by_regime:  Dict[int, np.ndarray]   # regime → float32 array
    sl_dists_by_regime: Dict[int, np.ndarray]   # regime → float32 array
    tpd_by_regime:      Dict[int, float]         # regime → avg trades per day
    entry_time_min:     int                      # minutes into session
    eval_risk:          float                    # full-slot ERP baseline (fraction of MLL)
    fund_risk:          float                    # full-slot FRP baseline (fraction of MLL)


@dataclass
class AccountSlot:
    configs: List[StrategyConfig]   # index 0 = highest priority


@dataclass
class AccountManagementStrategy:
    trigger:         str   # "greedy"|"on_fail"|"on_pass"|"on_close"|"on_payout"|"staggered"
    max_concurrent:  int   # 1–5
    reserve_n_evals: int   # eval fees to keep as cash cushion
    stagger_days:    int = 0   # days between opens; only for trigger="staggered"


def _regime_initial_dist(regime_result: RegimeResult) -> np.ndarray:
    """Derive empirical initial distribution from historical regime label counts."""
    counts = np.zeros(regime_result.n_states, dtype=np.float64)
    for r in regime_result.labels.values():
        counts[r] += 1
    total = counts.sum()
    return counts / total if total > 0 else np.full(regime_result.n_states, 1.0 / regime_result.n_states)


def _sample_regime_sequences(
    regime_result: RegimeResult,
    n_sims: int,
    horizon: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Pre-generate (n_sims, horizon) int8 regime array via Markov chain."""
    trans      = regime_result.transition_matrix
    n_regimes  = regime_result.n_states
    initial    = _regime_initial_dist(regime_result)
    seqs = np.empty((n_sims, horizon), dtype=np.int8)
    seqs[:, 0] = rng.choice(n_regimes, size=n_sims, p=initial)
    for d in range(1, horizon):
        prev = seqs[:, d - 1]
        for r in range(n_regimes):
            mask = prev == r
            if mask.any():
                seqs[mask, d] = rng.choice(
                    n_regimes, size=int(mask.sum()), p=trans[r]
                )
    return seqs


def _draw_day_trades(
    configs: List[StrategyConfig],
    regime: int,
    rng: np.random.Generator,
) -> Dict[str, Optional[Tuple[float, float]]]:
    """
    For each unique config (by name), determine whether a trade fires today.
    Returns {name: (pnl_pts, sl_dist)} or {name: None}.
    Draws are SHARED — all slots running the same config see the same outcome.
    """
    results: Dict[str, Optional[Tuple[float, float]]] = {}
    seen: set = set()
    for cfg in configs:
        if cfg.name in seen:
            continue
        seen.add(cfg.name)
        tpd = cfg.tpd_by_regime[regime]
        n_trades = int(rng.poisson(tpd))
        if n_trades == 0:
            results[cfg.name] = None
        else:
            pool_pnl = cfg.pnl_pts_by_regime[regime]
            pool_sl  = cfg.sl_dists_by_regime[regime]
            idx = int(rng.integers(0, len(pool_pnl)))
            results[cfg.name] = (float(pool_pnl[idx]), float(pool_sl[idx]))
    return results


def _resolve_slot_trade(
    slot: AccountSlot,
    day_trades: Dict[str, Optional[Tuple[float, float]]],
) -> Optional[Tuple[StrategyConfig, float, float]]:
    """
    Returns (config, pnl_pts, sl_dist) for the config that fires earliest today,
    or None if no config fired. Ties broken by list index (lower = higher priority).
    """
    candidates = [
        cfg for cfg in slot.configs
        if day_trades.get(cfg.name) is not None
    ]
    if not candidates:
        return None
    winner = min(
        candidates,
        key=lambda c: (c.entry_time_min, slot.configs.index(c)),
    )
    pnl, sl = day_trades[winner.name]  # type: ignore[misc]
    return (winner, pnl, sl)


# ---------------------------------------------------------------------------
# Internal account state dataclasses
# ---------------------------------------------------------------------------

@dataclass
class _EvalState:
    balance:        float
    mll:            float
    peak_eod:       float
    total_profit:   float = 0.0
    max_day_profit: float = 0.0
    n_prof_days:    int   = 0


@dataclass
class _FundedState:
    balance:         float
    mll:             float
    peak_eod:        float
    mll_locked:      bool = False
    cycle_prof_days: int  = 0
    payout_count:    int  = 0


# ---------------------------------------------------------------------------
# Sizing helper (nq_first: minis when possible, micros otherwise)
# ---------------------------------------------------------------------------

def _size_and_pnl(
    pnl_pts:      float,
    sl_dist:      float,
    risk_dollars: float,
    account:      LucidFlexAccount,
) -> float:
    """Compute dollar PnL for one trade using nq_first sizing."""
    sl_safe      = max(0.25, sl_dist)
    max_minis    = account.max_micros // 10
    mini_risk_1c = sl_safe * MINI_POINT_VALUE
    if max_minis > 0 and mini_risk_1c <= risk_dollars:
        n_c    = max(1, min(max_minis, int(risk_dollars / mini_risk_1c)))
        return pnl_pts * n_c * MINI_POINT_VALUE - MINI_COMM_RT * n_c
    else:
        n_c = max(1, min(account.max_micros,
                         int(risk_dollars / (sl_safe * MICRO_POINT_VALUE))))
        return pnl_pts * n_c * MICRO_POINT_VALUE - MICRO_COMM_RT * n_c


# ---------------------------------------------------------------------------
# Eval stepper
# ---------------------------------------------------------------------------

def _eval_step(
    state:        _EvalState,
    pnl_pts:      float,
    sl_dist:      float,
    risk_dollars: float,
    account:      LucidFlexAccount,
) -> Tuple[_EvalState, str]:
    """
    Apply one trade to eval state.
    Returns (new_state, event) where event ∈ {"running", "passed", "failed"}.
    Implements: trailing EOD MLL, consistency rule (max_day ≤ 50% total, ≥ 2 prof days).
    """
    dollar = _size_and_pnl(pnl_pts, sl_dist, risk_dollars, account)

    new_balance = state.balance + dollar
    if new_balance <= state.mll:
        return state, "failed"

    new_peak = max(state.peak_eod, new_balance)
    new_mll  = max(state.mll, new_peak - account.mll_amount)

    total_profit   = new_balance - account.starting_balance
    n_prof_days    = state.n_prof_days
    max_day_profit = state.max_day_profit
    if dollar > 0.0:
        n_prof_days    += 1
        max_day_profit  = max(max_day_profit, dollar)

    new_state = _EvalState(
        balance=new_balance, mll=new_mll, peak_eod=new_peak,
        total_profit=total_profit, max_day_profit=max_day_profit,
        n_prof_days=n_prof_days,
    )

    if (total_profit >= account.profit_target
            and max_day_profit <= total_profit * 0.5
            and n_prof_days >= 2):
        return new_state, "passed"

    return new_state, "running"


# ---------------------------------------------------------------------------
# Funded stepper
# ---------------------------------------------------------------------------

def _funded_step(
    state:        _FundedState,
    pnl_pts:      float,
    sl_dist:      float,
    risk_dollars: float,
    account:      LucidFlexAccount,
) -> Tuple[_FundedState, str, float]:
    """
    Apply one trade to funded state.
    Returns (new_state, event, payout_net) where:
      event ∈ {"running", "payout", "closed", "blown"}
      payout_net: net dollars received (0 if no payout this step).
    Implements: trailing MLL (locks at starting_balance−$100 once balance ≥ starting),
    5-profitable-day payout cycle, max 6 payouts then "closed".
    """
    dollar = _size_and_pnl(pnl_pts, sl_dist, risk_dollars, account)

    new_balance = state.balance + dollar
    mll_locked  = state.mll_locked

    if not mll_locked and new_balance >= account.starting_balance:
        mll_locked = True

    if mll_locked:
        new_mll = account.starting_balance - 100.0
    else:
        new_peak_pre = max(state.peak_eod, new_balance)
        new_mll      = max(state.mll, new_peak_pre - account.mll_amount)

    if new_balance <= new_mll:
        return state, "blown", 0.0

    new_peak        = new_peak_pre if not mll_locked else max(state.peak_eod, new_balance)
    cycle_prof_days = state.cycle_prof_days
    if dollar > 0.0:
        cycle_prof_days += 1

    payout_net   = 0.0
    payout_count = state.payout_count
    event        = "running"

    if cycle_prof_days >= 5 and payout_count < account.max_payouts:
        profits = max(0.0, new_balance - account.starting_balance)
        gross   = min(profits * 0.5, account.payout_cap)
        if gross >= 500.0:
            payout_net       = gross * account.split
            new_balance     -= gross
            payout_count    += 1
            cycle_prof_days  = 0
            event            = "closed" if payout_count >= account.max_payouts else "payout"

    new_state = _FundedState(
        balance=new_balance, mll=new_mll, peak_eod=new_peak,
        mll_locked=mll_locked, cycle_prof_days=cycle_prof_days,
        payout_count=payout_count,
    )
    return new_state, event, payout_net


# ---------------------------------------------------------------------------
# Account management trigger logic
# ---------------------------------------------------------------------------

def _trigger_fires(
    strategy:      AccountManagementStrategy,
    trigger_event: Optional[str],
    day:           int,
    last_open_day: int,
) -> bool:
    """
    Determine if the account opening trigger fires.

    Trigger semantics:
    - "greedy" → always True
    - "on_fail" → trigger_event == "fail"
    - "on_pass" → trigger_event == "pass"
    - "on_close" → trigger_event in ("fail", "closed", "blown")
    - "on_payout" → trigger_event == "payout"
    - "staggered" → (day - last_open_day) >= stagger_days
    """
    t = strategy.trigger
    if t == "greedy":
        return True
    if t == "on_fail":
        return trigger_event == "fail"
    if t == "on_pass":
        return trigger_event == "pass"
    if t == "on_close":
        return trigger_event in ("fail", "closed", "blown")
    if t == "on_payout":
        return trigger_event == "payout"
    if t == "staggered":
        return (day - last_open_day) >= strategy.stagger_days
    return False


def _should_open(
    strategy:         AccountManagementStrategy,
    trigger_event:    Optional[str],
    n_active:         int,
    cash:             float,
    eval_fee:         float,
    day:              int,
    last_open_day:    int,
    pending_in_queue: int,
) -> Tuple[bool, bool]:
    """
    Determine whether to open a new account and whether to queue the request.

    Returns (open_now, add_to_queue) where:
    - open_now: True if we should open an account right now
    - add_to_queue: True if we should queue the open request (only if full)

    Fallback rule: if n_active + pending_in_queue == 0 and cash >= eval_fee,
    always open one regardless of trigger or reserve (prevents dormancy).

    Queue rule: if at max_concurrent, don't open now but queue if trigger fires.

    Reserve rule: normally don't open if cash - eval_fee < reserve_n_evals × eval_fee,
    but fallback overrides this.
    """
    if cash < eval_fee:
        return False, False

    # Fallback: prevent dormancy
    if n_active + pending_in_queue == 0:
        return True, False

    if n_active >= strategy.max_concurrent:
        # Full — queue if trigger fires
        return False, _trigger_fires(strategy, trigger_event, day, last_open_day)

    reserve = strategy.reserve_n_evals * eval_fee
    if cash - eval_fee < reserve:
        return False, False

    return _trigger_fires(strategy, trigger_event, day, last_open_day), False
