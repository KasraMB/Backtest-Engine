"""
backtest/propfirm/correlated_mc.py
────────────────────────────────────
Regime-aware correlated reinvestment Monte Carlo.

Simulates N investors over `horizon` trading days. Accounts within one sim
share the same daily market draw (correlated). Different sims are independent.

Public API: correlated_reinvestment_mc(...)

Performance: vectorised batch simulation processes all n_sims simultaneously
using (n_sims, max_c) numpy arrays and pre-generated random draws.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from numba import njit, prange

from backtest.propfirm.lucidflex import (
    LucidFlexAccount,
    MICRO_POINT_VALUE,
    MINI_POINT_VALUE,
    MICRO_COMM_RT,
    MINI_COMM_RT,
)
from backtest.regime.hmm import RegimeResult


# ---------------------------------------------------------------------------
# Trigger encoding (string → int for Numba)
# ---------------------------------------------------------------------------
_TRIG_GREEDY    = 0
_TRIG_ON_FAIL   = 1
_TRIG_ON_PASS   = 2
_TRIG_ON_CLOSE  = 3
_TRIG_ON_PAYOUT = 4
_TRIG_STAGGERED = 5
_TRIG_MAP = {
    "greedy": _TRIG_GREEDY, "on_fail": _TRIG_ON_FAIL, "on_pass": _TRIG_ON_PASS,
    "on_close": _TRIG_ON_CLOSE, "on_payout": _TRIG_ON_PAYOUT, "staggered": _TRIG_STAGGERED,
}


@dataclass
class StrategyConfig:
    name:               str
    pnl_pts_by_regime:  Dict[int, np.ndarray]
    sl_dists_by_regime: Dict[int, np.ndarray]
    tpd_by_regime:      Dict[int, float]
    entry_time_min:     int
    eval_risk:          float   # full-slot ERP baseline (fraction of MLL)
    fund_risk:          float   # full-slot FRP baseline (fraction of MLL)


@dataclass
class AccountSlot:
    configs: List[StrategyConfig]   # index 0 = highest priority


@dataclass
class AccountManagementStrategy:
    trigger:         str   # "greedy"|"on_fail"|"on_pass"|"on_close"|"on_payout"|"staggered"
    max_concurrent:  int   # 1–5
    reserve_n_evals: int
    stagger_days:    int = 0


# ---------------------------------------------------------------------------
# Regime sequence generation
# ---------------------------------------------------------------------------

def _regime_initial_dist(regime_result: RegimeResult) -> np.ndarray:
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
    """Pre-generate (n_sims, horizon) int8 regime array via Markov chain.
    Fast-path for single-regime: returns all-zeros without RNG calls."""
    n_regimes = regime_result.n_states
    if n_regimes == 1:
        return np.zeros((n_sims, horizon), dtype=np.int8)

    trans   = regime_result.transition_matrix
    initial = _regime_initial_dist(regime_result)
    seqs    = np.empty((n_sims, horizon), dtype=np.int8)
    seqs[:, 0] = rng.choice(n_regimes, size=n_sims, p=initial)

    # Vectorised Markov stepping via cumulative-probability lookup
    cum_trans = np.cumsum(trans, axis=1)  # (n_regimes, n_regimes)
    for d in range(1, horizon):
        u = rng.random(n_sims)                           # (n_sims,) uniform
        thresholds = cum_trans[seqs[:, d - 1].astype(np.int32)]  # (n_sims, n_regimes)
        seqs[:, d] = np.argmax(thresholds > u[:, None], axis=1).astype(np.int8)

    return seqs


# ---------------------------------------------------------------------------
# Vectorised sizing helper
# ---------------------------------------------------------------------------

def _size_and_pnl_vec(
    pnl_pts:      np.ndarray,   # (n,) float
    sl_dist:      np.ndarray,   # (n,) float
    risk_dollars: np.ndarray,   # (n,) or scalar float
    account:      LucidFlexAccount,
    max_minis:    int,
) -> np.ndarray:
    """Vectorised nq_first sizing: minis when possible, micros otherwise."""
    sl_safe      = np.maximum(0.25, sl_dist)
    mini_risk_1c = sl_safe * MINI_POINT_VALUE
    use_mini     = (max_minis > 0) & (mini_risk_1c <= risk_dollars)

    n_mini = np.clip(np.floor(risk_dollars / mini_risk_1c).astype(np.int32), 1, max_minis)
    n_micro = np.clip(
        np.floor(risk_dollars / (sl_safe * MICRO_POINT_VALUE)).astype(np.int32),
        1, account.max_micros,
    )
    return np.where(
        use_mini,
        pnl_pts * n_mini  * MINI_POINT_VALUE  - MINI_COMM_RT  * n_mini,
        pnl_pts * n_micro * MICRO_POINT_VALUE - MICRO_COMM_RT * n_micro,
    )


# ---------------------------------------------------------------------------
# Scalar trade draw / resolve helpers (kept for tests that call them directly)
# ---------------------------------------------------------------------------

def _draw_day_trades(
    configs: List[StrategyConfig],
    regime: int,
    rng: np.random.Generator,
) -> Dict[str, Optional[Tuple[float, float]]]:
    """Per-sim version used by tests and _run_single_sim."""
    results: Dict[str, Optional[Tuple[float, float]]] = {}
    seen: set = set()
    for cfg in configs:
        if cfg.name in seen:
            continue
        seen.add(cfg.name)
        tpd = cfg.tpd_by_regime[regime]
        if int(rng.poisson(tpd)) == 0:
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
    """Per-sim version used by tests."""
    candidates = [
        cfg for cfg in slot.configs
        if day_trades.get(cfg.name) is not None
    ]
    if not candidates:
        return None
    winner = min(candidates, key=lambda c: (c.entry_time_min, slot.configs.index(c)))
    pnl, sl = day_trades[winner.name]  # type: ignore[misc]
    return (winner, pnl, sl)


# ---------------------------------------------------------------------------
# Scalar sizing + steppers (kept for tests that call them directly)
# ---------------------------------------------------------------------------

def _size_and_pnl(pnl_pts: float, sl_dist: float,
                  risk_dollars: float, account: LucidFlexAccount) -> float:
    sl_safe      = max(0.25, sl_dist)
    max_minis    = account.max_micros // 10
    mini_risk_1c = sl_safe * MINI_POINT_VALUE
    if max_minis > 0 and mini_risk_1c <= risk_dollars:
        n_c = max(1, min(max_minis, int(risk_dollars / mini_risk_1c)))
        return pnl_pts * n_c * MINI_POINT_VALUE - MINI_COMM_RT * n_c
    n_c = max(1, min(account.max_micros, int(risk_dollars / (sl_safe * MICRO_POINT_VALUE))))
    return pnl_pts * n_c * MICRO_POINT_VALUE - MICRO_COMM_RT * n_c


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


def _eval_step(state, pnl_pts, sl_dist, risk_dollars, account):
    dollar      = _size_and_pnl(pnl_pts, sl_dist, risk_dollars, account)
    new_balance = state.balance + dollar
    if new_balance <= state.mll:
        return state, "failed"
    new_peak = max(state.peak_eod, new_balance)
    new_mll  = max(state.mll, new_peak - account.mll_amount)
    total_profit   = new_balance - account.starting_balance
    n_prof_days    = state.n_prof_days
    max_day_profit = state.max_day_profit
    if dollar > 0.0:
        n_prof_days   += 1
        max_day_profit = max(max_day_profit, dollar)
    new_state = _EvalState(balance=new_balance, mll=new_mll, peak_eod=new_peak,
                           total_profit=total_profit, max_day_profit=max_day_profit,
                           n_prof_days=n_prof_days)
    if (total_profit >= account.profit_target
            and max_day_profit <= total_profit * 0.5
            and n_prof_days >= 2):
        return new_state, "passed"
    return new_state, "running"


def _funded_step(state, pnl_pts, sl_dist, risk_dollars, account):
    dollar      = _size_and_pnl(pnl_pts, sl_dist, risk_dollars, account)
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
            payout_net      = gross * account.split
            new_balance    -= gross
            payout_count   += 1
            cycle_prof_days = 0
            event = "closed" if payout_count >= account.max_payouts else "payout"
    new_state = _FundedState(balance=new_balance, mll=new_mll, peak_eod=new_peak,
                             mll_locked=mll_locked, cycle_prof_days=cycle_prof_days,
                             payout_count=payout_count)
    return new_state, event, payout_net


# ---------------------------------------------------------------------------
# Trigger helpers (scalar — used by tests and _run_single_sim)
# ---------------------------------------------------------------------------

def _trigger_fires(strategy, trigger_event, day, last_open_day) -> bool:
    t = strategy.trigger
    if t == "greedy":    return True
    if t == "on_fail":   return trigger_event == "fail"
    if t == "on_pass":   return trigger_event == "pass"
    if t == "on_close":  return trigger_event in ("fail", "closed", "blown")
    if t == "on_payout": return trigger_event == "payout"
    if t == "staggered": return (day - last_open_day) >= strategy.stagger_days
    return False


def _should_open(strategy, trigger_event, n_active, cash, eval_fee,
                 day, last_open_day, pending_in_queue) -> Tuple[bool, bool]:
    if cash < eval_fee:
        return False, False
    if n_active + pending_in_queue == 0:
        return True, False
    if n_active >= strategy.max_concurrent:
        return False, _trigger_fires(strategy, trigger_event, day, last_open_day)
    reserve = strategy.reserve_n_evals * eval_fee
    if cash - eval_fee < reserve:
        return False, False
    return _trigger_fires(strategy, trigger_event, day, last_open_day), False


# ---------------------------------------------------------------------------
# Scalar single-sim runner (kept for unit tests)
# ---------------------------------------------------------------------------

@dataclass
class _AccountInstance:
    template_idx:  int
    phase:         str
    eval_state:    Optional[_EvalState]   = None
    funded_state:  Optional[_FundedState] = None


def _run_single_sim(regime_seq, slot_template, account, eval_fee,
                    strategy, budget, horizon, rng) -> float:
    def _draw(configs, regime):
        results = {}
        seen = set()
        for cfg in configs:
            if cfg.name in seen:
                continue
            seen.add(cfg.name)
            tpd = cfg.tpd_by_regime[regime]
            if int(rng.poisson(tpd)) == 0:
                results[cfg.name] = None
            else:
                pool_pnl = cfg.pnl_pts_by_regime[regime]
                pool_sl  = cfg.sl_dists_by_regime[regime]
                idx = int(rng.integers(0, len(pool_pnl)))
                results[cfg.name] = (float(pool_pnl[idx]), float(pool_sl[idx]))
        return results

    def _resolve(slot, day_trades):
        candidates = [c for c in slot.configs if day_trades.get(c.name) is not None]
        if not candidates:
            return None
        winner = min(candidates, key=lambda c: (c.entry_time_min, slot.configs.index(c)))
        pnl, sl = day_trades[winner.name]
        return winner, pnl, sl

    cash = budget; accounts = []; queue = 0
    tpl_cycle = 0; last_open_day = -(strategy.stagger_days + 1)

    def _open(day):
        nonlocal cash, tpl_cycle, last_open_day
        if cash < eval_fee: return
        idx = tpl_cycle % len(slot_template); tpl_cycle += 1
        cash -= eval_fee; last_open_day = day
        accounts.append(_AccountInstance(
            template_idx=idx, phase="eval",
            eval_state=_EvalState(balance=account.starting_balance,
                                  mll=account.starting_balance - account.mll_amount,
                                  peak_eod=account.starting_balance)))

    while True:
        open_now, _ = _should_open(strategy, None, len(accounts), cash, eval_fee,
                                    0, last_open_day, queue)
        if not open_now or len(accounts) >= strategy.max_concurrent or cash < eval_fee:
            break
        _open(0)

    for day in range(horizon):
        regime = int(regime_seq[day])
        seen_names: set = set()
        active_configs = []
        for acc in accounts:
            for cfg in slot_template[acc.template_idx].configs:
                if cfg.name not in seen_names:
                    seen_names.add(cfg.name); active_configs.append(cfg)

        day_trades = _draw(active_configs, regime)
        closed_idxs = []; trigger_events = []

        for i, acc in enumerate(accounts):
            slot   = slot_template[acc.template_idx]
            n_cfgs = len(slot.configs)
            trade  = _resolve(slot, day_trades)
            if trade is None: continue
            cfg, pnl_pts, sl_dist = trade

            if acc.phase == "eval":
                risk_d = (cfg.eval_risk / n_cfgs) * account.mll_amount
                new_es, event = _eval_step(acc.eval_state, pnl_pts, sl_dist, risk_d, account)
                acc.eval_state = new_es
                if event == "passed":
                    acc.phase = "funded"
                    acc.funded_state = _FundedState(
                        balance=account.starting_balance,
                        mll=account.starting_balance - account.mll_amount,
                        peak_eod=account.starting_balance)
                    trigger_events.append("pass")
                elif event == "failed":
                    closed_idxs.append(i); trigger_events.append("fail")
            else:
                risk_d = (cfg.fund_risk / n_cfgs) * account.mll_amount
                new_fs, event, pay = _funded_step(acc.funded_state, pnl_pts, sl_dist, risk_d, account)
                acc.funded_state = new_fs
                if event in ("payout", "closed"):
                    cash += pay; trigger_events.append("payout")
                    if event == "closed":
                        closed_idxs.append(i); trigger_events.append("closed")
                elif event == "blown":
                    closed_idxs.append(i); trigger_events.append("blown")

        for i in sorted(set(closed_idxs), reverse=True):
            accounts.pop(i)

        while queue > 0 and len(accounts) < strategy.max_concurrent and cash >= eval_fee:
            _open(day); queue -= 1

        for ev in (trigger_events if trigger_events else [None]):
            open_now, add_q = _should_open(strategy, ev, len(accounts), cash,
                                            eval_fee, day, last_open_day, queue)
            if open_now and cash >= eval_fee and len(accounts) < strategy.max_concurrent:
                _open(day)
            elif add_q:
                queue += 1

        while True:
            open_now, _ = _should_open(strategy, None, len(accounts), cash,
                                        eval_fee, day, last_open_day, queue)
            if not open_now or len(accounts) >= strategy.max_concurrent or cash < eval_fee:
                break
            _open(day)

    return cash


# ---------------------------------------------------------------------------
# Vectorised batch runner  (processes all n_sims simultaneously)
# ---------------------------------------------------------------------------

def _run_batch(
    slot_template:  List[AccountSlot],
    regime_seqs:    np.ndarray,      # (n_sims, horizon) int8
    account:        LucidFlexAccount,
    eval_fee:       float,
    strategy:       AccountManagementStrategy,
    budget:         float,
    horizon:        int,
    rng:            np.random.Generator,
) -> np.ndarray:
    """
    Vectorised simulation: all n_sims investors run in parallel using
    (n_sims, max_c) state arrays. Pre-generates all random draws upfront
    to eliminate per-sim scalar RNG calls.
    """
    n_sims  = regime_seqs.shape[0]
    max_c   = strategy.max_concurrent
    n_tpl   = len(slot_template)
    SB      = float(account.starting_balance)
    MLL_AMT = float(account.mll_amount)
    PT      = float(account.profit_target)
    MAX_PAY = int(account.max_payouts)
    PAY_CAP = float(account.payout_cap)
    SPLIT   = float(account.split)
    MAX_MINIS = account.max_micros // 10
    n_states  = int(regime_seqs.max()) + 1

    # ── Collect unique configs across all slot templates ──────────────────
    unique_cfgs: Dict[str, StrategyConfig] = {}
    for slot in slot_template:
        for cfg in slot.configs:
            if cfg.name not in unique_cfgs:
                unique_cfgs[cfg.name] = cfg

    # ── Pre-generate random draws: (n_sims, horizon) per config ──────────
    # fired[name][s, d] = True if a trade fires for config 'name' in sim s on day d
    # pnl[name][s, d]   = PnL in points (0.0 if not fired)
    # sl[name][s, d]    = SL distance (1.0 if not fired)
    cfg_fired: Dict[str, np.ndarray] = {}
    cfg_pnl:   Dict[str, np.ndarray] = {}
    cfg_sl:    Dict[str, np.ndarray] = {}

    for name, cfg in unique_cfgs.items():
        tpd_lookup = np.array([cfg.tpd_by_regime.get(r, 0.0) for r in range(n_states)])
        tpd_arr    = tpd_lookup[regime_seqs.astype(np.int32)]      # (n_sims, horizon)
        fired      = rng.poisson(tpd_arr) > 0                      # (n_sims, horizon) bool

        max_pool  = max(len(v) for v in cfg.pnl_pts_by_regime.values())
        raw_idx   = rng.integers(0, max(max_pool, 1),
                                 size=(n_sims, horizon), dtype=np.int32)

        pnl_out = np.zeros((n_sims, horizon), dtype=np.float64)
        sl_out  = np.ones( (n_sims, horizon), dtype=np.float64)

        for r, pnl_pool in cfg.pnl_pts_by_regime.items():
            sl_pool = cfg.sl_dists_by_regime[r]
            mask_r  = (regime_seqs.astype(np.int32) == r) & fired
            if mask_r.any():
                idx_r          = raw_idx[mask_r] % len(pnl_pool)
                pnl_out[mask_r] = pnl_pool[idx_r]
                sl_out[mask_r]  = sl_pool[idx_r]

        cfg_fired[name] = fired
        cfg_pnl[name]   = pnl_out
        cfg_sl[name]    = sl_out

    # ── Pre-build per-slot trade arrays (fired/pnl/sl/risk) ──────────────
    # slot_data[t] = (fired, pnl, sl, risk_eval, risk_fund) for template index t
    # fired/pnl/sl are (n_sims, horizon); risks are scalars
    slot_data: List[Tuple] = []
    for slot in slot_template:
        n_cfgs = len(slot.configs)
        if n_cfgs == 1:
            cfg       = slot.configs[0]
            fired_s   = cfg_fired[cfg.name]
            pnl_s     = cfg_pnl[cfg.name]
            sl_s      = cfg_sl[cfg.name]
            risk_eval = cfg.eval_risk * account.mll_amount
            risk_fund = cfg.fund_risk * account.mll_amount
        else:
            # Multi-config: build merged arrays respecting entry_time_min priority.
            # Process configs from lowest to highest priority (so highest overwrites).
            sorted_cfgs = sorted(enumerate(slot.configs),
                                 key=lambda x: (-x[1].entry_time_min, x[0]))
            fired_s = np.zeros((n_sims, horizon), dtype=bool)
            pnl_s   = np.zeros((n_sims, horizon), dtype=np.float64)
            sl_s    = np.ones( (n_sims, horizon), dtype=np.float64)
            for _, cfg in sorted_cfgs:
                f = cfg_fired[cfg.name]
                fired_s = np.where(f, True, fired_s)
                pnl_s   = np.where(f, cfg_pnl[cfg.name], pnl_s)
                sl_s    = np.where(f, cfg_sl[cfg.name],  sl_s)
            # Scale risk by 1/n_cfgs using the primary (index-0) config's risk
            risk_eval = slot.configs[0].eval_risk / n_cfgs * account.mll_amount
            risk_fund = slot.configs[0].fund_risk / n_cfgs * account.mll_amount

        slot_data.append((fired_s, pnl_s, sl_s, risk_eval, risk_fund))

    # ── State arrays: (n_sims, max_c) ────────────────────────────────────
    phase    = np.zeros((n_sims, max_c), dtype=np.int8)   # 0=inactive,1=eval,2=funded
    tpl      = np.zeros((n_sims, max_c), dtype=np.int8)   # slot_template index

    ev_bal   = np.full( (n_sims, max_c), SB)
    ev_mll   = np.full( (n_sims, max_c), SB - MLL_AMT)
    ev_peak  = np.full( (n_sims, max_c), SB)
    ev_tot   = np.zeros((n_sims, max_c))
    ev_max   = np.zeros((n_sims, max_c))
    ev_np    = np.zeros((n_sims, max_c), dtype=np.int32)

    fu_bal    = np.full( (n_sims, max_c), SB)
    fu_mll    = np.full( (n_sims, max_c), SB - MLL_AMT)
    fu_peak   = np.full( (n_sims, max_c), SB)
    fu_locked = np.zeros((n_sims, max_c), dtype=bool)
    fu_cyc    = np.zeros((n_sims, max_c), dtype=np.int32)
    fu_cnt    = np.zeros((n_sims, max_c), dtype=np.int32)

    cash     = np.full(n_sims, budget)
    queue    = np.zeros(n_sims, dtype=np.int32)
    tpl_cyc  = np.zeros(n_sims, dtype=np.int32)
    last_day = np.full(n_sims, -(strategy.stagger_days + 1), dtype=np.int32)

    trigger  = strategy.trigger
    reserve  = float(strategy.reserve_n_evals) * eval_fee
    ac_range = np.arange(max_c)   # reused every day

    # ── Vectorised account opener ─────────────────────────────────────────
    def _open_vec(should: np.ndarray, day: int) -> None:
        """Open one account per iteration for sims where `should` is True.
        Loops at most max_c times (bounded), each fully vectorised."""
        nonlocal cash, tpl_cyc, last_day
        for _ in range(max_c):
            n_act = (phase > 0).sum(axis=1)
            can   = should & (n_act < max_c) & (cash >= eval_fee)
            if not can.any():
                break
            free       = phase == 0                              # (n_sims, max_c)
            first_free = np.argmax(free, axis=1)                # (n_sims,)
            open_m     = can[:, None] & free & (ac_range == first_free[:, None])

            new_tpl = (tpl_cyc % n_tpl).astype(np.int8)
            phase[open_m]   = 1
            tpl[open_m]     = np.broadcast_to(new_tpl[:, None], (n_sims, max_c))[open_m]
            ev_bal[open_m]  = SB
            ev_mll[open_m]  = SB - MLL_AMT
            ev_peak[open_m] = SB
            ev_tot[open_m]  = 0.0
            ev_max[open_m]  = 0.0
            ev_np[open_m]   = 0

            opened = can & open_m.any(axis=1)
            cash[opened]    -= eval_fee
            tpl_cyc[opened] += 1
            last_day[opened] = day

    # ── Vectorised trigger ────────────────────────────────────────────────
    def _trig_vec(day, any_fail, any_pass, any_payout, any_close) -> np.ndarray:
        if trigger == "greedy":    return np.ones(n_sims, dtype=bool)
        if trigger == "on_fail":   return any_fail
        if trigger == "on_pass":   return any_pass
        if trigger == "on_close":  return any_close
        if trigger == "on_payout": return any_payout
        if trigger == "staggered": return (day - last_day) >= strategy.stagger_days
        return np.zeros(n_sims, dtype=bool)

    # ── Day-0 initial opens ───────────────────────────────────────────────
    _open_vec(np.ones(n_sims, dtype=bool), 0)

    # ── Main day loop ─────────────────────────────────────────────────────
    for day in range(horizon):
        any_fail   = np.zeros(n_sims, dtype=bool)
        any_pass   = np.zeros(n_sims, dtype=bool)
        any_payout = np.zeros(n_sims, dtype=bool)
        any_close  = np.zeros(n_sims, dtype=bool)
        closed_m   = np.zeros((n_sims, max_c), dtype=bool)
        payout_amt = np.zeros(n_sims)

        for a in range(max_c):
            # Gather trade data for this slot, per sim (template may vary per sim)
            fired_a    = np.zeros(n_sims, dtype=bool)
            pnl_a      = np.zeros(n_sims)
            sl_a       = np.ones(n_sims)
            r_eval_a   = np.zeros(n_sims)
            r_fund_a   = np.zeros(n_sims)
            for t, (f_t, p_t, s_t, re_t, rf_t) in enumerate(slot_data):
                m = tpl[:, a] == t
                if m.any():
                    fired_a[m]  = f_t[m, day]
                    pnl_a[m]    = p_t[m, day]
                    sl_a[m]     = s_t[m, day]
                    r_eval_a[m] = re_t
                    r_fund_a[m] = rf_t

            # ── Eval accounts ─────────────────────────────────────────────
            is_eval  = phase[:, a] == 1
            act_eval = is_eval & fired_a
            if act_eval.any():
                dollar = _size_and_pnl_vec(pnl_a, sl_a, r_eval_a, account, MAX_MINIS)
                d      = np.where(act_eval, dollar, 0.0)
                new_b  = ev_bal[:, a] + d
                failed = act_eval & (new_b <= ev_mll[:, a])
                ok     = is_eval & ~failed

                new_peak = np.where(ok, np.maximum(ev_peak[:, a], new_b), ev_peak[:, a])
                new_mll  = np.where(ok, np.maximum(ev_mll[:, a], new_peak - MLL_AMT), ev_mll[:, a])
                new_tot  = np.where(ok, new_b - SB, ev_tot[:, a])
                won      = act_eval & (d > 0) & ~failed
                new_max  = np.where(won, np.maximum(ev_max[:, a], d), ev_max[:, a])
                new_np   = ev_np[:, a] + np.where(won, 1, 0)

                passed = (
                    ok & act_eval
                    & (new_tot >= PT)
                    & (new_max <= new_tot * 0.5)
                    & (new_np >= 2)
                )

                ev_bal[:, a]  = np.where(ok, new_b, ev_bal[:, a])
                ev_mll[:, a]  = np.where(ok, new_mll, ev_mll[:, a])
                ev_peak[:, a] = np.where(ok, new_peak, ev_peak[:, a])
                ev_tot[:, a]  = np.where(ok, new_tot, ev_tot[:, a])
                ev_max[:, a]  = np.where(ok, new_max, ev_max[:, a])
                ev_np[:, a]   = np.where(ok, new_np, ev_np[:, a])

                phase[passed, a] = 2
                fu_bal[passed, a]    = SB
                fu_mll[passed, a]    = SB - MLL_AMT
                fu_peak[passed, a]   = SB
                fu_locked[passed, a] = False
                fu_cyc[passed, a]    = 0
                fu_cnt[passed, a]    = 0

                closed_m[:, a] |= failed
                any_fail |= failed
                any_pass |= passed
                any_close |= failed

            # ── Funded accounts ───────────────────────────────────────────
            is_fund  = phase[:, a] == 2
            act_fund = is_fund & fired_a
            if act_fund.any():
                dollar = _size_and_pnl_vec(pnl_a, sl_a, r_fund_a, account, MAX_MINIS)
                d      = np.where(act_fund, dollar, 0.0)
                new_b  = fu_bal[:, a] + d

                new_locked = fu_locked[:, a] | (act_fund & (new_b >= SB))
                new_peak_pre = np.maximum(fu_peak[:, a], new_b)
                new_mll = np.where(
                    new_locked,
                    SB - 100.0,
                    np.maximum(fu_mll[:, a], new_peak_pre - MLL_AMT),
                )
                blown  = act_fund & (new_b <= new_mll)
                ok     = is_fund & ~blown
                new_peak = np.where(ok, new_peak_pre, fu_peak[:, a])

                won    = act_fund & (d > 0) & ~blown
                new_cyc = fu_cyc[:, a] + np.where(won, 1, 0)

                profits  = np.maximum(0.0, new_b - SB)
                gross    = np.minimum(profits * 0.5, PAY_CAP)
                can_pay  = ok & (new_cyc >= 5) & (fu_cnt[:, a] < MAX_PAY) & (gross >= 500.0)
                net_pay  = np.where(can_pay, gross * SPLIT, 0.0)
                new_b    = np.where(can_pay, new_b - gross, new_b)
                new_cnt  = fu_cnt[:, a] + np.where(can_pay, 1, 0)
                new_cyc  = np.where(can_pay, 0, new_cyc)
                f_closed = can_pay & (new_cnt >= MAX_PAY)

                payout_amt += np.where(ok, net_pay, 0.0)

                fu_bal[:, a]    = np.where(ok, new_b, fu_bal[:, a])
                fu_mll[:, a]    = np.where(ok, new_mll, fu_mll[:, a])
                fu_peak[:, a]   = np.where(ok, new_peak, fu_peak[:, a])
                fu_locked[:, a] = np.where(ok, new_locked, fu_locked[:, a])
                fu_cyc[:, a]    = np.where(ok, new_cyc, fu_cyc[:, a])
                fu_cnt[:, a]    = np.where(ok, new_cnt, fu_cnt[:, a])

                closed_m[:, a] |= blown | f_closed
                any_payout |= can_pay
                any_close  |= blown | f_closed

        cash += payout_amt
        phase[closed_m] = 0

        # ── Account management ────────────────────────────────────────────
        # 1. Drain queue into freed slots
        q_drain = (queue > 0) & ((phase > 0).sum(axis=1) < max_c) & (cash >= eval_fee)
        if q_drain.any():
            _open_vec(q_drain, day)
            queue -= q_drain.astype(np.int32)
            np.maximum(queue, 0, out=queue)

        # 2. Fallback: always open if completely dormant
        dormant = ((phase > 0).sum(axis=1) + queue == 0) & (cash >= eval_fee)
        if dormant.any():
            _open_vec(dormant, day)

        # 3. Trigger-based opens / queuing
        trig = _trig_vec(day, any_fail, any_pass, any_payout, any_close)
        n_act = (phase > 0).sum(axis=1)
        can_afford = (cash - eval_fee) >= reserve
        open_now = trig & (n_act < max_c) & can_afford & ~dormant
        add_queue = trig & (n_act >= max_c) & (cash >= eval_fee) & ~dormant
        if open_now.any():
            _open_vec(open_now, day)
        queue += add_queue.astype(np.int32)

    return cash


# ---------------------------------------------------------------------------
# Numba JIT kernel — fastest single-combo runner
# ---------------------------------------------------------------------------

@njit(parallel=True, cache=True, fastmath=True)
def _run_batch_jit(
    cfg_fired,        # (n_cfg, n_sims, horizon) bool
    cfg_pnl,          # (n_cfg, n_sims, horizon) float64
    cfg_sl,           # (n_cfg, n_sims, horizon) float64
    tpl_cfg_indices,  # (n_tpl, max_cfgs_per_tpl) int32 — priority-sorted, -1 = unused
    tpl_n_cfgs,       # (n_tpl,) int32
    tpl_risk_eval,    # (n_tpl,) float64
    tpl_risk_fund,    # (n_tpl,) float64
    SB,               # starting_balance
    MLL_AMT,          # mll_amount
    PT,               # profit_target
    MAX_PAY,          # max_payouts
    PAY_CAP,          # payout_cap
    SPLIT,            # 0.90
    MAX_MINIS,        # account.max_micros // 10
    MAX_MICROS,       # account.max_micros
    EVAL_FEE,         # eval_fee
    max_c,            # strategy.max_concurrent
    trigger_code,     # 0..5
    reserve,          # reserve_n_evals * eval_fee
    stagger_days,     # strategy.stagger_days
    n_tpl,            # number of slot templates
    budget,           # initial cash
    horizon,          # number of trading days
):
    """
    Per-sim scalar simulation loop, JIT-compiled and parallelised across sims.
    Equivalent to _run_batch / _run_single_sim but ~50–100x faster.
    """
    n_sims = cfg_fired.shape[1]
    cash_out = np.empty(n_sims, dtype=np.float64)

    for s in prange(n_sims):
        # ── Per-sim state ───────────────────────────────────────────
        cash       = budget
        queue      = 0
        tpl_cyc    = 0
        last_day   = -(stagger_days + 1)
        n_active   = 0

        phase     = np.zeros(max_c, dtype=np.int8)
        tpl_idx   = np.zeros(max_c, dtype=np.int8)
        ev_bal    = np.empty(max_c, dtype=np.float64)
        ev_mll    = np.empty(max_c, dtype=np.float64)
        ev_peak   = np.empty(max_c, dtype=np.float64)
        ev_tot    = np.zeros(max_c, dtype=np.float64)
        ev_max    = np.zeros(max_c, dtype=np.float64)
        ev_np     = np.zeros(max_c, dtype=np.int32)
        fu_bal    = np.empty(max_c, dtype=np.float64)
        fu_mll    = np.empty(max_c, dtype=np.float64)
        fu_peak   = np.empty(max_c, dtype=np.float64)
        fu_locked = np.zeros(max_c, dtype=np.bool_)
        fu_cyc    = np.zeros(max_c, dtype=np.int32)
        fu_cnt    = np.zeros(max_c, dtype=np.int32)

        # ── Day-0 initial opens (always greedy regardless of trigger) ──
        while n_active < max_c and cash >= EVAL_FEE:
            a = -1
            for i in range(max_c):
                if phase[i] == 0:
                    a = i
                    break
            if a < 0:
                break
            phase[a]   = 1
            tpl_idx[a] = tpl_cyc % n_tpl
            ev_bal[a]  = SB
            ev_mll[a]  = SB - MLL_AMT
            ev_peak[a] = SB
            ev_tot[a]  = 0.0
            ev_max[a]  = 0.0
            ev_np[a]   = 0
            cash      -= EVAL_FEE
            tpl_cyc   += 1
            last_day   = 0
            n_active  += 1

        # ── Day loop ────────────────────────────────────────────────
        for day in range(horizon):
            any_fail   = False
            any_pass   = False
            any_payout = False
            any_close  = False

            for a in range(max_c):
                if phase[a] == 0:
                    continue

                ti   = tpl_idx[a]
                ncfg = tpl_n_cfgs[ti]

                # Resolve which config fires (priority-sorted, first wins)
                won = -1
                for ci in range(ncfg):
                    cidx = tpl_cfg_indices[ti, ci]
                    if cfg_fired[cidx, s, day]:
                        won = cidx
                        break
                if won < 0:
                    continue

                pnl = cfg_pnl[won, s, day]
                sl  = cfg_sl[won, s, day]

                # nq_first sizing
                sl_safe = sl if sl > 0.25 else 0.25
                mini_risk_1c = sl_safe * 20.0
                if phase[a] == 1:
                    risk_d = tpl_risk_eval[ti]
                else:
                    risk_d = tpl_risk_fund[ti]

                if MAX_MINIS > 0 and mini_risk_1c <= risk_d:
                    n_c = int(risk_d / mini_risk_1c)
                    if n_c < 1: n_c = 1
                    if n_c > MAX_MINIS: n_c = MAX_MINIS
                    dollar = pnl * n_c * 20.0 - 3.5 * n_c
                else:
                    n_c = int(risk_d / (sl_safe * 2.0))
                    if n_c < 1: n_c = 1
                    if n_c > MAX_MICROS: n_c = MAX_MICROS
                    dollar = pnl * n_c * 2.0 - 1.0 * n_c

                if phase[a] == 1:
                    # ── EVAL step ──
                    new_b = ev_bal[a] + dollar
                    if new_b <= ev_mll[a]:
                        phase[a] = 0
                        any_fail = True
                        any_close = True
                        n_active -= 1
                    else:
                        ev_bal[a] = new_b
                        if new_b > ev_peak[a]:
                            ev_peak[a] = new_b
                        cand = ev_peak[a] - MLL_AMT
                        if cand > ev_mll[a]:
                            ev_mll[a] = cand
                        ev_tot[a] = new_b - SB
                        if dollar > 0.0:
                            ev_np[a] += 1
                            if dollar > ev_max[a]:
                                ev_max[a] = dollar
                        if (ev_tot[a] >= PT
                                and ev_max[a] <= ev_tot[a] * 0.5
                                and ev_np[a] >= 2):
                            phase[a]     = 2
                            fu_bal[a]    = SB
                            fu_mll[a]    = SB - MLL_AMT
                            fu_peak[a]   = SB
                            fu_locked[a] = False
                            fu_cyc[a]    = 0
                            fu_cnt[a]    = 0
                            any_pass     = True
                else:
                    # ── FUNDED step ──
                    new_b = fu_bal[a] + dollar
                    if (not fu_locked[a]) and new_b >= SB:
                        fu_locked[a] = True
                    if fu_locked[a]:
                        new_mll = SB - 100.0
                    else:
                        new_peak_pre = fu_peak[a] if fu_peak[a] > new_b else new_b
                        cand = new_peak_pre - MLL_AMT
                        new_mll = fu_mll[a] if fu_mll[a] > cand else cand
                    if new_b <= new_mll:
                        phase[a] = 0
                        any_close = True
                        n_active -= 1
                    else:
                        fu_bal[a]  = new_b
                        fu_mll[a]  = new_mll
                        if new_b > fu_peak[a]:
                            fu_peak[a] = new_b
                        if dollar > 0.0:
                            fu_cyc[a] += 1
                        if fu_cyc[a] >= 5 and fu_cnt[a] < MAX_PAY:
                            profits = new_b - SB
                            if profits < 0.0:
                                profits = 0.0
                            half = profits * 0.5
                            gross = half if half < PAY_CAP else PAY_CAP
                            if gross >= 500.0:
                                cash      += gross * SPLIT
                                fu_bal[a]  = new_b - gross
                                fu_cnt[a] += 1
                                fu_cyc[a]  = 0
                                any_payout = True
                                if fu_cnt[a] >= MAX_PAY:
                                    phase[a]  = 0
                                    any_close = True
                                    n_active -= 1

            # ── Drain queue ────────────────────────────────────────
            while queue > 0 and n_active < max_c and cash >= EVAL_FEE:
                a = -1
                for i in range(max_c):
                    if phase[i] == 0:
                        a = i
                        break
                if a < 0:
                    break
                phase[a]   = 1
                tpl_idx[a] = tpl_cyc % n_tpl
                ev_bal[a]  = SB
                ev_mll[a]  = SB - MLL_AMT
                ev_peak[a] = SB
                ev_tot[a]  = 0.0
                ev_max[a]  = 0.0
                ev_np[a]   = 0
                cash      -= EVAL_FEE
                tpl_cyc   += 1
                last_day   = day
                queue     -= 1
                n_active  += 1

            # ── Fallback: dormant ──────────────────────────────────
            dormant = (n_active + queue == 0) and (cash >= EVAL_FEE)
            if dormant:
                a = -1
                for i in range(max_c):
                    if phase[i] == 0:
                        a = i
                        break
                if a >= 0:
                    phase[a]   = 1
                    tpl_idx[a] = tpl_cyc % n_tpl
                    ev_bal[a]  = SB
                    ev_mll[a]  = SB - MLL_AMT
                    ev_peak[a] = SB
                    ev_tot[a]  = 0.0
                    ev_max[a]  = 0.0
                    ev_np[a]   = 0
                    cash      -= EVAL_FEE
                    tpl_cyc   += 1
                    last_day   = day
                    n_active  += 1

            # ── Trigger-based ──────────────────────────────────────
            trig = False
            if   trigger_code == 0: trig = True
            elif trigger_code == 1: trig = any_fail
            elif trigger_code == 2: trig = any_pass
            elif trigger_code == 3: trig = any_close
            elif trigger_code == 4: trig = any_payout
            elif trigger_code == 5: trig = (day - last_day) >= stagger_days

            if trig and not dormant:
                if n_active < max_c and (cash - EVAL_FEE) >= reserve:
                    a = -1
                    for i in range(max_c):
                        if phase[i] == 0:
                            a = i
                            break
                    if a >= 0:
                        phase[a]   = 1
                        tpl_idx[a] = tpl_cyc % n_tpl
                        ev_bal[a]  = SB
                        ev_mll[a]  = SB - MLL_AMT
                        ev_peak[a] = SB
                        ev_tot[a]  = 0.0
                        ev_max[a]  = 0.0
                        ev_np[a]   = 0
                        cash      -= EVAL_FEE
                        tpl_cyc   += 1
                        last_day   = day
                        n_active  += 1
                elif n_active >= max_c and cash >= EVAL_FEE:
                    queue += 1

            # ── Greedy daily fill ──────────────────────────────────
            if trigger_code == 0:
                while n_active < max_c and (cash - EVAL_FEE) >= reserve:
                    a = -1
                    for i in range(max_c):
                        if phase[i] == 0:
                            a = i
                            break
                    if a < 0:
                        break
                    phase[a]   = 1
                    tpl_idx[a] = tpl_cyc % n_tpl
                    ev_bal[a]  = SB
                    ev_mll[a]  = SB - MLL_AMT
                    ev_peak[a] = SB
                    ev_tot[a]  = 0.0
                    ev_max[a]  = 0.0
                    ev_np[a]   = 0
                    cash      -= EVAL_FEE
                    tpl_cyc   += 1
                    last_day   = day
                    n_active  += 1

        cash_out[s] = cash

    return cash_out


def _build_jit_inputs(slot_template, regime_seqs, account, rng):
    """Pre-compute flat numpy arrays for the Numba kernel."""
    n_sims, horizon = regime_seqs.shape

    unique_cfgs = []
    name_to_idx = {}
    for slot in slot_template:
        for cfg in slot.configs:
            if cfg.name not in name_to_idx:
                name_to_idx[cfg.name] = len(unique_cfgs)
                unique_cfgs.append(cfg)
    n_cfg = len(unique_cfgs)
    n_states = int(regime_seqs.max()) + 1

    cfg_fired = np.zeros((n_cfg, n_sims, horizon), dtype=np.bool_)
    cfg_pnl   = np.zeros((n_cfg, n_sims, horizon), dtype=np.float64)
    cfg_sl    = np.ones( (n_cfg, n_sims, horizon), dtype=np.float64)

    regime_int = regime_seqs.astype(np.int32)
    for ci, cfg in enumerate(unique_cfgs):
        tpd_lookup = np.array([cfg.tpd_by_regime.get(r, 0.0) for r in range(n_states)])
        tpd_arr    = tpd_lookup[regime_int]
        fired      = rng.poisson(tpd_arr) > 0
        cfg_fired[ci] = fired

        max_pool = max(len(v) for v in cfg.pnl_pts_by_regime.values())
        raw_idx  = rng.integers(0, max(max_pool, 1), size=(n_sims, horizon), dtype=np.int32)

        for r, pnl_pool in cfg.pnl_pts_by_regime.items():
            sl_pool = cfg.sl_dists_by_regime[r]
            mask_r  = (regime_int == r) & fired
            if mask_r.any():
                idx_r = raw_idx[mask_r] % len(pnl_pool)
                cfg_pnl[ci][mask_r] = pnl_pool[idx_r]
                cfg_sl[ci][mask_r]  = sl_pool[idx_r]

    n_tpl = len(slot_template)
    max_cfgs = max(len(s.configs) for s in slot_template)
    tpl_cfg_indices = np.full((n_tpl, max_cfgs), -1, dtype=np.int32)
    tpl_n_cfgs      = np.zeros(n_tpl, dtype=np.int32)
    tpl_risk_eval   = np.zeros(n_tpl, dtype=np.float64)
    tpl_risk_fund   = np.zeros(n_tpl, dtype=np.float64)

    for t, slot in enumerate(slot_template):
        n = len(slot.configs)
        tpl_n_cfgs[t] = n
        priorities = sorted(range(n), key=lambda i: (slot.configs[i].entry_time_min, i))
        for ci, orig_idx in enumerate(priorities):
            tpl_cfg_indices[t, ci] = name_to_idx[slot.configs[orig_idx].name]
        tpl_risk_eval[t] = slot.configs[0].eval_risk / n * account.mll_amount
        tpl_risk_fund[t] = slot.configs[0].fund_risk / n * account.mll_amount

    return (cfg_fired, cfg_pnl, cfg_sl,
            tpl_cfg_indices, tpl_n_cfgs, tpl_risk_eval, tpl_risk_fund)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def correlated_reinvestment_mc(
    slot_template:  List[AccountSlot],
    regime_result:  RegimeResult,
    account:        LucidFlexAccount,
    eval_fee:       float,
    strategy:       AccountManagementStrategy,
    budget:         float = 300.0,
    horizon:        int   = 84,
    n_sims:         int   = 1_000,
    seed:           int   = 42,
) -> np.ndarray:
    """
    Run correlated reinvestment MC.

    Returns (n_sims,) float64 array of final cash per simulated investor.
    Uses a Numba JIT-compiled kernel parallelised across sims via prange.
    """
    rng         = np.random.default_rng(seed)
    regime_seqs = _sample_regime_sequences(regime_result, n_sims, horizon, rng)

    (cfg_fired, cfg_pnl, cfg_sl,
     tpl_cfg_indices, tpl_n_cfgs, tpl_risk_eval, tpl_risk_fund
     ) = _build_jit_inputs(slot_template, regime_seqs, account, rng)

    return _run_batch_jit(
        cfg_fired, cfg_pnl, cfg_sl,
        tpl_cfg_indices, tpl_n_cfgs, tpl_risk_eval, tpl_risk_fund,
        float(account.starting_balance),
        float(account.mll_amount),
        float(account.profit_target),
        np.int32(account.max_payouts),
        float(account.payout_cap),
        float(account.split),
        np.int32(account.max_micros // 10),
        np.int32(account.max_micros),
        float(eval_fee),
        np.int32(strategy.max_concurrent),
        np.int32(_TRIG_MAP[strategy.trigger]),
        float(strategy.reserve_n_evals * eval_fee),
        np.int32(strategy.stagger_days),
        np.int32(len(slot_template)),
        float(budget),
        np.int32(horizon),
    )
