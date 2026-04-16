"""
backtest/propfirm/lucidflex.py
──────────────────────────────
LucidFlex account definitions and vectorised Monte Carlo prop-firm simulator.

Rules implemented (confirmed):
  EVAL
  ────
  • EOD trailing MLL — MLL rises when EOD balance sets a new high.
    Checked once per simulated trading day.
  • No daily loss limit.
  • Consistency: largest single day ≤ 50% of total profit (eval only).
    Minimum 2 profitable days required as a consequence.
  • No time limit (200-day simulation cap).

  FUNDED
  ──────
  • Same EOD trailing MLL. Locks permanently at (starting_balance - $100)
    once EOD balance ≥ starting_balance for the first time.
  • No consistency rule.
  • Payout: 5 profitable trading days per cycle (non-consecutive ok).
  • Min payout $500. Max payout = min(50% of profits above starting, cap).
  • Payout subtracts withdrawn amount from balance; MLL unchanged.
  • Max 6 payouts before LucidLive. Split: 90/10 (trader keeps 90%).

Sizing
──────
  All in MNQ micros by default (point_value = $2).
  AUTO mode: use minis ($20/pt) when target_risk ≥ 1 mini's risk at that SL,
             micros otherwise.

Performance
───────────
  Core loops are fully vectorised with numpy — all N_SIMS simulations run in
  parallel as (N_SIMS, max_trades_per_day) matrix operations. Typical runtime:
  ~0.2s per (scheme, risk_pct) cell at 2,000 sims.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np


# ---------------------------------------------------------------------------
# Account definitions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LucidFlexAccount:
    name: str
    starting_balance: float
    profit_target: float
    mll_amount: float
    max_micros: int
    eval_fee: float
    reset_fee: float
    payout_cap: float            # max gross withdrawal per cycle
    max_payouts: int = 6
    split: float = 0.90


LUCIDFLEX_ACCOUNTS: dict[str, LucidFlexAccount] = {
    "25K": LucidFlexAccount(
        name="25K", starting_balance=25_000,
        profit_target=1_250, mll_amount=1_000,
        max_micros=20, eval_fee=70, reset_fee=60,
        payout_cap=1_500,
    ),
    "50K": LucidFlexAccount(
        name="50K", starting_balance=50_000,
        profit_target=3_000, mll_amount=2_000,
        max_micros=40, eval_fee=130, reset_fee=105,
        payout_cap=2_000,
    ),
    "100K": LucidFlexAccount(
        name="100K", starting_balance=100_000,
        profit_target=6_000, mll_amount=3_000,
        max_micros=60, eval_fee=157.50, reset_fee=140,
        payout_cap=2_500,
    ),
    "150K": LucidFlexAccount(
        name="150K", starting_balance=150_000,
        profit_target=9_000, mll_amount=4_500,
        max_micros=100, eval_fee=241.50, reset_fee=225,
        payout_cap=3_000,
    ),
}

# ---------------------------------------------------------------------------
# Risk geometry and axis
# ---------------------------------------------------------------------------

RISK_GEOMETRIES = [
    "fixed_dollar",   # fixed $ risk per trade = base_risk (risk_pct × MLL)
    "pct_balance",    # % of current balance per trade
    "frac_dd",        # % of remaining drawdown buffer per trade
    "floor_aware",    # scale down near MLL, scale up after good runs
    "max_size",       # always max allowed contracts
    "martingale",     # double after loss, reset after win (capped at 4× base)
]

# Risk axis: % of MLL (same meaning across all account sizes)
RISK_PCT_OF_MLL       = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
EVAL_RISK_PCT_OF_MLL  = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
FUNDED_RISK_PCT_OF_MLL= [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]

MICRO_POINT_VALUE = 2.0
MINI_POINT_VALUE  = 20.0

# Integer scheme codes for Numba JIT (no string comparison in kernel)
_SCHEME_FIXED_DOLLAR = np.int32(0)
_SCHEME_PCT_BALANCE  = np.int32(1)
_SCHEME_FRAC_DD      = np.int32(2)
_SCHEME_FLOOR_AWARE  = np.int32(3)
_SCHEME_MAX_SIZE     = np.int32(4)
_SCHEME_MARTINGALE   = np.int32(5)

SCHEME_CODE: dict[str, int] = {
    'fixed_dollar': 0, 'pct_balance': 1, 'frac_dd': 2,
    'floor_aware': 3, 'max_size': 4, 'martingale': 5,
}

try:
    from numba import njit, prange
    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False
    def njit(*a, **kw):
        return lambda f: f
    def prange(n):
        return range(n)


# ---------------------------------------------------------------------------
# Trade normalisation
# ---------------------------------------------------------------------------

def extract_normalised_trades(
    trades: list,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert backtest Trade objects to two numpy arrays:
        pnl_pts   (n,) — signed points per trade
        sl_dists  (n,) — SL distance in points (>0); inferred if no SL

    Returns (pnl_pts, sl_dists).
    """
    from backtest.strategy.enums import ExitReason
    pnl_list, sl_list = [], []
    for t in trades:
        pnl = (t.exit_price - t.entry_price) * t.direction
        if t.sl_price is not None:
            sl_d = abs(t.entry_price - t.sl_price)
        else:
            # Proxy: use actual adverse move as risk estimate
            sl_d = max(0.25, abs(min(pnl, 0.0)))
            if sl_d == 0:
                sl_d = 1.0
        pnl_list.append(pnl)
        sl_list.append(max(0.25, sl_d))
    return np.array(pnl_list, dtype=np.float64), np.array(sl_list, dtype=np.float64)


# ---------------------------------------------------------------------------
# Contract sizing — vectorised
# ---------------------------------------------------------------------------

def resolve_contracts_vec(
    target_risk: np.ndarray,   # (N,) or scalar
    sl_dists: np.ndarray,      # (N,) or (N_SIMS, max_t)
    max_micros: int,
    mode: str = "micros",
) -> np.ndarray:
    """
    Vectorised contract sizing.  Returns integer array same shape as sl_dists.
    """
    sl_safe = np.where(sl_dists <= 0, 1.0, sl_dists)
    if mode == "auto":
        mini_risk = sl_safe * MINI_POINT_VALUE
        use_mini  = mini_risk <= target_risk
        pv        = np.where(use_mini, MINI_POINT_VALUE, MICRO_POINT_VALUE)
        cap       = np.where(use_mini, max_micros // 10, max_micros)
        raw       = np.where(pv > 0, target_risk / (sl_safe * pv), 1.0)
        return np.clip(np.floor(raw).astype(np.int32), 1, cap)
    else:
        raw = target_risk / (sl_safe * MICRO_POINT_VALUE)
        return np.clip(np.floor(raw).astype(np.int32), 1, max_micros)


# ---------------------------------------------------------------------------
# Risk geometry: target risk per trade — vectorised per-sim
# ---------------------------------------------------------------------------

def compute_target_risk_vec(
    scheme: str,
    risk_pct: float,
    account: LucidFlexAccount,
    balance: np.ndarray,    # (N_SIMS,)
    mll_level: np.ndarray,  # (N_SIMS,)
    last_won: np.ndarray,   # (N_SIMS,) bool
    prev_risk: np.ndarray,  # (N_SIMS,)
) -> np.ndarray:
    """
    Returns target dollar risk per trade for each sim — shape (N_SIMS,).

    risk_pct is the risk axis value expressed as a fraction of MLL
    (e.g. 0.10 = 10% of MLL).  base = risk_pct × account.mll_amount
    is the anchor dollar amount used consistently across schemes so that
    the same column in the heatmap is genuinely comparable.

    Schemes:
      fixed_dollar  — always risk exactly `base` dollars per trade.
      pct_balance   — risk the same *proportion* of current balance as
                      base is of starting_balance. Scales up as equity
                      grows, down as it shrinks.
      frac_dd       — risk `risk_pct` of the remaining drawdown buffer
                      (balance − mll_level). Automatically sizes down
                      when approaching the MLL.
      floor_aware   — base risk, scaled by buffer ratio. Below 50% of
                      full buffer: scale down to 0.25×. Above: up to 2×.
      max_size      — always max contracts regardless of SL distance.
      martingale    — double after a loss, reset after a win, cap at 4×.
    """
    base       = risk_pct * account.mll_amount
    remaining  = np.maximum(1.0, balance - mll_level)
    # Fraction of starting_balance that base represents — used by pct_balance
    base_frac  = base / account.starting_balance   # e.g. $50 / $25000 = 0.002

    if scheme == "fixed_dollar":
        return np.full(len(balance), base)

    elif scheme == "pct_balance":
        # Risk the same fraction of current balance as base is of starting_balance.
        # At starting_balance this equals base exactly; it scales proportionally.
        return balance * base_frac

    elif scheme == "frac_dd":
        return remaining * risk_pct

    elif scheme == "floor_aware":
        buffer_ratio = remaining / account.mll_amount
        scale = np.clip(buffer_ratio / 0.5, 0.25, 2.0)
        return base * scale

    elif scheme == "max_size":
        return np.full(len(balance), 1e9)

    elif scheme == "martingale":
        doubled = np.minimum(prev_risk * 2, base * 4)
        return np.where(last_won, base, doubled)

    return np.full(len(balance), base)


# ---------------------------------------------------------------------------
# Numba JIT eval kernel
# ---------------------------------------------------------------------------

@njit(parallel=True, cache=True, fastmath=True)
def _eval_loop_nb(
    pnl_pts:    np.ndarray,   # (n_pool,) float32
    sl_dists:   np.ndarray,   # (n_pool,) float32
    all_counts: np.ndarray,   # (n_sims, MAX_DAYS) int16
    all_idx:    np.ndarray,   # (n_sims, MAX_DAYS, MAX_T) int32
    regime_seq: np.ndarray,   # (n_sims, MAX_DAYS) int8
    pnl_r0: np.ndarray, pnl_r1: np.ndarray, pnl_r2: np.ndarray,
    sl_r0:  np.ndarray, sl_r1:  np.ndarray, sl_r2:  np.ndarray,
    n_pool_r0: int, n_pool_r1: int, n_pool_r2: int,
    starting_balance: float,
    mll_amount:       float,
    profit_target:    float,
    max_micros:       int,
    risk_pct:    float,
    scheme_code: int,
    point_value: float,
    max_t:       int,
):
    n_sims   = all_counts.shape[0]
    max_days = all_counts.shape[1]

    passed_out  = np.zeros(n_sims, dtype=np.bool_)
    balance_out = np.full(n_sims, np.float32(starting_balance))
    days_out    = np.zeros(n_sims, dtype=np.int32)

    base      = np.float32(risk_pct * mll_amount)
    base_frac = np.float32(base / starting_balance)

    for sim_i in prange(n_sims):
        balance      = np.float32(starting_balance)
        mll_level    = np.float32(starting_balance - mll_amount)
        peak_eod     = np.float32(starting_balance)
        total_profit = np.float32(0.0)
        max_day_pnl  = np.float32(0.0)
        n_prof_days  = np.int32(0)
        alive        = True
        passed       = False
        last_won     = True
        prev_risk    = base
        days_elapsed = np.int32(0)

        for day in range(max_days):
            if not alive:
                break

            n_today = int(all_counts[sim_i, day])
            if n_today == 0:
                continue

            # Select trade pool for today's regime
            regime = int(regime_seq[sim_i, day])
            if regime == 0:
                pool_pnl = pnl_r0; pool_sl = sl_r0; n_pool = n_pool_r0
            elif regime == 1:
                pool_pnl = pnl_r1; pool_sl = sl_r1; n_pool = n_pool_r1
            else:
                pool_pnl = pnl_r2; pool_sl = sl_r2; n_pool = n_pool_r2
            if n_pool == 0:
                pool_pnl = pnl_pts; pool_sl = sl_dists; n_pool = len(pnl_pts)

            # Target risk this day
            remaining = max(np.float32(1.0), balance - mll_level)
            if scheme_code == 0:
                target = base
            elif scheme_code == 1:
                target = balance * base_frac
            elif scheme_code == 2:
                target = np.float32(remaining * risk_pct)
            elif scheme_code == 3:
                buf_ratio = remaining / np.float32(mll_amount)
                scale = min(np.float32(2.0), max(np.float32(0.25), buf_ratio / np.float32(0.5)))
                target = base * scale
            elif scheme_code == 4:
                target = np.float32(1e9)
            else:  # martingale
                doubled = min(prev_risk * np.float32(2.0), base * np.float32(4.0))
                target  = base if last_won else doubled

            # Process trades
            day_pnl        = np.float32(0.0)
            cum_pnl        = np.float32(0.0)
            breached       = False
            last_trade_pnl = np.float32(0.0)

            for t in range(n_today):
                raw_idx = all_idx[sim_i, day, t] % n_pool
                pnl_t   = pool_pnl[raw_idx]
                sl_t    = max(np.float32(0.25), pool_sl[raw_idx])
                raw_c   = target / (sl_t * np.float32(point_value))
                n_c     = max(1, min(max_micros, int(raw_c)))
                dollar  = pnl_t * np.float32(n_c) * np.float32(point_value)
                cum_pnl += dollar
                day_pnl += dollar
                last_trade_pnl = dollar
                if balance + cum_pnl <= mll_level:
                    breached = True
                    break

            last_won  = last_trade_pnl > np.float32(0.0)
            prev_risk = target

            if breached:
                alive = False
                continue

            balance += day_pnl

            new_peak = max(peak_eod, balance)
            peak_eod = new_peak
            new_mll  = peak_eod - np.float32(mll_amount)
            if new_mll > mll_level:
                mll_level = new_mll

            if balance <= mll_level:
                alive = False
                continue

            total_profit = balance - np.float32(starting_balance)
            if day_pnl > np.float32(0.0):
                n_prof_days += 1
                if day_pnl > max_day_pnl:
                    max_day_pnl = day_pnl

            days_elapsed += 1

            if not passed and total_profit >= np.float32(profit_target):
                consistent = (
                    (max_day_pnl <= total_profit * np.float32(0.5))
                    and (n_prof_days >= 2)
                )
                if consistent:
                    passed       = True
                    days_elapsed = day + 1
                    alive        = False

        passed_out[sim_i]  = passed
        balance_out[sim_i] = balance
        days_out[sim_i]    = days_elapsed

    return passed_out, balance_out, days_out


# ---------------------------------------------------------------------------
# Regime helpers
# ---------------------------------------------------------------------------

def _build_regime_arrays(
    pnl_pts: np.ndarray,
    sl_dists: np.ndarray,
    regime_labels: Optional[np.ndarray],
) -> tuple:
    """Split pnl_pts/sl_dists by regime (0/1/2). If None, all pools are the full array."""
    if regime_labels is None:
        arr  = pnl_pts.astype(np.float32)
        sl_a = sl_dists.astype(np.float32)
        return arr, arr, arr, sl_a, sl_a, sl_a
    pools, sl_pools = [], []
    for r in (0, 1, 2):
        mask = regime_labels == r
        if mask.any():
            pools.append(pnl_pts[mask].astype(np.float32))
            sl_pools.append(sl_dists[mask].astype(np.float32))
        else:
            pools.append(pnl_pts.astype(np.float32))
            sl_pools.append(sl_dists.astype(np.float32))
    return (*pools, *sl_pools)


def _build_regime_seq(
    n_sims: int,
    max_days: int,
    transition_matrix: Optional[np.ndarray],
    rng: np.random.Generator,
) -> np.ndarray:
    """Pre-generate regime sequence via Markov chain. Returns int8 (n_sims, max_days)."""
    if transition_matrix is None:
        return np.zeros((n_sims, max_days), dtype=np.int8)
    n_states = transition_matrix.shape[0]
    try:
        vals, vecs = np.linalg.eig(transition_matrix.T)
        idx  = np.argmin(np.abs(vals - 1.0))
        stat = np.real(vecs[:, idx])
        stat = np.abs(stat) / np.abs(stat).sum()
    except Exception:
        stat = np.ones(n_states) / n_states
    seq = np.zeros((n_sims, max_days), dtype=np.int8)
    for sim_i in range(n_sims):
        regime = int(rng.choice(n_states, p=stat))
        for day in range(max_days):
            seq[sim_i, day] = np.int8(regime)
            regime = int(rng.choice(n_states, p=transition_matrix[regime]))
    return seq


def _build_transition_matrix_from_labels(labels: np.ndarray, n_states: int = 3) -> np.ndarray:
    mat = np.zeros((n_states, n_states))
    for a, b in zip(labels[:-1], labels[1:]):
        a, b = int(a), int(b)
        if 0 <= a < n_states and 0 <= b < n_states:
            mat[a, b] += 1
    row_sums = mat.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    return mat / row_sums


# ---------------------------------------------------------------------------
# Vectorised eval simulation
# ---------------------------------------------------------------------------

def simulate_eval_batch(
    pnl_pts: np.ndarray,      # (n_pool,)
    sl_dists: np.ndarray,     # (n_pool,)
    account: LucidFlexAccount,
    risk_pct: float,
    scheme: str,
    sizing_mode: str,
    rng: np.random.Generator,
    trades_per_day: float,
    n_sims: int,
    regime_labels: Optional[np.ndarray] = None,
    transition_matrix: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run n_sims eval simulations in parallel using Numba JIT kernel.
    Returns (passed: bool array (n_sims,), final_balance: float array (n_sims,),
             days_elapsed: int array (n_sims,)).
    """
    n_pool   = len(pnl_pts)
    MAX_DAYS = 200
    MAX_T    = min(30, max(3, int(trades_per_day * 4)))  # cap per-day trades

    # Pre-generate all random variates — eliminates per-iteration Python RNG overhead.
    all_counts = rng.poisson(trades_per_day, size=(n_sims, MAX_DAYS)).clip(0, MAX_T).astype(np.int16)
    all_idx    = rng.integers(0, n_pool, size=(n_sims, MAX_DAYS, MAX_T), dtype=np.int32)

    # Build regime-partitioned trade pools
    pnl_r0, pnl_r1, pnl_r2, sl_r0, sl_r1, sl_r2 = _build_regime_arrays(
        pnl_pts, sl_dists, regime_labels
    )

    # Build per-sim regime sequence (Markov chain)
    regime_seq = _build_regime_seq(n_sims, MAX_DAYS, transition_matrix, rng)

    passed, balance, days = _eval_loop_nb(
        pnl_pts.astype(np.float32), sl_dists.astype(np.float32),
        all_counts, all_idx, regime_seq,
        pnl_r0, pnl_r1, pnl_r2,
        sl_r0,  sl_r1,  sl_r2,
        len(pnl_r0), len(pnl_r1), len(pnl_r2),
        float(account.starting_balance),
        float(account.mll_amount),
        float(account.profit_target),
        int(account.max_micros),
        float(risk_pct),
        int(SCHEME_CODE.get(scheme, 0)),
        float(MICRO_POINT_VALUE),
        int(MAX_T),
    )
    return passed, balance, days


# ---------------------------------------------------------------------------
# Vectorised funded simulation
# ---------------------------------------------------------------------------

def simulate_funded_batch(
    pnl_pts: np.ndarray,
    sl_dists: np.ndarray,
    account: LucidFlexAccount,
    risk_pct: float,
    scheme: str,
    sizing_mode: str,
    rng: np.random.Generator,
    trades_per_day: float,
    starting_balances: np.ndarray,   # (n_passed,) — varies per sim
) -> np.ndarray:
    """
    Simulate funded phase for all sims that passed eval.
    Returns (total_w, days_to_w, n_payouts, payout_days, payout_amounts, funded_days_elapsed).
    funded_days_elapsed[i] = last trading day sim i was alive (whether it blew out or
    completed all payouts).  Used as the unconditional funded-days denominator.
    """
    n = len(starting_balances)
    if n == 0:
        empty = np.array([])
        return empty, empty, empty, np.full((0, 6), np.nan), np.full((0, 6), 0.0), empty

    MAX_DAYS  = 500
    MAX_T     = min(30, max(3, int(trades_per_day * 4)))
    n_pool    = len(pnl_pts)
    max_pay   = account.max_payouts   # 6

    balance    = starting_balances.copy().astype(np.float32)
    mll_level  = np.full(n, account.starting_balance - account.mll_amount, dtype=np.float32)
    peak_eod   = starting_balances.copy().astype(np.float32)
    mll_locked = np.zeros(n, dtype=bool)
    alive      = np.ones(n, dtype=bool)
    n_payouts  = np.zeros(n, dtype=np.int32)
    total_w    = np.zeros(n, dtype=np.float32)
    prof_days  = np.zeros(n, dtype=np.int32)
    last_won   = np.ones(n, dtype=bool)
    prev_risk  = np.full(n, risk_pct * account.mll_amount, dtype=np.float32)

    # Track day of each payout: shape (n, max_payouts), NaN = not reached
    payout_days    = np.full((n, max_pay), np.nan, dtype=np.float64)
    payout_amounts = np.zeros((n, max_pay), dtype=np.float64)
    # Keep first-payout shorthand for backward compat
    days_to_w   = np.full(n, np.nan, dtype=np.float64)
    # Last trading day each sim was alive (unconditional denominator for per-K ev/day)
    funded_days_elapsed = np.zeros(n, dtype=np.float64)

    # Pre-generate all random variates — eliminates per-iteration Python RNG overhead.
    all_counts = rng.poisson(trades_per_day, size=(n, MAX_DAYS)).clip(0, MAX_T).astype(np.int16)
    all_idx    = rng.integers(0, n_pool, size=(n, MAX_DAYS, MAX_T), dtype=np.int32)

    for day in range(MAX_DAYS):
        if not alive.any():
            break

        # Record last active day before any breach/payout checks so that sims
        # killed this day still count their day toward the denominator.
        funded_days_elapsed = np.where(alive, float(day + 1), funded_days_elapsed)

        n_today = all_counts[:, day].astype(np.int32)
        max_t   = int(n_today.max())
        if max_t == 0:
            continue

        idx      = all_idx[:, day, :max_t]
        t_pnl    = pnl_pts[idx]
        t_sl     = sl_dists[idx]
        target   = compute_target_risk_vec(
            scheme, risk_pct, account, balance, mll_level, last_won, prev_risk
        )
        n_c      = resolve_contracts_vec(target[:, None], t_sl, account.max_micros, sizing_mode)
        dollar_pnl = t_pnl * n_c * MICRO_POINT_VALUE
        trade_mask = np.arange(max_t)[None, :] < n_today[:, None]
        dollar_pnl = np.where(trade_mask, dollar_pnl, 0.0)

        intra_breach = alive & (
            (balance[:, None] + np.cumsum(dollar_pnl, axis=1)) <= mll_level[:, None]
        ).any(axis=1)

        day_pnl    = dollar_pnl.sum(axis=1)
        last_trade = (np.arange(max_t)[None, :] == (n_today-1).clip(0)[:, None])
        last_won   = np.where(alive, (dollar_pnl * last_trade).sum(axis=1) > 0, last_won)
        prev_risk  = np.where(alive, target, prev_risk)

        apply_mask = alive & ~intra_breach
        balance    = np.where(apply_mask, balance + day_pnl, balance)
        alive      = alive & ~intra_breach

        new_peak   = np.maximum(peak_eod, balance)
        peak_eod   = np.where(alive, new_peak, peak_eod)

        just_locked = alive & ~mll_locked & (balance >= account.starting_balance)
        mll_locked  = mll_locked | just_locked
        new_mll     = np.where(
            mll_locked,
            account.starting_balance - 100,
            np.maximum(mll_level, peak_eod - account.mll_amount),
        )
        mll_level = np.where(alive, new_mll, mll_level)

        eod_breach = alive & (balance <= mll_level)
        alive      = alive & ~eod_breach

        # Payout cycle
        prof_days = np.where(alive & (day_pnl > 0), prof_days + 1, prof_days)
        can_payout = alive & (prof_days >= 5)
        if can_payout.any():
            profits = np.maximum(0.0, balance - account.starting_balance)
            gross   = np.minimum(profits * 0.50, account.payout_cap)
            do_pay  = can_payout & (gross >= 500) & (n_payouts < max_pay)

            # Record day and withdrawal amount of each payout by index
            for p_idx in range(max_pay):
                this_pay = do_pay & (n_payouts == p_idx)
                if this_pay.any():
                    payout_days[:, p_idx] = np.where(
                        this_pay, day + 1, payout_days[:, p_idx]
                    )
                    payout_amounts[:, p_idx] = np.where(
                        this_pay, gross * account.split, payout_amounts[:, p_idx]
                    )

            first_pay  = do_pay & (n_payouts == 0)
            days_to_w  = np.where(first_pay, day + 1, days_to_w)
            balance    = np.where(do_pay, balance - gross, balance)
            total_w    = np.where(do_pay, total_w + gross * account.split, total_w)
            n_payouts  = np.where(do_pay, n_payouts + 1, n_payouts)
            prof_days  = np.where(do_pay, 0, prof_days)
            alive      = alive & (n_payouts < max_pay)

    return total_w, days_to_w, n_payouts, payout_days, payout_amounts, funded_days_elapsed


# ---------------------------------------------------------------------------
# Full MC grid — 3D sweep: scheme × eval_risk × funded_risk
# ---------------------------------------------------------------------------

def _run_scheme_worker(args: tuple) -> tuple[str, dict]:
    """
    Worker function for parallel scheme processing.
    Runs eval + funded simulations for a single scheme and returns
    (scheme_name, results_dict).
    Defined at module level so it can be pickled by ProcessPoolExecutor.
    """
    (scheme, pnl_pts, sl_dists, account, eval_risk_pcts, funded_risk_pcts,
     sizing_mode, n_sims, seed, BS, trades_per_day,
     regime_labels, transition_matrix) = args

    # Each worker gets its own RNG seeded deterministically per scheme
    scheme_seed = seed + hash(scheme) % 10_000
    rng = np.random.default_rng(scheme_seed)

    scheme_results = {}

    # ── Eval sims ────────────────────────────────────────────────────────────
    eval_cache: dict = {}
    for erp in eval_risk_pcts:
        passed, _, days_eval = simulate_eval_batch(
            pnl_pts, sl_dists, account, erp, scheme,
            sizing_mode, rng, trades_per_day, n_sims,
            regime_labels=regime_labels,
            transition_matrix=transition_matrix,
        )
        eval_cache[erp] = (passed, days_eval)

    # ── Funded sims ───────────────────────────────────────────────────────────
    funded_start = np.full(n_sims, account.starting_balance, dtype=np.float64)
    funded_cache: dict = {}
    for frp in funded_risk_pcts:
        w_all, dtw_all, n_pay_all, payout_days_all, payout_amounts_all, funded_days_all = simulate_funded_batch(
            pnl_pts, sl_dists, account, frp, scheme,
            sizing_mode, rng, trades_per_day, funded_start,
        )
        funded_cache[frp] = (w_all, dtw_all, n_pay_all, payout_days_all, payout_amounts_all, funded_days_all)

    # ── Combine eval × funded ─────────────────────────────────────────────────
    optimal_funded: dict = {}

    for erp in eval_risk_pcts:
        scheme_results[erp] = {}
        passed, days_eval = eval_cache[erp]

        pass_rate = float(passed.mean())
        if pass_rate > 0:
            e_attempts = 1.0 / pass_rate
            e_cost = account.eval_fee + max(0, e_attempts - 1) * account.reset_fee
        else:
            e_cost = account.eval_fee + 10 * account.reset_fee

        idx     = rng.integers(0, n_sims, size=(BS, n_sims))
        bs_pass = np.median(passed.astype(np.float32)[idx], axis=1)
        pr_ci   = (float(np.percentile(bs_pass, 2.5)),
                   float(np.percentile(bs_pass, 97.5)))

        median_days_pass = (float(np.median(days_eval[passed]))
                            if passed.any() else 0.0)

        best_ev  = -np.inf
        best_frp = funded_risk_pcts[0]

        for frp in funded_risk_pcts:
            w_all, dtw_all, n_pay_all, payout_days_all, payout_amounts_all, funded_days_all = funded_cache[frp]

            # ── Withdrawal metrics ────────────────────────────────────────────
            paid_mask     = w_all > 0
            survival_rate = float(paid_mask.mean())

            # Conditional median: what you earn IF you survive ≥1 payout
            median_w = (float(np.median(w_all[paid_mask]))
                        if paid_mask.any() else 0.0)

            # Unconditional mean total withdrawal across all funded sims
            # (includes zeros from blown accounts — correct input for EV)
            mean_w_uncond = float(w_all.mean())

            # Payout count stats
            median_n_payouts = float(np.median(n_pay_all))
            pct_full_payout  = float((n_pay_all >= account.max_payouts).mean())

            # ── Full-cycle EV (6-payout baseline, computed here, stored later) ──
            # EV = P(pass_eval) × E[total_withdrawal_if_funded] − E[eval_fees]
            # E[total_withdrawal_if_funded] = mean(w_all) unconditionally
            # — already prices in blowout risk via the zero values.
            net_ev_6cycle = pass_rate * mean_w_uncond - e_cost

            # Bootstrap CI on unconditional mean withdrawal
            w_idx = rng.integers(0, n_sims, size=(BS, n_sims))
            bs_w  = w_all[w_idx].mean(axis=1)
            w_ci  = (float(np.percentile(bs_w, 2.5)),
                     float(np.percentile(bs_w, 97.5)))

            valid_dtw     = dtw_all[~np.isnan(dtw_all) & (dtw_all > 0)]
            median_days_w = (float(np.median(valid_dtw))
                             if len(valid_dtw) > 0 else 0.0)

            # ── Full cycle timeline ───────────────────────────────────────────
            # payout_days_all: (n_sims, 6) — day of each payout, NaN if not reached
            # Median day of each payout across sims that reached it (conditional)
            median_payout_days = []
            median_gap_days    = []   # days between consecutive payouts
            for p in range(account.max_payouts):
                col = payout_days_all[:, p]
                valid = col[~np.isnan(col) & (col > 0)]
                median_payout_days.append(float(np.median(valid)) if len(valid) > 0 else None)
                # Gap from previous payout (or from funded start for payout 0)
                if p == 0:
                    median_gap_days.append(median_payout_days[0])
                else:
                    prev_col  = payout_days_all[:, p-1]
                    both_mask = ~np.isnan(col) & ~np.isnan(prev_col) & (col > 0) & (prev_col > 0)
                    gaps      = col[both_mask] - prev_col[both_mask]
                    median_gap_days.append(float(np.median(gaps)) if both_mask.any() else None)

            # Full cycle eval days — correct formula:
            # E[total eval days] = E[failed attempts] × median(days_per_failed_attempt)
            #                    + median(days_per_passing_attempt)
            # Failed sims: those where passed=False, days_elapsed = their failure day
            failed_mask = ~passed
            median_days_fail = (float(np.median(days_eval[failed_mask]))
                                if failed_mask.any() else median_days_pass)
            e_attempts       = 1.0 / pass_rate if pass_rate > 0 else 10.0
            n_failed_attempts = max(0.0, e_attempts - 1.0)
            total_eval_days  = round(
                n_failed_attempts * median_days_fail + median_days_pass, 1
            )
            # Days from funded start to last payout (for sims reaching all 6)
            last_col = payout_days_all[:, account.max_payouts - 1]
            valid_last = last_col[~np.isnan(last_col) & (last_col > 0)]
            median_funded_full_days = float(np.median(valid_last)) if len(valid_last) > 0 else None
            # Calendar day estimate (×1.4 for weekends/holidays)
            total_cycle_calendar = (
                round((total_eval_days + (median_funded_full_days or 0)) * 1.4)
                if median_funded_full_days else None
            )

            funded_full = median_funded_full_days or 0.0
            total_cycle_days = total_eval_days + funded_full

            # EV per day for the full 6-payout cycle (baseline comparison only)
            ev_per_day_6cycle = net_ev_6cycle / max(total_cycle_days, 1.0)

            # ── Per-K analysis: EV/day for targeting exactly K payouts ─────────
            # For each K in 1..max_payouts, compute:
            #   - P(reach K payouts) = fraction of sims with n_payouts >= K
            #   - E[total_net_withdrawal | targeting K] = mean of sum of first K payouts
            #   - E[cycle_days targeting K] = total_eval_days + median(payout_days[:,K-1])
            #   - ev_per_day_k = (P(reach_K) × E[sum_1_to_K] - e_cost) / cycle_days_k
            per_k = []
            for k in range(1, account.max_payouts + 1):
                # sims that reached at least K payouts
                reached_k = n_pay_all >= k
                p_reach_k = float(reached_k.mean())

                # sum of first K payout amounts for sims that reached K
                if reached_k.any():
                    sum_k = payout_amounts_all[reached_k, :k].sum(axis=1)
                    mean_sum_k = float(sum_k.mean())
                    # Unconditional mean (includes zeros from sims that didn't reach K)
                    mean_sum_k_uncond = float(
                        payout_amounts_all[:, :k].sum(axis=1).mean()
                    )
                else:
                    mean_sum_k_uncond = 0.0

                # Unconditional mean funded days when targeting exactly K payouts:
                #   sims that reached K  → days to their K-th payout
                #   sims that didn't     → days they ran before blowing out
                # This correctly penalises the many sims that die before K.
                funded_days_k = np.where(
                    n_pay_all >= k,
                    payout_days_all[:, k - 1],
                    funded_days_all,
                )
                mean_funded_days_k = float(funded_days_k.mean())

                cycle_days_k = total_eval_days + mean_funded_days_k if mean_funded_days_k > 0 else None
                ev_k = pass_rate * mean_sum_k_uncond - e_cost
                ev_per_day_k = (ev_k / cycle_days_k) if cycle_days_k and cycle_days_k > 0 else None

                per_k.append({
                    "k":              k,
                    "p_reach":        round(p_reach_k, 4),
                    "mean_total_w":   round(mean_sum_k_uncond, 2),
                    "ev":             round(ev_k, 2),
                    "cycle_days":     round(cycle_days_k, 1) if cycle_days_k else None,
                    "ev_per_day":     round(ev_per_day_k, 4) if ev_per_day_k else None,
                })
            # Best K by ev_per_day
            valid_k_rows = [r for r in per_k if r["ev_per_day"] is not None]
            best_k_row   = max(valid_k_rows, key=lambda r: r["ev_per_day"]) \
                           if valid_k_rows else per_k[-1]
            optimal_k        = best_k_row["k"]
            optimal_k_ev     = best_k_row["ev"]
            optimal_k_days   = best_k_row["cycle_days"] or total_cycle_days
            ev_per_day_opt   = best_k_row["ev_per_day"] or ev_per_day_6cycle

            scheme_results[erp][frp] = {
                "pass_rate":                 pass_rate,
                "pass_rate_ci":              pr_ci,
                "survival_rate":             survival_rate,
                "median_withdrawal":         median_w,
                "mean_withdrawal":           mean_w_uncond,
                "withdrawal_ci":             w_ci,
                # net_ev = optimal-K EV so it is consistent with ev_per_day.
                # net_ev_6cycle = full 6-payout EV for reference.
                "net_ev":                    round(optimal_k_ev, 2),
                "net_ev_6cycle":             round(net_ev_6cycle, 2),
                "ev_per_day":                round(ev_per_day_opt, 4),
                "optimal_k":                 optimal_k,
                "optimal_k_days":            round(optimal_k_days, 1),
                "per_k":                     per_k,
                "median_n_payouts":          median_n_payouts,
                "pct_full_payout":           pct_full_payout,
                "median_days_to_pass":       median_days_pass,
                "median_days_to_withdrawal": median_days_w,
                "median_payout_days":        median_payout_days,
                "median_gap_days":           median_gap_days,
                "total_eval_days":           round(total_eval_days, 1),
                "total_cycle_days":          round(total_cycle_days, 1),
                "median_funded_full_days":   median_funded_full_days,
                "total_cycle_calendar_days": total_cycle_calendar,
            }

            if ev_per_day_opt > best_ev:
                best_ev  = ev_per_day_opt
                best_frp = frp

        optimal_funded[erp] = best_frp

    scheme_results["optimal_funded_rp"] = optimal_funded
    return scheme, scheme_results


def run_propfirm_grid(
    trades: list,
    account: LucidFlexAccount,
    n_sims: int = 2_000,
    sizing_mode: str = "micros",
    eval_risk_pcts: list[float] = None,
    funded_risk_pcts: list[float] = None,
    schemes: list[str] = None,
    seed: int = 42,
    n_workers: int = None,
    regime_labels: Optional[np.ndarray] = None,
) -> dict:
    """
    Sweep scheme × eval_risk_pct × funded_risk_pct.

    Each scheme is simulated independently and run in parallel using
    ProcessPoolExecutor. On a 4-core machine this gives ~3-3.5× speedup.

    n_workers: number of parallel workers. Default = min(n_schemes, cpu_count).
               Set to 1 to disable parallelism (useful for debugging).
    """
    if eval_risk_pcts is None:
        eval_risk_pcts = EVAL_RISK_PCT_OF_MLL
    if funded_risk_pcts is None:
        funded_risk_pcts = FUNDED_RISK_PCT_OF_MLL
    if schemes is None:
        schemes = RISK_GEOMETRIES

    pnl_pts, sl_dists = extract_normalised_trades(trades)
    if len(pnl_pts) == 0:
        raise ValueError("No trades to simulate")

    if hasattr(trades[0], "entry_bar"):
        trades_per_day = max(0.5, len(trades) / max(1, _estimate_trading_days(trades)))
    else:
        trades_per_day = 1.0

    # ── Regime transition matrix ──────────────────────────────────────────────
    transition_matrix = None
    if regime_labels is not None and len(regime_labels) > 0:
        transition_matrix = _build_transition_matrix_from_labels(regime_labels)

    # ── Numba warmup ─────────────────────────────────────────────────────────
    if _NUMBA_AVAILABLE:
        _dummy = np.zeros(4, dtype=np.float32)
        _dummy_sl = np.ones(4, dtype=np.float32)
        _eval_loop_nb(
            _dummy, _dummy_sl,
            np.ones((2, 3), dtype=np.int16),
            np.zeros((2, 3, 3), dtype=np.int32),
            np.zeros((2, 3), dtype=np.int8),
            _dummy, _dummy, _dummy,
            _dummy_sl, _dummy_sl, _dummy_sl,
            4, 4, 4,
            25000.0, 1000.0, 1250.0, 20,
            0.20, 0, 2.0, 3,
        )

    BS = 500

    # ── Determine worker count ────────────────────────────────────────────────
    import os
    from concurrent.futures import ProcessPoolExecutor, as_completed
    cpu_count  = os.cpu_count() or 1
    # Use physical cores (cpu_count // 2) to avoid hyperthreading overhead on
    # numpy-heavy workloads. Cap at number of schemes.
    phys_cores = max(1, cpu_count // 2)
    if n_workers is None:
        n_workers = min(len(schemes), phys_cores)
    n_workers = max(1, n_workers)

    # ── Build worker args ─────────────────────────────────────────────────────
    worker_args = [
        (scheme, pnl_pts, sl_dists, account, eval_risk_pcts, funded_risk_pcts,
         sizing_mode, n_sims, seed, BS, trades_per_day,
         regime_labels, transition_matrix)
        for scheme in schemes
    ]

    # ── Run in parallel (or serial if n_workers=1) ────────────────────────────
    results: dict = {}
    if n_workers == 1:
        # Serial path — avoids process spawn overhead for small jobs
        for args in worker_args:
            scheme, scheme_results = _run_scheme_worker(args)
            results[scheme] = scheme_results
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_run_scheme_worker, args): args[0]
                       for args in worker_args}
            for future in as_completed(futures):
                scheme, scheme_results = future.result()
                results[scheme] = scheme_results

    _add_stability_scores_3d(results, eval_risk_pcts, funded_risk_pcts, schemes)
    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _estimate_trading_days(trades: list) -> int:
    try:
        bars = [t.entry_bar for t in trades]
        return max(1, (max(bars) - min(bars)) // 390)
    except Exception:
        return len(trades)


def _add_stability_scores_3d(
    results: dict,
    eval_risk_pcts: list[float],
    funded_risk_pcts: list[float],
    schemes: list[str],
) -> None:
    """
    Stability of net_ev with respect to eval_risk changes (holding funded_risk fixed).
    For each (scheme, erp, frp) cell: 1 − max_adjacent_drop / 0.20, clipped [0,1].
    """
    for scheme in schemes:
        n = len(eval_risk_pcts)
        for frp in funded_risk_pcts:
            evs = [results[scheme][erp][frp]["net_ev"] for erp in eval_risk_pcts]
            # Normalise drop relative to the range of EVs
            ev_range = max(abs(max(evs) - min(evs)), 1.0)
            for i, erp in enumerate(eval_risk_pcts):
                drops = []
                if i > 0:     drops.append(abs(evs[i] - evs[i-1]))
                if i < n - 1: drops.append(abs(evs[i] - evs[i+1]))
                max_drop  = max(drops) if drops else 0.0
                stability = max(0.0, 1.0 - max_drop / (ev_range * 0.20))
                results[scheme][erp][frp]["stability"] = stability