"""
Propfirm reinvestment simulation — calls lucidflex.py Numba kernels directly.
Reconstructed after merge loss.

Key exports: simulate_lifecycles, reinvestment_mc, BUDGET, HORIZON, GOAL.
"""
from __future__ import annotations
import numpy as np
from backtest.propfirm.lucidflex import (
    simulate_eval_batch, simulate_funded_batch,
    LUCIDFLEX_ACCOUNTS, MICRO_COMM_RT, MINI_COMM_RT,
)

BUDGET  = 300.0
HORIZON = 84        # trading days
GOAL    = 10_000.0

MAX_PAY = 6


def simulate_lifecycles(
    pnl_pts:       np.ndarray,
    sl_dists:      np.ndarray,
    tpd:           float,
    account,
    eval_risk:     float,
    fund_risk:     float,
    N:             int   = 40_000,
    max_eval_days: int   = 300,
    max_fund_days: int   = 250,
    seed:          int   = 42,
) -> dict:
    """
    Pre-simulate N propfirm account lifecycles using lucidflex Numba kernels.

    Returns dict:
      pass_day   (N,)      int32   — eval pass day (<0 = never)
      n_payouts  (N,)      int32   — number of funded payouts received
      pay_days   (N, 6)    int32   — days from account open to each payout
      pay_amts   (N, 6)    float64 — net cash per payout
      ev_per_day float     — mean total payout / HORIZON
    """
    rng        = np.random.default_rng(seed)
    max_minis  = int(account.max_micros) // 10

    # ── EVAL PHASE ────────────────────────────────────────────────────────────
    passed, _bal, eval_days = simulate_eval_batch(
        pnl_pts, sl_dists, account,
        risk_pct    = eval_risk,
        scheme      = "fixed_dollar",
        sizing_mode = "nq_first",
        rng         = rng,
        trades_per_day = tpd,
        n_sims      = N,
        max_minis   = max_minis,
        mini_comm_rt= MINI_COMM_RT,
    )

    pass_day  = np.full(N, -1,         dtype=np.int32)
    n_payouts = np.zeros(N,             dtype=np.int32)
    pay_days  = np.zeros((N, MAX_PAY),  dtype=np.int32)
    pay_amts  = np.zeros((N, MAX_PAY),  dtype=np.float64)

    pass_day[passed] = eval_days[passed].astype(np.int32)

    n_passed = int(passed.sum())
    if n_passed == 0:
        return dict(pass_day=pass_day, n_payouts=n_payouts,
                    pay_days=pay_days, pay_amts=pay_amts, ev_per_day=0.0)

    # ── FUNDED PHASE ──────────────────────────────────────────────────────────
    sb_f = np.full(n_passed, float(account.starting_balance), dtype=np.float32)

    total_w, _d2w, n_pyt, pyt_days, pyt_amts, _fd = simulate_funded_batch(
        pnl_pts, sl_dists, account,
        risk_pct          = fund_risk,
        scheme            = "fixed_dollar",
        sizing_mode       = "nq_first",
        rng               = rng,
        trades_per_day    = tpd,
        starting_balances = sb_f,
        max_minis         = max_minis,
        mini_comm_rt      = MINI_COMM_RT,
    )

    # Store results — pay_days relative to account open = eval_days + funded_day
    passed_idx = np.where(passed)[0]
    n_payouts[passed] = n_pyt.astype(np.int32)

    eval_d = pass_day[passed].astype(np.float64)   # eval days for passers
    for k in range(MAX_PAY):
        fd_k = pyt_days[:, k]                       # funded day of kth payout
        valid = ~np.isnan(fd_k)
        pay_days[passed_idx[valid], k] = (eval_d[valid] + fd_k[valid]).astype(np.int32)
        pay_amts[passed_idx[valid], k] = pyt_amts[:, k][valid]

    ev_per_day = float(pay_amts.sum(axis=1).mean()) / max(HORIZON, 1)

    return dict(
        pass_day  = pass_day,
        n_payouts = n_payouts,
        pay_days  = pay_days,
        pay_amts  = pay_amts,
        ev_per_day= ev_per_day,
    )


def reinvestment_mc(
    pool:     dict,
    eval_fee: float,
    budget:   float = BUDGET,
    horizon:  int   = HORIZON,
    n_sims:   int   = 10_000,
    seed:     int   = 99,
    _max_acc: int   = 300,
) -> np.ndarray:
    """
    Vectorised reinvestment MC over `horizon` trading days.
    Draws lifecycles from pool, opens new accounts whenever cash >= eval_fee.
    Returns final cash array of shape (n_sims,).
    """
    rng      = np.random.default_rng(seed)
    N_pool   = len(pool["pass_day"])
    pay_days = pool["pay_days"]   # (N_pool, MAX_PAY)
    pay_amts = pool["pay_amts"]   # (N_pool, MAX_PAY)
    n_pyts   = pool["n_payouts"]  # (N_pool,)

    N   = n_sims
    MAX = _max_acc

    active   = np.zeros((N, MAX), dtype=bool)
    nxt_abs  = np.full( (N, MAX), horizon + 1, dtype=np.int32)
    nxt_amt  = np.zeros((N, MAX), dtype=np.float64)
    nxt_k    = np.zeros((N, MAX), dtype=np.int8)
    max_k    = np.zeros((N, MAX), dtype=np.int8)
    open_d   = np.zeros((N, MAX), dtype=np.int32)
    life_i   = np.zeros((N, MAX), dtype=np.int32)
    nused    = np.zeros(N, dtype=np.int32)
    cash     = np.full(N, budget,  dtype=np.float64)
    wdrn     = np.zeros(N, dtype=np.float64)   # total payouts received

    def _open(day: int) -> None:
        can = (cash >= eval_fee) & (nused < MAX)
        while can.any():
            si  = np.where(can)[0]
            ac  = nused[si]
            idx = rng.integers(0, N_pool, len(si), dtype=np.int32)
            mk  = n_pyts[idx].astype(np.int8)
            has = mk > 0

            life_i[si, ac]  = idx
            open_d[si, ac]  = day
            nxt_k[si, ac]   = 0
            max_k[si, ac]   = mk

            sy = si[has];  ay = ac[has];  iy = idx[has]
            sn = si[~has]; an = ac[~has]

            active[sy, ay]  = True
            nxt_abs[sy, ay] = day + pay_days[iy, 0]
            nxt_amt[sy, ay] = pay_amts[iy, 0]

            active[sn, an]  = False
            nxt_abs[sn, an] = horizon + 1

            cash[si] -= eval_fee
            nused[si] += 1
            nused[sn] -= 1
            np.maximum(nused, 0, out=nused)

            can = (cash >= eval_fee) & (nused < MAX)

    _open(0)

    for day in range(1, horizon + 1):
        due = active & (nxt_abs == day)
        if due.any():
            pay = np.where(due, nxt_amt, 0.0).sum(axis=1)
            cash += pay
            wdrn += pay

            k1   = (nxt_k + 1).astype(np.int8)
            more = due & (k1 < max_k)
            done = due & ~more

            si_m, ac_m = np.where(more)
            if len(si_m):
                li = life_i[si_m, ac_m]
                k  = k1[si_m, ac_m]
                nxt_k[si_m, ac_m]   = k
                nxt_abs[si_m, ac_m] = open_d[si_m, ac_m] + pay_days[li, k]
                nxt_amt[si_m, ac_m] = pay_amts[li, k]

            si_d, ac_d = np.where(done)
            if len(si_d):
                active[si_d, ac_d]  = False
                nxt_abs[si_d, ac_d] = horizon + 1
                nused[si_d] -= 1
                np.maximum(nused, 0, out=nused)

        _open(day)

    return wdrn
