from collections import deque
from itertools import product

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import bicgstab
# round trip cost per trade
COST = 3


START_BALANCE = 0
START_TARGET = 3000
TRAIL_DISTANCE = 2000
TRAIL_LOCK = 2100
BASE_MAX_WIN = START_TARGET / 2

Q = 4  # cents; round to avoid float drift in state keys


def _barrier(highest):
    return (min(highest, TRAIL_LOCK) - TRAIL_DISTANCE)


def build_and_solve(INITIAL_RISK, RR):
    WIN_PROB = 1.0 / (1.0 + RR)
    LOSS_PROB = 1.0 - WIN_PROB

    def key(b, h, m):
        return (round(b * Q), round(h * Q), round(m * Q))

    start = (START_BALANCE, START_BALANCE, BASE_MAX_WIN)
    q = deque([start])
    state_idx = {key(*start): 0}
    states = [start]

    WIN_LIST, LOSS_LIST, TERM_P, IS_TERM_LIST = [], [], [], []

    while q:
        balance, highest, max_win_ref = q.popleft()
        barrier = _barrier(highest)

        if balance <= barrier:
            IS_TERM_LIST.append(True); TERM_P.append(0.0)
            WIN_LIST.append(0); LOSS_LIST.append(0); continue
        if balance >= max_win_ref * 2:
            IS_TERM_LIST.append(True); TERM_P.append(1.0)
            WIN_LIST.append(0); LOSS_LIST.append(0); continue

        r = balance - barrier
        risk = INITIAL_RISK if r > INITIAL_RISK else r
        win_pnl = risk * RR - COST
        loss_pnl = -risk - COST

        new_bal_w = balance + win_pnl
        new_high_w = new_bal_w if new_bal_w > highest else highest
        new_mwr = win_pnl if win_pnl > max_win_ref else max_win_ref

        w = (new_bal_w, new_high_w, new_mwr)
        l = (balance + loss_pnl, highest, max_win_ref)

        IS_TERM_LIST.append(False); TERM_P.append(0.0)

        for child in (w, l):
            k = key(*child)
            if k not in state_idx:
                state_idx[k] = len(states)
                states.append(child)
                q.append(child)
            (WIN_LIST if child is w else LOSS_LIST).append(state_idx[k])

    P = np.array(TERM_P, dtype=np.float64)
    WI = np.array(WIN_LIST, dtype=np.int32)
    LI = np.array(LOSS_LIST, dtype=np.int32)
    IS_T = np.array(IS_TERM_LIST, dtype=bool)

    N = len(states)
    nt = np.where(~IS_T)[0]
    n = len(nt)
    pos = -np.ones(N, dtype=np.int64); pos[nt] = np.arange(n)

    wi_f, li_f = WI[nt], LI[nt]
    wi_p, li_p = pos[wi_f], pos[li_f]
    w_nt, l_nt = wi_p >= 0, li_p >= 0

    # Build (I - M_nn) sparse
    diag_r = np.arange(n); diag_v = np.ones(n)
    w_r = np.where(w_nt)[0]; w_c = wi_p[w_nt]; w_v = np.full(w_r.size, -WIN_PROB)
    l_r = np.where(l_nt)[0]; l_c = li_p[l_nt]; l_v = np.full(l_r.size, -LOSS_PROB)
    A = csr_matrix(
        (np.concatenate([diag_v, w_v, l_v]),
         (np.concatenate([diag_r, w_r, l_r]), np.concatenate([diag_r, w_c, l_c]))),
        shape=(n, n),
    )

    # Terminal contributions for P's RHS (E, H, G terminals = 0)
    b_P = np.zeros(n)
    b_P[~w_nt] += WIN_PROB * P[wi_f[~w_nt]]
    b_P[~l_nt] += LOSS_PROB * P[li_f[~l_nt]]

    def solve(b):
        x, _ = bicgstab(A, b, rtol=1e-10, atol=1e-12, maxiter=10_000)
        return x

    P[nt] = solve(b_P)
    Q_ = 1.0 - P
    E = np.zeros(N); E[nt] = solve(np.ones(n))
    H = np.zeros(N); H[nt] = solve(P[nt])
    G = np.zeros(N); G[nt] = solve(Q_[nt])

    si = state_idx[key(*start)]
    profit_prob = P[si]; stop_prob = 1.0 - profit_prob

    return (
        profit_prob, stop_prob,
        H[si] / profit_prob if profit_prob > 0 else float("nan"),
        G[si] / stop_prob if stop_prob > 0 else float("nan"),
        E[si],
    )


RR_VALUES = [1, 1.5]
RISK_VALUES = [1505, 1503, 1000, 1003]

results = []
for rr, risk in product(RR_VALUES, RISK_VALUES):
    results.append((rr, risk, *build_and_solve(risk, rr)))

print("\nRESULTS")
print("RR\tRisk\tProfit%\t\tFail%\t\tE[T|Pass]\tE[T|Fail]\tE[T|All]")
print("-" * 100)
for rr, risk, p, f, ep, ef, ea in results:
    print(f"{rr:.2f}\t{risk}\t{p:.4%}\t{f:.4%}\t{ep:.3f}\t\t{ef:.3f}\t\t{ea:.3f}")
