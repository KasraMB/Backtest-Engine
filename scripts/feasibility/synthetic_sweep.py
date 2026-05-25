"""Feasibility sweep: parameterize per-trade edge directly and find what (p, RR,
lambda, MAE, rho) combinations cross P(profit) > 80% on LucidFlex 50K Flex at $1K.

Per the methodology-validation agent, the synthetic distribution includes:
  - Win rate p (Bernoulli per trade), with day-level stickiness via a hidden
    2-state Markov regime (GOOD: p*1.2, BAD: p*0.8; stickiness rho).
  - RR (win = +RR, loss = -1).
  - Trades/day lambda (Poisson, capped at max_trades=10).
  - MAE_R coupled to RR: on wins, mean = alpha*min(RR,1.0) + 0.2; on losses,
    1.0 + Exp(0.1) (slips past SL).
  - SL_dist fixed at 16 pts (sizing invariant for the R-based sim).

AMR anchor: AMR cfg495+ToD has p~0.55, RR=1.0, lambda~0.4, MAE_R~0.6 on wins.
The sweep includes this point to verify the sim reproduces ~54% p_profit.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from backtest.portfolio.sim import build_per_slot_arrays, propfirm_p_profit_slots

SL_PTS = 16.0
MAX_TRADES_PER_DAY = 10
N_TRAIN_DAYS = 1000
SEEDS = (2026, 2027)
ERP_GRID = (0.3, 0.4, 0.5, 0.7)


def gen_trades(p: float, RR: float, lam: float, mae_alpha: float,
               rho: float, n_days: int, seed: int) -> pd.DataFrame:
    """Generate a synthetic trade DataFrame (cols: date, R, sl_dist, MAE_R).

    Hidden 2-state Markov regime per day: stays with prob rho. GOOD scales p
    by 1.2 (capped 1), BAD by 0.8. Within day: n trades ~ Poisson(lam),
    capped. Per trade win ~ Bernoulli(p_today); R = +RR or -1; MAE_R coupled.
    """
    rng = np.random.default_rng(seed)
    base_date = pd.Timestamp('2019-01-01')
    state = 0   # 0=GOOD, 1=BAD
    rows = []
    for d in range(n_days):
        # Markov transition
        if rng.random() > rho:
            state = 1 - state
        p_today = min(max(p * (1.2 if state == 0 else 0.8), 0.01), 0.99)
        n_t = min(rng.poisson(lam), MAX_TRADES_PER_DAY)
        if n_t == 0:
            continue
        dt = base_date + pd.Timedelta(days=d)
        wins = rng.random(n_t) < p_today
        for w in wins:
            if w:
                R = float(RR)
                mae_mean = mae_alpha * min(RR, 1.0) + 0.2
                mae = max(0.0, mae_mean + rng.uniform(-0.1, 0.1))
            else:
                R = -1.0
                mae = 1.0 + float(rng.exponential(0.1))
            rows.append({'date': dt, 'R': R, 'sl_dist': SL_PTS, 'MAE_R': mae})
    return pd.DataFrame(rows)


def measure_pprofit(df: pd.DataFrame) -> float:
    if len(df) == 0:
        return 0.0
    arr = build_per_slot_arrays([df])
    best = 0.0
    for erp in ERP_GRID:
        pp = float(np.mean([propfirm_p_profit_slots(arr, erp, budget=1000, n_sims=8000,
                                                     seed=s)['p_profit'] for s in SEEDS]))
        if pp > best:
            best = pp
    return best


def main():
    # Sweep grid
    P_GRID   = (0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75)
    RR_GRID  = (1.0, 1.5, 2.0, 3.0)
    LAM_GRID = (0.5, 1, 2, 5)
    MAE_REGIMES = (('TIGHT', 0.0), ('REAL', 0.5))    # alpha=0.0 -> mean=0.2; alpha=0.5 -> mean=0.7
    RHO_GRID = (0.5, 0.7)    # 0.5 = i.i.d. days, 0.7 = sticky regimes

    # Anchor: AMR cfg495+ToD-like profile
    print('=== ANCHOR check (AMR cfg495+ToD ~ p=0.56, RR=1.0, lambda=0.4, MAE=0.6, rho=0.5) ===')
    anchor = gen_trades(p=0.56, RR=1.0, lam=0.4, mae_alpha=0.4, rho=0.5,
                        n_days=1000, seed=2026)
    anchor_pp = measure_pprofit(anchor)
    print(f'  n={len(anchor)} mean_R={anchor.R.mean():+.4f} | sim p_profit={anchor_pp:.4f}')
    print(f'  (empirical AMR cfg495+ToD = ~0.54 at $1K; calibration {"OK" if 0.40 < anchor_pp < 0.65 else "OFF"})')
    print()

    print('=== FEASIBILITY SWEEP ===')
    print('p_profit > 0.80 cells flagged with ***; > 0.70 with **; > 0.60 with *')
    rows = []
    for mae_name, alpha in MAE_REGIMES:
        for rho in RHO_GRID:
            for lam in LAM_GRID:
                for RR in RR_GRID:
                    for p in P_GRID:
                        df = gen_trades(p, RR, lam, alpha, rho,
                                        n_days=N_TRAIN_DAYS, seed=2026)
                        pp = measure_pprofit(df)
                        rows.append({'MAE': mae_name, 'rho': rho, 'lam': lam,
                                     'RR': RR, 'p': p, 'p_profit': pp,
                                     'n_trades': len(df)})
    df_all = pd.DataFrame(rows)
    df_all.to_csv(REPO / 'results' / 'feasibility_sweep.csv', index=False)

    # Frontier: rows where p_profit > 0.80
    hits = df_all[df_all['p_profit'] > 0.80].sort_values('p_profit', ascending=False)
    print(f"\nCombos hitting p_profit > 0.80: {len(hits)} / {len(df_all)}")
    if len(hits):
        print(hits.head(20).to_string(index=False))
    print(f"\nCombos hitting > 0.70: {(df_all['p_profit'] > 0.70).sum()}")
    print(f"Combos hitting > 0.60: {(df_all['p_profit'] > 0.60).sum()}")
    print(f"Max p_profit observed: {df_all['p_profit'].max():.4f}")
    print(f"\nSaved -> {REPO / 'results' / 'feasibility_sweep.csv'}")


if __name__ == '__main__':
    main()
