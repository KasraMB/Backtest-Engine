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
from concurrent.futures import ProcessPoolExecutor
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import time
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from backtest.portfolio.sim import build_per_slot_arrays, propfirm_p_profit_slots

SL_PTS = 16.0
SLIPPAGE_PTS = 0.5             # 2-tick/side RT on NQ -> baked into R like backtest_cell does
SLIP_R = SLIPPAGE_PTS / SL_PTS # ~0.0313 R drag per trade (win AND loss)
MAX_TRADES_PER_DAY = 10
N_TRAIN_DAYS = 1000
SEEDS = (2026, 2027)
ERP_GRID = (0.3, 0.4, 0.5, 0.7)


def gen_trades(
    p: float,
    RR: float,
    lam: float,
    mae_alpha: float,
    rho: float,
    n_days: int,
    seed: int,
) -> pd.DataFrame:
    """
    Vectorized synthetic trade generator.
    """

    rng = np.random.default_rng(seed)

    # -------------------------------------------------
    # Daily hidden Markov state
    # -------------------------------------------------
    states = np.empty(n_days, dtype=np.int8)

    state = 0  # 0=GOOD, 1=BAD

    for d in range(n_days):
        if rng.random() > rho:
            state ^= 1
        states[d] = state

    # GOOD=1.2x, BAD=0.8x
    p_days = np.clip(
        p * np.where(states == 0, 1.2, 0.8),
        0.01,
        0.99,
    )

    # -------------------------------------------------
    # Trades/day
    # -------------------------------------------------
    n_trades = np.minimum(
        rng.poisson(lam, size=n_days),
        MAX_TRADES_PER_DAY,
    )

    total_trades = int(n_trades.sum())

    if total_trades == 0:
        return pd.DataFrame(
            columns=['date', 'R', 'sl_dist', 'MAE_R']
        )

    # -------------------------------------------------
    # Expand day index -> per-trade rows
    # -------------------------------------------------
    day_idx = np.repeat(np.arange(n_days), n_trades)

    p_trade = p_days[day_idx]

    # -------------------------------------------------
    # Wins/losses
    # -------------------------------------------------
    wins = rng.random(total_trades) < p_trade

    # -------------------------------------------------
    # R values
    # -------------------------------------------------
    R = np.where(
        wins,
        RR - SLIP_R,
        -1.0 - SLIP_R,
    )

    # -------------------------------------------------
    # MAE
    # -------------------------------------------------
    mae = np.empty(total_trades, dtype=np.float32)

    win_mask = wins
    loss_mask = ~wins

    mae_mean = mae_alpha * min(RR, 1.0) + 0.2

    n_wins = int(win_mask.sum())
    n_losses = total_trades - n_wins

    mae[win_mask] = np.maximum(
        0.0,
        mae_mean + rng.uniform(-0.1, 0.1, size=n_wins),
    )

    mae[loss_mask] = (
        1.0
        + rng.exponential(0.1, size=n_losses)
    )

    # -------------------------------------------------
    # Dates
    # -------------------------------------------------
    base_date = np.datetime64('2019-01-01')

    dates = (
        base_date
        + day_idx.astype('timedelta64[D]')
    )

    # -------------------------------------------------
    # Build DataFrame once
    # -------------------------------------------------
    return pd.DataFrame({
        'date': dates,
        'R': R.astype(np.float32),
        'sl_dist': np.full(total_trades, SL_PTS, dtype=np.float32),
        'MAE_R': mae,
    })

def measure_pprofit(df: pd.DataFrame) -> float:
    if len(df) == 0:
        return 0.0
    t0 = time.perf_counter()
    arr = build_per_slot_arrays([df])
    print(f'build_per_slot_arrays: {time.perf_counter() - t0:.3f}s')
    best = 0.0
    for erp in ERP_GRID:
        pp = float(np.mean([propfirm_p_profit_slots(arr, erp, budget=1000, n_sims=8000,
                                                     seed=s)['p_profit'] for s in SEEDS]))
        if pp > best:
            best = pp
    return best

def run_cell(args):
    mae_name, alpha, rho, lam, RR, p = args

    df = gen_trades(
        p=p,
        RR=RR,
        lam=lam,
        mae_alpha=alpha,
        rho=rho,
        n_days=N_TRAIN_DAYS,
        seed=2026,
    )

    pp = measure_pprofit(df)

    return {
        'MAE': mae_name,
        'rho': rho,
        'lam': lam,
        'RR': RR,
        'p': p,
        'p_profit': pp,
        'n_trades': len(df),
    }


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
    t0 = time.perf_counter()
    anchor_pp = measure_pprofit(anchor)
    print(f'measure_pprofit time = {time.perf_counter() - t0:.2f}s')
    print(f'  n={len(anchor)} mean_R={anchor.R.mean():+.4f} | sim p_profit={anchor_pp:.4f}')
    print(f'  (empirical AMR cfg495+ToD = ~0.54 at $1K; calibration {"OK" if 0.40 < anchor_pp < 0.65 else "OFF"})')
    print()

    print('=== FEASIBILITY SWEEP ===')
    print('p_profit > 0.80 cells flagged with ***; > 0.70 with **; > 0.60 with *')


    params = [
    (mae_name, alpha, rho, lam, RR, p)
        for mae_name, alpha in MAE_REGIMES
        for rho in RHO_GRID
        for lam in LAM_GRID
        for RR in RR_GRID
        for p in P_GRID
    ]

    with ProcessPoolExecutor() as ex:
        rows = list(ex.map(run_cell, params))

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
