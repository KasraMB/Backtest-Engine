"""
Push p_profit at $1K via grind_risk_pct (lower risk in funded phase).

Hypothesis: keep eval risk high to pass quickly, but lower grind risk to
preserve funded gains and accumulate more payouts before any blow-up.

Best so far: cfg_406, erp=0.7, max_c=1 → p_profit=0.7489 at BUDGET=$1K.
Gap to 80%: 5 pts.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from backtest.propfirm.three_phase_mc import ThreePhaseConfig, three_phase_reinvestment_mc
from backtest.propfirm.lucidflex import LUCIDFLEX_ACCOUNTS

ACCOUNT = LUCIDFLEX_ACCOUNTS["50K"]
HORIZON = 84
N_SIMS  = 20_000
SEEDS   = [2026, 2027, 2028]
BUDGET  = 1000.0
MIN_PAYOUT = 500.0
RESET_FEE_DELTA = 7.0


def build_oos_config(cfg_id, amr, oos_start='2023-01-01'):
    data = amr[cfg_id]
    trade_df = data['trades'].copy()
    trade_df['date'] = pd.to_datetime(trade_df['date'])
    days_df = data['days'].copy()
    days_df['date'] = pd.to_datetime(days_df['date'])
    oos_t = trade_df[trade_df['date'] >= oos_start]
    oos_d = days_df[days_df['date'] >= oos_start]
    if len(oos_d) < 50:
        return None
    return ThreePhaseConfig(
        name=f"{cfg_id}_OOS",
        pnl_pts=oos_t['pnl_pts'].to_numpy(np.float32),
        sl_dists=oos_t['sl_dist'].to_numpy(np.float32),
        tpd=float(len(oos_t)) / float(len(oos_d)),
    )


def evaluate(cfg, erp, grind_risk_pct, max_c, seed):
    out = three_phase_reinvestment_mc(
        eval_grind_config=cfg, payout_config=cfg,
        eval_grind_risk=erp,
        grind_risk_pct=grind_risk_pct,
        account=ACCOUNT,
        budget=BUDGET, horizon=HORIZON,
        max_concurrent=max_c, n_sims=N_SIMS, seed=seed,
        enforce_consistency_rule=True,
    )
    cash = out['cash'].astype(np.float64)
    n_evals = out['eval_fees_paid'] / 98.0
    reopens = np.maximum(n_evals - max_c, 0.0)
    cash_adj = cash - RESET_FEE_DELTA * reopens
    return {
        'p_profit': float((cash_adj >= BUDGET + MIN_PAYOUT).mean()),
        'p_zero':   float((cash_adj <= 0).mean()),
        'mean_payouts': float(out['n_payouts'].mean()),
    }


def main():
    amr = pickle.load(open(REPO / 'cache' / 'amr_per_day_trades.pkl', 'rb'))

    print(f"\n{'='*100}")
    print(f"BUDGET=$1K push: eval_risk × grind_risk separation, max_concurrent=1, OOS 2023+")
    print(f"{'='*100}")

    for cfg_id in ['cfg_406', 'cfg_029', 'cfg_495']:
        cfg = build_oos_config(cfg_id, amr)
        if cfg is None:
            continue
        print(f"\n--- {cfg_id}_OOS ---")
        print(f"{'erp':<5} | " + " | ".join(f"grind={gr:.2f}" for gr in [0.05, 0.10, 0.20, 0.30, 0.50, 0.70]))
        for erp in [0.5, 0.7, 0.9]:
            row = []
            for grind in [0.05, 0.10, 0.20, 0.30, 0.50, 0.70]:
                if grind > erp:
                    row.append('  -  ')
                    continue
                res = [evaluate(cfg, erp, grind, 1, s) for s in SEEDS]
                mean_pp = np.mean([r['p_profit'] for r in res])
                row.append(f"{mean_pp:.3f}")
            print(f"{erp:.1f}   | " + " | ".join(f"{v:>9s}" for v in row))


if __name__ == '__main__':
    main()
