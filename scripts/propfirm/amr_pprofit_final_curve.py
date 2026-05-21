"""
Final BUDGET-sensitivity curve at OPTIMAL settings (mc=1, all corrections).

User asked for P(profit) > 80% at BUDGET=$1K.
Best at $1K = 75.2% (cfg_406, erp=0.7, mc=1).

Question: at what BUDGET does p_profit cross 80%? Report curve so user can
choose either accept lower target or higher budget.
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
MIN_PAYOUT = 500.0
RESET_FEE_DELTA = -3.0   # $95 reset - $98 eval (confirmed pricing)


def build_oos(cfg_id, amr):
    data = amr[cfg_id]
    td = data['trades'].copy()
    td['date'] = pd.to_datetime(td['date'])
    dd = data['days'].copy()
    dd['date'] = pd.to_datetime(dd['date'])
    oos_t = td[td['date'] >= '2023-01-01']
    oos_d = dd[dd['date'] >= '2023-01-01']
    return ThreePhaseConfig(
        name=f"{cfg_id}_OOS",
        pnl_pts=oos_t['pnl_pts'].to_numpy(np.float32),
        sl_dists=oos_t['sl_dist'].to_numpy(np.float32),
        tpd=float(len(oos_t)) / float(len(oos_d)),
    )


def evaluate(cfg, erp, budget, mc, seed):
    out = three_phase_reinvestment_mc(
        eval_grind_config=cfg, payout_config=cfg,
        eval_grind_risk=erp, account=ACCOUNT,
        budget=budget, horizon=HORIZON,
        max_concurrent=mc, n_sims=N_SIMS, seed=seed,
        enforce_consistency_rule=True,
    )
    cash = out['cash'].astype(np.float64)
    n_evals = out['eval_fees_paid'] / 98.0
    reopens = np.maximum(n_evals - mc, 0.0)
    cash_adj = cash - RESET_FEE_DELTA * reopens
    return float((cash_adj >= budget + MIN_PAYOUT).mean())


def main():
    amr = pickle.load(open(REPO / 'cache' / 'amr_per_day_trades.pkl', 'rb'))
    cfg_406 = build_oos('cfg_406', amr)
    cfg_029 = build_oos('cfg_029', amr)

    print(f"\n{'='*90}")
    print(f"FINAL BUDGET-SENSITIVITY: erp=0.7, max_concurrent=1, OOS 2023+, all corrections applied")
    print(f"LucidFlex 50K Flex (verified rules: $98 eval, $95 reset, consistency ON)")
    print(f"Pass criterion: cash_adj >= BUDGET + $500 (one real payout cleared)")
    print(f"{'='*90}")

    BUDGETS = [1000, 1500, 2000, 2500, 3000, 4000, 5000, 7500, 10000]
    print(f"\n{'BUDGET':<10s} | {'cfg_406':<15s} | {'cfg_029':<15s} | crosses 80%?")
    print('-' * 70)
    for budget in BUDGETS:
        pp_406 = np.mean([evaluate(cfg_406, 0.7, float(budget), 1, s) for s in SEEDS])
        pp_029 = np.mean([evaluate(cfg_029, 0.7, float(budget), 1, s) for s in SEEDS])
        tag_406 = " <-PASS" if pp_406 > 0.80 else ""
        tag_029 = " <-PASS" if pp_029 > 0.80 else ""
        crossed = "YES" if (pp_406 > 0.80 or pp_029 > 0.80) else "no"
        print(f"${budget:<9d} | {pp_406:.4f}{tag_406:<8s} | {pp_029:.4f}{tag_029:<8s} | {crossed}")

    # Also check mc=2 at higher budgets
    print(f"\n--- Same curve at max_concurrent=2 ---")
    print(f"{'BUDGET':<10s} | {'cfg_406_mc2':<15s} | {'cfg_029_mc2':<15s}")
    print('-' * 50)
    for budget in BUDGETS:
        pp_406 = np.mean([evaluate(cfg_406, 0.7, float(budget), 2, s) for s in SEEDS])
        pp_029 = np.mean([evaluate(cfg_029, 0.7, float(budget), 2, s) for s in SEEDS])
        print(f"${budget:<9d} | {pp_406:.4f}        | {pp_029:.4f}")


if __name__ == '__main__':
    main()
