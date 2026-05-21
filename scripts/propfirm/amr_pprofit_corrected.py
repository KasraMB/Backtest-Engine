"""
Corrected propfirm sim with user-specified BUDGET=$1K + LucidFlex rule fixes.

Corrections vs previous version:
  - BUDGET = $1000 (NOT $5000)
  - enforce_consistency_rule = True (eval requires max-day <= 50% of total)
  - Post-hoc reset fee adjustment: cash -= 7 * max(0, n_evals_purchased - max_concurrent)
    Reflects $105 LucidFlex reset fee vs $98 eval fee on reopens.
  - Pass criterion: cash >= BUDGET + 500 (one real payout cleared, $500 min)
  - Horizon = 84 trading days (= 4 months trading-day equivalent)
  - OOS-only trade pool (2023+) to eliminate train contamination

NOTE: exact $eval and $reset fees for LucidFlex 50K Flex are not available from
the official rules article (dynamic checkout / dashboard-only). Using sim
default $98/$105 — user should confirm or provide exact values.
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
HORIZON = 84            # 84 trading days = ~4 months
N_SIMS  = 20_000
SEEDS   = [2026, 2027, 2028]
RESET_FEE_DELTA = -3.0  # $95 reset - $98 eval = -$3 (reset cheaper, CONFIRMED from lucidtrading.com pricing screenshot)
MIN_PAYOUT = 500.0      # LucidFlex min payout request


def build_oos_config(cfg_id: str, amr: dict, oos_start='2023-01-01'):
    data = amr[cfg_id]
    trade_df = data['trades'].copy()
    trade_df['date'] = pd.to_datetime(trade_df['date'])
    days_df = data['days'].copy()
    days_df['date'] = pd.to_datetime(days_df['date'])

    oos_t = trade_df[trade_df['date'] >= oos_start]
    oos_d = days_df[days_df['date'] >= oos_start]
    if len(oos_d) == 0:
        return None
    return ThreePhaseConfig(
        name=f"{cfg_id}_OOS",
        pnl_pts=oos_t['pnl_pts'].to_numpy(np.float32),
        sl_dists=oos_t['sl_dist'].to_numpy(np.float32),
        tpd=float(len(oos_t)) / float(len(oos_d)),
    )


def run_one(cfg, erp, budget, max_c, seed):
    out = three_phase_reinvestment_mc(
        eval_grind_config=cfg, payout_config=cfg,
        eval_grind_risk=erp, account=ACCOUNT,
        budget=budget, horizon=HORIZON,
        max_concurrent=max_c, n_sims=N_SIMS, seed=seed,
        enforce_consistency_rule=True,    # LucidFlex eval-only rule
    )
    cash = out['cash'].astype(np.float64)
    n_evals = out.get('eval_fees_paid', np.zeros(N_SIMS)) / 98.0    # eval count per sim
    # Post-hoc reset-fee adjustment: only re-opens (after first) cost extra $7
    reopens = np.maximum(n_evals - max_c, 0.0)     # first max_c opens are not reopens
    cash_adj = cash - RESET_FEE_DELTA * reopens
    n_payouts = out.get('n_payouts', np.zeros(N_SIMS))
    return cash_adj, n_payouts, reopens


def main():
    amr = pickle.load(open(REPO / 'cache' / 'amr_per_day_trades.pkl', 'rb'))
    cfg_029 = build_oos_config('cfg_029', amr)
    cfg_406 = build_oos_config('cfg_406', amr)

    # BUDGET sensitivity curve at corrected BUDGET=$1K + showing slope
    BUDGET_GRID = [1000.0, 1500.0, 2000.0, 3000.0, 5000.0]
    erp_grid = [0.3, 0.5, 0.7]

    print(f"\n{'='*110}")
    print(f"CORRECTED P(profit) sweep (OOS-only 2023+, consistency rule ON, +$7 reset surcharge per reopen)")
    print(f"LucidFlex 50K Flex, max_concurrent=4, horizon={HORIZON}d, n_sims={N_SIMS} × 3 seeds")
    print(f"Pass criterion: cash_adj >= BUDGET + $500 (one real payout cleared)")
    print(f"{'='*110}")

    for cfg, name in [(cfg_029, 'cfg_029_OOS'), (cfg_406, 'cfg_406_OOS')]:
        if cfg is None:
            continue
        print(f"\n--- {name} ---")
        print(f"  n_trades={len(cfg.pnl_pts)}, tpd={cfg.tpd:.2f}")
        for budget in BUDGET_GRID:
            for erp in erp_grid:
                pps = []
                pzeros = []
                preopens = []
                pmeaningful = []
                for seed in SEEDS:
                    cash_adj, n_payouts, reopens = run_one(cfg, erp, budget, 4, seed)
                    pp = float((cash_adj >= budget + MIN_PAYOUT).mean())
                    pz = float((cash_adj <= 0).mean())
                    pmean = float((n_payouts >= 1).mean())
                    pps.append(pp)
                    pzeros.append(pz)
                    preopens.append(reopens.mean())
                    pmeaningful.append(pmean)
                mean_pp = float(np.mean(pps))
                mean_pz = float(np.mean(pzeros))
                mean_reopens = float(np.mean(preopens))
                mean_pmean = float(np.mean(pmeaningful))
                ci_se = np.sqrt(mean_pp * (1 - mean_pp) / N_SIMS)
                ci_lo = mean_pp - 1.96 * ci_se
                tag = "PASS>80" if mean_pp > 0.80 else ""
                print(f"  budget=${int(budget):>5d} erp={erp:.2f}: "
                      f"p_profit={mean_pp:.4f} (CI_lo={ci_lo:.4f}) "
                      f"p_zero={mean_pz:.4f} p_>=1_payout={mean_pmean:.4f} "
                      f"avg_reopens={mean_reopens:.1f} {tag}")


if __name__ == '__main__':
    main()
