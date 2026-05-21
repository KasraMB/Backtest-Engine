"""
Adaptive max_concurrent: start small, grow per payout collected.

Best so far at BUDGET=$1K: cfg_406 erp=0.7 max_c=1 → p_profit=0.749.
Gap to 80%: 5 pts.

Hypothesis: starting with max_c=1 minimises early-burn risk; growing per
payout uses cash pool gains to fund additional concurrent accounts.
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


def build_oos(cfg_id, amr, oos='2023-01-01'):
    data = amr[cfg_id]
    trade_df = data['trades'].copy()
    trade_df['date'] = pd.to_datetime(trade_df['date'])
    days_df = data['days'].copy()
    days_df['date'] = pd.to_datetime(days_df['date'])
    oos_t = trade_df[trade_df['date'] >= oos]
    oos_d = days_df[days_df['date'] >= oos]
    if len(oos_d) < 50:
        return None
    return ThreePhaseConfig(
        name=f"{cfg_id}_OOS",
        pnl_pts=oos_t['pnl_pts'].to_numpy(np.float32),
        sl_dists=oos_t['sl_dist'].to_numpy(np.float32),
        tpd=float(len(oos_t)) / float(len(oos_d)),
    )


def evaluate(cfg, erp, mc_init, grow, mc_max, seed):
    out = three_phase_reinvestment_mc(
        eval_grind_config=cfg, payout_config=cfg,
        eval_grind_risk=erp, account=ACCOUNT,
        budget=BUDGET, horizon=HORIZON,
        max_concurrent=mc_max, n_sims=N_SIMS, seed=seed,
        enforce_consistency_rule=True,
        max_concurrent_initial=mc_init,
        max_concurrent_grow_per_payout=grow,
    )
    cash = out['cash'].astype(np.float64)
    n_evals = out['eval_fees_paid'] / 98.0
    reopens = np.maximum(n_evals - mc_init, 0.0)
    cash_adj = cash - RESET_FEE_DELTA * reopens
    return float((cash_adj >= BUDGET + MIN_PAYOUT).mean()), float((cash_adj <= 0).mean()), float(out['n_payouts'].mean())


def main():
    amr = pickle.load(open(REPO / 'cache' / 'amr_per_day_trades.pkl', 'rb'))

    print(f"\n{'='*110}")
    print(f"ADAPTIVE max_concurrent at BUDGET=$1K, OOS 2023+")
    print(f"Format: p_profit / p_zero / mean_payouts")
    print(f"{'='*110}")

    for cfg_id in ['cfg_406', 'cfg_029']:
        cfg = build_oos(cfg_id, amr)
        if cfg is None:
            continue
        print(f"\n--- {cfg_id}_OOS ---")
        configs = [
            ('mc_init=1, grow=0, max=1', 1, 0, 1),
            ('mc_init=1, grow=1, max=2', 1, 1, 2),
            ('mc_init=1, grow=1, max=4', 1, 1, 4),
            ('mc_init=1, grow=2, max=4', 1, 2, 4),
            ('mc_init=2, grow=0, max=2', 2, 0, 2),
            ('mc_init=2, grow=1, max=4', 2, 1, 4),
        ]
        for erp in [0.5, 0.7, 0.9]:
            print(f"  erp={erp}:")
            for label, mc_init, grow, mc_max in configs:
                pps, pzs, npays = zip(*[evaluate(cfg, erp, mc_init, grow, mc_max, s) for s in SEEDS])
                mean_pp = float(np.mean(pps))
                tag = " <-- PASS >80%" if mean_pp > 0.80 else ""
                print(f"    {label:<30s}  p_profit={mean_pp:.4f}  "
                      f"p_zero={np.mean(pzs):.3f}  npay={np.mean(npays):.1f}{tag}")


if __name__ == '__main__':
    main()
