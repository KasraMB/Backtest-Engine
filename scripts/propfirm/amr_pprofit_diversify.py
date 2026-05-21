"""
AMR propfirm P(profit) push toward 80%: multi-config diversification + scaling.

Phase 1 sweep showed best 76.1% at cfg_029 erp=0.7. Plateaued.
Try: (a) multi-config slots (cfg_029 + cfg_406 diversification),
     (b) larger max_concurrent (more accounts),
     (c) different account sizes.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from backtest.propfirm.three_phase_mc import ThreePhaseConfig, three_phase_reinvestment_mc
from backtest.propfirm.lucidflex import LUCIDFLEX_ACCOUNTS

ACCOUNT_25K  = LUCIDFLEX_ACCOUNTS["25K"]
ACCOUNT_50K  = LUCIDFLEX_ACCOUNTS["50K"]
ACCOUNT_100K = LUCIDFLEX_ACCOUNTS["100K"]
BUDGET  = 3_000.0
HORIZON = 84
N_SIMS  = 5000
SEED    = 2026


def build_config_from_amr(cfg_id: str, amr_trades: dict) -> ThreePhaseConfig:
    data = amr_trades[cfg_id]
    trade_df = data['trades']
    days_df  = data['days']
    return ThreePhaseConfig(
        name=cfg_id,
        pnl_pts=trade_df['pnl_pts'].to_numpy(np.float32),
        sl_dists=trade_df['sl_dist'].to_numpy(np.float32),
        tpd=float(data['n_trades']) / float(len(days_df)),
    )


def run_once(label, eval_cfg, payout_cfg, account, erp, max_c, slot_assignments=None):
    out = three_phase_reinvestment_mc(
        eval_grind_config=eval_cfg if slot_assignments is None else None,
        payout_config=payout_cfg if slot_assignments is None else None,
        slot_assignments=slot_assignments,
        eval_grind_risk=erp,
        account=account, budget=BUDGET, horizon=HORIZON,
        max_concurrent=max_c, n_sims=N_SIMS, seed=SEED,
    )
    cash = out['cash']
    res = {
        'label':       label,
        'p_profit':    float((cash > BUDGET).mean()),
        'p_zero':      float((cash <= 0).mean()),
        'p_goal10':    float((cash >= 10_000.0).mean()),
        'p_goal20':    float((cash >= 20_000.0).mean()),
        'mean_cash':   float(cash.mean()),
        'median_cash': float(np.median(cash)),
    }
    return res


def main():
    print(f"Loading AMR trade cache...", flush=True)
    amr = pickle.load(open(REPO / 'cache' / 'amr_per_day_trades.pkl', 'rb'))

    cfg_029 = build_config_from_amr('cfg_029', amr)
    cfg_406 = build_config_from_amr('cfg_406', amr)
    cfg_495 = build_config_from_amr('cfg_495', amr)
    cfg_475 = build_config_from_amr('cfg_475', amr)

    print(f"\n{'='*100}", flush=True)
    print(f"Experiment 1: SOLO baseline at best erp=0.7 (n_sims={N_SIMS})", flush=True)
    print(f"{'='*100}", flush=True)
    for cfg, name in [(cfg_029, 'cfg_029'), (cfg_406, 'cfg_406')]:
        for max_c in [4, 6, 8, 10]:
            r = run_once(f"{name}_mc{max_c}", cfg, cfg, ACCOUNT_50K, 0.7, max_c)
            print(f"  {r['label']:<20s}  p_profit={r['p_profit']:.4f}  p_goal10={r['p_goal10']:.4f}  "
                  f"mean=${r['mean_cash']:,.0f}  median=${r['median_cash']:,.0f}", flush=True)

    print(f"\n{'='*100}", flush=True)
    print(f"Experiment 2: Account size variation (25K, 50K, 100K) at best erp=0.7", flush=True)
    print(f"{'='*100}", flush=True)
    for cfg, name in [(cfg_029, 'cfg_029'), (cfg_406, 'cfg_406')]:
        for acc, acc_name in [(ACCOUNT_25K, '25K'), (ACCOUNT_50K, '50K'), (ACCOUNT_100K, '100K')]:
            for erp in [0.5, 0.7, 0.9]:
                r = run_once(f"{name}_{acc_name}_erp{erp}", cfg, cfg, acc, erp, 4)
                print(f"  {r['label']:<22s}  p_profit={r['p_profit']:.4f}  p_goal10={r['p_goal10']:.4f}  "
                      f"mean=${r['mean_cash']:,.0f}  median=${r['median_cash']:,.0f}", flush=True)

    print(f"\n{'='*100}", flush=True)
    print(f"Experiment 3: Effect of larger BUDGET (more eval-purchase capacity)", flush=True)
    print(f"{'='*100}", flush=True)
    # Override BUDGET globally for this experiment
    global BUDGET
    saved_budget = BUDGET
    for budget_val in [1500.0, 3000.0, 5000.0, 10_000.0]:
        BUDGET = budget_val
        for max_c in [4, 8]:
            r = run_once(f"cfg_029_b${int(budget_val)}_mc{max_c}", cfg_029, cfg_029, ACCOUNT_50K, 0.7, max_c)
            print(f"  {r['label']:<24s}  p_profit={r['p_profit']:.4f}  p_goal10={r['p_goal10']:.4f}  "
                  f"mean=${r['mean_cash']:,.0f}", flush=True)
    BUDGET = saved_budget


if __name__ == '__main__':
    main()
