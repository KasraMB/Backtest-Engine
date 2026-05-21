"""
Push p_profit higher at BUDGET=$1K via parameter variations.

Constraints (USER):
  - BUDGET = $1000 (HARD)
  - horizon = 84 trading days (HARD)
  - LucidFlex 50K Flex rules

Variables to explore:
  - max_concurrent: 1 / 2 / 3 / 4   (fewer = lower burn rate, but slower compounding)
  - erp: 0.3 / 0.5 / 0.7 / 0.9
  - Top configs across the cache (not just cfg_029/cfg_406)
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
N_SIMS  = 10_000
SEEDS   = [2026, 2027]
BUDGET  = 1000.0
MIN_PAYOUT = 500.0
RESET_FEE_DELTA = -3.0   # $95 reset - $98 eval (CONFIRMED from lucidtrading pricing)


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


def evaluate(cfg, erp, max_c, seed=2026):
    out = three_phase_reinvestment_mc(
        eval_grind_config=cfg, payout_config=cfg,
        eval_grind_risk=erp, account=ACCOUNT,
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
        'p_>=1_payout': float((out['n_payouts'] >= 1).mean()),
        'mean_payouts': float(out['n_payouts'].mean()),
        'mean_reopens': float(reopens.mean()),
    }


def main():
    amr = pickle.load(open(REPO / 'cache' / 'amr_per_day_trades.pkl', 'rb'))

    # All top configs by mean R/day
    cfg_stats = []
    for cfg_id, data in amr.items():
        if len(data['days']) < 200:
            continue
        cfg_stats.append((cfg_id, data['days']['R'].mean()))
    cfg_stats.sort(key=lambda x: -x[1])
    top_cfgs = [c[0] for c in cfg_stats[:6]]
    print(f"Top 6 configs: {top_cfgs}")

    print(f"\n{'='*110}")
    print(f"BUDGET=$1K Push: max_concurrent × erp sweep, all top OOS configs")
    print(f"Goal: p_profit > 0.80")
    print(f"{'='*110}")

    best_overall = (None, None, None, 0.0)
    for cfg_id in top_cfgs:
        cfg = build_oos_config(cfg_id, amr)
        if cfg is None:
            continue
        print(f"\n--- {cfg_id} OOS (n_trades={len(cfg.pnl_pts)}, tpd={cfg.tpd:.2f}) ---")
        for max_c in [1, 2, 3, 4]:
            row = []
            for erp in [0.3, 0.5, 0.7, 0.9]:
                results = [evaluate(cfg, erp, max_c, s) for s in SEEDS]
                mean_pp = np.mean([r['p_profit'] for r in results])
                row.append((erp, mean_pp, results[0]['p_zero'], results[0]['mean_payouts']))
                if mean_pp > best_overall[3]:
                    best_overall = (cfg_id, erp, max_c, mean_pp)
            cells = "  ".join(f"erp={r[0]:.1f}:{r[1]:.3f}(pz={r[2]:.2f},npay={r[3]:.1f})" for r in row)
            print(f"  mc={max_c}: {cells}")

    print(f"\n{'='*110}")
    print(f"BEST: cfg={best_overall[0]}, erp={best_overall[1]}, max_c={best_overall[2]}, p_profit={best_overall[3]:.4f}")
    if best_overall[3] > 0.80:
        print(f"  -> PASSED 80% target")
    else:
        print(f"  -> FAILED 80% target (gap = {0.80 - best_overall[3]:.4f})")


if __name__ == '__main__':
    main()
