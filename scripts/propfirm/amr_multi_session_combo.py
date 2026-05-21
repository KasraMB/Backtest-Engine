"""
Multi-session AMR combination at $1K BUDGET.

KEY FINDING: cfg_406 (session 2000), cfg_152 (session 300), cfg_495 (session
930) have near-zero correlations (max 0.045) on OOS daily R. Combined mean
R/day = +0.188 (vs +0.142 best single config). Diversification works.

Test approach: build a "combined" ThreePhaseConfig that pools trades from
multiple session-configs. The sim's Poisson(tpd) day-trade model samples
from the combined pool; cross-session decorrelation means combined daily
R variance < sum of individual variances (key for surviving MLL).
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

ACCOUNT = LUCIDFLEX_ACCOUNTS['50K']
BUDGET = 1000.0
HORIZON = 84
N_SIMS = 20_000
SEEDS = [2026, 2027, 2028]
MIN_PAYOUT = 500.0
RESET_DELTA = -3.0


def build_combined_config(cfg_ids: list, amr: dict, oos_start='2023-01-01') -> ThreePhaseConfig:
    """Concat trade pools from multiple AMR configs into one combined config.

    Each cfg trades a different session, so trades don't collide on the
    same calendar day. Combined tpd = total OOS trades / total OOS days
    (counting EACH calendar day once regardless of which config used it).
    """
    all_pnl  = []
    all_sl   = []
    all_dates = set()
    n_trades_total = 0
    for cfg_id in cfg_ids:
        d = amr[cfg_id]
        td = d['trades'].copy(); td['date'] = pd.to_datetime(td['date'])
        dd = d['days'].copy();   dd['date'] = pd.to_datetime(dd['date'])
        oos_t = td[td['date'] >= oos_start]
        oos_d = dd[dd['date'] >= oos_start]
        all_pnl.append(oos_t['pnl_pts'].to_numpy(np.float32))
        all_sl.append(oos_t['sl_dist'].to_numpy(np.float32))
        all_dates.update(pd.to_datetime(oos_d['date']).tolist())
        n_trades_total += len(oos_t)
    combined_pnl = np.concatenate(all_pnl)
    combined_sl  = np.concatenate(all_sl)
    n_days_total = len(all_dates)
    tpd = float(n_trades_total) / float(n_days_total)
    return ThreePhaseConfig(
        name='+'.join(cfg_ids),
        pnl_pts=combined_pnl,
        sl_dists=combined_sl,
        tpd=tpd,
    )


def evaluate(cfg, erp, mc, seed):
    out = three_phase_reinvestment_mc(
        eval_grind_config=cfg, payout_config=cfg,
        eval_grind_risk=erp, account=ACCOUNT,
        budget=BUDGET, horizon=HORIZON,
        max_concurrent=mc, n_sims=N_SIMS, seed=seed,
        enforce_consistency_rule=True,
    )
    cash = out['cash'].astype(np.float64)
    n_evals = out['eval_fees_paid'] / 98.0
    reopens = np.maximum(n_evals - mc, 0.0)
    cash_adj = cash - RESET_DELTA * reopens
    growth = cash_adj - BUDGET
    downside = growth[growth < 0]
    return {
        'p_profit': float((cash_adj >= BUDGET + MIN_PAYOUT).mean()),
        'p_zero':   float((cash_adj <= 0).mean()),
        'mean_g':   float(growth.mean()),
        'median_g': float(np.median(growth)),
        'sharpe':   float(growth.mean() / max(growth.std(), 1e-9)),
        'sortino':  float(growth.mean() / max(downside.std() if len(downside) else 1.0, 1e-9)),
    }


def main():
    amr = pickle.load(open(REPO / 'cache' / 'amr_per_day_trades.pkl', 'rb'))

    combos = [
        ['cfg_406'],                                  # single (baseline)
        ['cfg_495'],                                  # single best 930
        ['cfg_152'],                                  # single best 300
        ['cfg_406', 'cfg_495'],                       # 2000 + 930
        ['cfg_406', 'cfg_152'],                       # 2000 + 300
        ['cfg_495', 'cfg_152'],                       # 930 + 300
        ['cfg_406', 'cfg_495', 'cfg_152'],            # ALL THREE
    ]

    print(f"\n{'='*100}")
    print(f"MULTI-SESSION AMR DIVERSIFICATION at $1K, 50K Flex, OOS 2023+")
    print(f"{'='*100}")

    for combo in combos:
        print(f"\n--- {'+'.join(combo)} ---")
        cfg = build_combined_config(combo, amr)
        print(f"  Combined: n_trades={len(cfg.pnl_pts)}, tpd={cfg.tpd:.3f}, "
              f"mean_R/trade={cfg.pnl_pts.sum()/cfg.sl_dists.sum():+.4f}")
        for erp in [0.5, 0.7, 0.9]:
            for mc in [1, 2]:
                results = [evaluate(cfg, erp, mc, s) for s in SEEDS]
                pp = np.mean([r['p_profit'] for r in results])
                sortino = np.mean([r['sortino'] for r in results])
                sharpe = np.mean([r['sharpe'] for r in results])
                mean_g = np.mean([r['mean_g'] for r in results])
                median_g = np.mean([r['median_g'] for r in results])
                pz = np.mean([r['p_zero'] for r in results])
                tag = " <-PASS>80%" if pp > 0.80 else ""
                print(f"  erp={erp:.1f} mc={mc}: p_profit={pp:.4f}{tag}, "
                      f"Sharpe={sharpe:+.3f}, Sortino={sortino:+.3f}, "
                      f"mean=${mean_g:+,.0f}, median=${median_g:+,.0f}")


if __name__ == '__main__':
    main()
