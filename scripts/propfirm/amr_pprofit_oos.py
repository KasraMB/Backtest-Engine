"""
OOS-only re-verification of p_profit > 80% finding.

Slice each top config's trade pool to 2023+ trades only (out of train window
2019-2022). Re-run propfirm sim with same params. If p_profit stays > 80% on
OOS-only data, the result is robust to train-selection contamination.
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
BUDGET = 5000.0
ERP = 0.7
MAX_C = 4
HORIZON = 84
N_SIMS = 20_000
OOS_DATE = '2023-01-01'


def build_config_oos(cfg_id, amr_trades, oos_start):
    """Build ThreePhaseConfig using ONLY trades after oos_start."""
    data = amr_trades[cfg_id]
    trade_df = data['trades'].copy()
    trade_df['date'] = pd.to_datetime(trade_df['date'])
    days_df = data['days'].copy()
    days_df['date'] = pd.to_datetime(days_df['date'])

    # Slice
    oos_trades = trade_df[trade_df['date'] >= oos_start]
    oos_days   = days_df[days_df['date'] >= oos_start]
    n_trades = len(oos_trades)
    n_days   = len(oos_days)
    if n_days == 0:
        return None, 0, 0
    return ThreePhaseConfig(
        name=f"{cfg_id}_OOS",
        pnl_pts=oos_trades['pnl_pts'].to_numpy(np.float32),
        sl_dists=oos_trades['sl_dist'].to_numpy(np.float32),
        tpd=float(n_trades) / float(n_days),
    ), n_trades, n_days


def main():
    amr = pickle.load(open(REPO / 'cache' / 'amr_per_day_trades.pkl', 'rb'))

    print(f"\n{'='*100}")
    print(f"OOS-ONLY VERIFICATION ({OOS_DATE}+ trades, no train data leak)")
    print(f"erp={ERP}, BUDGET=${BUDGET}, max_concurrent={MAX_C}, horizon={HORIZON}d, n_sims={N_SIMS}")
    print(f"{'='*100}")

    for cfg_id in ['cfg_029', 'cfg_406', 'cfg_495']:
        cfg, n_t, n_d = build_config_oos(cfg_id, amr, OOS_DATE)
        if cfg is None or n_t < 100:
            print(f"\n{cfg_id}_OOS: insufficient trades ({n_t})")
            continue
        # OOS mean R
        mean_R_per_trade = cfg.pnl_pts.mean() / cfg.sl_dists.mean()
        mean_R_per_day = mean_R_per_trade * cfg.tpd

        print(f"\n--- {cfg_id}_OOS (n_trades={n_t}, n_days={n_d}, tpd={cfg.tpd:.2f}, "
              f"OOS mean R/trade ~{mean_R_per_trade:+.3f}, /day ~{mean_R_per_day:+.3f}) ---")

        seeds = [2026, 2027, 2028]
        pps = []
        for seed in seeds:
            out = three_phase_reinvestment_mc(
                eval_grind_config=cfg, payout_config=cfg,
                eval_grind_risk=ERP, account=ACCOUNT,
                budget=BUDGET, horizon=HORIZON,
                max_concurrent=MAX_C, n_sims=N_SIMS, seed=seed,
            )
            cash = out['cash']
            pp = float((cash > BUDGET).mean())
            pz = float((cash <= 0).mean())
            p10 = float((cash >= 10_000.0).mean())
            pps.append(pp)
            print(f"  seed={seed}: p_profit={pp:.4f}, p_zero={pz:.4f}, p>=10K={p10:.4f}, "
                  f"mean=${cash.mean():,.0f}, median=${np.median(cash):,.0f}")
        mean_pp = np.mean(pps)
        ci_se = np.sqrt(mean_pp * (1 - mean_pp) / N_SIMS)
        ci_lo = mean_pp - 1.96 * ci_se
        verdict = "PASSED >80%" if mean_pp > 0.80 else f"FAILED ({mean_pp:.4f} < 0.80)"
        print(f"  OOS mean p_profit = {mean_pp:.4f} (CI lo {ci_lo:.4f}) - {verdict}")


if __name__ == '__main__':
    main()
