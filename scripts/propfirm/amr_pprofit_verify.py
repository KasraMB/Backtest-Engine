"""
Verify P(profit) > 80% finding for cfg_029 at LucidFlex 50K, erp=0.7, BUDGET=$5000.

Phase 2 confirmation:
  - Larger n_sims (20000) for tighter p_profit estimate
  - Multiple configs tested
  - Verify result is robust to seed change
  - Report 95% confidence interval on p_profit
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

ACCOUNT = LUCIDFLEX_ACCOUNTS["50K"]
HORIZON = 84
N_SIMS_VERIFY = 20_000


def build_config_from_amr(cfg_id, amr_trades):
    data = amr_trades[cfg_id]
    return ThreePhaseConfig(
        name=cfg_id,
        pnl_pts=data['trades']['pnl_pts'].to_numpy(np.float32),
        sl_dists=data['trades']['sl_dist'].to_numpy(np.float32),
        tpd=float(data['n_trades']) / float(len(data['days'])),
    )


def main():
    amr = pickle.load(open(REPO / 'cache' / 'amr_per_day_trades.pkl', 'rb'))
    cfg_029 = build_config_from_amr('cfg_029', amr)
    cfg_406 = build_config_from_amr('cfg_406', amr)
    cfg_495 = build_config_from_amr('cfg_495', amr)

    BUDGET = 5000.0   # locked from sweep
    ERP = 0.7
    MAX_C = 4

    print(f"\n{'='*100}")
    print(f"PHASE 2 VERIFICATION: P(profit) > 80% target")
    print(f"LucidFlex 50K Flex, erp={ERP}, max_concurrent={MAX_C}, BUDGET=${BUDGET}, horizon={HORIZON}d, n_sims={N_SIMS_VERIFY}")
    print(f"{'='*100}")

    configs_to_test = [(cfg_029, 'cfg_029'), (cfg_406, 'cfg_406'), (cfg_495, 'cfg_495')]

    for cfg, name in configs_to_test:
        print(f"\n--- {name} ---")
        seeds = [2026, 2027, 2028]
        p_profits = []
        for seed in seeds:
            out = three_phase_reinvestment_mc(
                eval_grind_config=cfg, payout_config=cfg,
                eval_grind_risk=ERP, account=ACCOUNT,
                budget=BUDGET, horizon=HORIZON,
                max_concurrent=MAX_C, n_sims=N_SIMS_VERIFY, seed=seed,
            )
            cash = out['cash']
            pp = float((cash > BUDGET).mean())
            pz = float((cash <= 0).mean())
            p10 = float((cash >= 10_000.0).mean())
            p20 = float((cash >= 20_000.0).mean())
            p_profits.append(pp)
            print(f"  seed={seed}: p_profit={pp:.4f}, p_zero={pz:.4f}, "
                  f"p_cash>=$10K={p10:.4f}, p_cash>=$20K={p20:.4f}, "
                  f"mean=${cash.mean():,.0f}, median=${np.median(cash):,.0f}")
        pp_arr = np.array(p_profits)
        mean_pp = pp_arr.mean()
        # 95% CI on p_profit via binomial approx
        se = np.sqrt(mean_pp * (1 - mean_pp) / N_SIMS_VERIFY)
        ci_lo = mean_pp - 1.96 * se
        ci_hi = mean_pp + 1.96 * se
        print(f"  AVERAGE p_profit across 3 seeds: {mean_pp:.4f}  (95% CI per-seed: [{ci_lo:.4f}, {ci_hi:.4f}])")
        if mean_pp > 0.80:
            print(f"  PASSED: EXCEEDS 80% TARGET")
        else:
            print(f"  FAILED: below 80% target")


if __name__ == '__main__':
    main()
