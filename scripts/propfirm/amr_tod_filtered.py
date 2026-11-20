"""
Test AMR cfg_495 with TIME-OF-DAY filter (9:35-10:05 ET sweet spot).

ToD pre-screen showed 9:40-10:05 has +0.133 mean_R vs +0.077 baseline.
Build pool with skip_first_mins=5, window_mins_per_session={'930':35} = 9:35-10:05.
Run audited bootstrap sim at LucidFlex 50K Flex + $1K + intraday MLL + consistency.
"""
from __future__ import annotations

import datetime
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from backtest.propfirm.lucidflex import LUCIDFLEX_ACCOUNTS, extract_normalised_trades
from backtest.propfirm.three_phase_mc import three_phase_reinvestment_mc
from backtest.runner.config import RunConfig
from backtest.runner.runner import run_backtest
from strategies.amr_strategy import AnchoredMeanReversionStrategy
from scripts.propfirm.rebuild_realistic_pools import build_per_day_arrays


def main():
    md = pickle.load(open(REPO / 'md_cache.pkl', 'rb'))['md']
    amr = pickle.load(open(REPO / 'cache' / 'amr_per_day_trades.pkl', 'rb'))
    base = amr['cfg_495']['params']

    print("Rebuilding cfg_495_TOD with skip=5, window=35 (9:35-10:05 ET)...", flush=True)
    p = dict(base)
    p['skip_first_mins'] = 5
    p['window_mins_per_session'] = {'930': 35}
    cfg = RunConfig(starting_capital=100_000, commission_per_contract=3.5,
                    slippage_points=0.5, fill_mode='next_open',
                    order_cancel_time=datetime.time(23, 59), params=p)
    t0 = time.time()
    res = run_backtest(AnchoredMeanReversionStrategy, cfg, data=md, validate=False)
    per_day = build_per_day_arrays(res.trades, md)
    print(f"  {per_day['total_trades']} trades, mean_R/trade={per_day['mean_R_trade']:+.4f}, "
          f"WR={per_day['wr']*100:.1f}%, days={(per_day['fired']).sum()}, took {time.time()-t0:.1f}s")

    # Filter to OOS 2023+ and prepare bootstrap arrays
    dates = per_day['dates']
    oos_mask = dates >= pd.Timestamp('2023-01-01')
    n_days = int(oos_mask.sum())
    print(f"  OOS days: {n_days}")
    if n_days < 50:
        print("  TOO FEW OOS days — skipping sim")
        return

    R_arr     = per_day['R'][oos_mask][None, :]
    sl_arr    = per_day['sl_dists'][oos_mask][None, :]
    fired_arr = per_day['fired'][oos_mask][None, :]
    nt_arr    = per_day['n_trades'][oos_mask][None, :]
    Rt_arr    = per_day['R_per_trade'][oos_mask][None, :, :]
    MAE_arr   = per_day['MAE_R_per_trade'][oos_mask][None, :, :]

    ACCOUNT = LUCIDFLEX_ACCOUNTS['50K']
    BUDGET = 1000.0
    MIN_PAYOUT = 500.0
    print(f"\n{'='*80}")
    print(f"PROPFIRM SIM: LucidFlex 50K Flex, $1K, 84d, OOS 2023+ (ToD-filtered)")
    print(f"{'='*80}")
    print(f"{'erp':<5} {'mc':<3} {'p_profit':>10} {'Sortino':>9} {'mean_g':>10} {'median':>10} {'npay':>5}")
    for erp in [0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.70]:
        for mc in [1]:
            pps, sortinos, mgs, medgs, npays = [], [], [], [], []
            for seed in [2026, 2027, 2028]:
                # tile arrays if mc > 1
                R_in     = R_arr if mc == 1 else np.tile(R_arr, (mc, 1))[:mc]
                sl_in    = sl_arr if mc == 1 else np.tile(sl_arr, (mc, 1))[:mc]
                fired_in = fired_arr if mc == 1 else np.tile(fired_arr, (mc, 1))[:mc]
                nt_in    = nt_arr if mc == 1 else np.tile(nt_arr, (mc, 1))[:mc]
                Rt_in    = Rt_arr if mc == 1 else np.tile(Rt_arr, (mc, 1, 1))[:mc]
                MAE_in   = MAE_arr if mc == 1 else np.tile(MAE_arr, (mc, 1, 1))[:mc]
                out = three_phase_reinvestment_mc(
                    account=ACCOUNT, budget=BUDGET, horizon=84,
                    max_concurrent=mc, n_sims=15000, seed=seed,
                    eval_grind_risk=erp, bootstrap_block_len=5.0,
                    bootstrap_oos_R_per_slot=R_in,
                    bootstrap_oos_sl_dists_per_slot=sl_in,
                    bootstrap_oos_fired_per_slot=fired_in,
                    bootstrap_oos_n_trades_per_slot=nt_in,
                    bootstrap_oos_R_per_trade_per_slot=Rt_in,
                    bootstrap_oos_MAE_R_per_trade_per_slot=MAE_in,
                    bootstrap_intraday_blow_check=True,
                    enforce_consistency_rule=True,
                )
                cash = out['cash'].astype(np.float64)
                n_evals = out['eval_fees_paid'] / ACCOUNT.eval_fee
                reopens = np.maximum(n_evals - mc, 0.0)
                cash_adj = cash - (ACCOUNT.reset_fee - ACCOUNT.eval_fee) * reopens
                growth = cash_adj - BUDGET
                pps.append(float((cash_adj >= BUDGET + MIN_PAYOUT).mean()))
                downside_std = growth[growth < 0].std() if (growth < 0).any() else 1.0
                sortinos.append(float(growth.mean() / max(downside_std, 1e-9)))
                mgs.append(float(growth.mean()))
                medgs.append(float(np.median(growth)))
                npays.append(float(out['n_payouts'].mean()))
            pp = np.mean(pps)
            tag = ' <-PASS' if pp > 0.80 else (' (>70)' if pp > 0.70 else '')
            print(f"{erp:<5.2f} {mc:<3} {pp:>10.4f}{tag} {np.mean(sortinos):>+9.2f} "
                  f"${np.mean(mgs):>+8,.0f} ${np.mean(medgs):>+8,.0f} {np.mean(npays):>5.1f}")


if __name__ == '__main__':
    main()
