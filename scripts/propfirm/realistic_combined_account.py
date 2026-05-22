"""
Combined-daily-R strategy: one account trades cfg_406 (8 PM) + cfg_495 (9:30 AM)
+ cfg_152 (3 AM) — three different sessions per day, summing R per day.

Different from 3-slot diversification (which uses 3 accounts).
Here: 1 account, all 3 strategies. Same MLL constraint applies to combined R.

If session-level decorrelation is real, combined daily variance is LOWER than
individual variances summed (Sharpe goes up). Variance reduction reduces MLL
blow rate without burning extra eval fees.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from backtest.propfirm.lucidflex import LUCIDFLEX_ACCOUNTS
from backtest.propfirm.three_phase_mc import three_phase_reinvestment_mc

ACCOUNT = LUCIDFLEX_ACCOUNTS['50K']
BUDGET = 1000.0
HORIZON = 84
N_SIMS = 10_000
SEEDS = [2026, 2027]
MIN_PAYOUT = 500.0
RESET_DELTA = -3.0


def build_combined_slot(pools: dict, cfg_ids: list, oos_start='2023-01-01'):
    """Build per-day arrays for a SINGLE slot that trades all cfg_ids on the
    same calendar day. R per day = sum across strategies. SL = mean weighted by
    n_trades. Per-trade arrays: concat trades across the strategies that fired."""
    # Per-cfg OOS dates + arrays
    cfg_data = {}
    all_dates = set()
    max_trades_per_cfg = 0
    for cfg_id in cfg_ids:
        pd_data = pools[cfg_id]['per_day']
        dates = pd_data['dates']
        oos_mask = dates >= pd.Timestamp(oos_start)
        oos_dates = dates[oos_mask]
        cfg_data[cfg_id] = {
            'dates':       oos_dates,
            'R':           pd_data['R'][oos_mask],
            'sl_dists':    pd_data['sl_dists'][oos_mask],
            'n_trades':    pd_data['n_trades'][oos_mask],
            'R_per_trade': pd_data['R_per_trade'][oos_mask],
            'MAE_per_trade': pd_data['MAE_R_per_trade'][oos_mask],
        }
        all_dates.update(oos_dates.tolist())
        max_trades_per_cfg = max(max_trades_per_cfg, pd_data['R_per_trade'].shape[1])

    sorted_dates = sorted(all_dates)
    n_days = len(sorted_dates)
    date_to_idx = {d: i for i, d in enumerate(sorted_dates)}
    max_trades = max_trades_per_cfg * len(cfg_ids)   # sum across strategies

    R_arr     = np.zeros((1, n_days), dtype=np.float64)
    sl_arr    = np.full((1, n_days), 16.5, dtype=np.float64)
    fired_arr = np.zeros((1, n_days), dtype=bool)
    nt_arr    = np.zeros((1, n_days), dtype=np.int64)
    Rt_arr    = np.zeros((1, n_days, max_trades), dtype=np.float64)
    MAE_arr   = np.zeros((1, n_days, max_trades), dtype=np.float64)

    for cfg_id in cfg_ids:
        c = cfg_data[cfg_id]
        for j, d in enumerate(c['dates']):
            day_i = date_to_idx[d]
            n_cfg_trades = c['n_trades'][j]
            R_arr[0, day_i]   += c['R'][j]
            fired_arr[0, day_i] = True
            # Append per-trade arrays
            existing_k = nt_arr[0, day_i]
            new_k = existing_k + n_cfg_trades
            if new_k <= max_trades:
                Rt_arr[0, day_i, existing_k:new_k]   = c['R_per_trade'][j, :n_cfg_trades]
                MAE_arr[0, day_i, existing_k:new_k]  = c['MAE_per_trade'][j, :n_cfg_trades]
            nt_arr[0, day_i] = new_k
            # SL: weighted mean (re-weight as we go)
            if existing_k == 0:
                sl_arr[0, day_i] = c['sl_dists'][j]
            else:
                sl_arr[0, day_i] = (sl_arr[0, day_i] * existing_k + c['sl_dists'][j] * n_cfg_trades) / new_k
    return R_arr, sl_arr, fired_arr, nt_arr, Rt_arr, MAE_arr, n_days


def run_combined(cfg_ids, pools, erp, seed=2026):
    R_arr, sl_arr, fired_arr, nt_arr, Rt_arr, MAE_arr, n_days = build_combined_slot(pools, cfg_ids)
    out = three_phase_reinvestment_mc(
        account=ACCOUNT, budget=BUDGET, horizon=HORIZON,
        max_concurrent=1, n_sims=N_SIMS, seed=seed,
        eval_grind_risk=erp,
        bootstrap_block_len=5.0,
        bootstrap_oos_R_per_slot=R_arr,
        bootstrap_oos_sl_dists_per_slot=sl_arr,
        bootstrap_oos_fired_per_slot=fired_arr,
        bootstrap_oos_n_trades_per_slot=nt_arr,
        bootstrap_oos_R_per_trade_per_slot=Rt_arr,
        bootstrap_oos_MAE_R_per_trade_per_slot=MAE_arr,
        bootstrap_intraday_blow_check=True,
        enforce_consistency_rule=True,
    )
    cash = out['cash'].astype(np.float64)
    n_evals = out['eval_fees_paid'] / 98.0
    reopens = np.maximum(n_evals - 1, 0.0)
    cash_adj = cash - RESET_DELTA * reopens
    growth = cash_adj - BUDGET
    return {
        'p_profit':   float((cash_adj >= BUDGET + MIN_PAYOUT).mean()),
        'p_zero':     float((cash_adj <= 0).mean()),
        'mean_g':     float(growth.mean()),
        'median_g':   float(np.median(growth)),
        'sortino':    float(growth.mean() / max(growth[growth<0].std() if (growth<0).any() else 1.0, 1e-9)),
        'mean_payouts': float(out['n_payouts'].mean()),
        'n_oos_days': n_days,
    }


def main():
    pools = pickle.load(open(REPO / 'cache' / 'amr_realistic_pools.pkl', 'rb'))

    combos = [
        ['cfg_406'],                                  # baseline
        ['cfg_495'],
        ['cfg_406', 'cfg_495'],                       # 2-session combined
        ['cfg_406', 'cfg_152'],
        ['cfg_495', 'cfg_152'],
        ['cfg_406', 'cfg_495', 'cfg_152'],            # all 3 sessions combined on 1 account
    ]
    print(f"\n{'='*100}")
    print(f"COMBINED-ACCOUNT: 1 account trades multiple sessions, R/day summed, $1K, OOS 2023+")
    print(f"{'='*100}")
    for combo in combos:
        print(f"\n--- {'+'.join(combo)} ({len(combo)} session(s)) ---")
        for erp in [0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.70]:
            results = [run_combined(combo, pools, erp, seed=s) for s in SEEDS]
            pp = np.mean([r['p_profit'] for r in results])
            sortino = np.mean([r['sortino'] for r in results])
            mean_g = np.mean([r['mean_g'] for r in results])
            median_g = np.mean([r['median_g'] for r in results])
            npay = np.mean([r['mean_payouts'] for r in results])
            n_days = results[0]['n_oos_days']
            tag = ' <-PASS>80%' if pp > 0.80 else (' (>70%)' if pp > 0.70 else '')
            print(f"  erp={erp:.2f}: p_profit={pp:.4f}{tag}, Sortino={sortino:+.2f}, "
                  f"mean=${mean_g:+,.0f}, median=${median_g:+,.0f}, npay={npay:.1f} [{n_days} days]")


if __name__ == '__main__':
    main()
