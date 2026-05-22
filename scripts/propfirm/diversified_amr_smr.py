"""
Diversified AMR cfg_495(ToD) + SessionMeanRev (NY) on LucidFlex 50K Flex at $1K.

Train (2019-2022): select erp via 2-slot per-slot bootstrap.
TRUE OOS (2024+): single-shot validation.

Both strategies have real OOS edge:
- AMR cfg_495+ToD: +0.128 R OOS, p<0.05
- SMR (NY momentum, sweep4 config): +0.152 R OOS
- Daily R correlation: +0.14 (low — genuine diversification potential)
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from backtest.propfirm.lucidflex import LUCIDFLEX_ACCOUNTS
from backtest.propfirm.three_phase_mc import three_phase_reinvestment_mc

A = LUCIDFLEX_ACCOUNTS['50K']
BUDGET = 1000.0
MIN_PAYOUT = 500.0
N_SIMS = 10000
SEEDS = [2026, 2027, 2028]
TRAIN_HI = pd.Timestamp('2022-12-31')
OOS_LO = pd.Timestamp('2024-01-01')


def slice_pool(df, date_lo=None, date_hi=None):
    if date_lo is not None:
        df = df[df['date'] >= date_lo]
    if date_hi is not None:
        df = df[df['date'] <= date_hi]
    return df


def build_per_slot_arrays(slot_dfs):
    """Build (mc, n_days, ...) arrays with shared OOS day union across slots.
    slot_dfs: list of trade-DataFrames with 'date','R','sl_dist','MAE_R' columns.
    """
    mc = len(slot_dfs)
    all_dates = set()
    for df in slot_dfs:
        all_dates.update(pd.to_datetime(df['date'].unique()).tolist())
    sorted_dates = sorted(all_dates)
    nd = len(sorted_dates)
    d2i = {d: i for i, d in enumerate(sorted_dates)}
    max_t = max(int(df.groupby('date').size().max()) if len(df) else 1 for df in slot_dfs)
    R = np.zeros((mc, nd)); sl = np.full((mc, nd), 16.5); fired = np.zeros((mc, nd), bool)
    nt = np.zeros((mc, nd), np.int64); Rt = np.zeros((mc, nd, max_t)); MAE = np.zeros((mc, nd, max_t))
    for si, df in enumerate(slot_dfs):
        if len(df) == 0:
            continue
        for dt, sub in df.groupby('date'):
            i = d2i[pd.Timestamp(dt)]
            R[si, i] = sub['R'].sum(); sl[si, i] = sub['sl_dist'].mean(); fired[si, i] = True
            nt[si, i] = len(sub); k = min(len(sub), max_t)
            Rt[si, i, :k] = sub['R'].values[:k]; MAE[si, i, :k] = sub['MAE_R'].values[:k]
    return R, sl, fired, nt, Rt, MAE, nd


def run_pp(arrays, erp, mc, seed):
    R, sl, fired, nt, Rt, MAE, nd = arrays
    out = three_phase_reinvestment_mc(
        account=A, budget=BUDGET, horizon=84, max_concurrent=mc, n_sims=N_SIMS, seed=seed,
        eval_grind_risk=erp, bootstrap_block_len=5.0,
        bootstrap_oos_R_per_slot=R, bootstrap_oos_sl_dists_per_slot=sl,
        bootstrap_oos_fired_per_slot=fired, bootstrap_oos_n_trades_per_slot=nt,
        bootstrap_oos_R_per_trade_per_slot=Rt, bootstrap_oos_MAE_R_per_trade_per_slot=MAE,
        bootstrap_intraday_blow_check=True, enforce_consistency_rule=True,
    )
    cash = out['cash'].astype(np.float64)
    n_e = out['eval_fees_paid'] / A.eval_fee
    reopens = np.maximum(n_e - mc, 0.0)
    cash_adj = cash - (A.reset_fee - A.eval_fee) * reopens
    growth = cash_adj - BUDGET
    return {
        'p_profit': float((cash_adj >= BUDGET + MIN_PAYOUT).mean()),
        'p_zero':   float((cash_adj <= 0).mean()),
        'mean_g':   float(growth.mean()),
        'median_g': float(np.median(growth)),
        'sortino':  float(growth.mean() / max(growth[growth<0].std() if (growth<0).any() else 1.0, 1e-9)),
        'mean_payouts': float(out['n_payouts'].mean()),
    }


def main():
    amr = pickle.load(open(REPO / 'cache' / 'amr_cfg495_tod_realistic_trades.pkl', 'rb'))
    smr = pickle.load(open(REPO / 'cache' / 'smr_realistic_trades.pkl', 'rb'))
    print(f"AMR pool: {len(amr)} trades; SMR pool: {len(smr)} trades", flush=True)

    # TRAIN
    amr_train = slice_pool(amr, date_hi=TRAIN_HI)
    smr_train = slice_pool(smr, date_hi=TRAIN_HI)
    print(f"Train: AMR n={len(amr_train)}, SMR n={len(smr_train)}", flush=True)

    # Configurations to test on train:
    configs = [
        ('amr_only',   [amr_train], 1),
        ('smr_only',   [smr_train], 1),
        ('amr+smr',    [amr_train, smr_train], 2),
    ]
    train_results = {}
    print("\n=== TRAIN: erp selection per config ===", flush=True)
    for name, dfs, mc in configs:
        train_arr = build_per_slot_arrays(dfs)
        best_erp, best_pp = None, -1.0
        for erp in [0.20, 0.30, 0.40, 0.50, 0.70]:
            pps = [run_pp(train_arr, erp, mc, s)['p_profit'] for s in [2026, 2027]]
            pp = float(np.mean(pps))
            if pp > best_pp:
                best_pp, best_erp = pp, erp
            print(f"  {name:<12s} erp={erp}: train_pp={pp:.4f}", flush=True)
        train_results[name] = (best_erp, best_pp, dfs, mc)
        print(f"  -> BEST {name}: erp={best_erp}, train_pp={best_pp:.4f}", flush=True)

    # SINGLE-SHOT OOS
    print("\n=== TRUE OOS (2024+) SINGLE-SHOT ===", flush=True)
    oos_results = {}
    for name, (erp, _, dfs_train, mc) in train_results.items():
        # Use OOS data, same mc
        if name == 'amr_only':
            oos_dfs = [slice_pool(amr, date_lo=OOS_LO)]
        elif name == 'smr_only':
            oos_dfs = [slice_pool(smr, date_lo=OOS_LO)]
        else:
            oos_dfs = [slice_pool(amr, date_lo=OOS_LO), slice_pool(smr, date_lo=OOS_LO)]
        oos_arr = build_per_slot_arrays(oos_dfs)
        if oos_arr[6] < 50:
            print(f"  {name}: OOS insufficient ({oos_arr[6]} days)", flush=True)
            continue
        results = [run_pp(oos_arr, erp, mc, s) for s in SEEDS]
        pp = float(np.mean([r['p_profit'] for r in results]))
        sortino = float(np.mean([r['sortino'] for r in results]))
        mean_g = float(np.mean([r['mean_g'] for r in results]))
        median_g = float(np.mean([r['median_g'] for r in results]))
        oos_results[name] = pp
        tag = " <-PASS>80%" if pp > 0.80 else (' >70%' if pp > 0.70 else '')
        print(f"  {name:<12s} erp={erp} mc={mc}: OOS_pp={pp:.4f}{tag} Sortino={sortino:+.2f} "
              f"mean=${mean_g:+,.0f} median=${median_g:+,.0f}", flush=True)

    with open(REPO / 'results' / 'diversified_amr_smr_oos.json', 'w') as f:
        json.dump({
            'train_selections': {k: {'erp': v[0], 'train_pp': v[1]} for k, v in train_results.items()},
            'oos_results':      oos_results,
        }, f, indent=2, default=str)
    print("\nWrote results/diversified_amr_smr_oos.json", flush=True)


if __name__ == '__main__':
    main()
