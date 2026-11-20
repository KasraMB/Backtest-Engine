"""
HONEST out-of-sample AMR validation — corrects prior overfitting to 2023.

Protocol (per agent design):
- TRAIN: 2019-2022. Select bundle (ToD filter, HMM filter, erp) by train p_profit.
- TRUE OOS: 2024+. Single-shot the locked bundle. 2023 dropped (burned).
- Bundle grid (small, pre-registered DoF):
    ToD ∈ {none, 9:35-10:05}
    HMM ∈ {none, drop-bull}
    erp ∈ {0.25, 0.35, 0.45}
- Report OOS p_profit + permutation p-value (sign-flip daily R, N=1000).

All realistic execution: next_open fill + 0.5pt slippage + intraday MLL + consistency.
"""
from __future__ import annotations

import datetime
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from datetime import timedelta

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from backtest.propfirm.lucidflex import LUCIDFLEX_ACCOUNTS, extract_normalised_trades
from backtest.propfirm.three_phase_mc import three_phase_reinvestment_mc
from backtest.runner.config import RunConfig
from backtest.runner.runner import run_backtest
from strategies.amr_strategy import AnchoredMeanReversionStrategy

A = LUCIDFLEX_ACCOUNTS['50K']
BUDGET = 1000.0
MIN_PAYOUT = 500.0
N_SIMS = 15000
SEEDS = [2026, 2027, 2028]
TRAIN_LO = pd.Timestamp('2019-01-01'); TRAIN_HI = pd.Timestamp('2022-12-31')
OOS_LO = pd.Timestamp('2024-01-01')


def get_cfg495_trades(tod_filter: bool):
    md = pickle.load(open(REPO / 'md_cache.pkl', 'rb'))['md']
    amr = pickle.load(open(REPO / 'cache' / 'amr_per_day_trades.pkl', 'rb'))
    p = dict(amr['cfg_495']['params'])
    if tod_filter:
        p['skip_first_mins'] = 5
        p['window_mins_per_session'] = {'930': 35}
    cfg = RunConfig(starting_capital=100_000, commission_per_contract=3.5,
                    slippage_points=0.5, fill_mode='next_open',
                    order_cancel_time=datetime.time(23, 59), params=p)
    res = run_backtest(AnchoredMeanReversionStrategy, cfg, data=md, validate=False)
    pnl, sl = extract_normalised_trades(res.trades)
    R = pnl / np.maximum(sl, 1e-6)
    mae = np.array([float(getattr(t, 'mae_points', 0.0) or 0.0) for t in res.trades])
    dates = [md.df_1m.index[t.entry_bar].date() for t in res.trades]
    df = pd.DataFrame({'date': pd.to_datetime(dates), 'R': R, 'sl_dist': sl,
                       'MAE_R': mae / np.maximum(sl, 1e-6)})
    hmm = pickle.load(open(REPO / 'cache' / 'amr_hmm.pkl', 'rb'))
    labels = hmm.labels
    df['hmm'] = [labels.get(d - timedelta(days=1), 1) for d in dates]
    return df


def build_arrays(df_slice):
    """Per-day arrays (1 slot) from a trade DataFrame slice."""
    if len(df_slice) == 0:
        return None
    udays = pd.to_datetime(df_slice['date'].unique())
    nd = len(udays)
    d2i = {d: i for i, d in enumerate(udays)}
    mt = int(df_slice.groupby('date').size().max())
    R_a = np.zeros((1, nd)); sl_a = np.full((1, nd), 16.5); fired = np.ones((1, nd), bool)
    nt = np.zeros((1, nd), np.int64); Rt = np.zeros((1, nd, mt)); MAE = np.zeros((1, nd, mt))
    for dt, sub in df_slice.groupby('date'):
        i = d2i[dt]
        R_a[0, i] = sub['R'].sum(); sl_a[0, i] = sub['sl_dist'].mean(); nt[0, i] = len(sub)
        k = min(len(sub), mt)
        Rt[0, i, :k] = sub['R'].values[:k]; MAE[0, i, :k] = sub['MAE_R'].values[:k]
    return R_a, sl_a, fired, nt, Rt, MAE, nd


def run_pp(arrays, erp, seeds=SEEDS, n_sims=N_SIMS):
    R_a, sl_a, fired, nt, Rt, MAE, nd = arrays
    pps = []
    for seed in seeds:
        out = three_phase_reinvestment_mc(
            account=A, budget=BUDGET, horizon=84, max_concurrent=1, n_sims=n_sims, seed=seed,
            eval_grind_risk=erp, bootstrap_block_len=5.0,
            bootstrap_oos_R_per_slot=R_a, bootstrap_oos_sl_dists_per_slot=sl_a,
            bootstrap_oos_fired_per_slot=fired, bootstrap_oos_n_trades_per_slot=nt,
            bootstrap_oos_R_per_trade_per_slot=Rt, bootstrap_oos_MAE_R_per_trade_per_slot=MAE,
            bootstrap_intraday_blow_check=True, enforce_consistency_rule=True,
        )
        cash = out['cash'].astype(np.float64)
        n_e = out['eval_fees_paid'] / A.eval_fee
        reopens = np.maximum(n_e - 1, 0.0)
        cash_adj = cash - (A.reset_fee - A.eval_fee) * reopens
        pps.append(float((cash_adj >= BUDGET + MIN_PAYOUT).mean()))
    return float(np.mean(pps))


def main():
    print("Building cfg_495 trades (with/without ToD filter)...", flush=True)
    df_full = get_cfg495_trades(tod_filter=False)
    df_tod  = get_cfg495_trades(tod_filter=True)

    bundles = []
    for tod_name, df in [('none', df_full), ('tod935', df_tod)]:
        for hmm_name in ['none', 'dropbull']:
            d = df if hmm_name == 'none' else df[df['hmm'].isin([0, 1])]
            bundles.append((tod_name, hmm_name, d))

    print("\n=== TRAIN SELECTION (2019-2022) ===", flush=True)
    train_results = []
    for tod_name, hmm_name, d in bundles:
        d_train = d[(d['date'] >= TRAIN_LO) & (d['date'] <= TRAIN_HI)]
        arr = build_arrays(d_train)
        if arr is None or arr[6] < 100:
            continue
        for erp in [0.25, 0.35, 0.45]:
            pp = run_pp(arr, erp, seeds=[2026, 2027])
            train_results.append((tod_name, hmm_name, erp, pp, arr[6]))
            print(f"  ToD={tod_name:<7} HMM={hmm_name:<9} erp={erp}: train_pp={pp:.4f} ({arr[6]} days)", flush=True)

    best = max(train_results, key=lambda x: x[3])
    print(f"\n=== SELECTED (train-best): ToD={best[0]}, HMM={best[1]}, erp={best[2]}, train_pp={best[3]:.4f} ===", flush=True)

    # ── SINGLE-SHOT OOS 2024+ ──
    print("\n=== TRUE OOS SINGLE-SHOT (2024+) ===", flush=True)
    tod_name, hmm_name, erp = best[0], best[1], best[2]
    df_sel = dict([('none', df_full), ('tod935', df_tod)])[tod_name]
    if hmm_name == 'dropbull':
        df_sel = df_sel[df_sel['hmm'].isin([0, 1])]
    d_oos = df_sel[df_sel['date'] >= OOS_LO]
    arr_oos = build_arrays(d_oos)
    if arr_oos is None or arr_oos[6] < 30:
        print(f"  OOS insufficient ({arr_oos[6] if arr_oos else 0} days)", flush=True)
        return
    oos_pp = run_pp(arr_oos, erp, seeds=SEEDS)
    n_oos_days = arr_oos[6]

    # Daily R permutation null (sign-flip)
    daily_R = d_oos.groupby('date')['R'].sum().values
    obs_mean = float(daily_R.mean())
    rng = np.random.default_rng(2026)
    signs = rng.choice([-1.0, 1.0], size=(1000, len(daily_R)))
    null_means = (signs * daily_R[None, :]).mean(axis=1)
    perm_p = float((null_means >= obs_mean).mean())

    print(f"  OOS days={n_oos_days}, daily_mean_R={obs_mean:+.4f}, perm_p={perm_p:.4f}", flush=True)
    print(f"  OOS p_profit = {oos_pp:.4f}", flush=True)
    print(f"\n=== HONEST DEPLOYABLE NUMBER: {oos_pp:.1%} p_profit (2024+ OOS, single-shot) ===", flush=True)

    out = {
        'selected_bundle': {'ToD': tod_name, 'HMM': hmm_name, 'erp': erp},
        'train_pp': best[3],
        'oos_pp': oos_pp,
        'oos_days': int(n_oos_days),
        'oos_daily_mean_R': obs_mean,
        'oos_permutation_p': perm_p,
    }
    import json
    with open(REPO / 'results' / 'amr_honest_oos.json', 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f"Wrote results/amr_honest_oos.json", flush=True)


if __name__ == '__main__':
    main()
