"""
Realistic propfirm sim: 3-slot bootstrap of (cfg_406, cfg_495, cfg_152) at $1K BUDGET.

Uses the AUDITED sim path with all realistic flags ON:
  - bootstrap mode (per-slot arrays, shared day-indices → preserves correlation)
  - bootstrap_intraday_blow_check=True (real-MAE intraday MLL touch detection)
  - enforce_consistency_rule=True (LucidFlex eval 50% rule)
  - Trade pools rebuilt with fill_mode='next_open' + slippage_points=0.5

This is THE deployable number — if it clears 80%, ship; if not, structure pivot needed.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from backtest.propfirm.lucidflex import LUCIDFLEX_ACCOUNTS
from backtest.propfirm.three_phase_mc import three_phase_reinvestment_mc

ACCOUNT = LUCIDFLEX_ACCOUNTS['50K']
BUDGET = 1000.0
HORIZON = 84
N_SIMS = 10_000
SEEDS = [2026, 2027, 2028]
MIN_PAYOUT = 500.0
RESET_DELTA = -3.0   # $95 reset - $98 eval


def align_arrays_to_common_days(pools: dict, cfg_ids: list):
    """Align per-day arrays of 3 configs to the COMMON set of OOS days
    (2023+) so all slots have the same (n_oos_days,) shape.

    Days where a config didn't trade get R=0, fired=False, n_trades=0.
    """
    # Collect each cfg's OOS dates and R series
    all_dates = set()
    cfg_data = {}
    for cfg_id in cfg_ids:
        pd_data = pools[cfg_id]['per_day']
        dates = pd_data['dates']  # pandas DatetimeIndex
        # Filter to OOS (>=2023)
        import pandas as pd
        oos_mask = dates >= pd.Timestamp('2023-01-01')
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

    sorted_dates = sorted(all_dates)
    n_days = len(sorted_dates)
    date_to_idx = {d: i for i, d in enumerate(sorted_dates)}
    max_trades = max(c['R_per_trade'].shape[1] for c in cfg_data.values())
    max_c = len(cfg_ids)

    R_arr     = np.zeros((max_c, n_days), dtype=np.float64)
    sl_arr    = np.full((max_c, n_days), 16.5, dtype=np.float64)   # default reasonable SL
    fired_arr = np.zeros((max_c, n_days), dtype=bool)
    nt_arr    = np.zeros((max_c, n_days), dtype=np.int64)
    Rt_arr    = np.zeros((max_c, n_days, max_trades), dtype=np.float64)
    MAE_arr   = np.zeros((max_c, n_days, max_trades), dtype=np.float64)

    for slot_idx, cfg_id in enumerate(cfg_ids):
        c = cfg_data[cfg_id]
        for j, d in enumerate(c['dates']):
            day_i = date_to_idx[d]
            R_arr[slot_idx, day_i]   = c['R'][j]
            sl_arr[slot_idx, day_i]  = c['sl_dists'][j]
            fired_arr[slot_idx, day_i] = (c['n_trades'][j] > 0)
            nt_arr[slot_idx, day_i]  = c['n_trades'][j]
            Rt_arr[slot_idx, day_i, :max_trades]   = c['R_per_trade'][j, :max_trades]
            MAE_arr[slot_idx, day_i, :max_trades]  = c['MAE_per_trade'][j, :max_trades]
    return R_arr, sl_arr, fired_arr, nt_arr, Rt_arr, MAE_arr, n_days


def run_3slot(cfg_ids, pools, erp, n_sims=N_SIMS, seed=2026):
    R_arr, sl_arr, fired_arr, nt_arr, Rt_arr, MAE_arr, n_days = align_arrays_to_common_days(pools, cfg_ids)
    out = three_phase_reinvestment_mc(
        account=ACCOUNT,
        budget=BUDGET,
        horizon=HORIZON,
        max_concurrent=len(cfg_ids),
        n_sims=n_sims,
        seed=seed,
        eval_grind_risk=erp,
        bootstrap_block_len=5.0,
        bootstrap_oos_R_per_slot=R_arr,
        bootstrap_oos_sl_dists_per_slot=sl_arr,
        bootstrap_oos_fired_per_slot=fired_arr,
        bootstrap_oos_n_trades_per_slot=nt_arr,
        bootstrap_oos_R_per_trade_per_slot=Rt_arr,
        bootstrap_oos_MAE_R_per_trade_per_slot=MAE_arr,
        bootstrap_intraday_blow_check=True,
        bootstrap_cap_daily_risk=False,
        enforce_consistency_rule=True,
    )
    cash = out['cash'].astype(np.float64)
    n_evals = out['eval_fees_paid'] / 98.0
    reopens = np.maximum(n_evals - len(cfg_ids), 0.0)
    cash_adj = cash - RESET_DELTA * reopens
    growth = cash_adj - BUDGET
    return {
        'p_profit': float((cash_adj >= BUDGET + MIN_PAYOUT).mean()),
        'p_zero':   float((cash_adj <= 0).mean()),
        'mean_g':   float(growth.mean()),
        'median_g': float(np.median(growth)),
        'sharpe':   float(growth.mean() / max(growth.std(), 1e-9)),
        'sortino':  float(growth.mean() / max(growth[growth<0].std() if (growth<0).any() else 1.0, 1e-9)),
        'mean_payouts': float(out['n_payouts'].mean()),
        'n_oos_days_pool': n_days,
    }


def main():
    pools = pickle.load(open(REPO / 'cache' / 'amr_realistic_pools.pkl', 'rb'))
    print(f"Meta: {pools['meta']}")
    for k, v in pools.items():
        if k == 'meta': continue
        pd = v['per_day']
        print(f"  {k}: {pd['total_trades']} trades, mean_R={pd['mean_R_trade']:+.4f}, WR={pd['wr']*100:.1f}%")

    print(f"\n{'='*100}")
    print(f"REALISTIC 3-SLOT BOOTSTRAP at $1K, 50K Flex, intraday MLL + consistency ON, n_sims={N_SIMS}")
    print(f"{'='*100}")

    # Single-slot baselines
    print("\n--- Single-slot baselines (max_c=1) ---")
    for cfg_id in ['cfg_406', 'cfg_495', 'cfg_152']:
        for erp in [0.5, 0.7]:
            results = [run_3slot([cfg_id], pools, erp, seed=s) for s in SEEDS]
            pp = np.mean([r['p_profit'] for r in results])
            sortino = np.mean([r['sortino'] for r in results])
            mean_g = np.mean([r['mean_g'] for r in results])
            tag = " <-PASS>80%" if pp > 0.80 else ""
            print(f"  {cfg_id} mc=1 erp={erp}: p_profit={pp:.4f}{tag}, Sortino={sortino:+.2f}, mean=${mean_g:+,.0f}")

    # 2-slot combos
    print("\n--- 2-slot diversified ---")
    for combo in [['cfg_406','cfg_495'], ['cfg_406','cfg_152'], ['cfg_495','cfg_152']]:
        for erp in [0.5, 0.7]:
            results = [run_3slot(combo, pools, erp, seed=s) for s in SEEDS]
            pp = np.mean([r['p_profit'] for r in results])
            sortino = np.mean([r['sortino'] for r in results])
            mean_g = np.mean([r['mean_g'] for r in results])
            tag = " <-PASS>80%" if pp > 0.80 else ""
            print(f"  {'+'.join(combo)} erp={erp}: p_profit={pp:.4f}{tag}, Sortino={sortino:+.2f}, mean=${mean_g:+,.0f}")

    # 3-slot full diversification
    print("\n--- 3-slot full diversification ---")
    combo = ['cfg_406','cfg_495','cfg_152']
    for erp in [0.3, 0.4, 0.5, 0.6, 0.7, 0.9]:
        results = [run_3slot(combo, pools, erp, seed=s) for s in SEEDS]
        pp = np.mean([r['p_profit'] for r in results])
        sortino = np.mean([r['sortino'] for r in results])
        sharpe = np.mean([r['sharpe'] for r in results])
        mean_g = np.mean([r['mean_g'] for r in results])
        median_g = np.mean([r['median_g'] for r in results])
        npay = np.mean([r['mean_payouts'] for r in results])
        tag = " <-PASS>80%" if pp > 0.80 else ""
        print(f"  3-slot erp={erp}: p_profit={pp:.4f}{tag}, Sharpe={sharpe:+.3f}, Sortino={sortino:+.2f}, "
              f"mean=${mean_g:+,.0f}, median=${median_g:+,.0f}, npay={npay:.1f}")


if __name__ == '__main__':
    main()
