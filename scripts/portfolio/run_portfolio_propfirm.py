"""Phase 4/5: equal-RISK multi-strategy propfirm portfolio.

Implements the "combine uncorrelated strategies, judge at the propfirm level"
thesis: each candidate cell is a SLOT sized to the same dollar risk per trade
(equal risk, not equal-cell-weight-on-R), so low-correlation cells genuinely
reduce combined variance.

Candidate set (relaxed, propfirm-level — NOT strong-edge): positive expectancy,
consistent across years, enough trades, valid R. We do NOT pre-dismiss weak edges.

TRAIN (2019-2023): sweep erp, report train p_profit + concentration.   [default]
OOS  (2024+):      single-shot at locked erp.                          [--oos]
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from backtest.portfolio import (
    backtest_cell, build_per_slot_arrays, load_instrument_window,
    propfirm_p_profit_slots,
)

CACHE = REPO / 'cache' / 'portfolio'
OOS_LO = pd.Timestamp('2024-01-01', tz='America/New_York')
ERP_GRID = [0.20, 0.30, 0.40, 0.50, 0.70]


def select_candidates(results: dict, min_n=50, min_mean_R=1e-9, min_years=4):
    """Relaxed, propfirm-level candidate set: positive + consistent + enough
    trades. No strong-edge or per-cell significance gate (judged in the portfolio)."""
    out = []
    for name, c in results.items():
        if c.n_trades >= min_n and c.mean_R > min_mean_R and c.years_positive >= min_years:
            out.append(c)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--oos', action='store_true', help='Touch OOS (single-shot)')
    ap.add_argument('--ex-ict', action='store_true', help='Drop degenerate ICT slot')
    args = ap.parse_args()

    results = pickle.load(open(CACHE / 'cell_results.pkl', 'rb'))
    cands = select_candidates(results)
    if args.ex_ict:
        cands = [c for c in cands if 'ICTSMC' not in c.cell.name]
    cands.sort(key=lambda c: -c.mean_R)
    names = [c.cell.name for c in cands]
    print(f"Candidate slots ({len(cands)}): {names}", flush=True)
    for c in cands:
        print(f"  {c.cell.name:30s} n={c.n_trades:5d} mean_R={c.mean_R:+.4f} "
              f"yrs+={c.years_positive}/{len(c.by_year)} proxy_sl={c.proxy_sl_fraction:.2f}", flush=True)

    # --- TRAIN: erp sweep on cached (2019-2023) trades ---
    train_dfs = [c.trades for c in cands]
    train_arr = build_per_slot_arrays(train_dfs)
    print(f"\n=== TRAIN erp sweep ({train_arr['nd']} union days) ===", flush=True)
    best_erp, best_pp = None, -1.0
    for erp in ERP_GRID:
        pp = float(np.mean([propfirm_p_profit_slots(train_arr, erp, seed=s)['p_profit']
                            for s in (2026, 2027)]))
        if pp > best_pp:
            best_pp, best_erp = pp, erp
        print(f"  erp={erp}: train p_profit={pp:.4f}", flush=True)
    print(f"  -> LOCKED erp={best_erp} (train p_profit={best_pp:.4f})", flush=True)

    if not args.oos:
        print("\n(TRAIN only. Re-run with --oos to spend the single-shot OOS.)", flush=True)
        return

    # --- OOS: single-shot, backtest each candidate on 2024+ ---
    print("\n=== OOS (2024+) SINGLE-SHOT ===", flush=True)
    md_cache = {}
    oos_dfs = []
    for c in cands:
        inst = c.cell.instrument
        if inst not in md_cache:
            md_cache[inst] = load_instrument_window(inst, OOS_LO, None)
        res = backtest_cell(c.cell, md_cache[inst])
        oos_dfs.append(res.trades)
        print(f"  {c.cell.name:30s} OOS n={res.n_trades:5d} mean_R={res.mean_R:+.4f}", flush=True)
    oos_arr = build_per_slot_arrays(oos_dfs)
    if oos_arr['nd'] < 50:
        print(f"  OOS insufficient ({oos_arr['nd']} days)", flush=True)
        return
    res = [propfirm_p_profit_slots(oos_arr, best_erp, seed=s) for s in (2026, 2027, 2028)]
    pp = float(np.mean([r['p_profit'] for r in res]))
    sortino = float(np.mean([r['sortino'] for r in res]))
    mean_g = float(np.mean([r['mean_growth'] for r in res]))
    median_g = float(np.mean([r['median_growth'] for r in res]))
    tag = " <- PASS >80%" if pp > 0.80 else (" >70%" if pp > 0.70 else "")
    print(f"\n  OOS p_profit={pp:.4f}{tag}  erp={best_erp}  Sortino={sortino:+.2f}  "
          f"mean=${mean_g:+,.0f} median=${median_g:+,.0f}", flush=True)


if __name__ == '__main__':
    main()
