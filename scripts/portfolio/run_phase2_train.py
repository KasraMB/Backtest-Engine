"""Phase 2: backtest all pre-registered cells on TRAIN data (2019-2023).

Runs one cell per call to backtest_cell(). Per-cell cache to cache/portfolio/
so re-runs are incremental. Failures are recorded but don't stop the loop.

Output:
  cache/portfolio/cell_results.pkl   — dict[cell_name -> CellResult]
  results/portfolio_phase2_summary.csv
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
import traceback
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from backtest.portfolio import backtest_cell, load_instrument, summarize_cells
from backtest.portfolio.registry import CELLS

CACHE_DIR = REPO / 'cache' / 'portfolio'
CACHE_DIR.mkdir(parents=True, exist_ok=True)
# Pre-registered TRAIN window: 2019-01-01 .. 2023-12-31. Data starts ~2014, so a
# lower bound is required — without it pre-2019 years leak into the filter and
# break the ">=4 of 5 years" rule.
TRAIN_LO = pd.Timestamp('2019-01-01 00:00', tz='America/New_York')
TRAIN_HI = pd.Timestamp('2023-12-31 23:59', tz='America/New_York')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--retry-failed', action='store_true',
                    help='Re-run cells that previously errored')
    ap.add_argument('--only', default=None,
                    help='Comma-separated cell name substrings to run (default: all)')
    args = ap.parse_args()

    results_path = CACHE_DIR / 'cell_results.pkl'
    failed_path  = CACHE_DIR / 'cell_failures.json'
    results: dict = pickle.load(open(results_path, 'rb')) if results_path.exists() else {}
    failures: dict = json.load(open(failed_path)) if failed_path.exists() else {}

    # Group cells by instrument to load MarketData ONCE per symbol
    by_inst: dict = {}
    for cell in CELLS:
        if args.only and not any(s in cell.name for s in args.only.split(',')):
            continue
        if cell.name in results:
            continue
        if cell.name in failures and not args.retry_failed:
            continue
        by_inst.setdefault(cell.instrument, []).append(cell)

    total = sum(len(v) for v in by_inst.values())
    if total == 0:
        print("All pre-registered cells already cached. Nothing to do.", flush=True)
        return

    print(f"Running {total} cells across {len(by_inst)} instruments...", flush=True)
    t_start = time.time()
    done = 0
    for inst, cells in by_inst.items():
        print(f"\nLoading {inst}...", flush=True)
        md_full = load_instrument(inst)
        # Slice to TRAIN-only window [TRAIN_LO, TRAIN_HI]
        mask_1m = (md_full.df_1m.index >= TRAIN_LO) & (md_full.df_1m.index <= TRAIN_HI)
        mask_5m = (md_full.df_5m.index >= TRAIN_LO) & (md_full.df_5m.index <= TRAIN_HI)
        from backtest.data.market_data import MarketData
        from backtest.data.loader      import DataLoader
        df_1m = md_full.df_1m[mask_1m]
        df_5m = md_full.df_5m[mask_5m]
        loader = DataLoader(data_dir=str(REPO/'data'))
        import numpy as np
        a1 = {c: df_1m[c].to_numpy(np.float64) for c in ('open','high','low','close','volume')}
        a5 = {c: df_5m[c].to_numpy(np.float64) for c in ('open','high','low','close','volume')}
        from datetime import time as _time
        rth = (df_1m.index.time >= _time(9,30)) & (df_1m.index.time <= _time(17,0))
        md = MarketData(
            df_1m=df_1m, df_5m=df_5m,
            open_1m=a1['open'], high_1m=a1['high'], low_1m=a1['low'],
            close_1m=a1['close'], volume_1m=a1['volume'],
            open_5m=a5['open'], high_5m=a5['high'], low_5m=a5['low'],
            close_5m=a5['close'], volume_5m=a5['volume'],
            bar_map=loader._build_bar_map(df_1m, df_5m),
            trading_dates=sorted(set(df_1m[rth].index.date)),
        )
        print(f"  {inst} TRAIN: {len(md.df_1m):,} 1m bars", flush=True)

        for cell in cells:
            done += 1
            t0 = time.time()
            try:
                res = backtest_cell(cell, md)
                results[cell.name] = res
                dt = time.time() - t0
                print(f"  [{done}/{total}] {cell.name}: n={res.n_trades:>5}, "
                      f"mean_R={res.mean_R:+.4f}, yc={res.years_positive}/{len(res.by_year)} "
                      f"({dt:.1f}s)", flush=True)
                # Incremental save every 5 cells
                if done % 5 == 0:
                    pickle.dump(results, open(results_path, 'wb'))
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                failures[cell.name] = err
                print(f"  [{done}/{total}] {cell.name}: FAILED — {err[:80]}", flush=True)
                json.dump(failures, open(failed_path, 'w'), indent=2)

    # Final save
    pickle.dump(results, open(results_path, 'wb'))
    json.dump(failures, open(failed_path, 'w'), indent=2)

    # Summary CSV
    summary = summarize_cells(list(results.values()))
    summary.to_csv(REPO/'results'/'portfolio_phase2_summary.csv', index=False)

    print(f"\n=== Phase 2 complete in {(time.time()-t_start)/60:.1f} min ===", flush=True)
    print(f"  Successful cells: {len(results)} / {len(CELLS)}", flush=True)
    print(f"  Failures:         {len(failures)}", flush=True)


if __name__ == '__main__':
    main()
