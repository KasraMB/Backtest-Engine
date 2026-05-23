"""NQ bit-for-bit regression guard for the multi-instrument refactor.

The make_baseline.py tool only covers ICTSMC and is currently broken by the
strategies/ folder reorg. This guard snapshots EVERY strategy's NQ trades so we
can prove the per-instrument changes (which default to NQ economics) leave NQ
results bit-for-bit identical.

Usage:
  python scripts/portfolio/nq_regression.py snapshot pre     # before changes
  python scripts/portfolio/nq_regression.py snapshot post    # after changes
  python scripts/portfolio/nq_regression.py compare pre post # assert identical
"""
from __future__ import annotations

import hashlib
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from backtest.portfolio import backtest_cell, load_instrument_window
from backtest.portfolio.registry import CELLS

CACHE = REPO / 'cache' / 'portfolio'
TRAIN_LO = pd.Timestamp('2019-01-01 00:00', tz='America/New_York')
TRAIN_HI = pd.Timestamp('2023-12-31 23:59', tz='America/New_York')


def _digest(cell, md) -> dict:
    res = backtest_cell(cell, md)
    if res.n_trades == 0:
        h = 'EMPTY'
    else:
        arr = res.trades[['pnl_pts', 'sl_dist']].to_numpy(np.float64)
        h = hashlib.sha1(arr.tobytes()).hexdigest()[:16]
    return {'n': res.n_trades, 'sum_R': round(res.sum_R, 6),
            'mean_R': round(res.mean_R, 6), 'hash': h}


def snapshot(tag: str) -> None:
    md = load_instrument_window('NQ', TRAIN_LO, TRAIN_HI)
    nq_cells = [c for c in CELLS if c.instrument == 'NQ']
    out = {}
    for i, cell in enumerate(nq_cells, 1):
        try:
            out[cell.name] = _digest(cell, md)
            print(f"  [{i}/{len(nq_cells)}] {cell.name}: {out[cell.name]}", flush=True)
        except Exception as e:
            out[cell.name] = {'error': f"{type(e).__name__}: {e}"}
            print(f"  [{i}/{len(nq_cells)}] {cell.name}: ERROR {e}", flush=True)
    path = CACHE / f'nq_regression_{tag}.pkl'
    pickle.dump(out, open(path, 'wb'))
    print(f"\nSaved {len(out)} cells -> {path}")


def compare(tag_a: str, tag_b: str) -> None:
    a = pickle.load(open(CACHE / f'nq_regression_{tag_a}.pkl', 'rb'))
    b = pickle.load(open(CACHE / f'nq_regression_{tag_b}.pkl', 'rb'))
    keys = sorted(set(a) | set(b))
    diffs = []
    for k in keys:
        if a.get(k) != b.get(k):
            diffs.append((k, a.get(k), b.get(k)))
    if not diffs:
        print(f"IDENTICAL: all {len(keys)} NQ cells match between '{tag_a}' and '{tag_b}'. NQ bit-for-bit preserved.")
        return
    print(f"REGRESSION: {len(diffs)} / {len(keys)} NQ cells differ:")
    for k, va, vb in diffs:
        print(f"  {k}\n    {tag_a}: {va}\n    {tag_b}: {vb}")
    sys.exit(1)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    mode = sys.argv[1]
    if mode == 'snapshot':
        snapshot(sys.argv[2])
    elif mode == 'compare':
        compare(sys.argv[2], sys.argv[3])
    else:
        print(f"unknown mode {mode!r}"); sys.exit(1)
