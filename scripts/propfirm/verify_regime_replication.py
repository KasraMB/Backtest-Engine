"""
verify_regime_replication.py — confirm daemon labels match backtest labels

Refits the HMM exactly as regime_update.py does, then diffs every (date, label)
pair against the saved backtest cache `cache/amr_hmm.pkl`. Any mismatch means
the live daemon would produce different regime labels than the backtest used —
which would invalidate the playbook's expected EV numbers.

If you change data sources, data adjustment conventions, or HMM hyperparams,
run this script and confirm 0 mismatches before going live.

Run:
  python scripts/propfirm/verify_regime_replication.py
  python scripts/propfirm/verify_regime_replication.py --strict   # exit 1 on any mismatch
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.propfirm.regime_update import compute_regime, PARQUET_PATH, CACHE_PATH   # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument('--parquet', default=str(PARQUET_PATH))
    ap.add_argument('--cache',   default=str(CACHE_PATH))
    ap.add_argument('--strict',  action='store_true',
                    help='Exit with non-zero status if any mismatches found')
    args = ap.parse_args()

    cache_path = Path(args.cache)
    if not cache_path.exists():
        print(f"ERROR: backtest cache not found at {cache_path}", file=sys.stderr)
        sys.exit(2)

    print(f"Loading backtest cache: {cache_path}", flush=True)
    with open(cache_path, 'rb') as f:
        backtest_result = pickle.load(f)
    backtest_labels = backtest_result.labels

    print(f"Refitting HMM from {args.parquet}...", flush=True)
    _, live_result = compute_regime(Path(args.parquet))
    live_labels = live_result.labels

    # Compare on the intersection of dates
    backtest_dates = set(backtest_labels.keys())
    live_dates     = set(live_labels.keys())
    common = sorted(backtest_dates & live_dates)
    only_backtest = backtest_dates - live_dates
    only_live     = live_dates - backtest_dates

    n_match    = sum(1 for d in common if backtest_labels[d] == live_labels[d])
    n_total    = len(common)
    mismatches = [
        (d, backtest_labels[d], live_labels[d])
        for d in common
        if backtest_labels[d] != live_labels[d]
    ]

    print(f"\n=== Replication check ===")
    print(f"  Backtest labelled days:  {len(backtest_dates):>5}")
    print(f"  Live labelled days:      {len(live_dates):>5}")
    print(f"  Common dates:            {n_total:>5}")
    print(f"  Only-in-backtest:        {len(only_backtest):>5}")
    print(f"  Only-in-live (new):      {len(only_live):>5}")
    print(f"  Match:                   {n_match:>5} / {n_total}  ({n_match/max(n_total,1):.2%})")
    print(f"  Mismatch:                {len(mismatches):>5}")

    # Compare HMM params
    print(f"\n  Transition matrix diff (backtest vs live, abs max):")
    import numpy as np
    tm_diff = np.abs(backtest_result.transition_matrix - live_result.transition_matrix)
    print(f"    max |diff| = {tm_diff.max():.6f}")
    sm_diff = np.abs(np.array(backtest_result.state_means) - np.array(live_result.state_means))
    print(f"  State-means diff (abs max): {sm_diff.max():.6f}")

    if mismatches:
        print(f"\nFirst {min(20, len(mismatches))} mismatches (date, backtest, live):")
        for d, b, l in mismatches[:20]:
            print(f"  {d}: backtest={backtest_result.label_names[b]:>7}  "
                  f"live={live_result.label_names[l]:>7}")

    if only_live:
        print(f"\nNew dates available in live ({len(only_live)}): {sorted(only_live)[:10]}...")

    if mismatches and args.strict:
        sys.exit(1)


if __name__ == '__main__':
    main()
