"""
regime_lookup.py — query HMM regime labels for any past date(s)

Reads the cached RegimeResult (cache/amr_hmm.pkl) which contains a label for
every NQ daily bar in the parquet. Useful for verifying historical regimes,
debugging a specific day, or seeing the regime context around an interesting
date.

Both date conventions are accepted:
  --parquet     interpret arguments as Sierra Chart session-start dates
  (default)     interpret arguments as Tradovate session-end dates

Examples:
  python scripts/propfirm/regime_lookup.py 2026-03-13                  # single Tradovate date
  python scripts/propfirm/regime_lookup.py 2026-03-12 --parquet        # same bar, parquet convention
  python scripts/propfirm/regime_lookup.py 2026-03-01 2026-03-31       # range, Tradovate
  python scripts/propfirm/regime_lookup.py last 30                     # last 30 labelled days
  python scripts/propfirm/regime_lookup.py last 30 --csv > out.csv     # save range to CSV
"""
from __future__ import annotations

import argparse
import pickle
import sys
from datetime import date, timedelta
from pathlib import Path

REPO_ROOT  = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))   # so pickle can resolve backtest.regime.hmm.RegimeResult
CACHE_PATH = REPO_ROOT / 'cache' / 'amr_hmm.pkl'

WEEKDAY_NAME = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}


def parse_date(s: str) -> date:
    parts = s.replace('/', '-').split('-')
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"Bad date {s!r}; expected YYYY-MM-DD")
    y, m, d = (int(p) for p in parts)
    return date(y, m, d)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument('args', nargs='+',
                    help='Single date, two dates (range), or "last N" for last N labelled days')
    ap.add_argument('--parquet', action='store_true',
                    help='Interpret date arguments as parquet (session-start) dates. '
                         'Default: Tradovate (session-end) dates.')
    ap.add_argument('--csv', action='store_true', help='Emit as CSV (with header)')
    args = ap.parse_args()

    if not CACHE_PATH.exists():
        print(f"ERROR: HMM cache not found at {CACHE_PATH}. "
              f"Run scripts/propfirm/regime_update.py first.", file=sys.stderr)
        sys.exit(1)

    with open(CACHE_PATH, 'rb') as f:
        result = pickle.load(f)
    labels      = result.labels             # {parquet_date: int}
    label_names = result.label_names        # {0:'bear', 1:'neutral', 2:'bull'}
    all_dates   = sorted(labels.keys())

    # ── Resolve query into a list of parquet dates ───────────────────────
    if len(args.args) == 2 and args.args[0] == 'last':
        try:
            n = int(args.args[1])
        except ValueError:
            ap.error("`last` must be followed by an integer (count of days)")
            return
        target_parquet_dates = all_dates[-n:]
    else:
        try:
            input_dates = [parse_date(s) for s in args.args]
        except argparse.ArgumentTypeError as e:
            ap.error(str(e))
            return
        if len(input_dates) == 1:
            input_dates = input_dates * 2  # treat as range of 1
        if len(input_dates) != 2:
            ap.error("Provide a single date, two dates (range), or 'last N'")
            return

        # If input is Tradovate convention, shift to parquet (-1 day)
        if args.parquet:
            start, end = input_dates
        else:
            start = input_dates[0] - timedelta(days=1)
            end   = input_dates[1] - timedelta(days=1)
        if start > end:
            start, end = end, start
        target_parquet_dates = [d for d in all_dates if start <= d <= end]

    if not target_parquet_dates:
        print("No matching labelled dates in that window.", file=sys.stderr)
        sys.exit(2)

    # ── Emit ──────────────────────────────────────────────────────────────
    if args.csv:
        print("parquet_date,tradovate_date,weekday,regime,label_id")
        for d in target_parquet_dates:
            td = d + timedelta(days=1)
            lab = labels[d]
            print(f"{d},{td},{WEEKDAY_NAME[d.weekday()]},{label_names[lab]},{lab}")
    else:
        print(f"{'parquet_date':<13} {'tradovate_date':<15} {'weekday':<5} {'regime':<8} {'label_id'}")
        print(f"{'-'*13} {'-'*15} {'-'*5} {'-'*8} {'-'*8}")
        for d in target_parquet_dates:
            td = d + timedelta(days=1)
            lab = labels[d]
            print(f"{d!s:<13} {td!s:<15} {WEEKDAY_NAME[d.weekday()]:<5} "
                  f"{label_names[lab]:<8} {lab}")


if __name__ == '__main__':
    main()
