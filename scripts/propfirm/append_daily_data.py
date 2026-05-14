"""
append_daily_data.py — extend NQ_1day_full_data.parquet with new daily bars

Validates a candidate data source against the existing parquet's overlap rows
before appending. Refuses to corrupt the parquet if the new source disagrees
with the existing data.

Two data-source modes:

  --source yahoo   (requires yfinance + internet)
      Pulls NQ=F daily bars from Yahoo Finance for the gap window.

  --source csv --csv-path PATH
      Reads bars from a local CSV. Must have columns: Date, Open, High, Low,
      Last (or Close), Volume.

Workflow:
  1. Determine existing parquet's last date to that's the gap-start.
  2. Pull new source from (gap-start - overlap_days) to today.
  3. Compare overlap rows: max |relative diff| in Last (close) price.
  4. If PASS (<0.05%): append the post-overlap rows to the parquet.
  5. If WARN (0.05% - 0.5%): print diff table, refuse, recommend manual fix.
  6. If FAIL (>0.5%): print diff table, refuse, suggest different source.

Run:
  python scripts/propfirm/append_daily_data.py --dry-run                 # validate only
  python scripts/propfirm/append_daily_data.py --source yahoo            # fetch + append
  python scripts/propfirm/append_daily_data.py --source csv --csv-path data/new_bars.csv
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
PARQUET   = REPO_ROOT / 'data' / 'NQ_1day_full_data.parquet'

OVERLAP_DAYS  = 30      # calendar days of overlap to compare
TOL_PASS_REL  = 0.0005  # 0.05% — auto-append below this
TOL_WARN_REL  = 0.005   # 0.5%  — refuse above this


def _load_existing() -> pd.DataFrame:
    df = pd.read_parquet(PARQUET)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    return df


def _fetch_yahoo(start: date, end: date) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError:
        raise SystemExit("yfinance not installed. Run: pip install yfinance")

    print(f"Fetching NQ=F from Yahoo {start} to {end} ...", flush=True)
    df = yf.download('NQ=F', start=start.isoformat(), end=(end + timedelta(days=1)).isoformat(),
                     interval='1d', progress=False, auto_adjust=False)
    if df.empty:
        raise SystemExit(f"Yahoo returned no data for NQ=F {start}..{end}. "
                         f"Network unreachable or ticker delisted.")
    # Flatten potential MultiIndex columns (newer yfinance versions)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    out = pd.DataFrame({
        'Date':   pd.to_datetime(df.index.date),
        'Time':   '18:00:00',
        'Open':   df['Open'].values.astype(float),
        'High':   df['High'].values.astype(float),
        'Low':    df['Low'].values.astype(float),
        'Last':   df['Close'].values.astype(float),
        'Volume': df['Volume'].values.astype(int),
        'NumTrades': 0,
        'BidVol':    0,
        'AskVol':    0,
    })
    return out.sort_values('Date').reset_index(drop=True)


def _fetch_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Normalize column names
    cols_lower = {c.lower(): c for c in df.columns}
    if 'date' not in cols_lower:
        raise SystemExit(f"CSV must have a 'Date' column. Got: {list(df.columns)}")
    df['Date'] = pd.to_datetime(df[cols_lower['date']])
    close_col = cols_lower.get('last') or cols_lower.get('close')
    if close_col is None:
        raise SystemExit(f"CSV must have 'Last' or 'Close' column. Got: {list(df.columns)}")
    out = pd.DataFrame({
        'Date':   df['Date'],
        'Time':   '18:00:00',
        'Open':   df[cols_lower.get('open', close_col)].astype(float),
        'High':   df[cols_lower.get('high', close_col)].astype(float),
        'Low':    df[cols_lower.get('low',  close_col)].astype(float),
        'Last':   df[close_col].astype(float),
        'Volume': df[cols_lower.get('volume', close_col)].fillna(0).astype(int) if 'volume' in cols_lower else 0,
        'NumTrades': 0,
        'BidVol':    0,
        'AskVol':    0,
    })
    return out.sort_values('Date').reset_index(drop=True)


def _compare_overlap(existing: pd.DataFrame, new_src: pd.DataFrame, overlap_start: pd.Timestamp):
    e = existing[existing['Date'] >= overlap_start].set_index('Date')
    n = new_src[new_src['Date'] >= overlap_start].set_index('Date')
    common = e.index.intersection(n.index)
    if len(common) == 0:
        print("ERROR: no overlapping dates between existing parquet and new source.", file=sys.stderr)
        sys.exit(2)

    cmp = pd.DataFrame({
        'existing_Last': e.loc[common, 'Last'].values,
        'new_Last':      n.loc[common, 'Last'].values,
    }, index=common)
    cmp['abs_diff'] = np.abs(cmp['existing_Last'] - cmp['new_Last'])
    cmp['rel_diff'] = cmp['abs_diff'] / cmp['existing_Last']

    max_abs = float(cmp['abs_diff'].max())
    max_rel = float(cmp['rel_diff'].max())
    median_rel = float(cmp['rel_diff'].median())

    print(f"\n=== Overlap comparison ({len(common)} common dates) ===")
    print(f"  Max  |abs diff|: ${max_abs:.4f}")
    print(f"  Max  rel diff:   {max_rel:.4%}")
    print(f"  Median rel diff: {median_rel:.4%}")
    # Always show worst 5 rows
    worst = cmp.sort_values('rel_diff', ascending=False).head(5)
    print(f"\n  Worst 5 mismatches:")
    for d, row in worst.iterrows():
        print(f"    {d.date()}: existing=${row['existing_Last']:.2f}  "
              f"new=${row['new_Last']:.2f}  "
              f"|abs|=${row['abs_diff']:.2f}  rel={row['rel_diff']:.4%}")

    if max_rel < TOL_PASS_REL:
        verdict = 'PASS'
    elif max_rel < TOL_WARN_REL:
        verdict = 'WARN'
    else:
        verdict = 'FAIL'

    print(f"\n  Verdict: {verdict}  (PASS<{TOL_PASS_REL:.2%}, WARN<{TOL_WARN_REL:.2%}, else FAIL)")
    return verdict, cmp


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument('--source', choices=['yahoo', 'csv'], default='yahoo')
    ap.add_argument('--csv-path', help='Required if --source csv')
    ap.add_argument('--overlap-days', type=int, default=OVERLAP_DAYS)
    ap.add_argument('--dry-run', action='store_true',
                    help='Validate overlap only, do not append even if PASS')
    ap.add_argument('--force', action='store_true',
                    help='Append even on WARN (do NOT use unless you understand the diff)')
    ap.add_argument('--date-shift', type=int, default=0,
                    help='Calendar-day offset applied to source dates BEFORE comparison. '
                         'Use -1 for Tradovate (session-end-date) -> Sierra Chart (session-start-date).')
    args = ap.parse_args()

    if args.source == 'csv' and not args.csv_path:
        ap.error("--source csv requires --csv-path")

    existing = _load_existing()
    last_date = existing['Date'].iloc[-1]
    today = pd.Timestamp(date.today())

    print(f"Existing parquet: {len(existing)} rows, last date = {last_date.date()}")
    print(f"Today:            {today.date()}")
    days_missing = (today - last_date).days
    print(f"Gap:              {days_missing} calendar days")
    if days_missing <= 0:
        print("Parquet already current — nothing to do.")
        return

    overlap_start = last_date - timedelta(days=args.overlap_days)
    if args.source == 'yahoo':
        new_src = _fetch_yahoo(overlap_start.date(), today.date())
    else:
        new_src = _fetch_csv(Path(args.csv_path))

    # Apply date shift (e.g. -1 to translate Tradovate session-end dates
    # to Sierra Chart's session-start convention)
    if args.date_shift != 0:
        new_src = new_src.copy()
        new_src['Date'] = new_src['Date'] + pd.Timedelta(days=args.date_shift)
        print(f"Applied --date-shift {args.date_shift:+d} days to source dates.")

    print(f"New source: {len(new_src)} rows, {new_src['Date'].min().date()} to {new_src['Date'].max().date()}")

    verdict, _ = _compare_overlap(existing, new_src, overlap_start)
    if verdict == 'FAIL':
        print("\nRefusing to append — data sources disagree by >0.5%.", file=sys.stderr)
        print("Use your original data source (Sierra Chart, IB, etc.) to fill the gap.",
              file=sys.stderr)
        sys.exit(1)
    if verdict == 'WARN' and not args.force:
        print("\nRefusing to append — data sources disagree by 0.05-0.5%.", file=sys.stderr)
        print("Re-run with --force only if you understand the diff above is acceptable.",
              file=sys.stderr)
        sys.exit(1)

    if args.dry_run:
        print("\nDry run — not modifying parquet. Would have appended:")
        to_append = new_src[new_src['Date'] > last_date]
        print(f"  {len(to_append)} new rows ({to_append['Date'].min().date() if len(to_append) else 'n/a'}"
              f" to {to_append['Date'].max().date() if len(to_append) else 'n/a'})")
        return

    to_append = new_src[new_src['Date'] > last_date].copy()
    if len(to_append) == 0:
        print("No new rows beyond existing parquet's last date — nothing to append.")
        return

    print(f"\nAppending {len(to_append)} rows ({to_append['Date'].min().date()} "
          f"to {to_append['Date'].max().date()}) ...")
    # Ensure column order matches existing parquet
    to_append = to_append[list(existing.columns)]
    combined = pd.concat([existing, to_append], ignore_index=True)

    # Backup existing parquet before overwriting
    backup = PARQUET.with_suffix(f'.parquet.bak.{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    PARQUET.rename(backup)
    print(f"  Backed up old parquet to {backup.name}")

    combined.to_parquet(PARQUET, index=False)
    print(f"  Wrote {PARQUET} ({len(combined)} rows total).")
    print(f"\nNow re-run scripts/propfirm/regime_update.py to get the latest regime.")


if __name__ == '__main__':
    main()
