"""
daily_update_from_tradovate.py — guided daily NQ data entry from Tradovate

Walks you through entering today's (or catching up multiple days') NQ daily
bars from a Tradovate continuous-NQ chart. For each missing bar, prints the
*Tradovate date* you should look up (Tradovate uses session-end dates), then
accepts the OHLC + volume in one line, validates, and stores it. After all
rows are entered, runs append_daily_data.py with --date-shift -1 to fold the
data into the parquet, then runs regime_update.py to compute today's regime.

Usage:
  python scripts/propfirm/daily_update_from_tradovate.py            # auto-detect missing days
  python scripts/propfirm/daily_update_from_tradovate.py --one-day  # just today

Input format per row:
  Open High Low Close Volume
  e.g. 25259.00 25465.25 25058.25 25281.50 524535

Type 'q' or Ctrl+C to stop entering early; rows already entered will still be
processed.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

REPO_ROOT  = Path(__file__).resolve().parents[2]
PARQUET    = REPO_ROOT / 'data' / 'NQ_1day_full_data.parquet'
APPEND_PY  = REPO_ROOT / 'scripts' / 'propfirm' / 'append_daily_data.py'
DAEMON_PY  = REPO_ROOT / 'scripts' / 'propfirm' / 'regime_update.py'
TMP_CSV    = REPO_ROOT / 'tmp_tradovate_input.csv'

WEEKDAY_NAME = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}


def next_session_start(d: date) -> date:
    """
    Next valid session-start date in parquet convention.
    CME futures weekly cycle: Sun-Thu session-start dates (skip Fri+Sat).
    The session "starting" on a given day opens at 18:00 ET and closes
    17:00 ET the next calendar day.
    """
    nxt = d + timedelta(days=1)
    # Skip Friday (4) and Saturday (5) — no session-start on those days
    while nxt.weekday() in (4, 5):
        nxt += timedelta(days=1)
    return nxt


def parquet_to_tradovate(parq_date: date) -> date:
    """Tradovate labels a bar with its session-end date = parquet_date + 1."""
    return parq_date + timedelta(days=1)


def prompt_one(parq_date: date, n_done: int, n_total: int) -> dict | None:
    """Prompt user for OHLC+Volume of one bar. Returns dict or None to abort."""
    trad_date = parquet_to_tradovate(parq_date)
    print()
    print(f"=== Bar {n_done + 1} / {n_total} ===")
    print(f"  Look up on Tradovate continuous-NQ chart:  "
          f"{WEEKDAY_NAME[trad_date.weekday()]} {trad_date.isoformat()}")
    print(f"  (this maps to parquet date {parq_date.isoformat()})")

    while True:
        line = input("  Enter  Open High Low Close Volume  (or 'q' to stop): ").strip()
        if line.lower() in ('q', 'quit', 'exit'):
            return None
        parts = line.split()
        if len(parts) != 5:
            print(f"    ERROR: expected 5 values, got {len(parts)}. Try again.")
            continue
        try:
            o = float(parts[0]); h = float(parts[1])
            l = float(parts[2]); c = float(parts[3])
            v = int(float(parts[4]))   # accept "524535" or "524535.0"
        except ValueError as e:
            print(f"    ERROR: parse failure ({e}). Try again.")
            continue
        # Sanity checks
        if l > o or l > c or h < o or h < c or h < l:
            print(f"    ERROR: low/high inconsistent (L={l}, H={h}, O={o}, C={c}). Try again.")
            continue
        if not (1_000 < o < 100_000):
            print(f"    ERROR: Open={o} is way out of typical NQ range. Did you mistype?")
            continue
        if v < 0:
            print(f"    ERROR: Volume can't be negative.")
            continue

        return {
            'Date':   trad_date.isoformat(),
            'Open':   o,
            'High':   h,
            'Low':    l,
            'Close':  c,
            'Volume': v,
        }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument('--one-day', action='store_true',
                    help='Only ask for one bar, not the full catch-up window')
    ap.add_argument('--skip-daemon', action='store_true',
                    help='Skip running regime_update.py at the end')
    args = ap.parse_args()

    # ── Load parquet, find missing dates ──────────────────────────────────
    df = pd.read_parquet(PARQUET)
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    last_parq_date = df['Date'].max()
    today = date.today()

    missing = []
    cur = next_session_start(last_parq_date)
    # Stop one day before today, because today's session (18:00 today → 17:00 tomorrow)
    # hasn't closed yet. Last fillable parquet date is the session that ended at
    # 17:00 ET today, which has start date = today - 1 day (or earlier weekend).
    last_fillable = today - timedelta(days=1)
    while cur <= last_fillable:
        missing.append(cur)
        cur = next_session_start(cur)

    print(f"Parquet last date: {last_parq_date.isoformat()} ({WEEKDAY_NAME[last_parq_date.weekday()]})")
    print(f"Today:             {today.isoformat()} ({WEEKDAY_NAME[today.weekday()]})")

    if not missing:
        print("\nParquet is current — no missing bars.")
        if not args.skip_daemon:
            _run_daemon()
        return

    if args.one_day and len(missing) > 1:
        print(f"\n--one-day specified; only entering most recent missing bar "
              f"({missing[-1].isoformat()}) out of {len(missing)} total missing.")
        missing = missing[-1:]

    print(f"\nMissing bars to enter: {len(missing)}")
    print(f"  First: parquet {missing[0]}  (Tradovate {parquet_to_tradovate(missing[0])})")
    print(f"  Last:  parquet {missing[-1]} (Tradovate {parquet_to_tradovate(missing[-1])})")

    rows: list[dict] = []
    try:
        for parq_date in missing:
            row = prompt_one(parq_date, len(rows), len(missing))
            if row is None:
                print("\nStopping early. Rows entered so far will still be processed.")
                break
            rows.append(row)
            print(f"    Recorded: O={row['Open']} H={row['High']} L={row['Low']} "
                  f"C={row['Close']} V={row['Volume']}")
    except KeyboardInterrupt:
        print("\n\nCtrl+C — stopping. Rows entered so far will still be processed.")

    if not rows:
        print("\nNo rows entered. Nothing to do.")
        return

    # ── Write temp CSV ────────────────────────────────────────────────────
    pd.DataFrame(rows).to_csv(TMP_CSV, index=False)
    print(f"\nWrote {len(rows)} rows to {TMP_CSV}")

    # ── Run appender ──────────────────────────────────────────────────────
    print("\n=== Running append_daily_data.py ===")
    rc = subprocess.call([
        sys.executable, str(APPEND_PY),
        '--source', 'csv',
        '--csv-path', str(TMP_CSV),
        '--date-shift', '-1',
        '--no-validate',   # manual entry; overlap validation isn't meaningful
    ])
    if rc != 0:
        print(f"\nAppend failed (exit {rc}). Check the output above.")
        print(f"The temp CSV is preserved at {TMP_CSV} — fix and re-run manually if needed.")
        return

    # ── Run daemon ────────────────────────────────────────────────────────
    if not args.skip_daemon:
        _run_daemon()


def _run_daemon():
    print("\n=== Running regime_update.py ===")
    rc = subprocess.call([sys.executable, str(DAEMON_PY)])
    if rc != 0:
        print(f"\nDaemon failed (exit {rc}).")
        return
    out_path = REPO_ROOT / 'regime_today.json'
    if out_path.exists():
        print(f"\n=== regime_today.json ===")
        print(out_path.read_text())


if __name__ == '__main__':
    main()
