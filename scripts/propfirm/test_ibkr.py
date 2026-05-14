"""
test_ibkr.py — verify IBKR TWS can supply NQ daily bars matching the parquet

Prereqs:
  * IBKR account (paper is fine), TWS or IB Gateway running
  * API enabled in TWS, port 7497 (paper) or 7496 (live) by default
  * 127.0.0.1 in Trusted IPs
  * "Read-Only API" ticked (we never write)
  * ib_insync installed

What this script does:
  1. Connect to TWS at 127.0.0.1:PORT (default paper port 7497)
  2. Set market data type to delayed-frozen (3) — works WITHOUT a paid CME sub
  3. Pull ~30 days of NQ continuous-front-month daily bars
  4. Compare to data/NQ_1day_full_data.parquet's overlap rows
  5. Auto-detect whether IBKR uses session-end dates (like Tradovate) and shift
  6. Print PASS / WARN / FAIL verdict

Run:
  python scripts/propfirm/test_ibkr.py
  python scripts/propfirm/test_ibkr.py --port 7496        # live account port
  python scripts/propfirm/test_ibkr.py --port 4002        # IB Gateway paper
  python scripts/propfirm/test_ibkr.py --port 4001        # IB Gateway live
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
PARQUET   = REPO_ROOT / 'data' / 'NQ_1day_full_data.parquet'

TOL_PASS_REL = 0.0005   # 0.05%
TOL_WARN_REL = 0.005    # 0.5%


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=7497, help='7497=TWS paper, 7496=TWS live, 4002=GW paper, 4001=GW live')
    ap.add_argument('--client-id', type=int, default=42)
    ap.add_argument('--days', type=str, default='120 D',
                    help='Lookback window passed to reqHistoricalData. '
                         'Must be long enough to reach the parquet\'s last date for overlap.')
    args = ap.parse_args()

    try:
        from ib_insync import IB, ContFuture, util
    except ImportError:
        print("ib_insync not installed. Run: pip install --user ib_insync", file=sys.stderr)
        sys.exit(1)

    ib = IB()
    print(f"Connecting to TWS at {args.host}:{args.port} (client_id={args.client_id})...", flush=True)
    try:
        ib.connect(args.host, args.port, clientId=args.client_id, timeout=15)
    except Exception as e:
        print(f"\nFAILED to connect: {e}", file=sys.stderr)
        print(f"\nChecklist:", file=sys.stderr)
        print(f"  - Is TWS or IB Gateway actually running?", file=sys.stderr)
        print(f"  - Is the port correct? (TWS paper=7497, TWS live=7496, GW paper=4002, GW live=4001)", file=sys.stderr)
        print(f"  - Is 'Enable ActiveX and Socket Clients' ticked in API settings?", file=sys.stderr)
        print(f"  - Is 127.0.0.1 in 'Trusted IPs'?", file=sys.stderr)
        sys.exit(2)
    print(f"  Connected. Server version: {ib.client.serverVersion()}", flush=True)

    # Use delayed-frozen data — works on paper accounts WITHOUT a paid CME sub.
    #   1 = live (needs subscription)
    #   2 = frozen (last close, no live)
    #   3 = delayed (~15 min)
    #   4 = delayed-frozen
    ib.reqMarketDataType(3)
    print("  Market data type: delayed", flush=True)

    # NQ continuous front-month on CME
    contract = ContFuture('NQ', 'CME')
    print(f"\nResolving contract {contract} ...", flush=True)
    qualified = ib.qualifyContracts(contract)
    if not qualified:
        print(f"FAILED: could not qualify ContFuture('NQ','CME').", file=sys.stderr)
        print(f"Possible causes: incorrect exchange, account lacks instrument data permissions.", file=sys.stderr)
        ib.disconnect()
        sys.exit(3)
    q = qualified[0]
    print(f"  Qualified: conId={q.conId}, symbol={q.symbol}, "
          f"lastTradeDateOrContractMonth={q.lastTradeDateOrContractMonth}, "
          f"localSymbol={q.localSymbol}", flush=True)

    print(f"\nRequesting {args.days} of daily bars (RTH=False, TRADES) ...", flush=True)
    try:
        bars = ib.reqHistoricalData(
            q,
            endDateTime='',
            durationStr=args.days,
            barSizeSetting='1 day',
            whatToShow='TRADES',
            useRTH=False,
            formatDate=1,
            keepUpToDate=False,
        )
    except Exception as e:
        print(f"FAILED to fetch historical data: {e}", file=sys.stderr)
        ib.disconnect()
        sys.exit(4)

    if not bars:
        print("FAILED: no bars returned.", file=sys.stderr)
        ib.disconnect()
        sys.exit(5)

    df = util.df(bars)
    df['date'] = pd.to_datetime(df['date']).dt.date
    print(f"\nGot {len(df)} bars  ({df['date'].min()} -> {df['date'].max()})\n")
    print("Last 10 bars from IBKR:")
    print(df.tail(10)[['date', 'open', 'high', 'low', 'close', 'volume']].to_string(index=False))

    ib.disconnect()

    # ── Compare to parquet ────────────────────────────────────────────────
    print("\n--- Loading parquet for overlap comparison ---", flush=True)
    pq = pd.read_parquet(PARQUET)
    pq['Date'] = pd.to_datetime(pq['Date']).dt.date
    pq = pq.sort_values('Date').reset_index(drop=True)
    print(f"  Parquet: {len(pq)} rows, {pq['Date'].min()} to {pq['Date'].max()}")

    # Try both date conventions and see which one matches
    for shift, label in [(0, "no shift"), (-1, "session-end -> session-start (-1 day)")]:
        df_test = df.copy()
        df_test['date'] = pd.to_datetime(df_test['date']) + pd.Timedelta(days=shift)
        df_test['date'] = df_test['date'].dt.date
        merged = pq.merge(df_test, left_on='Date', right_on='date', how='inner', suffixes=('_pq', '_ib'))
        if len(merged) == 0:
            print(f"\n  Shift={shift:+d} ({label}): no overlapping dates")
            continue
        abs_diff = (merged['Last'] - merged['close']).abs()
        rel_diff = abs_diff / merged['Last']
        max_rel = float(rel_diff.max())
        med_rel = float(rel_diff.median())
        max_abs = float(abs_diff.max())

        if   max_rel < TOL_PASS_REL: verdict = 'PASS'
        elif max_rel < TOL_WARN_REL: verdict = 'WARN'
        else:                         verdict = 'FAIL'

        print(f"\n  Shift={shift:+d} ({label}): {len(merged)} overlapping dates")
        print(f"    max |abs| diff:  ${max_abs:.4f}")
        print(f"    max rel diff:    {max_rel:.4%}")
        print(f"    median rel diff: {med_rel:.4%}")
        print(f"    Verdict: {verdict}")
        if verdict in ('WARN', 'FAIL') or shift == -1:
            worst = merged.assign(abs_diff=abs_diff, rel_diff=rel_diff).sort_values(
                'rel_diff', ascending=False
            ).head(5)
            print(f"    Worst 5 (date_in_parquet  IBKR_close  parquet_close  abs  rel):")
            for _, r in worst.iterrows():
                print(f"      {r['Date']}  IB=${r['close']:>10.2f}  "
                      f"PQ=${r['Last']:>10.2f}  "
                      f"|abs|=${r['abs_diff']:>6.2f}  rel={r['rel_diff']:.4%}")

    print("\nDONE")


if __name__ == '__main__':
    main()
