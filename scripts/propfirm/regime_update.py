"""
regime_update.py — daily HMM regime daemon

Reads the latest NQ daily bars from `data/NQ_1day_full_data.parquet`, refits
the walk-forward HMM (same params as the backtest), and writes today's regime
label to `regime_today.json`. The strategy bot reads that file each day to
pick which session configs to run.

Schedule this script to run once per weekday after the RTH close (~16:30 ET).
Before scheduling, make sure `data/NQ_1day_full_data.parquet` is being updated
with new daily bars from your own data source (Sierra Chart export, IB API,
etc.). The file must contain a 'Date' column and a 'Last' column (close).

Output:
  regime_today.json   — { date, regime, label_id, latest_close }
  cache/amr_hmm.pkl   — refreshed RegimeResult pickle (overwrites)

Run:
  python scripts/propfirm/regime_update.py
  python scripts/propfirm/regime_update.py --no-write-cache    # don't overwrite pkl
  python scripts/propfirm/regime_update.py --out path/to/regime.json
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running this script directly from repo root or any subdir
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from backtest.regime.hmm import fit_regimes   # noqa: E402

PARQUET_PATH = REPO_ROOT / 'data' / 'NQ_1day_full_data.parquet'
CACHE_PATH   = REPO_ROOT / 'cache' / 'amr_hmm.pkl'
DEFAULT_OUT  = REPO_ROOT / 'regime_today.json'

# HMM params — MUST match backtest exactly for label replication
HMM_PARAMS = dict(
    n_states=3,
    train_ratio=0.5,
    mode='rolling',
    seed=42,
    volume_window=30,         # rolling z-score window for volume feature
)

USE_VOLUME_FEATURE = True     # set False for pure-returns (legacy) HMM


def compute_regime(parquet_path: Path) -> tuple[dict, object]:
    if not parquet_path.exists():
        raise FileNotFoundError(f"NQ daily parquet not found at {parquet_path}")

    df = pd.read_parquet(parquet_path)
    if 'Date' not in df.columns or 'Last' not in df.columns:
        raise ValueError(f"Expected 'Date' and 'Last' columns in {parquet_path}, "
                         f"got {list(df.columns)}")

    df = df.sort_values('Date').reset_index(drop=True)
    close = df['Last'].astype(float)
    daily_log_ret = np.log(close / close.shift(1)).dropna()
    daily_dates   = pd.to_datetime(df['Date'].iloc[1:].values).date.tolist()
    daily_rets    = daily_log_ret.values.astype(np.float64)

    daily_vols = None
    if USE_VOLUME_FEATURE and 'Volume' in df.columns:
        daily_vols = df['Volume'].iloc[1:].values.astype(np.float64)

    result = fit_regimes(
        daily_returns=daily_rets,
        daily_dates=daily_dates,
        daily_volumes=daily_vols,
        **HMM_PARAMS,
    )

    latest_date  = daily_dates[-1]
    latest_label = result.labels[latest_date]
    latest_close = float(close.iloc[-1])

    # Parquet uses session-START date convention (Sierra Chart). Tradovate
    # (and most retail chart platforms) label the same bar with the session-END
    # date, i.e. parquet_date + 1 calendar day. Surface both so the bot/user
    # can cross-reference whichever they're looking at.
    tradovate_date = latest_date + timedelta(days=1)

    payload = {
        'parquet_date':  str(latest_date),       # session-start (Sierra Chart) date
        'tradovate_date': str(tradovate_date),   # session-end date (what Tradovate shows)
        'date':          str(latest_date),        # legacy alias = parquet_date
        'regime':        result.label_names[latest_label],
        'label_id':      int(latest_label),
        'latest_close':  latest_close,
        'state_means':   [float(x) for x in result.state_means],
        'transition_matrix': result.transition_matrix.tolist(),
        'n_labelled_days':   len(result.labels),
    }
    return payload, result


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument('--parquet', default=str(PARQUET_PATH),
                    help=f'Path to NQ daily parquet (default: {PARQUET_PATH})')
    ap.add_argument('--out', default=str(DEFAULT_OUT),
                    help=f'Output JSON path (default: {DEFAULT_OUT})')
    ap.add_argument('--no-write-cache', action='store_true',
                    help='Skip overwriting cache/amr_hmm.pkl')
    args = ap.parse_args()

    parquet_path = Path(args.parquet)
    out_path     = Path(args.out)

    print(f"Loading {parquet_path}...", flush=True)
    payload, result = compute_regime(parquet_path)

    print(f"\n=== Regime update ===")
    print(f"  Parquet date (Sierra/session-start):  {payload['parquet_date']}")
    print(f"  Tradovate date (session-end):         {payload['tradovate_date']}")
    print(f"  Regime:                               {payload['regime']} (label_id={payload['label_id']})")
    print(f"  Latest close:                         {payload['latest_close']:.2f}")
    print(f"  Total days:                           {payload['n_labelled_days']}")
    print(f"  State means:                          {payload['state_means']}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote regime payload to {out_path}")

    if not args.no_write_cache:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE_PATH, 'wb') as f:
            pickle.dump(result, f)
        print(f"Refreshed HMM cache at {CACHE_PATH}")


if __name__ == '__main__':
    main()
