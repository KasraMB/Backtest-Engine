"""
AMR x IVB consolidation classifier — does filtering by ORB-range help AMR?

cfg_406 fires at session 2000 ET (evening); ORB at 9:31-10:00 AM ET precedes
entry by ~10 hours → forward-safe.

Vectorised classification: per OOS day, fetch 1m bars 9:31-10:00 ET, compute
ORB range, lookup ATR_20_daily. Classify each AMR trade's day as consolidation
or trending.

Then bucket-stat AMR R/day, plus continuous regression of mean_R vs
ORB_range/ATR_20 ratio. If meaningful effect → propose filtered config.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from backtest.indicators.atr import atr_daily


def main():
    amr = pickle.load(open(REPO / 'cache' / 'amr_per_day_trades.pkl', 'rb'))
    df1m = pd.read_parquet(REPO / 'data' / 'NQ_1m.parquet')

    # Build per-day ORB-range and ATR_20 lookups vectorised
    print("Computing ORB range per day (9:31-10:00 ET)...", flush=True)
    idx = pd.DatetimeIndex(df1m.index)
    bar_min = idx.hour.to_numpy(np.int32) * 60 + idx.minute.to_numpy(np.int32)
    orb_mask = (bar_min >= (9*60+31)) & (bar_min <= 10*60)
    day_naive = idx.tz_localize(None).normalize() if idx.tz is not None else idx.normalize()
    day_ord = day_naive.asi8
    unique_days, inv = np.unique(day_ord, return_inverse=True)
    n_days = len(unique_days)

    high = df1m['high'].to_numpy(np.float64)
    low  = df1m['low'].to_numpy(np.float64)
    orb_high = np.full(n_days, -np.inf)
    orb_low  = np.full(n_days,  np.inf)
    orb_bar_count = np.zeros(n_days, dtype=np.int64)
    np.maximum.at(orb_high, inv[orb_mask], high[orb_mask])
    np.minimum.at(orb_low,  inv[orb_mask], low[orb_mask])
    np.add.at(orb_bar_count, inv[orb_mask], 1)
    orb_range = orb_high - orb_low
    valid_orb = (orb_bar_count >= 25) & np.isfinite(orb_range) & (orb_range > 0)
    print(f"  {valid_orb.sum()} days with valid ORB out of {n_days}", flush=True)

    # ATR_20_daily
    print("Computing ATR_20_daily...", flush=True)
    atr_s = atr_daily(df1m, window=20)
    atr_d_naive = pd.DatetimeIndex(atr_s.index).tz_localize(None).normalize() \
                  if atr_s.index.tz is not None else pd.DatetimeIndex(atr_s.index).normalize()
    atr_lookup = pd.Series(atr_s.to_numpy(), index=atr_d_naive)
    atr_per_day = atr_lookup.reindex(pd.DatetimeIndex(unique_days)).to_numpy(np.float64)

    # Day-level table: date -> (orb_range, atr_20, ratio)
    day_table = pd.DataFrame({
        'date': pd.to_datetime(unique_days),
        'orb_range': orb_range,
        'atr_20':    atr_per_day,
        'valid':     valid_orb & np.isfinite(atr_per_day),
    })
    day_table['ratio'] = day_table['orb_range'] / day_table['atr_20']

    # Join cfg_406 OOS days
    cfg = 'cfg_406'
    days_df = amr[cfg]['days'].copy()
    days_df['date'] = pd.to_datetime(days_df['date'])
    oos = days_df[days_df['date'] >= '2023-01-01']
    merged = oos.merge(day_table, on='date', how='inner')
    valid = merged[merged['valid']]
    print(f"\nMerged: {len(valid)} OOS days with valid ORB/ATR (out of {len(oos)} cfg days)")

    # Bucket analysis at threshold 0.5 (v11 default)
    thr = 0.5
    cons = valid[valid['ratio'] < thr]
    trend = valid[valid['ratio'] >= thr]
    print(f"\n=== Bucket analysis (threshold ratio={thr}) ===")
    print(f"  Consolidation days (ratio<{thr}): n={len(cons)}, mean_R={cons['R'].mean():+.4f}, "
          f"win%={(cons['R']>0).mean()*100:.1f}, n_trades={cons['n_trades'].sum()}")
    print(f"  Trending days (ratio>={thr}): n={len(trend)}, mean_R={trend['R'].mean():+.4f}, "
          f"win%={(trend['R']>0).mean()*100:.1f}, n_trades={trend['n_trades'].sum()}")
    diff = trend['R'].mean() - cons['R'].mean()
    pooled_se = np.sqrt(cons['R'].var() / len(cons) + trend['R'].var() / len(trend))
    t_stat = diff / pooled_se if pooled_se > 0 else np.nan
    print(f"  Trend - Cons = {diff:+.4f}, t-stat = {t_stat:+.2f}")

    # Try multiple thresholds
    print(f"\n=== Threshold sweep ===")
    for thr in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2, 1.5]:
        below = valid[valid['ratio'] < thr]
        above = valid[valid['ratio'] >= thr]
        if len(below) < 30 or len(above) < 30:
            continue
        mean_below = below['R'].mean()
        mean_above = above['R'].mean()
        print(f"  thr={thr:.2f}: below(n={len(below):>3}) mean_R={mean_below:+.4f} "
              f"| above(n={len(above):>3}) mean_R={mean_above:+.4f} | "
              f"delta={mean_above-mean_below:+.4f}")

    # Continuous regression
    from scipy import stats
    valid_R = valid[~valid['ratio'].isna()]
    slope, intercept, r_val, p_val, se = stats.linregress(valid_R['ratio'].values,
                                                            valid_R['R'].values)
    print(f"\n=== Continuous regression (mean_R/day ~ ratio) ===")
    print(f"  slope={slope:+.4f} ± {se:.4f}, intercept={intercept:+.4f}, r={r_val:+.3f}, p={p_val:.4f}")


if __name__ == '__main__':
    main()
