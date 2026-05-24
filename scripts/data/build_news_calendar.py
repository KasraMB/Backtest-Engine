"""Build a clean, look-ahead-safe US news calendar from the HuggingFace FF dump.

Input:  data/forex_factory_cache.csv (HF Ehsanrs2 scrape; Tehran-time, jittered).
Output: data/news_calendar_us.csv with columns:
        etime (ET), event, impact, forecast_str, actual_str, previous_str,
        forecast, actual, previous, surprise, z, has_forecast.

Per the design-validation agent:
- Drop rows whose RAW Tehran timestamp is within +-2h of midnight (the ambiguous
  ones that can be +-1 day mis-dated under TZ conversion).
- Override event TIMES with canonical public release times (BLS/BEA 8:30,
  PMI 9:45, ISM/sentiment 10:00, FOMC 14:00); take only DATE + values from data.
- Parse forecast/actual with K/M/B/%/commas/signs.
- Per-event-type rolling 24-month std of (actual-forecast), CAUSAL (using only
  rows strictly BEFORE the event itself); z = surprise / std.
- has_forecast = forecast is not nan (missing-forecast events get z=nan and the
  strategy must treat them as "no re-anchor", NOT as "unexpected").
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RAW  = REPO / 'data' / 'forex_factory_cache.csv'
OUT  = REPO / 'data' / 'news_calendar_us.csv'

# Canonical ET release times by event-name substring (lowercase match).
CANON_TIME = [
    # 2pm FOMC family
    (['fomc statement', 'federal funds rate', 'fomc economic projections',
      'fomc minutes', 'fomc press conference'], (14, 0)),
    # 9:45 PMI flash
    (['flash manufacturing pmi', 'flash services pmi', 'flash pmi',
      'chicago pmi'], (9, 45)),
    # 10:00 ISM / sentiment / housing
    (['ism manufacturing', 'ism services', 'ism non-manufacturing',
      'consumer confidence', 'consumer sentiment', 'jolts', 'factory orders',
      'home sales', 'business inventories', 'wholesale'], (10, 0)),
    # 8:30 BLS / BEA / etc.
    (['non-farm', 'nonfarm', 'cpi', 'ppi', 'retail sales', 'gdp', 'pce',
      'durable', 'unemployment claims', 'jobless', 'average hourly',
      'unemployment rate', 'empire state', 'philly fed', 'philadelphia fed',
      'trade balance', 'personal income', 'personal spending',
      'employment cost'], (8, 30)),
]


def canon_time(event: str):
    e = event.lower()
    for keys, t in CANON_TIME:
        if any(k in e for k in keys):
            return t
    return None


_VAL_RE = re.compile(r'^\s*([+-]?)(\d[\d,]*\.?\d*)\s*([%KMBTkmbt]?)\s*$')


def parse_val(x):
    """Parse FF-style values: '312K', '-0.2%', '2.50%', '5.25', '167K', '' -> float."""
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if not s or s in ('-', '--'):
        return np.nan
    m = _VAL_RE.match(s)
    if not m:
        return np.nan
    sign, num, suf = m.groups()
    try:
        v = float(num.replace(',', ''))
    except ValueError:
        return np.nan
    if sign == '-':
        v = -v
    mult = {'k': 1e3, 'K': 1e3, 'm': 1e6, 'M': 1e6,
            'b': 1e9, 'B': 1e9, 't': 1e12, 'T': 1e12}.get(suf, 1.0)
    return v * mult


def main():
    print(f"Loading {RAW}...", flush=True)
    df = pd.read_csv(RAW)
    print(f"  raw rows: {len(df):,}", flush=True)

    # Filter near-Tehran-midnight rows by extracting hour directly from the
    # raw string (avoids the mixed-tz parse problem). Format: "YYYY-MM-DDTHH:MM:SS+HH:MM"
    s = df['DateTime'].astype(str)
    df['_raw_hour'] = pd.to_numeric(s.str.slice(11, 13), errors='coerce')
    df = df.dropna(subset=['_raw_hour'])
    df['_raw_hour'] = df['_raw_hour'].astype(int)

    # Drop Tehran 00:00-03:00 rows: these are the "tentative/all-day" placeholders
    # AND any event whose Tehran-local hour falls in the pre-dawn window where
    # the TZ conversion to ET shifts the DATE backward by one calendar day
    # (Tehran 00:00-03:00 maps to ET 15:30-19:30 of the PREVIOUS day).
    # Real US releases (8:30/10:00 ET = Tehran ~17:00/18:30; 2pm ET = Tehran ~22:30)
    # are NOT in this window, so we keep them.
    near_midnight = df['_raw_hour'] <= 3
    n_dropped = int(near_midnight.sum())
    print(f"  dropping {n_dropped:,} near-Tehran-midnight rows (date ambiguous)", flush=True)
    df = df[~near_midnight].copy()

    # Convert remaining to ET using their explicit (Tehran) offset.
    dt = pd.to_datetime(df['DateTime'], utc=True, errors='coerce')
    df['et'] = dt.dt.tz_convert('America/New_York')
    df = df.dropna(subset=['et'])

    # USD + high-impact + 2019-2024 window
    df = df[df['Currency'] == 'USD']
    df = df[df['Impact'].str.contains('High', na=False)]
    df = df[(df['et'] >= pd.Timestamp('2019-01-01', tz='America/New_York')) &
            (df['et'] <  pd.Timestamp('2025-01-01', tz='America/New_York'))]
    print(f"  USD high-impact 2019-2024 (after midnight filter): {len(df):,}", flush=True)

    # Assign canonical ET release times
    df['ct'] = df['Event'].astype(str).map(canon_time)
    df = df[df['ct'].notna()].copy()
    df['date'] = df['et'].dt.date
    df['_h'] = df['ct'].map(lambda x: x[0])
    df['_m'] = df['ct'].map(lambda x: x[1])
    df['etime'] = df.apply(
        lambda r: pd.Timestamp(f"{r.date} {int(r._h):02d}:{int(r._m):02d}",
                               tz='America/New_York'), axis=1)
    df = df.sort_values('etime').reset_index(drop=True)

    # Parse values
    df['forecast'] = df['Forecast'].map(parse_val)
    df['actual']   = df['Actual'].map(parse_val)
    df['previous'] = df['Previous'].map(parse_val)
    df['surprise'] = df['actual'] - df['forecast']
    df['has_forecast'] = df['forecast'].notna() & df['actual'].notna()

    # Per-event-type CAUSAL z-score: at row i, scale = std of past surprises
    # for the same event (shift(1) excludes current). Positional 24-event window
    # (~2 years for monthly events; ~6 months for weekly) for simplicity/robustness.
    df['z'] = np.nan
    for ev_name, sub in df[df['has_forecast']].groupby('Event'):
        sub_sorted = sub.sort_values('etime')
        scale = sub_sorted['surprise'].shift(1).rolling(24, min_periods=8).std()
        # Replace zero-std with NaN to avoid inf; z is NaN there (treated as
        # "no info" by the strategy, i.e. no re-anchor).
        scale = scale.where(scale > 0)
        z = sub_sorted['surprise'] / scale
        df.loc[sub_sorted.index, 'z'] = z.values

    out_cols = ['etime', 'Event', 'Impact', 'forecast', 'actual', 'previous',
                'surprise', 'z', 'has_forecast']
    out = df[out_cols].rename(columns={'Event': 'event', 'Impact': 'impact'})
    out.to_csv(OUT, index=False)
    print(f"\nSaved {len(out):,} events -> {OUT}")
    print('time distribution:'); print(out['etime'].dt.strftime('%H:%M').value_counts())
    print()
    print('z-score stats (events with forecast):')
    print(out[out['has_forecast']]['z'].describe())
    print()
    # Spot-check known events
    for kw, label in [('Non-Farm Employment Change', 'NFP'),
                      ('CPI m/m', 'CPI m/m'),
                      ('Federal Funds Rate', 'FOMC'),
                      ('ISM Manufacturing PMI', 'ISM Mfg')]:
        sub = out[out['event'].str.contains(kw, na=False) & out['has_forecast']]
        n = len(sub)
        n_big = int((sub['z'].abs() > 1).sum())
        if n:
            sub_with_z = sub[sub['z'].notna()]
            r0 = sub_with_z.iloc[0] if len(sub_with_z) else sub.iloc[0]
            zval = r0['z']
            zstr = f"{zval:.2f}" if pd.notna(zval) else "nan"
            print(f"  {label}: n={n:3d} ({n/6:.1f}/yr), |z|>1: {n_big} ({n_big/n*100:.0f}%) "
                  f"e.g. {r0['etime']} actual={r0['actual']} fcst={r0['forecast']} z={zstr}")
        else:
            print(f"  {label}: n=0")


if __name__ == '__main__':
    main()
