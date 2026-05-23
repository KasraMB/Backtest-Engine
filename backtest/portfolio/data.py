"""Generic instrument loader: parquet -> MarketData.

Supports the Sierra-Chart-export parquets in data/ (NQ_1m_full_data.parquet,
ES_1m_full_data.parquet, YM_1m_full_data.parquet, etc.) which all share the
same schema (Date, Time, Open, High, Low, Last, Volume, NumTrades, BidVol, AskVol).

Cached to disk on first build per (symbol, timeframe_pair) so subsequent
loads are instant.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.data.loader import DataLoader
from backtest.data.market_data import MarketData

REPO  = Path(__file__).resolve().parents[2]
DATA  = REPO / 'data'
CACHE = REPO / 'cache' / 'portfolio_md'
CACHE.mkdir(parents=True, exist_ok=True)

_CANON = {
    'Open': 'open', 'High': 'high', 'Low': 'low', 'Last': 'close',
    'Volume': 'volume', 'NumTrades': 'num_trades',
    'BidVol': 'bid_volume', 'AskVol': 'ask_volume',
}


def _read_full_parquet(path: Path) -> pd.DataFrame:
    """Load Sierra-export parquet, join Date+Time -> ET-aware DatetimeIndex,
    canonicalize columns to lowercase OHLCV+aux."""
    df = pd.read_parquet(path)
    # Combine Date (Timestamp at midnight) + Time (string HH:MM:SS) into tz-aware ET
    times = pd.to_timedelta(df['Time'].astype(str))
    ts = (df['Date'] + times).dt.tz_localize('America/New_York', ambiguous='infer',
                                              nonexistent='shift_forward')
    df = df.rename(columns=_CANON).drop(columns=['Date', 'Time'])
    df.index = pd.DatetimeIndex(ts, name='datetime')
    # Cast numeric to float64 / int64 for downstream numpy
    for c in ('open', 'high', 'low', 'close'):
        df[c] = df[c].astype(np.float64)
    for c in ('volume', 'num_trades', 'bid_volume', 'ask_volume'):
        df[c] = df[c].astype(np.int64)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep='first')]
    return df


def load_instrument(symbol: str, cache_ok: bool = True) -> MarketData:
    """Build (or load cached) MarketData for an instrument.

    Args:
        symbol: 'NQ', 'ES', 'YM'
        cache_ok: if True, load from cache when available

    Returns MarketData with df_1m + df_5m + arrays + bar_map.
    """
    cache_path = CACHE / f'{symbol}.pkl'
    if cache_ok and cache_path.exists():
        return pickle.load(open(cache_path, 'rb'))

    df_1m = _read_full_parquet(DATA / f'{symbol}_1m_full_data.parquet')
    df_5m = _read_full_parquet(DATA / f'{symbol}_5m_full_data.parquet')

    # Reuse DataLoader's bar_map builder (canonical implementation)
    loader  = DataLoader(data_dir=str(DATA))
    bar_map = loader._build_bar_map(df_1m, df_5m)

    a1 = {c: df_1m[c].to_numpy(np.float64) for c in ('open', 'high', 'low', 'close', 'volume')}
    a5 = {c: df_5m[c].to_numpy(np.float64) for c in ('open', 'high', 'low', 'close', 'volume')}

    from datetime import time as _time
    rth_mask  = (df_1m.index.time >= _time(9, 30)) & (df_1m.index.time <= _time(17, 0))
    trading_dates = sorted(set(df_1m[rth_mask].index.date))

    md = MarketData(
        df_1m=df_1m, df_5m=df_5m,
        open_1m=a1['open'], high_1m=a1['high'], low_1m=a1['low'],
        close_1m=a1['close'], volume_1m=a1['volume'],
        open_5m=a5['open'], high_5m=a5['high'], low_5m=a5['low'],
        close_5m=a5['close'], volume_5m=a5['volume'],
        bar_map=bar_map, trading_dates=trading_dates,
    )
    with open(cache_path, 'wb') as f:
        pickle.dump(md, f)
    return md


def _build_md_from_frames(df_1m: pd.DataFrame, df_5m: pd.DataFrame) -> MarketData:
    """Construct a MarketData from (already-sliced) 1m/5m frames. bar_map and all
    arrays are rebuilt from the SAME frames so indices stay aligned."""
    from datetime import time as _time
    loader = DataLoader(data_dir=str(DATA))
    a1 = {c: df_1m[c].to_numpy(np.float64) for c in ('open', 'high', 'low', 'close', 'volume')}
    a5 = {c: df_5m[c].to_numpy(np.float64) for c in ('open', 'high', 'low', 'close', 'volume')}
    rth = (df_1m.index.time >= _time(9, 30)) & (df_1m.index.time <= _time(17, 0))
    return MarketData(
        df_1m=df_1m, df_5m=df_5m,
        open_1m=a1['open'], high_1m=a1['high'], low_1m=a1['low'],
        close_1m=a1['close'], volume_1m=a1['volume'],
        open_5m=a5['open'], high_5m=a5['high'], low_5m=a5['low'],
        close_5m=a5['close'], volume_5m=a5['volume'],
        bar_map=loader._build_bar_map(df_1m, df_5m),
        trading_dates=sorted(set(df_1m[rth].index.date)),
    )


def load_instrument_window(
    symbol: str,
    lo: 'pd.Timestamp | None' = None,
    hi: 'pd.Timestamp | None' = None,
    cache_ok: bool = True,
) -> MarketData:
    """Load an instrument and slice it to [lo, hi] (inclusive), returning a
    MarketData whose 1m/5m frames, arrays, and bar_map are all consistent with
    the sliced window. Shared by the Phase 2 driver and the NQ regression guard
    so the slicing logic lives in exactly one place."""
    md_full = load_instrument(symbol, cache_ok=cache_ok)
    if lo is None and hi is None:
        return md_full
    m1 = np.ones(len(md_full.df_1m), dtype=bool)
    m5 = np.ones(len(md_full.df_5m), dtype=bool)
    if lo is not None:
        m1 &= md_full.df_1m.index >= lo
        m5 &= md_full.df_5m.index >= lo
    if hi is not None:
        m1 &= md_full.df_1m.index <= hi
        m5 &= md_full.df_5m.index <= hi
    return _build_md_from_frames(md_full.df_1m[m1], md_full.df_5m[m5])
