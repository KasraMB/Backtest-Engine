"""
Phase 1 Tests — Data Layer

Run with:
    cd nq_backtest
    python -m pytest tests/test_data.py -v

Or run the manual verification script at the bottom directly:
    python tests/test_data.py
"""

import pytest
import pandas as pd
import numpy as np
from datetime import date
from pathlib import Path
import sys

# Allow running from the nq_backtest root
sys.path.insert(0, str(Path(__file__).parent.parent))

from backtest.data.loader import DataLoader
from backtest.data.cleaner import DataCleaner
from backtest.data.market_data import MarketData


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_sample_df(n_bars: int = 100, freq: str = "1min", start: str = "2024-01-02 09:30") -> pd.DataFrame:
    """Build a minimal synthetic OHLCV DataFrame for testing."""
    idx = pd.date_range(start=start, periods=n_bars, freq=freq, tz="America/New_York")
    np.random.seed(42)
    closes = 17000 + np.cumsum(np.random.randn(n_bars) * 2)
    opens = closes + np.random.randn(n_bars) * 0.5
    highs = np.maximum(opens, closes) + np.abs(np.random.randn(n_bars))
    lows = np.minimum(opens, closes) - np.abs(np.random.randn(n_bars))
    volumes = np.random.randint(10, 500, n_bars).astype(float)
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes},
        index=idx,
    )


# ---------------------------------------------------------------------------
# DataCleaner tests
# ---------------------------------------------------------------------------

class TestDataCleaner:

    def test_removes_duplicate_timestamps(self):
        df = make_sample_df(10)
        df = pd.concat([df, df.iloc[[3]]])  # inject duplicate
        cleaner = DataCleaner()
        cleaned, report = cleaner.clean(df, timeframe_minutes=1)
        assert report.duplicate_timestamps_removed == 1
        assert cleaned.index.is_unique

    def test_flags_zero_volume_bars(self):
        df = make_sample_df(20)
        df.iloc[5, df.columns.get_loc("volume")] = 0
        cleaner = DataCleaner()
        cleaned, report = cleaner.clean(df, timeframe_minutes=1)
        assert cleaned["anomalous"].iloc[5] == True
        assert report.anomalous_bars_flagged >= 1

    def test_flags_invalid_ohlc(self):
        df = make_sample_df(10)
        # Make high < close (invalid)
        df.iloc[2, df.columns.get_loc("high")] = df.iloc[2]["close"] - 5
        cleaner = DataCleaner()
        cleaned, report = cleaner.clean(df, timeframe_minutes=1)
        assert cleaned["anomalous"].iloc[2] == True

    def test_anomalous_column_added(self):
        df = make_sample_df(10)
        cleaner = DataCleaner()
        cleaned, _ = cleaner.clean(df, timeframe_minutes=1)
        assert "anomalous" in cleaned.columns

    def test_clean_data_has_no_anomalies(self):
        df = make_sample_df(50)
        cleaner = DataCleaner()
        cleaned, report = cleaner.clean(df, timeframe_minutes=1)
        assert report.anomalous_bars_flagged == 0
        assert report.duplicate_timestamps_removed == 0


# ---------------------------------------------------------------------------
# Bar map tests
# ---------------------------------------------------------------------------

class TestBarMap:

    def _make_loader(self):
        return DataLoader(verbose=False)

    def test_bar_map_length_matches_1m(self):
        loader = self._make_loader()
        df_1m = make_sample_df(100, freq="1min", start="2024-01-02 09:30")
        df_5m = make_sample_df(20, freq="5min", start="2024-01-02 09:30")
        bar_map = loader._build_bar_map(df_1m, df_5m)
        assert len(bar_map) == len(df_1m)

    def test_bar_map_starts_at_minus_one(self):
        """No completed 5m bar should be visible for the first few 1m bars."""
        loader = self._make_loader()
        df_1m = make_sample_df(100, freq="1min", start="2024-01-02 09:30")
        df_5m = make_sample_df(20, freq="5min", start="2024-01-02 09:30")
        bar_map = loader._build_bar_map(df_1m, df_5m)
        # First 5 bars of 1m are inside the first 5m bar — no completed 5m bar yet
        assert bar_map[0] == -1
        assert bar_map[4] == -1

    def test_bar_map_advances_correctly(self):
        """After 5+ minutes, the first 5m bar should be visible."""
        loader = self._make_loader()
        df_1m = make_sample_df(20, freq="1min", start="2024-01-02 09:30")
        df_5m = make_sample_df(4, freq="5min", start="2024-01-02 09:30")
        bar_map = loader._build_bar_map(df_1m, df_5m)
        # At bar index 5 (09:35), the first 5m bar (09:30–09:35) is complete
        assert bar_map[5] == 0

    def test_bar_map_never_uses_future_5m(self):
        """bar_map[i] must always point to a 5m bar that started before 1m bar i."""
        loader = self._make_loader()
        df_1m = make_sample_df(60, freq="1min", start="2024-01-02 09:30")
        df_5m = make_sample_df(12, freq="5min", start="2024-01-02 09:30")
        bar_map = loader._build_bar_map(df_1m, df_5m)

        for i, ts_1m in enumerate(df_1m.index):
            j = bar_map[i]
            if j == -1:
                continue
            ts_5m_start = df_5m.index[j]
            ts_5m_end = ts_5m_start + pd.Timedelta(minutes=5)
            assert ts_5m_end <= ts_1m, (
                f"bar_map[{i}] = {j} points to 5m bar ending at {ts_5m_end} "
                f"but 1m bar is at {ts_1m} — lookahead!"
            )


# ---------------------------------------------------------------------------
# MarketData tests
# ---------------------------------------------------------------------------

class TestMarketData:

    def _make_market_data(self):
        loader = DataLoader(verbose=False)
        df_1m = make_sample_df(100, freq="1min", start="2024-01-02 09:30")
        df_5m = make_sample_df(20, freq="5min", start="2024-01-02 09:30")
        # Add anomalous column (cleaner adds it)
        cleaner = DataCleaner()
        df_1m, _ = cleaner.clean(df_1m, timeframe_minutes=1)
        df_5m, _ = cleaner.clean(df_5m, timeframe_minutes=5)
        bar_map = loader._build_bar_map(df_1m, df_5m)
        arrays_1m = loader._extract_arrays(df_1m)
        arrays_5m = loader._extract_arrays(df_5m)
        return MarketData(
            df_1m=df_1m,
            df_5m=df_5m,
            open_1m=arrays_1m["open"],
            high_1m=arrays_1m["high"],
            low_1m=arrays_1m["low"],
            close_1m=arrays_1m["close"],
            volume_1m=arrays_1m["volume"],
            open_5m=arrays_5m["open"],
            high_5m=arrays_5m["high"],
            low_5m=arrays_5m["low"],
            close_5m=arrays_5m["close"],
            volume_5m=arrays_5m["volume"],
            bar_map=bar_map,
            trading_dates=[date(2024, 1, 2)],
        )

    def test_market_data_creation(self):
        md = self._make_market_data()
        assert md.n_bars_1m == 100
        assert md.n_bars_5m == 20

    def test_numpy_arrays_match_dataframe(self):
        md = self._make_market_data()
        np.testing.assert_array_equal(md.close_1m, md.df_1m["close"].to_numpy())
        np.testing.assert_array_equal(md.high_5m, md.df_5m["high"].to_numpy())

    def test_bar_map_length(self):
        md = self._make_market_data()
        assert len(md.bar_map) == md.n_bars_1m


# ---------------------------------------------------------------------------
# Manual verification — run directly against your real data
# ---------------------------------------------------------------------------

def manual_verify(path_1m: str, path_5m: str, data_dir: str = "data/"):
    """
    Run this manually to verify your real data loads correctly.

    Usage:
        python tests/test_data.py
    """
    print("=" * 60)
    print("MANUAL DATA VERIFICATION")
    print("=" * 60)

    loader = DataLoader(data_dir=data_dir, verbose=True)
    md = loader.load(path_1m=path_1m, path_5m=path_5m)

    print("\n--- 1m DataFrame head ---")
    print(md.df_1m.head(10).to_string())

    print("\n--- 1m DataFrame tail ---")
    print(md.df_1m.tail(5).to_string())

    print("\n--- 5m DataFrame head ---")
    print(md.df_5m.head(5).to_string())

    print("\n--- Bar map first 20 values ---")
    print(md.bar_map[:20])

    print("\n--- Bar map verification (first 10 1m bars) ---")
    for i in range(min(15, md.n_bars_1m)):
        j = md.bar_map[i]
        ts_1m = md.df_1m.index[i]
        if j == -1:
            print(f"  1m[{i:3d}] {ts_1m}  ->  no completed 5m bar yet")
        else:
            ts_5m = md.df_5m.index[j]
            print(f"  1m[{i:3d}] {ts_1m}  ->  5m[{j:3d}] {ts_5m}")

    print("\n--- Numpy array spot check ---")
    print(f"  close_1m[0]  = {md.close_1m[0]}")
    print(f"  df_1m close[0] = {md.df_1m['close'].iloc[0]}")
    match = np.isclose(md.close_1m[0], md.df_1m["close"].iloc[0])
    print(f"  Match: {match}")

    print("\n--- Anomalous bars ---")
    anomalous_1m = md.df_1m[md.df_1m["anomalous"]]
    print(f"  1m anomalous bars: {len(anomalous_1m)}")
    if len(anomalous_1m) > 0:
        print(anomalous_1m.head(5).to_string())

    print("\nManual verification complete")


if __name__ == "__main__":
    # Edit these filenames to match yours
    manual_verify(path_1m="NQ_1m.txt", path_5m="NQ_5m.txt", data_dir="data/")