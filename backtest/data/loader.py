import pandas as pd
import numpy as np
from pathlib import Path

from backtest.data.market_data import MarketData
from backtest.data.cleaner import DataCleaner


# NQ point value: 1 point = $20
NQ_POINT_VALUE = 20.0

# Column rename map — handles the leading spaces from the raw CSV headers
RAW_COLUMN_MAP = {
    "Date": "date_raw",
    " Date": "date_raw",
    "Time": "time_raw",
    " Time": "time_raw",
    "Open": "open",
    " Open": "open",
    "High": "high",
    " High": "high",
    "Low": "low",
    " Low": "low",
    "Last": "close",        # NinjaTrader uses 'Last' for close
    " Last": "close",
    "Close": "close",
    " Close": "close",
    "Volume": "volume",
    " Volume": "volume",
    "NumberOfTrades": "num_trades",
    " NumberOfTrades": "num_trades",
    "BidVolume": "bid_volume",
    " BidVolume": "bid_volume",
    "AskVolume": "ask_volume",
    " AskVolume": "ask_volume",
}

# Columns required for the engine to function
REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume"]


class DataLoader:
    """
    Loads NQ 1m and 5m data from NinjaTrader-exported text files.

    Usage:
        loader = DataLoader(data_dir="data/")
        market_data = loader.load(
            path_1m="NQ_1m.txt",
            path_5m="NQ_5m.txt",
            start_date="2024-01-01",   # optional
            end_date="2024-12-31",     # optional
        )
    """

    def __init__(self, data_dir: str = "data/", verbose: bool = True):
        self.data_dir = Path(data_dir)
        self.verbose = verbose
        self.cleaner = DataCleaner()

    def load(
        self,
        path_1m: str,
        path_5m: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> MarketData:
        """
        Load, clean, and package 1m and 5m data into a MarketData object.

        Args:
            path_1m: Filename of the 1m data file (relative to data_dir)
            path_5m: Filename of the 5m data file (relative to data_dir)
            start_date: Optional filter start date, format 'YYYY-MM-DD'
            end_date: Optional filter end date, format 'YYYY-MM-DD'

        Returns:
            MarketData ready for use by the engine
        """
        if self.verbose:
            print(f"Loading 1m data from: {self.data_dir / path_1m}")
        df_1m = self._load_file(self.data_dir / path_1m)

        if self.verbose:
            print(f"Loading 5m data from: {self.data_dir / path_5m}")
        df_5m = self._load_file(self.data_dir / path_5m)

        # Apply date range filter before cleaning
        if start_date or end_date:
            df_1m = self._filter_dates(df_1m, start_date, end_date)
            df_5m = self._filter_dates(df_5m, start_date, end_date)

        # Clean both timeframes
        if self.verbose:
            print("\nCleaning 1m data...")
        df_1m, report_1m = self.cleaner.clean(df_1m, timeframe_minutes=1)
        if self.verbose:
            report_1m.print()

        if self.verbose:
            print("Cleaning 5m data...")
        df_5m, report_5m = self.cleaner.clean(df_5m, timeframe_minutes=5)
        if self.verbose:
            report_5m.print()

        # Build bar map
        bar_map = self._build_bar_map(df_1m, df_5m)

        # Pre-extract numpy arrays
        arrays_1m = self._extract_arrays(df_1m)
        arrays_5m = self._extract_arrays(df_5m)

        # Get unique RTH trading dates (09:30–17:00 ET only) so overnight
        # globex sessions don't inflate the trading day count.
        from datetime import time as _time
        rth_mask      = (df_1m.index.time >= _time(9, 30)) & (df_1m.index.time <= _time(17, 0))
        trading_dates = sorted(set(df_1m[rth_mask].index.date))

        if self.verbose:
            print(f"\nLoaded {len(df_1m):,} 1m bars and {len(df_5m):,} 5m bars")
            print(f"Date range: {df_1m.index[0]} -> {df_1m.index[-1]}")
            print(f"Trading dates: {len(trading_dates)}")

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
            trading_dates=trading_dates,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_file(self, path: Path) -> pd.DataFrame:
        """Parse a NinjaTrader CSV export into a clean DataFrame with DatetimeIndex."""
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")

        # Read raw CSV
        df = pd.read_csv(path, header=0)

        # Strip whitespace from column names
        df.columns = [c.strip() for c in df.columns]

        # Rename columns to standard names
        rename = {}
        for col in df.columns:
            if col in RAW_COLUMN_MAP:
                rename[col] = RAW_COLUMN_MAP[col]
        df = df.rename(columns=rename)

        # Validate required columns exist after rename
        missing = [c for c in ["date_raw", "time_raw"] + REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns after rename: {missing}. Available: {list(df.columns)}")

        # Combine date and time into a single datetime column
        # Strip whitespace from values too
        df["date_raw"] = df["date_raw"].astype(str).str.strip()
        df["time_raw"] = df["time_raw"].astype(str).str.strip()
        df["datetime"] = pd.to_datetime(
            df["date_raw"] + " " + df["time_raw"],
            format="%Y/%m/%d %H:%M:%S"
        )

        # Set datetime as index, localize to New York time
        df = df.set_index("datetime")
        df.index = df.index.tz_localize("America/New_York", ambiguous="infer", nonexistent="shift_forward")

        # Drop raw date/time columns
        df = df.drop(columns=["date_raw", "time_raw"], errors="ignore")

        # Keep only relevant columns, in order
        keep = REQUIRED_COLUMNS + [c for c in ["num_trades", "bid_volume", "ask_volume", "anomalous"] if c in df.columns]
        df = df[[c for c in keep if c in df.columns]]

        # Ensure numeric types
        for col in REQUIRED_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    def _filter_dates(
        self,
        df: pd.DataFrame,
        start_date: str | None,
        end_date: str | None,
    ) -> pd.DataFrame:
        if start_date:
            df = df[df.index >= pd.Timestamp(start_date, tz="America/New_York")]
        if end_date:
            df = df[df.index <= pd.Timestamp(end_date + " 23:59:59", tz="America/New_York")]
        return df

    def _build_bar_map(self, df_1m: pd.DataFrame, df_5m: pd.DataFrame) -> np.ndarray:
        """
        For each 1m bar at index i, find the index of the last COMPLETED 5m bar.

        A 5m bar starting at T is complete once we reach T + 5min.
        Uses vectorized numpy searchsorted — O(N log M) instead of O(N) Python loop.

        Returns an int array of length len(df_1m).
        -1 means no completed 5m bar is visible yet.
        """
        ts_1m = df_1m.index.asi8  # int64 ticks since epoch (resolution varies by pandas version)
        ts_5m = df_5m.index.asi8

        # Auto-detect tick resolution by comparing to known nanosecond value.
        # Pandas < 2.0 uses nanoseconds; Pandas >= 2.0 uses the index's native resolution
        # (often microseconds for tz-aware DatetimeIndex).
        known_ns = int(df_5m.index[0].timestamp() * 1e9)
        ratio     = known_ns / ts_5m[0]           # 1000 = us ticks, 1 = ns ticks
        five_min_ticks = np.int64(round(5 * 60 * 1e9 / ratio))

        completion_times = ts_5m + five_min_ticks

        # count = number of 5m bars whose completion time <= current 1m timestamp
        # last completed index = count - 1  (-1 = none yet)
        counts  = np.searchsorted(completion_times, ts_1m, side="right")
        bar_map = counts.astype(np.int64) - 1

        return bar_map

    def build_market_data(self, df_1m: pd.DataFrame, df_5m: pd.DataFrame) -> MarketData:
        """
        Build a MarketData object from already-cleaned DataFrames.
        Used when loading from parquet cache — skips file parsing and cleaning.
        """
        bar_map = self._build_bar_map(df_1m, df_5m)
        arrays_1m = self._extract_arrays(df_1m)
        arrays_5m = self._extract_arrays(df_5m)
        from datetime import time as _time
        rth_mask      = (df_1m.index.time >= _time(9, 30)) & (df_1m.index.time <= _time(17, 0))
        trading_dates = sorted(set(df_1m[rth_mask].index.date))

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
            trading_dates=trading_dates,
        )

    def _extract_arrays(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        """Pre-extract OHLCV as numpy arrays for fast loop access."""
        return {
            "open": df["open"].to_numpy(dtype=np.float64),
            "high": df["high"].to_numpy(dtype=np.float64),
            "low": df["low"].to_numpy(dtype=np.float64),
            "close": df["close"].to_numpy(dtype=np.float64),
            "volume": df["volume"].to_numpy(dtype=np.float64),
        }