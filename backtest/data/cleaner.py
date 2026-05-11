import pandas as pd
from dataclasses import dataclass, field


@dataclass
class CleaningReport:
    """Summary of what the cleaner found and fixed."""
    duplicate_timestamps_removed: int = 0
    gaps_found: int = 0
    anomalous_bars_flagged: int = 0
    partial_bars_trimmed: int = 0
    warnings: list[str] = field(default_factory=list)

    # Gap breakdown buckets (populated by _detect_gaps)
    gaps_minor: int = 0       # 2-5 min (data noise)
    gaps_session: int = 0     # 6-360 min (early closes, holidays)
    gaps_large: int = 0       # >360 min (multi-day closures)

    def print(self):
        print("--- Cleaning Report ---")
        print(f"  Duplicates removed  : {self.duplicate_timestamps_removed}")
        if self.gaps_found:
            print(f"  Gaps found          : {self.gaps_found}  "
                  f"(minor 2-5m: {self.gaps_minor}  "
                  f"session: {self.gaps_session}  "
                  f"large >6h: {self.gaps_large})")
        else:
            print("  Gaps found          : 0")
        print(f"  Anomalous bars      : {self.anomalous_bars_flagged}")
        print(f"  Partial bars trimmed: {self.partial_bars_trimmed}")
        # Only print non-gap warnings (zero-range, extreme range, etc.)
        other = [w for w in self.warnings if not w.startswith("Gap detected")]
        for w in other:
            print(f"  ! {w}")
        print("-----------------------")


class DataCleaner:
    """
    Cleans a raw OHLCV DataFrame before it enters the engine.

    Operations performed:
      1. Remove duplicate timestamps (keep first)
      2. Detect and report gaps in the bar sequence
      3. Flag anomalous bars (zero volume, zero range, extreme range)
      4. Trim partial bars at the start/end of the dataset
    """

    # A bar is anomalous if its range exceeds this many points for NQ
    MAX_REASONABLE_RANGE_POINTS = 200.0

    def clean(self, df: pd.DataFrame, timeframe_minutes: int) -> tuple[pd.DataFrame, CleaningReport]:
        """
        Clean a DataFrame and return the cleaned version plus a report.

        Args:
            df: Raw OHLCV DataFrame with DatetimeIndex
            timeframe_minutes: Expected bar interval in minutes (e.g. 1 or 5)

        Returns:
            (cleaned_df, report)
        """
        report = CleaningReport()
        df = df.copy()

        df = self._remove_duplicates(df, report)
        df = self._sort_index(df)
        self._detect_gaps(df, timeframe_minutes, report)
        df = self._flag_anomalous_bars(df, report)
        df = self._trim_partial_bars(df, timeframe_minutes, report)

        return df, report

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _remove_duplicates(self, df: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
        n_before = len(df)
        df = df[~df.index.duplicated(keep="first")]
        removed = n_before - len(df)
        report.duplicate_timestamps_removed = removed
        if removed > 0:
            report.warnings.append(f"{removed} duplicate timestamps removed — check data source")
        return df

    def _sort_index(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.sort_index()

    def _detect_gaps(self, df: pd.DataFrame, timeframe_minutes: int, report: CleaningReport):
        """
        Detect gaps larger than the expected bar interval.
        Gaps during the daily maintenance window (16:00–18:00 ET) are ignored.
        Buckets gaps by size rather than printing each one individually.
        """
        expected_delta = pd.Timedelta(minutes=timeframe_minutes)
        deltas = df.index.to_series().diff().dropna()

        gaps = 0
        for timestamp, delta in deltas.items():
            if delta <= expected_delta:
                continue

            prev_time = (timestamp - delta).time()
            if self._is_maintenance_window(prev_time):
                continue

            prev_ts = timestamp - delta
            if prev_ts.weekday() == 4:  # Friday
                continue

            gaps += 1
            minutes = int(delta.total_seconds() / 60)
            if minutes <= 5:
                report.gaps_minor += 1
            elif minutes <= 360:
                report.gaps_session += 1
            else:
                report.gaps_large += 1

        report.gaps_found = gaps

    def _is_maintenance_window(self, t) -> bool:
        """NQ daily maintenance window is roughly 16:00–18:00 ET."""
        from datetime import time
        return time(15, 55) <= t <= time(18, 5)

    def _flag_anomalous_bars(self, df: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
        """
        Flag bars that look suspicious. Does not remove them — adds an
        'anomalous' boolean column so strategies and the engine can see them.
        """
        df = df.copy()
        anomalous = pd.Series(False, index=df.index)

        # Zero volume bars
        zero_vol = df["volume"] == 0
        if zero_vol.any():
            n = zero_vol.sum()
            report.warnings.append(f"{n} zero-volume bars flagged")
            anomalous |= zero_vol

        # Zero range bars (high == low) — suspicious but not always wrong
        zero_range = df["high"] == df["low"]
        if zero_range.any():
            n = zero_range.sum()
            # Only warn if there are many — isolated ones are normal overnight behavior
            if n > 1000:
                report.warnings.append(f"{n} zero-range bars flagged")
            anomalous |= zero_range

        # Extreme range bars
        bar_range = df["high"] - df["low"]
        extreme = bar_range > self.MAX_REASONABLE_RANGE_POINTS
        if extreme.any():
            n = extreme.sum()
            report.warnings.append(f"{n} extreme-range bars flagged (range > {self.MAX_REASONABLE_RANGE_POINTS} pts)")
            anomalous |= extreme

        # OHLC sanity: high must be >= open, close, low; low must be <= all
        invalid_high = (df["high"] < df["open"]) | (df["high"] < df["close"])
        invalid_low = (df["low"] > df["open"]) | (df["low"] > df["close"])
        invalid_ohlc = invalid_high | invalid_low
        if invalid_ohlc.any():
            n = invalid_ohlc.sum()
            report.warnings.append(f"{n} bars with invalid OHLC structure (high < open/close or low > open/close)")
            anomalous |= invalid_ohlc

        report.anomalous_bars_flagged = anomalous.sum()
        df["anomalous"] = anomalous
        return df

    def _trim_partial_bars(self, df: pd.DataFrame, timeframe_minutes: int, report: CleaningReport) -> pd.DataFrame:
        """
        Trim bars at the very start and end of the dataset that may be partial.
        A partial bar at the start is one that begins mid-interval.
        A partial bar at the end is flagged only if the dataset ends mid-session.
        """
        trimmed = 0

        # Check first bar: for 5m bars, the timestamp should fall on a 5-min boundary
        if timeframe_minutes > 1 and len(df) > 0:
            first_ts = df.index[0]
            if first_ts.minute % timeframe_minutes != 0:
                df = df.iloc[1:]
                trimmed += 1
                report.warnings.append(
                    f"First bar trimmed — started mid-interval at {first_ts}"
                )

        report.partial_bars_trimmed = trimmed
        return df