"""
Validator — Phase 5
────────────────────
Runs pre-flight checks on a strategy before the main backtest loop.
Called automatically by run_backtest(). Hard failures abort the run.
Warnings are printed but do not abort.

Checks performed:
  1. Lookahead bias     — strategy reads future bars
  2. NaN propagation    — indicators return NaN on early bars
  3. Order sanity       — malformed orders from generate_signals
  4. PositionUpdate sanity — malformed updates from manage_position
  5. Session state leak — state carries over between runs
  6. Signals outside trading hours — generate_signals fires when it shouldn't
"""
from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from datetime import date, time, datetime, timedelta
from typing import Optional, Type

import numpy as np
import pandas as pd

from backtest.data.market_data import MarketData
from backtest.strategy.base import BaseStrategy
from backtest.strategy.enums import OrderType, SizeType
from backtest.strategy.order import Order
from backtest.strategy.update import OpenPosition, PositionUpdate
from backtest.runner.config import RunConfig


# ---------------------------------------------------------------------------
# ValidationReport
# ---------------------------------------------------------------------------

@dataclass
class ValidationReport:
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return len(self.failures) == 0

    def print(self, strategy_name: str) -> None:
        print(f"\nValidating {strategy_name}...")
        checks = [
            ("Lookahead bias",            self._check_result("lookahead")),
            ("NaN propagation",           self._check_result("nan")),
            ("Order sanity",              self._check_result("order")),
            ("PositionUpdate sanity",     self._check_result("update")),
            ("Session state leak",        self._check_result("state")),
            ("Signals outside hours",     self._check_result("hours")),
        ]
        for label, status in checks:
            print(f"  {'✓' if status else '✗'} {label}")

        if self.warnings:
            for w in self.warnings:
                print(f"  ⚠ {w}")

        if self.passed:
            print("  Validation passed.\n")
        else:
            print("\n  FAILURES:")
            for f in self.failures:
                print(f"    - {f}")
            print()

    def _check_result(self, key: str) -> bool:
        return not any(key in f.lower() for f in self.failures)


class ValidationError(Exception):
    pass


# ---------------------------------------------------------------------------
# Lookahead-safe MarketData proxy
# ---------------------------------------------------------------------------

class _LockedArray:
    """Numpy array proxy that raises if index > current bar."""

    def __init__(self, arr: np.ndarray, name: str):
        self._arr = arr
        self._name = name
        self._limit = 0

    def set_limit(self, i: int):
        self._limit = i

    def __getitem__(self, idx):
        # Handle slices and negative indices conservatively
        if isinstance(idx, slice):
            stop = idx.stop
            if stop is not None and stop > self._limit + 1:
                raise LookaheadError(
                    f"Lookahead: {self._name}[{idx}] accesses future data at bar {self._limit}"
                )
        elif isinstance(idx, int):
            actual = idx if idx >= 0 else len(self._arr) + idx
            if actual > self._limit:
                raise LookaheadError(
                    f"Lookahead: {self._name}[{idx}] accesses bar {actual}, current bar is {self._limit}"
                )
        return self._arr[idx]

    def __len__(self):
        return len(self._arr)


class LookaheadError(Exception):
    pass


class _ProxyMarketData:
    """MarketData wrapper that intercepts numpy array access to detect lookahead."""

    def __init__(self, data: MarketData):
        self._data = data
        self._arrays = {
            "close_1m":  _LockedArray(data.close_1m,  "close_1m"),
            "open_1m":   _LockedArray(data.open_1m,   "open_1m"),
            "high_1m":   _LockedArray(data.high_1m,   "high_1m"),
            "low_1m":    _LockedArray(data.low_1m,    "low_1m"),
            "volume_1m": _LockedArray(data.volume_1m, "volume_1m"),
            "close_5m":  _LockedArray(data.close_5m,  "close_5m"),
            "open_5m":   _LockedArray(data.open_5m,   "open_5m"),
            "high_5m":   _LockedArray(data.high_5m,   "high_5m"),
            "low_5m":    _LockedArray(data.low_5m,    "low_5m"),
        }

    def set_bar(self, i: int):
        for arr in self._arrays.values():
            arr.set_limit(i)

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        if name in self._arrays:
            return self._arrays[name]
        return getattr(self._data, name)


# ---------------------------------------------------------------------------
# Synthetic data builder
# ---------------------------------------------------------------------------

def _make_synthetic_data(n_bars: int = 100, price: float = 19000.0) -> MarketData:
    """Build a tiny synthetic MarketData for validation checks."""
    np.random.seed(0)
    closes = price + np.cumsum(np.random.randn(n_bars) * 2)
    opens  = closes + np.random.randn(n_bars) * 0.5
    highs  = np.maximum(opens, closes) + np.abs(np.random.randn(n_bars))
    lows   = np.minimum(opens, closes) - np.abs(np.random.randn(n_bars))
    vols   = np.ones(n_bars) * 100.0

    # Build a multi-day index: 390 bars/day from 09:30
    idx = pd.date_range(
        start="2024-01-02 09:30", periods=n_bars, freq="1min",
        tz="America/New_York"
    )

    df_1m = pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes,
         "volume": vols, "anomalous": np.zeros(n_bars, dtype=bool)},
        index=idx,
    )
    df_5m = df_1m.resample("5min").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum", "anomalous": "any",
    }).dropna()

    bar_map = np.full(n_bars, -1, dtype=np.int64)
    trading_dates = sorted(set(ts.date() for ts in idx))

    return MarketData(
        df_1m=df_1m, df_5m=df_5m,
        open_1m=opens.astype(np.float64),
        high_1m=highs.astype(np.float64),
        low_1m=lows.astype(np.float64),
        close_1m=closes.astype(np.float64),
        volume_1m=vols.astype(np.float64),
        open_5m=df_5m["open"].to_numpy(np.float64),
        high_5m=df_5m["high"].to_numpy(np.float64),
        low_5m=df_5m["low"].to_numpy(np.float64),
        close_5m=df_5m["close"].to_numpy(np.float64),
        volume_5m=df_5m["volume"].to_numpy(np.float64),
        bar_map=bar_map,
        trading_dates=trading_dates,
    )


def _make_synthetic_position(entry_price: float = 19000.0, direction: int = 1) -> OpenPosition:
    return OpenPosition(
        direction=direction,
        entry_price=entry_price,
        entry_bar=0,
        contracts=1,
        sl_price=entry_price - 20.0 * direction,
        tp_price=entry_price + 40.0 * direction,
    )


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

class Validator:
    """
    Runs pre-flight checks on a strategy class before a full backtest.

    Usage:
        report = Validator().run(MyStrategy, config, data)
        if not report.passed:
            raise ValidationError(...)
    """

    N_LOOKAHEAD_BARS = 50
    N_NAN_BARS_EXTRA = 10

    def run(
        self,
        strategy_class: Type[BaseStrategy],
        config: RunConfig,
        data: MarketData,
    ) -> ValidationReport:
        report = ValidationReport()

        self._check_lookahead(strategy_class, config, report)
        self._check_nan(strategy_class, config, data, report)
        self._check_order_sanity(strategy_class, config, report)
        self._check_update_sanity(strategy_class, config, report)
        self._check_state_leak(strategy_class, config, report)
        self._check_signals_outside_hours(strategy_class, config, report)

        return report

    # ── Check 1: Lookahead bias ────────────────────────────────────────────

    def _check_lookahead(
        self,
        strategy_class: Type[BaseStrategy],
        config: RunConfig,
        report: ValidationReport,
    ) -> None:
        synth = _make_synthetic_data(self.N_LOOKAHEAD_BARS + 10)
        proxy = _ProxyMarketData(synth)
        strategy = strategy_class(config.params)

        for i in range(self.N_LOOKAHEAD_BARS):
            proxy.set_bar(i)
            try:
                strategy.generate_signals(proxy, i)
            except LookaheadError as e:
                report.failures.append(f"Lookahead bias in generate_signals: {e}")
                return
            except Exception:
                pass  # other errors caught by order sanity check

            pos = _make_synthetic_position(synth.close_1m[i])
            try:
                strategy.manage_position(proxy, i, pos)
            except LookaheadError as e:
                report.failures.append(f"Lookahead bias in manage_position: {e}")
                return
            except Exception:
                pass

    # ── Check 2: NaN propagation ───────────────────────────────────────────

    def _check_nan(
        self,
        strategy_class: Type[BaseStrategy],
        config: RunConfig,
        data: MarketData,
        report: ValidationReport,
    ) -> None:
        lookback = getattr(strategy_class, "min_lookback", 0)
        n_bars = min(lookback + self.N_NAN_BARS_EXTRA, len(data.df_1m))
        strategy = strategy_class(config.params)

        for i in range(n_bars):
            try:
                result = strategy.generate_signals(data, i)
                if result is not None:
                    # Check all numeric fields for NaN
                    for fname, val in vars(result).items():
                        if isinstance(val, float) and np.isnan(val):
                            report.warnings.append(
                                f"NaN in Order.{fname} at bar {i} — check indicator warmup"
                            )
            except Exception:
                pass  # other checks handle these

    # ── Check 3: Order sanity ──────────────────────────────────────────────

    def _check_order_sanity(
        self,
        strategy_class: Type[BaseStrategy],
        config: RunConfig,
        report: ValidationReport,
    ) -> None:
        synth = _make_synthetic_data(60)
        strategy = strategy_class(config.params)
        found_order = False

        for i in range(50):
            try:
                order = strategy.generate_signals(synth, i)
            except Exception as e:
                report.warnings.append(
                    f"generate_signals raised at bar {i}: {type(e).__name__}: {e}"
                )
                continue

            if order is None:
                continue

            found_order = True
            close = synth.close_1m[i]
            errs = self._validate_order(order, close)
            for err in errs:
                report.failures.append(f"Order sanity: {err} (bar {i})")

        if not found_order:
            report.warnings.append(
                "generate_signals returned None for all 50 validation bars — "
                "strategy may have long warmup or rarely signals"
            )

    def _validate_order(self, order: Order, current_price: float) -> list[str]:
        errors = []

        # SL/TP side checks (market orders only — limit/stop deferred to fill)
        if order.order_type == OrderType.MARKET:
            if order.sl_price is not None:
                if order.direction == 1 and order.sl_price >= current_price:
                    errors.append(
                        f"Long SL {order.sl_price} is at or above entry {current_price}"
                    )
                elif order.direction == -1 and order.sl_price <= current_price:
                    errors.append(
                        f"Short SL {order.sl_price} is at or below entry {current_price}"
                    )
            if order.tp_price is not None:
                if order.direction == 1 and order.tp_price <= current_price:
                    errors.append(
                        f"Long TP {order.tp_price} is at or below entry {current_price}"
                    )
                elif order.direction == -1 and order.tp_price >= current_price:
                    errors.append(
                        f"Short TP {order.tp_price} is at or above entry {current_price}"
                    )
            if (order.sl_price is not None and order.tp_price is not None
                    and order.sl_price == order.tp_price):
                errors.append("SL and TP are equal")

        # Limit price side check
        if order.order_type in (OrderType.LIMIT, OrderType.STOP_LIMIT):
            if order.limit_price is not None:
                if order.direction == 1 and order.limit_price >= current_price:
                    errors.append(
                        f"Long limit {order.limit_price} is at or above current price {current_price} "
                        f"— use STOP order to enter above current price"
                    )
                elif order.direction == -1 and order.limit_price <= current_price:
                    errors.append(
                        f"Short limit {order.limit_price} is at or below current price {current_price}"
                    )

        return errors

    # ── Check 4: PositionUpdate sanity ─────────────────────────────────────

    def _check_update_sanity(
        self,
        strategy_class: Type[BaseStrategy],
        config: RunConfig,
        report: ValidationReport,
    ) -> None:
        synth = _make_synthetic_data(60)
        strategy = strategy_class(config.params)

        for i in range(50):
            pos = _make_synthetic_position(synth.close_1m[max(0, i - 1)])
            try:
                update = strategy.manage_position(synth, i, pos)
            except Exception as e:
                report.warnings.append(
                    f"manage_position raised at bar {i}: {type(e).__name__}: {e}"
                )
                continue

            if update is None:
                continue

            errs = self._validate_update(update, pos)
            for err in errs:
                report.failures.append(f"PositionUpdate sanity: {err} (bar {i})")

    def _validate_update(self, update: PositionUpdate, pos: OpenPosition) -> list[str]:
        errors = []

        sl = update.new_sl_price
        tp = update.new_tp_price

        if sl is not None and tp is not None:
            if pos.direction == 1 and sl >= tp:
                errors.append(f"new_sl {sl} >= new_tp {tp} on long position")
            elif pos.direction == -1 and sl <= tp:
                errors.append(f"new_sl {sl} <= new_tp {tp} on short position")

        return errors

    # ── Check 5: Session state leak ────────────────────────────────────────

    def _check_state_leak(
        self,
        strategy_class: Type[BaseStrategy],
        config: RunConfig,
        report: ValidationReport,
    ) -> None:
        synth = _make_synthetic_data(60)

        def _run_and_collect(strat):
            signals = []
            for i in range(50):
                try:
                    order = strat.generate_signals(synth, i)
                    signals.append(order is not None)
                except Exception:
                    signals.append(None)
            return signals

        # Run a fresh instance, record signals.
        # Then run it again from scratch — a clean strategy must produce identical results.
        # If class-level state exists, the second fresh instance will behave differently.
        run1 = _run_and_collect(strategy_class(config.params))

        # Pollute by running another instance (simulates a previous backtest run)
        _run_and_collect(strategy_class(config.params))

        # Now run a third fresh instance — should match run1 if no class-level leak
        run3 = _run_and_collect(strategy_class(config.params))

        if run1 != run3:
            report.failures.append(
                "Session state leak: a fresh strategy instance produced different signals "
                "after another instance had already run — check __init__ for mutable "
                "class-level state (e.g. class-level lists, dicts, or cached values)"
            )

    # ── Check 6: Signals outside trading hours ─────────────────────────────

    def _check_signals_outside_hours(
        self,
        strategy_class: Type[BaseStrategy],
        config: RunConfig,
        report: ValidationReport,
    ) -> None:
        trading_hours = getattr(strategy_class, "trading_hours", None)
        if trading_hours is None:
            return  # no restriction declared — nothing to check

        synth = _make_synthetic_data(60)
        strategy = strategy_class(config.params)

        for i in range(50):
            bar_time = synth.df_1m.index[i].time()
            in_window = any(start <= bar_time <= end for start, end in trading_hours)
            if in_window:
                continue

            try:
                order = strategy.generate_signals(synth, i)
                if order is not None:
                    report.warnings.append(
                        f"generate_signals returned an Order at bar {i} (time {bar_time}) "
                        f"which is outside declared trading_hours {trading_hours} — "
                        f"the engine would ignore it, but this suggests a logic error"
                    )
            except Exception:
                pass