"""
Noise-Boundary Intraday Momentum — Zarattini, Aziz & Barbon (2024)
"Beat the Market: An Effective Intraday Momentum Strategy for S&P500 ETF (SPY)"
Swiss Finance Institute Research Paper No. 24-97

Independently replicated on NQ by Quantitativo (Jan 2025):
  Annualised return 24.3%, Sharpe 1.67, max DD 24% vs 35% benchmark.

Mechanism
─────────
Each session, compute a "noise boundary" — a band around the RTH open price
based on recent historical daily volatility. Prices inside the band are noise.
When price breaks outside the band at a scheduled half-hour check (HH:00 or
HH:30), it signals a genuine supply/demand imbalance → enter in that direction.

Noise boundary:
    upper = session_open × (1 + noise_mult × avg_abs_daily_return_lookback)
    lower = session_open × (1 - noise_mult × avg_abs_daily_return_lookback)

Entry:  STOP order at boundary level (fills only if price continues through it)
SL:     boundary level (re-enters the noise = trade invalidated)
TP:     configured as risk_reward or ATR multiple
Trail:  optional trailing stop activated after trail_activation_points profit

Exit:   SL, TP, trailing stop, or EOD — whichever first
One trade per session (configurable via max_trades_per_day).

Params dict keys:
    lookback_days       : int   — days of daily returns for volatility estimate. Default 14
    noise_mult          : float — boundary width multiplier. Default 1.0
    check_times_min     : list  — minutes past hour to check (e.g. [0, 30]). Default [0, 30]
    sl_type             : str   — "boundary"|"atr_multiple"|"fixed_points". Default "boundary"
    sl_value            : float — used when sl_type != "boundary". Default 2.0
    tp_type             : str   — "risk_reward"|"atr_multiple"|"fixed_points". Default "risk_reward"
    tp_value            : float — TP multiplier. Default 2.0
    atr_length          : int   — ATR period for SL/TP. Default 14
    trail_points        : float — trailing stop distance. None = no trail. Default None
    trail_activation    : float — profit needed before trail activates. Default 0.0
    max_trades_per_day  : int   — max entries per session. Default 1
    contracts           : int   — fixed contracts. Default 1
"""
from __future__ import annotations

from datetime import time, date
from typing import Optional

import numpy as np

from backtest.data.market_data import MarketData
from backtest.strategy.base import BaseStrategy
from backtest.strategy.enums import ExitReason, OrderType, SizeType
from backtest.strategy.order import Order
from backtest.strategy.update import OpenPosition, PositionUpdate


def _atr(highs, lows, closes, period, i) -> float:
    if i < 1:
        return 0.0
    start = max(1, i - period * 3)
    trs = np.maximum(
        highs[start:i+1] - lows[start:i+1],
        np.abs(highs[start:i+1] - closes[start-1:i]),
        np.abs(lows[start:i+1]  - closes[start-1:i]),
    )
    if len(trs) == 0:
        return 0.0
    atr = float(trs[0])
    for tr in trs[1:]:
        atr = (atr * (period - 1) + float(tr)) / period
    return atr


def _avg_abs_daily_return(closes: np.ndarray, i: int, lookback: int) -> float:
    """
    Average absolute daily return over the last `lookback` 1m bars that
    correspond to a close-to-close return.  We approximate by looking at
    the last `lookback` × 390 bars and computing bar-to-bar returns,
    but since we only have 1m data we instead track daily closes externally.
    Here we use a fast approximation: std of the last lookback×390 1m returns
    scaled to daily (×sqrt(390)) — gives a good volatility estimate.
    """
    window = lookback * 390
    start = max(1, i - window)
    if i - start < 2:
        return 0.001   # fallback: 0.1%
    log_rets = np.diff(np.log(closes[start:i+1]))
    daily_vol = float(np.std(log_rets)) * np.sqrt(390)
    return max(daily_vol, 0.0001)


class NoiseBoundaryMomentum(BaseStrategy):
    """
    Noise-boundary intraday momentum (Zarattini et al. 2024), adapted for NQ.
    Enters when price breaks out of the historical-volatility noise band
    at scheduled half-hour check points.
    """

    trading_hours = [(time(9, 30), time(16, 0))]
    min_lookback  = 14 * 390 + 10   # 14 trading days of 1m bars

    def __init__(self, params: dict = None):
        super().__init__(params)
        p = params or {}

        self.lookback_days:      int   = p.get("lookback_days",    14)
        self.noise_mult:         float = p.get("noise_mult",        1.0)
        self.check_times_min:    list  = p.get("check_times_min",   [0, 30])
        self.sl_type:            str   = p.get("sl_type",           "boundary")
        self.sl_value:           float = p.get("sl_value",          2.0)
        self.tp_type:            str   = p.get("tp_type",           "risk_reward")
        self.tp_value:           float = p.get("tp_value",          2.0)
        self.atr_length:         int   = p.get("atr_length",        14)
        self.trail_points:       Optional[float] = p.get("trail_points", None)
        self.trail_activation:   float = p.get("trail_activation",  0.0)
        self.max_trades_per_day: int   = p.get("max_trades_per_day", 1)
        self.contracts:          int   = p.get("contracts",         1)

        # Build set of check times: (hour, minute) where minute in check_times_min
        # We check every hour from 09:30–15:30 at the configured minutes
        self._check_times: set[time] = set()
        for h in range(9, 16):
            for m in self.check_times_min:
                t_candidate = time(h, m)
                if time(9, 30) <= t_candidate <= time(15, 30):
                    self._check_times.add(t_candidate)

        # Per-day state
        self._current_date:    Optional[date]  = None
        self._session_open:    float           = 0.0
        self._upper_boundary:  float           = 0.0
        self._lower_boundary:  float           = 0.0
        self._boundary_set:    bool            = False
        self._trades_today:    int             = 0
        self._in_position:     bool            = False

    def _reset_day(self) -> None:
        self._session_open   = 0.0
        self._upper_boundary = 0.0
        self._lower_boundary = 0.0
        self._boundary_set   = False
        self._trades_today   = 0
        self._in_position    = False

    def generate_signals(self, data: MarketData, i: int) -> Optional[Order]:
        ts       = data.df_1m.index[i]
        bar_time = ts.time()
        bar_date = ts.date()
        close_i  = float(data.close_1m[i])

        # ── Day rollover ──────────────────────────────────────────────────────
        if bar_date != self._current_date:
            self._reset_day()
            self._current_date = bar_date

        # ── Capture session open and compute boundaries ───────────────────────
        if bar_time == time(9, 30):
            self._session_open = float(data.open_1m[i])
            if self._session_open > 0:
                avg_vol = _avg_abs_daily_return(
                    data.close_1m, i, self.lookback_days
                )
                offset = self._session_open * self.noise_mult * avg_vol
                self._upper_boundary = self._session_open + offset
                self._lower_boundary = self._session_open - offset
                self._boundary_set   = True

        if not self._boundary_set:
            return None
        if self._trades_today >= self.max_trades_per_day:
            return None
        if bar_time not in self._check_times:
            return None

        # ── Breakout check ────────────────────────────────────────────────────
        if close_i > self._upper_boundary:
            direction = 1
            stop_px   = self._upper_boundary   # STOP entry just above boundary
        elif close_i < self._lower_boundary:
            direction = -1
            stop_px   = self._lower_boundary
        else:
            return None   # inside noise band — no signal

        self._trades_today += 1
        self._in_position   = True

        return Order(
            direction   = direction,
            order_type  = OrderType.STOP,
            size_type   = SizeType.CONTRACTS,
            size_value  = self.contracts,
            stop_price  = stop_px,
            expiry_bars = 30,   # cancel if not filled within 30 bars
        )

    def on_fill(self, position: OpenPosition, data: MarketData,
                bar_index: int) -> None:
        entry   = position.entry_price
        is_long = position.is_long()
        atr     = _atr(data.high_1m, data.low_1m, data.close_1m,
                       self.atr_length, bar_index)

        # ── Stop loss ─────────────────────────────────────────────────────────
        if self.sl_type == "boundary":
            sl = self._lower_boundary if is_long else self._upper_boundary
        elif self.sl_type == "atr_multiple":
            offset = atr * self.sl_value
            sl = entry - offset if is_long else entry + offset
        elif self.sl_type == "fixed_points":
            sl = entry - self.sl_value if is_long else entry + self.sl_value
        else:
            sl = self._lower_boundary if is_long else self._upper_boundary

        # ── Take profit ───────────────────────────────────────────────────────
        risk = abs(entry - sl) if sl is not None else (atr * 1.5)
        if self.tp_type == "risk_reward":
            tp = entry + risk * self.tp_value if is_long else entry - risk * self.tp_value
        elif self.tp_type == "atr_multiple":
            offset = atr * self.tp_value
            tp = entry + offset if is_long else entry - offset
        elif self.tp_type == "fixed_points":
            tp = entry + self.tp_value if is_long else entry - self.tp_value
        else:
            tp = entry + risk * self.tp_value if is_long else entry - risk * self.tp_value

        position.set_initial_sl_tp(sl, tp)

        # ── Trailing stop ─────────────────────────────────────────────────────
        if self.trail_points is not None:
            position.trail_points            = self.trail_points
            position.trail_activation_points = self.trail_activation

    def manage_position(self, data: MarketData, i: int,
                        position: OpenPosition) -> Optional[PositionUpdate]:
        self._in_position = True
        return None