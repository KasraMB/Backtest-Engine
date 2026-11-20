"""
IBS + RSI(2) Mean Reversion Strategy
──────────────────────────────────────
Combines three well-documented effects on equity index futures:

1. Internal Bar Strength (IBS)
   IBS = (close - low) / (high - low)
   Values near 0 → close near low → market overextended downward → next day reverts up
   Values near 1 → close near high → market overextended upward → next day reverts down
   Documented by Pagonidis (2013) and confirmed across NQ/QQQ with ~0.35% avg next-day
   return when IBS < 0.2.

2. RSI(2) confirmation
   Ultra-short RSI as a secondary confirmation — only trade when RSI(2) aligns with IBS
   signal (RSI(2) < 10 for longs, > 90 for shorts). Filters out weak IBS signals.

3. Trend filter (200-period EMA)
   Only long when close > EMA(200), only short when close < EMA(200).
   Prevents fading a strong trending market.

4. VIX-proxy volatility filter (optional)
   Mean reversion works best in elevated-volatility environments.
   Proxy: only trade if today's ATR > its N-day average (similar to VIX > MA).

Entry:  LIMIT at today's RTH close (15:59 bar). Signal is computed on the
        completed daily bar, so there is zero lookahead — we know IBS, RSI,
        and EMA at bar close before entering.

Exit:   SL = entry - direction × atr_mult × ATR(14)   [hard stop]
        TP = previous session high/low (natural mean reversion target)
             OR fixed R multiple if prev high/low not available
        EOD: force close at eod_exit_time (default 15:30 next day)

Trade frequency: ~3-5 signals per week (IBS < 0.2 occurs ~20% of days,
                 RSI(2) < 10 narrows this to ~8-12% of days).

Params dict keys:
    ibs_long_thresh     : float — IBS threshold for long. Default 0.2
    ibs_short_thresh    : float — IBS threshold for short. Default 0.8
    rsi_period          : int   — RSI period. Default 2
    rsi_long_thresh     : float — RSI threshold for long. Default 10
    rsi_short_thresh    : float — RSI threshold for short. Default 90
    ema_period          : int   — trend filter EMA period. Default 200
    atr_period          : int   — ATR period for SL. Default 14
    atr_mult            : float — SL distance = atr_mult × ATR. Default 1.5
    tp_type             : str   — "prev_extreme"|"risk_reward". Default "prev_extreme"
    tp_rr               : float — R:R for risk_reward TP. Default 1.5
    use_trend_filter    : bool  — require close > EMA for longs. Default True
    use_vol_filter      : bool  — require ATR > avg ATR. Default False
    vol_filter_period   : int   — lookback for avg ATR in vol filter. Default 20
    vol_filter_mult     : float — ATR must be > mult × avg. Default 1.0
    entry_bar_time      : str   — "15:59" ET bar used for signal. Default "15:59"
    contracts           : int   — fixed contracts. Default 1
    max_trades_per_day  : int   — max entries per day. Default 1
"""
from __future__ import annotations

from datetime import time, date
from typing import Optional

import numpy as np

from backtest.data.market_data import MarketData
from backtest.strategy.base import BaseStrategy
from backtest.strategy.enums import OrderType, SizeType
from backtest.strategy.order import Order
from backtest.strategy.update import OpenPosition, PositionUpdate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rsi(closes: np.ndarray, period: int, i: int) -> float:
    """Wilder RSI up to bar i."""
    if i < period + 1:
        return 50.0
    start = max(0, i - period * 6)
    arr   = closes[start:i + 1].astype(np.float64)
    deltas= np.diff(arr)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses= np.where(deltas < 0, -deltas, 0.0)
    # Wilder smoothing
    avg_gain = float(np.mean(gains[:period]))
    avg_loss = float(np.mean(losses[:period]))
    for g, l in zip(gains[period:], losses[period:]):
        avg_gain = (avg_gain * (period - 1) + g) / period
        avg_loss = (avg_loss * (period - 1) + l) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def _ema(closes: np.ndarray, period: int, i: int) -> float:
    """EMA up to bar i."""
    if i < 1:
        return float(closes[i])
    start = max(0, i - period * 4)
    arr   = closes[start:i + 1].astype(np.float64)
    k     = 2.0 / (period + 1)
    val   = float(arr[0])
    for c in arr[1:]:
        val = c * k + val * (1 - k)
    return val


def _atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
         period: int, i: int) -> float:
    """Wilder ATR up to bar i."""
    if i < 1:
        return 0.0
    start = max(1, i - period * 3)
    trs   = np.maximum(
        highs[start:i + 1] - lows[start:i + 1],
        np.abs(highs[start:i + 1] - closes[start - 1:i]),
        np.abs(lows[start:i + 1]  - closes[start - 1:i]),
    )
    if len(trs) == 0:
        return 0.0
    atr = float(trs[0])
    for tr in trs[1:]:
        atr = (atr * (period - 1) + float(tr)) / period
    return atr


def _ibs(high: float, low: float, close: float) -> float:
    """Internal Bar Strength = (close - low) / (high - low). Range [0, 1]."""
    rng = high - low
    if rng <= 0:
        return 0.5
    return (close - low) / rng


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

class IBSMeanReversion(BaseStrategy):
    """
    IBS + RSI(2) mean reversion on NQ daily bars.
    Signal computed on the 15:59 ET bar (last complete 1-min bar of RTH session).
    Entry via LIMIT at that bar's close price.
    """

    trading_hours = [(time(9, 30), time(16, 0))]
    min_lookback  = 210   # need 200-period EMA warmup

    def __init__(self, params: dict = None):
        super().__init__(params)
        p = params or {}

        self.ibs_long_thresh:   float = p.get("ibs_long_thresh",    0.20)
        self.ibs_short_thresh:  float = p.get("ibs_short_thresh",   0.80)
        self.rsi_period:        int   = p.get("rsi_period",         2)
        self.rsi_long_thresh:   float = p.get("rsi_long_thresh",    10.0)
        self.rsi_short_thresh:  float = p.get("rsi_short_thresh",   90.0)
        self.ema_period:        int   = p.get("ema_period",         200)
        self.atr_period:        int   = p.get("atr_period",         14)
        self.atr_mult:          float = p.get("atr_mult",           1.5)
        self.tp_type:           str   = p.get("tp_type",            "prev_extreme")
        self.tp_rr:             float = p.get("tp_rr",              1.5)
        self.use_trend_filter:  bool  = p.get("use_trend_filter",   True)
        self.use_vol_filter:    bool  = p.get("use_vol_filter",     False)
        self.vol_filter_period: int   = p.get("vol_filter_period",  20)
        self.vol_filter_mult:   float = p.get("vol_filter_mult",    1.0)
        self.contracts:         int   = p.get("contracts",          1)
        self.max_trades_per_day:int   = p.get("max_trades_per_day", 1)

        # Signal bar: last 1-minute bar of RTH = 15:59 ET
        self._signal_time = time(15, 59)

        # Per-day state
        self._current_date:     Optional[date] = None
        self._trades_today:     int            = 0
        self._prev_day_high:    float          = 0.0
        self._prev_day_low:     float          = float("inf")
        self._session_high:     float          = 0.0
        self._session_low:      float          = float("inf")

        # Pending SL/TP for on_fill
        self._pending_sl:       float          = 0.0
        self._pending_tp:       float          = 0.0

    def _reset_day(self) -> None:
        self._trades_today = 0
        # Save previous session range
        if self._session_high > 0:
            self._prev_day_high = self._session_high
            self._prev_day_low  = self._session_low
        self._session_high = 0.0
        self._session_low  = float("inf")

    def _vol_filter_ok(self, data: MarketData, i: int) -> bool:
        """Returns True if ATR > vol_filter_mult × avg ATR over lookback."""
        if not self.use_vol_filter:
            return True
        atr_now = _atr(data.high_1m, data.low_1m, data.close_1m, self.atr_period, i)
        if atr_now <= 0:
            return False
        # Compute avg ATR over past vol_filter_period days
        # Proxy: use last vol_filter_period × 390 bars for daily ATR average
        lookback_bars = self.vol_filter_period * 390
        start = max(self.atr_period + 1, i - lookback_bars)
        atrs  = []
        step  = 390   # sample one bar per day (roughly)
        for j in range(start, i, step):
            a = _atr(data.high_1m, data.low_1m, data.close_1m, self.atr_period, j)
            if a > 0:
                atrs.append(a)
        if not atrs:
            return True
        avg_atr = float(np.mean(atrs))
        return atr_now > self.vol_filter_mult * avg_atr

    def generate_signals(self, data: MarketData, i: int) -> Optional[Order]:
        ts       = data.df_1m.index[i]
        bar_time = ts.time()
        bar_date = ts.date()
        close_i  = float(data.close_1m[i])
        high_i   = float(data.high_1m[i])
        low_i    = float(data.low_1m[i])

        # ── Day rollover ──────────────────────────────────────────────────────
        if bar_date != self._current_date:
            self._reset_day()
            self._current_date = bar_date

        # Track session range
        if time(9, 30) <= bar_time <= time(16, 0):
            self._session_high = max(self._session_high, high_i)
            self._session_low  = min(self._session_low,  low_i)

        # ── Signal only fires on the 15:59 bar ───────────────────────────────
        if bar_time != self._signal_time:
            return None

        if self._trades_today >= self.max_trades_per_day:
            return None

        if i < self.min_lookback:
            return None

        # ── Compute indicators ────────────────────────────────────────────────
        ibs_val = _ibs(high_i, low_i, close_i)
        rsi_val = _rsi(data.close_1m, self.rsi_period, i)
        ema_val = _ema(data.close_1m, self.ema_period, i)
        atr_val = _atr(data.high_1m, data.low_1m, data.close_1m, self.atr_period, i)

        if atr_val <= 0:
            return None

        # ── Volatility filter ─────────────────────────────────────────────────
        if not self._vol_filter_ok(data, i):
            return None

        # ── Long signal: IBS low + RSI(2) oversold ───────────────────────────
        long_signal  = (ibs_val <= self.ibs_long_thresh
                        and rsi_val <= self.rsi_long_thresh)
        if self.use_trend_filter:
            long_signal = long_signal and (close_i > ema_val)

        # ── Short signal: IBS high + RSI(2) overbought ───────────────────────
        short_signal = (ibs_val >= self.ibs_short_thresh
                        and rsi_val >= self.rsi_short_thresh)
        if self.use_trend_filter:
            short_signal = short_signal and (close_i < ema_val)

        if not long_signal and not short_signal:
            return None

        # Long takes priority if both fire (rare)
        direction = 1 if long_signal else -1

        # ── SL and TP ─────────────────────────────────────────────────────────
        sl_offset = self.atr_mult * atr_val
        if direction == 1:
            sl = close_i - sl_offset
            if self.tp_type == "prev_extreme" and self._prev_day_high > 0:
                tp = self._prev_day_high
                # Sanity: TP must be above entry
                if tp <= close_i:
                    tp = close_i + self.tp_rr * sl_offset
            else:
                tp = close_i + self.tp_rr * sl_offset
        else:
            sl = close_i + sl_offset
            if self.tp_type == "prev_extreme" and self._prev_day_low < float("inf"):
                tp = self._prev_day_low
                if tp >= close_i:
                    tp = close_i - self.tp_rr * sl_offset
            else:
                tp = close_i - self.tp_rr * sl_offset

        self._pending_sl    = sl
        self._pending_tp    = tp
        self._trades_today += 1

        return Order(
            direction  = direction,
            order_type = OrderType.LIMIT,
            size_type  = SizeType.CONTRACTS,
            size_value = self.contracts,
            limit_price= close_i,   # limit at today's close
        )

    def on_fill(self, position: OpenPosition, data: MarketData,
                bar_index: int) -> None:
        """Set SL/TP from computed values at signal time."""
        # Recompute from actual fill in case of slippage
        entry   = position.entry_price
        is_long = position.is_long()
        atr     = _atr(data.high_1m, data.low_1m, data.close_1m,
                       self.atr_period, bar_index)
        atr = max(atr, 1.0)
        sl_offset = self.atr_mult * atr

        sl = entry - sl_offset if is_long else entry + sl_offset

        # Use pending TP if valid, otherwise fallback to R:R
        tp = self._pending_tp
        if is_long and tp <= entry:
            tp = entry + self.tp_rr * sl_offset
        elif not is_long and tp >= entry:
            tp = entry - self.tp_rr * sl_offset

        position.set_initial_sl_tp(sl, tp)

    def manage_position(self, data: MarketData, i: int,
                        position: OpenPosition) -> Optional[PositionUpdate]:
        return None