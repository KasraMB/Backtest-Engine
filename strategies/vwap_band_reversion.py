"""
VWAP Band Mean Reversion — with Trend Filter & Previous Day Structure
─────────────────────────────────────────────────────────────────────
Mechanism
─────────
NQ institutions use VWAP as their execution benchmark — price gravitates back
to it throughout the session. When price extends to ±2σ from VWAP it is
statistically overextended and likely to revert. The reversion is higher
probability when:
  1. It aligns with the intraday trend (EMA filter)
  2. Price is near a previous-day high/low level (structural support/resistance)

Signal
──────
  VWAP + rolling standard deviation bands are computed on every RTH bar.
  Band touch:   price crosses into ±2σ band (overextension)
  Trend filter: 50-bar EMA — only trade longs when close > EMA, shorts when < EMA
  Structure:    previous day high/low within `structure_buffer` points
  Time filter:  entries only 09:45–15:00 ET (avoid open noise and close risk)

Entry:  LIMIT at the 2σ band level (better fill than chasing)
SL:     band_level − direction × atr_mult × ATR  (beyond the band → breakdown)
TP:     VWAP midline (natural mean reversion target, typically 1.5-2× risk)
Max 2 trades per day.  One at a time.

Params dict keys:
    ema_period       : int   — trend EMA period. Default 50
    atr_period       : int   — ATR period for SL sizing. Default 14
    atr_mult         : float — SL distance = atr_mult × ATR beyond band. Default 0.75
    band_mult        : float — band width in std devs. Default 2.0
    vwap_lookback    : int   — bars used for rolling VWAP/std computation. Default 30
    structure_buffer : float — max pts from prev day H/L to count as structure. Default 30.0
    require_structure: bool  — require proximity to prev day H/L. Default False
    entry_start_h    : int   — earliest entry hour ET. Default 9
    entry_start_m    : int   — earliest entry minute ET. Default 45
    entry_end_h      : int   — latest entry hour ET. Default 15
    entry_end_m      : int   — latest entry minute ET. Default 0
    max_trades       : int   — max trades per day. Default 2
    contracts        : int   — fixed contracts. Default 1
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


def _ema(closes, period, i) -> float:
    """Simple EMA over last `period` bars ending at i."""
    start = max(0, i - period * 3)
    arr   = closes[start:i+1]
    if len(arr) == 0:
        return float(closes[i])
    k   = 2.0 / (period + 1)
    val = float(arr[0])
    for c in arr[1:]:
        val = float(c) * k + val * (1 - k)
    return val


class VWAPBandMeanReversion(BaseStrategy):
    """
    VWAP band mean reversion with EMA trend filter.

    Enters on limit at ±2σ VWAP band when price is overextended in the
    direction opposite to the move (i.e. fading the extension back to VWAP).
    Only trades with the intraday trend (EMA filter).
    """

    trading_hours = [(time(9, 30), time(16, 0))]
    min_lookback  = 60

    def __init__(self, params: dict = None):
        super().__init__(params)
        p = params or {}

        self.ema_period:        int   = p.get("ema_period",        50)
        self.atr_period:        int   = p.get("atr_period",        14)
        self.atr_mult:          float = p.get("atr_mult",           0.75)
        self.band_mult:         float = p.get("band_mult",          2.0)
        self.vwap_lookback:     int   = p.get("vwap_lookback",      30)
        self.structure_buffer:  float = p.get("structure_buffer",   30.0)
        self.require_structure: bool  = p.get("require_structure",  False)
        self.entry_start_h:     int   = p.get("entry_start_h",      9)
        self.entry_start_m:     int   = p.get("entry_start_m",      45)
        self.entry_end_h:       int   = p.get("entry_end_h",        15)
        self.entry_end_m:       int   = p.get("entry_end_m",        0)
        self.max_trades:        int   = p.get("max_trades",         2)
        self.contracts:         int   = p.get("contracts",          1)

        self._entry_start = time(self.entry_start_h, self.entry_start_m)
        self._entry_end   = time(self.entry_end_h,   self.entry_end_m)

        # Per-day state
        self._current_date:   Optional[date]  = None
        self._trades_today:   int             = 0
        self._prev_day_high:  float           = 0.0
        self._prev_day_low:   float           = float("inf")
        self._session_high:   float           = 0.0
        self._session_low:    float           = float("inf")
        self._pending_dir:    int             = 0     # direction of pending limit
        self._pending_band:   float           = 0.0   # band level for limit
        self._pending_sl:     float           = 0.0
        self._pending_tp:     float           = 0.0

    def _reset_day(self) -> None:
        self._trades_today  = 0
        self._session_high  = 0.0
        self._session_low   = float("inf")
        self._pending_dir   = 0
        self._pending_band  = 0.0
        self._pending_sl    = 0.0
        self._pending_tp    = 0.0

    def _compute_vwap_bands(self, data: MarketData, i: int) -> tuple[float, float, float]:
        """
        Compute intraday VWAP and ±band_mult std dev bands using last
        vwap_lookback bars within the current RTH session.
        Returns (vwap, upper_band, lower_band).
        """
        start = max(0, i - self.vwap_lookback + 1)
        closes = data.close_1m[start:i+1].astype(np.float64)
        highs  = data.high_1m[start:i+1].astype(np.float64)
        lows   = data.low_1m[start:i+1].astype(np.float64)
        vols   = data.volume_1m[start:i+1].astype(np.float64)

        typicals = (highs + lows + closes) / 3.0
        vol_sum  = vols.sum()
        if vol_sum <= 0:
            vwap = float(typicals[-1])
            std  = float(np.std(typicals)) if len(typicals) > 1 else 1.0
        else:
            vwap = float((typicals * vols).sum() / vol_sum)
            # Volume-weighted standard deviation
            var = float(((typicals - vwap) ** 2 * vols).sum() / vol_sum)
            std = float(np.sqrt(var)) if var > 0 else float(np.std(typicals))

        std = max(std, 0.25)  # floor to avoid zero-width bands
        upper = vwap + self.band_mult * std
        lower = vwap - self.band_mult * std
        return vwap, upper, lower

    def generate_signals(self, data: MarketData, i: int) -> Optional[Order]:
        ts       = data.df_1m.index[i]
        bar_time = ts.time()
        bar_date = ts.date()
        close_i  = float(data.close_1m[i])
        high_i   = float(data.high_1m[i])
        low_i    = float(data.low_1m[i])

        # ── Day rollover ──────────────────────────────────────────────────────
        if bar_date != self._current_date:
            # Save previous session range as structure levels
            if self._session_high > 0 and self._session_low < float("inf"):
                self._prev_day_high = self._session_high
                self._prev_day_low  = self._session_low
            self._reset_day()
            self._current_date = bar_date

        # Track session range for next day's structure
        if bar_time >= time(9, 30):
            self._session_high = max(self._session_high, high_i)
            self._session_low  = min(self._session_low,  low_i)

        # ── Gate: time window and trade cap ──────────────────────────────────
        if self._trades_today >= self.max_trades:
            return None
        if not (self._entry_start <= bar_time <= self._entry_end):
            return None

        # ── Compute VWAP bands and EMA ────────────────────────────────────────
        if i < self.vwap_lookback + self.ema_period:
            return None

        vwap, upper, lower = self._compute_vwap_bands(data, i)
        ema = _ema(data.close_1m, self.ema_period, i)
        atr = _atr(data.high_1m, data.low_1m, data.close_1m, self.atr_period, i)
        if atr <= 0:
            return None

        # ── Signal logic ─────────────────────────────────────────────────────
        # Long setup: price at or below lower band AND close > EMA (uptrend)
        # Short setup: price at or above upper band AND close < EMA (downtrend)

        direction   = 0
        band_level  = 0.0

        if close_i <= lower and close_i > ema:
            # Price overextended downward but trend is still up — long reversion
            direction  =  1
            band_level = lower
        elif close_i >= upper and close_i < ema:
            # Price overextended upward but trend is down — short reversion
            direction  = -1
            band_level = upper

        if direction == 0:
            return None

        # ── Structure filter (optional) ───────────────────────────────────────
        if self.require_structure and self._prev_day_high > 0:
            near_prev_high = abs(close_i - self._prev_day_high) <= self.structure_buffer
            near_prev_low  = abs(close_i - self._prev_day_low)  <= self.structure_buffer
            if not (near_prev_high or near_prev_low):
                return None

        # ── Compute SL and TP ─────────────────────────────────────────────────
        sl_offset = self.atr_mult * atr
        if direction == 1:
            sl = band_level - sl_offset   # below the lower band
            tp = vwap                     # mean reversion target
        else:
            sl = band_level + sl_offset   # above the upper band
            tp = vwap

        # Sanity: ensure TP is on the correct side
        risk = abs(band_level - sl)
        reward = abs(tp - band_level)
        if reward < risk * 0.5:
            # VWAP too close — not enough reward
            return None

        self._trades_today += 1

        return Order(
            direction  = direction,
            order_type = OrderType.LIMIT,
            size_type  = SizeType.CONTRACTS,
            size_value = self.contracts,
            limit_price= band_level,
            sl_price   = None,   # set in on_fill from actual fill
            tp_price   = None,
            expiry_bars= 10,     # cancel if not filled in 10 bars
        )

    def on_fill(self, position: OpenPosition, data: MarketData,
                bar_index: int) -> None:
        """Set SL/TP from actual fill price."""
        entry   = position.entry_price
        is_long = position.is_long()
        atr     = _atr(data.high_1m, data.low_1m, data.close_1m,
                       self.atr_period, bar_index)
        atr     = max(atr, 5.0)

        # SL: atr_mult × ATR below/above fill
        sl_offset = self.atr_mult * atr
        sl = entry - sl_offset if is_long else entry + sl_offset

        # TP: recompute VWAP as target
        vwap, _, _ = self._compute_vwap_bands(data, bar_index)
        tp = vwap

        # Ensure TP is on the right side of entry
        if is_long and tp <= entry:
            tp = entry + sl_offset * 2
        elif not is_long and tp >= entry:
            tp = entry - sl_offset * 2

        position.set_initial_sl_tp(sl, tp)

    def manage_position(self, data: MarketData, i: int,
                        position: OpenPosition) -> Optional[PositionUpdate]:
        """
        Trail the TP to updated VWAP as it evolves intraday.
        Only move TP in the profitable direction (ratchet).
        """
        vwap, _, _ = self._compute_vwap_bands(data, i)
        is_long    = position.is_long()

        if position.tp_price is None:
            return None

        new_tp = vwap
        if is_long and new_tp > position.tp_price:
            return PositionUpdate(new_tp_price=new_tp)
        elif not is_long and new_tp < position.tp_price:
            return PositionUpdate(new_tp_price=new_tp)
        return None