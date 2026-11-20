"""
Double Session Sweep Strategy
──────────────────────────────
Exact Python replica of the PineScript "Double Session Sweep Strategy".

Logic summary:
  1. Track Asia session (20:00-00:00 ET) high/low
  2. Track London session (02:00-05:00 ET) — detect single-sided Asia sweep
  3. During NY (09:30-16:00 ET):
     a. Invalidation: if price sweeps the non-swept side before the double sweep
     b. Double sweep trigger: NY sweeps the same side London swept
     c. Market structure: track HH/HL (short) or LL/LH (long) pattern
     d. BoS: close breaks the most recent HL (short) or LH (long) with 4-bar gap
     e. Entry: MARKET at BoS close
     f. SL: NY range low/high ± ATR × multiplier
     g. TP: asiaFinalHigh (long) or asiaFinalLow (short), or fixed RR
  4. Force close 30 min before NY end (default 15:30 ET)
  One trade per day maximum.

Rule-based trade filters (applied at entry, no lookahead):
  ATR filter    : skip if today's ATR > atr_filter_mult × 20-day avg ATR
                  High-vol days → sweeps become genuine breakouts, not reversions
  Range filter  : skip if Asia range > range_filter_mult × 30-day median Asia range
                  Unusually wide Asia ranges → directional intent, not consolidation

Params dict keys:
    atr_length          : int   — ATR period. Default 14
    atr_multiplier      : float — SL distance multiplier. Default 1.5
    use_fixed_rr        : bool  — use fixed R:R instead of Asia level for TP. Default False
    risk_reward_ratio   : float — R:R if use_fixed_rr=True. Default 2.0
    force_close_ny      : bool  — force close before NY end. Default True
    ny_close_minutes    : int   — minutes before NY end to force close. Default 30
    contracts           : int   — fixed contracts. Default 5
    atr_filter_mult     : float — skip if ATR > mult × 20d avg ATR. None = disabled. Default None
    atr_filter_lookback : int   — days to average ATR over for filter. Default 20
    range_filter_mult   : float — skip if Asia range > mult × 30d median. None = disabled. Default None
    range_filter_lookback: int  — days of Asia range history for median. Default 30
    min_rr              : float — minimum R:R required to take the trade. If the
                                  natural TP (Asia level) gives a lower R:R, TP is
                                  pushed out to exactly entry ± risk × min_rr.
                                  Only applies when use_fixed_rr=False.
                                  None = disabled (default).
    trail_atr_mult      : float — base trail distance as ATR multiple (e.g. 1.5).
                                  None = trailing disabled (default).
    trail_activation_mult: float — ATR multiples of unrealised profit required
                                   before the trail activates. Default 1.0.
    trail_aggression    : float — 0.0–1.0. How aggressively the trail tightens as
                                  price approaches TP.
                                    0.0 = fixed trail distance throughout
                                    1.0 = trail shrinks linearly from full width
                                          at activation down to trail_min_mult × ATR
                                          by the time price reaches TP.
                                  Default 0.0.
    trail_min_mult      : float — ATR multiple floor the trail can never go below,
                                  regardless of aggression. Default 0.25.
"""
from __future__ import annotations

from collections import deque
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


def _in_session(bar_time: time, start: time, end: time) -> bool:
    """Check if bar_time falls within [start, end). Handles overnight sessions."""
    if start < end:
        return start <= bar_time < end
    else:
        return bar_time >= start or bar_time < end


# Session boundaries (ET / America/New_York)
ASIA_START   = time(20, 0)
ASIA_END     = time(0, 0)
LONDON_START = time(2, 0)
LONDON_END   = time(5, 0)
NY_START     = time(9, 30)
NY_END       = time(11, 0)


class DoubleSessionSweep(BaseStrategy):
    """
    Double Session Sweep — exact replica of the PineScript original,
    with optional rule-based ATR and Asia range filters.
    """

    trading_hours = None
    min_lookback  = 20

    def __init__(self, params: dict = None):
        super().__init__(params)
        p = params or {}

        self.atr_length:           int   = p.get("atr_length",            14)
        self.atr_multiplier:       float = p.get("atr_multiplier",        1.5)
        self.use_fixed_rr:         bool  = p.get("use_fixed_rr",          False)
        self.risk_reward_ratio:    float = p.get("risk_reward_ratio",      2.0)
        self.force_close_ny:       bool  = p.get("force_close_ny",        True)
        self.ny_close_minutes:     int   = p.get("ny_close_minutes",       30)
        self.contracts:            int   = p.get("contracts",              5)

        # ── Rule-based filters ────────────────────────────────────────────────
        # ATR filter: skip if today's ATR > mult × rolling avg ATR
        self.atr_filter_mult:      Optional[float] = p.get("atr_filter_mult",      None)
        self.atr_filter_lookback:  int             = p.get("atr_filter_lookback",   20)
        # Range filter: skip if Asia range > mult × rolling median Asia range
        self.range_filter_mult:    Optional[float] = p.get("range_filter_mult",     None)
        self.range_filter_lookback:int             = p.get("range_filter_lookback",  30)
        # Minimum R:R — push TP out if natural TP gives worse R:R (use_fixed_rr=False only)
        self.min_rr:               Optional[float] = p.get("min_rr",               None)
        # Dynamic trailing stop
        self.trail_atr_mult:       Optional[float] = p.get("trail_atr_mult",       None)
        self.trail_activation_mult: float          = p.get("trail_activation_mult", 1.0)
        self.trail_aggression:     float           = p.get("trail_aggression",      0.0)
        self.trail_min_mult:       float           = p.get("trail_min_mult",        0.25)

        # Force-close time
        fc_total = 16 * 60 - self.ny_close_minutes
        self._force_close_time = time(fc_total // 60, fc_total % 60)

        # Rolling histories for filters (persist across days)
        self._atr_history:   deque = deque(maxlen=self.atr_filter_lookback)
        self._range_history: deque = deque(maxlen=self.range_filter_lookback)

        self._reset_day()
        self._current_trade_date:    Optional[date] = None
        self._current_calendar_date: Optional[date] = None

    def _reset_day(self) -> None:
        # Asia session
        self._asia_high:         float = 0.0
        self._asia_low:          float = float("inf")
        self._asia_final_high:   float = 0.0
        self._asia_final_low:    float = float("inf")
        self._in_asia_prev:      bool  = False
        self._asia_done:         bool  = False

        # London session
        self._london_high:       float = 0.0
        self._london_low:        float = float("inf")
        self._london_final_high: float = 0.0
        self._london_final_low:  float = float("inf")
        self._in_london_prev:    bool  = False
        self._london_done:       bool  = False

        # Sweep detection
        self._london_swept_asia_low:  bool = False
        self._london_swept_asia_high: bool = False
        self._london_low_only:        bool = False
        self._london_high_only:       bool = False

        # NY state
        self._ny_invalid_low:    bool  = False
        self._ny_invalid_high:   bool  = False
        self._ny_low_triggered:  bool  = False
        self._ny_high_triggered: bool  = False
        self._setup_direction:   str   = ""    # "long" or "short" or ""
        self._ny_range_high:     float = 0.0
        self._ny_range_low:      float = float("inf")
        self._in_ny_prev:        bool  = False

        # Structure tracking
        self._initial_high:      float = 0.0
        self._initial_low:       float = float("inf")
        self._last_hh:           float = 0.0
        self._last_hl:           float = float("inf")
        self._last_ll:           float = float("inf")
        self._last_lh:           float = 0.0
        self._last_swing_high:   float = 0.0
        self._last_swing_low:    float = float("inf")
        self._struct_count:      int   = 0
        self._last_struct_type:  str   = ""  # "HH","HL","LL","LH"
        self._last_high_bar:     int   = -999
        self._last_low_bar:      int   = -999
        self._most_recent_hl:    float = float("inf")
        self._most_recent_lh:    float = 0.0
        self._most_recent_hl_bar: int  = -999
        self._most_recent_lh_bar: int  = -999

        # Trade state
        self._bos_detected:      bool  = False
        self._trade_triggered:   bool  = False
        self._trade_taken_today: bool  = False

        # SL/TP anchors stored for on_fill
        self._pending_sl:        float = 0.0
        self._pending_tp:        float = 0.0

    def generate_signals(self, data: MarketData, i: int) -> Optional[Order]:
        ts       = data.df_1m.index[i]
        bar_time = ts.time()
        bar_date = ts.date()
        high_i   = float(data.high_1m[i])
        low_i    = float(data.low_1m[i])
        close_i  = float(data.close_1m[i])
        open_i   = float(data.open_1m[i])

        in_asia   = _in_session(bar_time, ASIA_START, ASIA_END)
        in_london = _in_session(bar_time, LONDON_START, LONDON_END)
        in_ny     = _in_session(bar_time, NY_START, NY_END)

        # ── Calendar date rollover ────────────────────────────────────────────
        # Pine resets on timeframe.change("D") which fires at midnight ET.
        # Reset ALL session state when the calendar date changes.
        #
        # IMPORTANT: Asia ends at 00:00 ET — same bar that triggers the reset.
        # We must save Asia finals BEFORE resetting, otherwise they get wiped.
        # At the midnight bar: in_asia=False, _in_asia_prev=True → save finals first.
        if bar_date != self._current_calendar_date:
            # Save Asia finals before reset if Asia just ended
            asia_was_running = self._in_asia_prev
            asia_high_save   = self._asia_high
            asia_low_save    = self._asia_low
            self._reset_day()
            self._current_calendar_date = bar_date
            # Restore Asia finals if Asia was running when date changed
            if asia_was_running and asia_high_save > 0 and asia_low_save < float("inf"):
                self._asia_final_high = asia_high_save
                self._asia_final_low  = asia_low_save
                self._asia_done       = True

        # ── Trade date: used only for "one trade per day" guard ───────────────
        # The trade date is the NY calendar date (same as bar_date during NY).
        if in_ny:
            self._current_trade_date = bar_date

        # ── ASIA SESSION ──────────────────────────────────────────────────────
        if in_asia:
            if not self._in_asia_prev:
                # Session start
                self._asia_high = high_i
                self._asia_low  = low_i
            else:
                self._asia_high = max(self._asia_high, high_i)
                self._asia_low  = min(self._asia_low,  low_i)
        elif self._in_asia_prev and not in_asia and not self._asia_done:
            # Asia just ended — save finals and record range for filter history
            self._asia_final_high = self._asia_high
            self._asia_final_low  = self._asia_low
            self._asia_done = True
            asia_range = self._asia_final_high - self._asia_final_low
            if asia_range > 0:
                self._range_history.append(asia_range)
        self._in_asia_prev = in_asia

        # ── LONDON SESSION ────────────────────────────────────────────────────
        if in_london:
            if not self._in_london_prev:
                self._london_high = high_i
                self._london_low  = low_i
            else:
                self._london_high = max(self._london_high, high_i)
                self._london_low  = min(self._london_low,  low_i)
            # Check sweeps vs Asia range
            if self._asia_done:
                if low_i < self._asia_final_low:
                    self._london_swept_asia_low  = True
                if high_i > self._asia_final_high:
                    self._london_swept_asia_high = True
        elif self._in_london_prev and not in_london and not self._london_done:
            # London just ended
            self._london_final_high = self._london_high
            self._london_final_low  = self._london_low
            self._london_low_only   = self._london_swept_asia_low  and not self._london_swept_asia_high
            self._london_high_only  = self._london_swept_asia_high and not self._london_swept_asia_low
            self._london_done = True
        self._in_london_prev = in_london

        # ── NY SESSION ────────────────────────────────────────────────────────
        if in_ny:
            # NY range tracking
            if not self._in_ny_prev:
                # NY just opened — record today's ATR into history for filter
                atr_now = _atr(data.high_1m, data.low_1m, data.close_1m,
                                self.atr_length, i)
                if atr_now > 0:
                    self._atr_history.append(atr_now)

                self._ny_range_high = high_i
                self._ny_range_low  = low_i
                # Structure init
                self._initial_high    = high_i
                self._initial_low     = low_i
                self._last_swing_high = high_i
                self._last_swing_low  = low_i
            else:
                self._ny_range_high = max(self._ny_range_high, high_i)
                self._ny_range_low  = min(self._ny_range_low,  low_i)

            # ── Force close ───────────────────────────────────────────────────
            # Handled via RunConfig.eod_exit_time = 15:30, so no action needed here.
            # (The engine closes the position automatically.)

            # ── NY Invalidation ───────────────────────────────────────────────
            if self._london_low_only and not self._ny_low_triggered and not self._ny_invalid_low:
                if (self._asia_done and high_i > self._asia_final_high) or \
                   (self._london_done and high_i > self._london_final_high):
                    self._ny_invalid_low = True

            if self._london_high_only and not self._ny_high_triggered and not self._ny_invalid_high:
                if (self._asia_done and low_i < self._asia_final_low) or \
                   (self._london_done and low_i < self._london_final_low):
                    self._ny_invalid_high = True

            # ── Double Sweep Detection ─────────────────────────────────────────
            if (self._london_low_only and not self._ny_low_triggered
                    and not self._ny_invalid_low and self._london_done
                    and low_i < self._london_final_low):
                self._ny_low_triggered = True
                self._setup_direction  = "long"

            if (self._london_high_only and not self._ny_high_triggered
                    and not self._ny_invalid_high and self._london_done
                    and high_i > self._london_final_high):
                self._ny_high_triggered = True
                self._setup_direction   = "short"

            # ── Market Structure + BoS ─────────────────────────────────────────
            if (not self._bos_detected and self._setup_direction
                    and i >= 1 and self._in_ny_prev):
                self._update_structure(i, high_i, low_i, close_i, open_i,
                                       float(data.high_1m[i-1]),
                                       float(data.low_1m[i-1]),
                                       float(data.close_1m[i-1]),
                                       float(data.open_1m[i-1]))

            # ── Entry ─────────────────────────────────────────────────────────
            if (self._bos_detected and not self._trade_triggered
                    and not self._trade_taken_today and self._setup_direction):

                # ── Rule-based filters ─────────────────────────────────────────
                if self._filters_reject(data, i):
                    self._in_ny_prev = True
                    return None
                atr = _atr(data.high_1m, data.low_1m, data.close_1m, self.atr_length, i)

                if self._setup_direction == "long":
                    sl = self._ny_range_low - atr * self.atr_multiplier
                    if self.use_fixed_rr:
                        tp = close_i + abs(close_i - sl) * self.risk_reward_ratio
                    else:
                        tp = self._asia_final_high
                elif self._setup_direction == "short":
                    sl = self._ny_range_high + atr * self.atr_multiplier
                    if self.use_fixed_rr:
                        tp = close_i - abs(close_i - sl) * self.risk_reward_ratio
                    else:
                        tp = self._asia_final_low
                else:
                    self._in_ny_prev = True
                    return None

                if sl == 0.0 or tp == 0.0:
                    self._in_ny_prev = True
                    return None

                # ── Minimum R:R enforcement ───────────────────────────────────
                # If the natural TP (Asia level) gives a worse R:R than min_rr,
                # push TP out to exactly entry ± risk × min_rr instead.
                # Only applied when use_fixed_rr=False; fixed-RR already
                # guarantees an exact ratio.
                if self.min_rr is not None and not self.use_fixed_rr:
                    risk = abs(close_i - sl)
                    if risk > 0 and abs(tp - close_i) / risk < self.min_rr:
                        if self._setup_direction == "long":
                            tp = close_i + risk * self.min_rr
                        else:
                            tp = close_i - risk * self.min_rr

                self._pending_sl       = sl
                self._pending_tp       = tp
                self._trade_triggered  = True
                self._trade_taken_today = True
                direction = 1 if self._setup_direction == "long" else -1

                self._in_ny_prev = True
                return Order(
                    direction  = direction,
                    order_type = OrderType.MARKET,
                    size_type  = SizeType.CONTRACTS,
                    size_value = self.contracts,
                )

        self._in_ny_prev = in_ny
        return None

    def _filters_reject(self, data: MarketData, i: int) -> bool:
        """
        Returns True if the trade should be SKIPPED based on rule-based filters.
        Both filters use only historical data — no lookahead.

        ATR filter:
            Skip if today's ATR > atr_filter_mult × avg(ATR over last N days).
            High-volatility days are when sweeps tend to become genuine breakouts
            rather than reverting — the stop hunt becomes a trend continuation.

        Range filter:
            Skip if today's Asia range > range_filter_mult × median(Asia range
            over last N days). Unusually wide Asia ranges suggest directional
            institutional intent rather than consolidation liquidity hunting.
        """
        # ── ATR filter ────────────────────────────────────────────────────────
        if self.atr_filter_mult is not None and len(self._atr_history) >= 5:
            current_atr = _atr(data.high_1m, data.low_1m, data.close_1m,
                               self.atr_length, i)
            # Use all history EXCEPT the most recent entry (today's) so it's
            # a true prior-days average
            prior = list(self._atr_history)[:-1] if len(self._atr_history) > 1 \
                    else list(self._atr_history)
            avg_atr = float(np.mean(prior))
            if avg_atr > 0 and current_atr > self.atr_filter_mult * avg_atr:
                return True   # skip — unusually high volatility today

        # ── Asia range filter ─────────────────────────────────────────────────
        if self.range_filter_mult is not None and len(self._range_history) >= 5:
            if self._asia_done:
                today_range = self._asia_final_high - self._asia_final_low
                # Use all history EXCEPT today's range (already appended when
                # Asia closed) — exclude last entry for unbiased prior median
                prior = list(self._range_history)[:-1] if len(self._range_history) > 1 \
                        else list(self._range_history)
                median_range = float(np.median(prior))
                if median_range > 0 and today_range > self.range_filter_mult * median_range:
                    return True   # skip — unusually wide Asia range today

        return False   # no filter triggered — take the trade

    def _update_structure(self, bar_idx: int,
                          high_i: float, low_i: float,
                          close_i: float, open_i: float,
                          high_p: float, low_p: float,
                          close_p: float, open_p: float) -> None:
        """
        Update market structure tracking — Pine uses bar[1] for pattern detection.
        high_p/low_p/close_p/open_p = previous bar values (bar[1] in Pine).
        """
        sd = self._setup_direction

        # ── SHORT setup: look for HH → HL ────────────────────────────────────
        if sd == "short":
            pattern_high = max(high_p, high_i)
            prev_hh = self._last_hh if self._last_hh != 0.0 else self._last_swing_high

            # HH: (no struct OR last was HL) AND bullish prev bar AND bearish curr bar
            #     AND pattern_high > prev_hh AND close_p > prev_hh
            if (not self._last_struct_type or self._last_struct_type == "HL"):
                if (close_p > open_p and close_i < open_i
                        and pattern_high > prev_hh and close_p > prev_hh):
                    too_close = (bar_idx - self._last_high_bar) <= 4
                    create = True
                    if too_close and self._struct_count > 0 and pattern_high <= self._last_hh:
                        create = False
                    if create:
                        self._struct_count    += 1
                        self._last_hh          = pattern_high
                        self._last_swing_high  = pattern_high
                        self._last_high_bar    = bar_idx
                        self._last_struct_type = "HH"

            # HL: last was HH AND bearish prev bar AND bullish curr bar
            #     AND pattern_low > prev_hl AND close_p > prev_hl
            if self._last_struct_type == "HH":
                pattern_low = min(low_p, low_i)
                prev_hl = self._last_hl if self._last_hl != float("inf") else self._last_swing_low
                if (close_p < open_p and close_i > open_i
                        and pattern_low > prev_hl and close_p > prev_hl):
                    too_close = (bar_idx - self._last_low_bar) <= 4
                    create = True
                    if too_close and pattern_low <= self._last_hl:
                        create = False
                    if create:
                        self._struct_count      += 1
                        self._last_hl            = pattern_low
                        self._most_recent_hl     = pattern_low
                        self._most_recent_hl_bar = bar_idx - 1
                        self._last_low_bar       = bar_idx
                        self._last_struct_type   = "HL"

            # BoS for short: close < mostRecentHl with 4-bar gap
            if (self._most_recent_hl != float("inf")
                    and self._most_recent_hl_bar != -999
                    and close_i < self._most_recent_hl
                    and (bar_idx - self._most_recent_hl_bar) >= 4):
                self._bos_detected = True

        # ── LONG setup: look for LL → LH ─────────────────────────────────────
        elif sd == "long":
            pattern_low = min(low_p, low_i)
            prev_ll = self._last_ll if self._last_ll != float("inf") else self._last_swing_low

            # LL: (no struct OR last was LH) AND bearish prev bar AND bullish curr bar
            #     AND pattern_low < prev_ll AND close_p < prev_ll
            if (not self._last_struct_type or self._last_struct_type == "LH"):
                if (close_p < open_p and close_i > open_i
                        and pattern_low < prev_ll and close_p < prev_ll):
                    too_close = (bar_idx - self._last_low_bar) <= 4
                    create = True
                    if too_close and self._struct_count > 0 and pattern_low >= self._last_ll:
                        create = False
                    if create:
                        self._struct_count    += 1
                        self._last_ll          = pattern_low
                        self._last_swing_low   = pattern_low
                        self._last_low_bar     = bar_idx
                        self._last_struct_type = "LL"

            # LH: last was LL AND bullish prev bar AND bearish curr bar
            #     AND pattern_high < prev_lh AND close_p < prev_lh
            if self._last_struct_type == "LL":
                pattern_high = max(high_p, high_i)
                prev_lh = self._last_lh if self._last_lh != 0.0 else self._last_swing_high
                if (close_p > open_p and close_i < open_i
                        and pattern_high < prev_lh and close_p < prev_lh):
                    too_close = (bar_idx - self._last_high_bar) <= 4
                    create = True
                    if too_close and pattern_high >= self._last_lh:
                        create = False
                    if create:
                        self._struct_count      += 1
                        self._last_lh            = pattern_high
                        self._most_recent_lh     = pattern_high
                        self._most_recent_lh_bar = bar_idx - 1
                        self._last_high_bar      = bar_idx
                        self._last_struct_type   = "LH"

            # BoS for long: close > mostRecentLh with 4-bar gap
            if (self._most_recent_lh != 0.0
                    and self._most_recent_lh_bar != -999
                    and close_i > self._most_recent_lh
                    and (bar_idx - self._most_recent_lh_bar) >= 4):
                self._bos_detected = True

    def on_fill(self, position: OpenPosition, data: MarketData,
                bar_index: int) -> None:
        position.set_initial_sl_tp(self._pending_sl, self._pending_tp)

    def manage_position(self, data: MarketData, i: int,
                        position: OpenPosition) -> Optional[PositionUpdate]:
        if self.trail_atr_mult is None:
            return None

        atr = _atr(data.high_1m, data.low_1m, data.close_1m, self.atr_length, i)
        if atr <= 0:
            return None

        close_i    = float(data.close_1m[i])
        entry      = position.entry_price
        profit_pts = (close_i - entry) * position.direction

        # Trail doesn't activate until price has moved trail_activation_mult × ATR
        if profit_pts < atr * self.trail_activation_mult:
            return None

        # Base trail distance, then tighten based on progress toward TP
        trail_dist = atr * self.trail_atr_mult

        if self.trail_aggression > 0 and position.tp_price is not None:
            total_pts = abs(position.tp_price - entry)
            if total_pts > 0:
                progress   = min(profit_pts / total_pts, 1.0)
                trail_dist *= (1.0 - self.trail_aggression * progress)

        # Floor — trail never collapses below trail_min_mult × ATR
        trail_dist = max(trail_dist, atr * self.trail_min_mult)

        new_sl = close_i - trail_dist if position.is_long() else close_i + trail_dist

        # Only submit an update when the new SL is actually better than current
        # (engine enforces this too, but avoids unnecessary PositionUpdate churn)
        sl = position.sl_price
        if position.is_long():
            if sl is not None and new_sl <= sl:
                return None
        else:
            if sl is not None and new_sl >= sl:
                return None

        return PositionUpdate(new_sl_price=new_sl)