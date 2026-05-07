"""
AnchoredMeanReversion Strategy (NQ Futures, 1-min bars)
Ported from spec.md PineScript indicator.

Entry logic:
  1. Displacement candle: close-position ratio >= threshold on the current bar.
  2. BOS: that bar's close also breaks at least one unbroken swing high or low.
  3. Session window: bar falls within `window_mins` of a tracked session open.

SL: ATR-tiered tick distance (three tiers based on ATR level).
TP: SL × tp_multiple.

All signals are pre-computed in one forward pass during _setup so that
generate_signals() is a pure O(1) array lookup.

Default sessions: 9:30 AM ("930") and 2:00 PM ("1400") ET, 90-min window.
"""
from __future__ import annotations

from datetime import time as _time
from typing import Optional

import numpy as np

from backtest.data.market_data import MarketData
from backtest.strategy.base import BaseStrategy
from backtest.strategy.enums import OrderType, SizeType
from backtest.strategy.order import Order
from backtest.strategy.update import OpenPosition, PositionUpdate

POINT_VALUE = 20.0
TICK        = 0.25

# Session opens in minutes since midnight ET
_SESSION_MAP: dict[str, int] = {
    "930":  9  * 60 + 30,
    "1400": 14 * 60,
    "2000": 20 * 60,
    "300":  3  * 60,
}

try:
    from numba import njit as _numba_njit
    _njit = lambda fn: _numba_njit(fn, cache=True)
except ImportError:
    _njit = lambda fn: fn


@_njit
def _wilder_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                period: int) -> np.ndarray:
    n   = len(close)
    out = np.zeros(n)
    if n < period + 1:
        return out
    seed = high[0] - low[0]
    for k in range(1, period):
        hl = high[k] - low[k]
        hc = abs(high[k] - close[k - 1])
        lc = abs(low[k]  - close[k - 1])
        seed += hl if (hl >= hc and hl >= lc) else (hc if hc >= lc else lc)
    out[period - 1] = seed / period
    inv   = 1.0 - 1.0 / period
    alpha = 1.0 / period
    for i in range(period, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i]  - close[i - 1])
        tr = hl if (hl >= hc and hl >= lc) else (hc if hc >= lc else lc)
        out[i] = out[i - 1] * inv + tr * alpha
    return out


@_njit
def _compute_signals(
    highs:      np.ndarray,
    lows:       np.ndarray,
    closes:     np.ndarray,
    opens:      np.ndarray,
    times_min:  np.ndarray,
    atr:        np.ndarray,
    n:          int,
    threshold:  float,
    atr_mult:   float,
    session_opens: np.ndarray,  # int32 array of session open minutes
    window_mins: int,
    atr_sl_low:  float,
    atr_sl_high: float,
    sl_ticks_low:  int,
    sl_ticks_mid:  int,
    sl_ticks_high: int,
    tp_multiple: float,
) -> tuple:
    """
    Numba JIT forward pass. Returns (long_sig, short_sig, sl_arr, tp_arr).
    Swing state uses fixed-size arrays (MAX_SW=200) with compact-on-write
    semantics to avoid Python list overhead.
    """
    MAX_SW = 200
    N      = len(closes)
    long_sig  = np.zeros(N, dtype=np.bool_)
    short_sig = np.zeros(N, dtype=np.bool_)
    sl_arr    = np.zeros(N, dtype=np.float64)
    tp_arr    = np.zeros(N, dtype=np.float64)

    # Only unbroken swing levels are stored; broken ones are removed immediately
    # so the lists stay small and never fill the MAX_SW cap.
    sh_prices = np.zeros(MAX_SW, dtype=np.float64)
    sh_count  = 0

    sl_prices = np.zeros(MAX_SW, dtype=np.float64)
    sl_count  = 0

    n_sess   = len(session_opens)
    last_open = np.full(n_sess, -1, dtype=np.int64)

    for i in range(N):
        t  = times_min[i]
        cl = closes[i]
        op = opens[i]
        hi = highs[i]
        lo = lows[i]
        av = atr[i]

        # ── Session open tracking ──────────────────────────────────────────
        for k in range(n_sess):
            if t == session_opens[k]:
                last_open[k] = i

        # ── Within session window ──────────────────────────────────────────
        within = False
        for k in range(n_sess):
            if last_open[k] >= 0 and (i - last_open[k]) <= window_mins:
                within = True
                break

        # ── Swing detection at bar i-n ─────────────────────────────────────
        if i >= 2 * n:
            si = i - n

            is_hh = True
            for k in range(1, n + 1):
                if highs[si] <= highs[si - k] or highs[si] <= highs[si + k]:
                    is_hh = False
                    break
            if is_hh:
                new_h = highs[si]
                # Keep only unbroken highs above new_h (lower ones are dominated)
                w = 0
                for j in range(sh_count):
                    if sh_prices[j] >= new_h:
                        sh_prices[w] = sh_prices[j]
                        w += 1
                sh_count = w
                if sh_count < MAX_SW:
                    sh_prices[sh_count] = new_h
                    sh_count += 1

            is_ll = True
            for k in range(1, n + 1):
                if lows[si] >= lows[si - k] or lows[si] >= lows[si + k]:
                    is_ll = False
                    break
            if is_ll:
                new_l = lows[si]
                # Keep only unbroken lows below new_l
                w = 0
                for j in range(sl_count):
                    if sl_prices[j] <= new_l:
                        sl_prices[w] = sl_prices[j]
                        w += 1
                sl_count = w
                if sl_count < MAX_SW:
                    sl_prices[sl_count] = new_l
                    sl_count += 1

        # ── BOS: remove broken levels immediately to keep list compact ─────
        broke_high = False
        for j in range(sh_count):
            if cl > sh_prices[j]:
                broke_high = True
                break
        if broke_high:
            w = 0
            for j in range(sh_count):
                if sh_prices[j] >= cl:   # keep levels NOT yet broken
                    sh_prices[w] = sh_prices[j]
                    w += 1
            sh_count = w

        broke_low = False
        for j in range(sl_count):
            if cl < sl_prices[j]:
                broke_low = True
                break
        if broke_low:
            w = 0
            for j in range(sl_count):
                if sl_prices[j] <= cl:   # keep levels NOT yet broken
                    sl_prices[w] = sl_prices[j]
                    w += 1
            sl_count = w

        # ── Displacement candle ────────────────────────────────────────────
        candle_range = hi - lo
        atr_ok = True if atr_mult == 0.0 else (candle_range >= av * atr_mult)

        bull_range = hi - op
        bull_pos   = (cl - op) / bull_range if bull_range > 0.0 else 0.0
        bull_cond  = (cl >= op) and (bull_pos >= threshold)

        bear_range = op - lo
        bear_pos   = (op - cl) / bear_range if bear_range > 0.0 else 0.0
        bear_cond  = (cl < op) and (bear_pos >= threshold)

        is_disp = (bull_cond or bear_cond) and atr_ok

        # ── SL / TP ───────────────────────────────────────────────────────
        if av < atr_sl_low:
            sl_ticks = sl_ticks_low
        elif av < atr_sl_high:
            sl_ticks = sl_ticks_mid
        else:
            sl_ticks = sl_ticks_high
        sl_d = sl_ticks * TICK
        tp_d = sl_d * tp_multiple
        sl_arr[i] = sl_d
        tp_arr[i] = tp_d

        # ── Signal ────────────────────────────────────────────────────────
        if is_disp and (broke_high or broke_low) and within and av > 0.0:
            if bull_cond:
                long_sig[i]  = True
            elif bear_cond:
                short_sig[i] = True

    return long_sig, short_sig, sl_arr, tp_arr


class AnchoredMeanReversionStrategy(BaseStrategy):
    """
    Displacement + BOS + session-window momentum/mean-rev strategy.
    """

    trading_hours = None   # 24h; session filtering handled internally
    min_lookback  = 30     # need ATR to warm up

    def __init__(self, params: dict = None) -> None:
        super().__init__(params)
        p = params or {}

        self.n              = int(p.get("swing_periods",      1))
        self.threshold      = float(p.get("threshold",        0.8))
        self.atr_len        = int(p.get("atr_len",            14))
        self.atr_mult       = float(p.get("atr_mult",         0.0))
        self.tp_multiple    = float(p.get("tp_multiple",      1.5))
        self.atr_sl_low     = float(p.get("atr_sl_low",       7.0))
        self.atr_sl_high    = float(p.get("atr_sl_high",      20.0))
        self.sl_ticks_low   = int(p.get("sl_ticks_low",       66))
        self.sl_ticks_mid   = int(p.get("sl_ticks_mid",       100))
        self.sl_ticks_high  = int(p.get("sl_ticks_high",      200))
        self.window_mins    = int(p.get("window_mins",         90))
        self.sessions       = list(p.get("sessions",          ["930", "1400"]))

        # Per-session trade constraints.
        # Format: {"930": {"max_trades": 2, "stop_if_positive": True}, ...}
        # Omit a session or key to leave that constraint inactive.
        self.session_config: dict = dict(p.get("session_config", {}))

        self._times_min:  Optional[np.ndarray] = None
        self._long_sig:   Optional[np.ndarray] = None
        self._short_sig:  Optional[np.ndarray] = None
        self._sl_arr:     Optional[np.ndarray] = None
        self._tp_arr:     Optional[np.ndarray] = None
        self._sig_mask:   Optional[np.ndarray] = None

        # Runtime session-constraint state (only used when session_config is set)
        self._sess_open_bars:       dict[str, np.ndarray] = {}
        self._sess_current_open_bar: dict[str, int]       = {}
        self._sess_trade_count:      dict[str, int]       = {}
        self._sess_net_pnl:          dict[str, float]     = {}
        self._pending_trade:         Optional[dict]       = None

    def _setup(self, data: MarketData) -> None:
        if self._long_sig is not None:
            return

        idx = data.df_1m.index
        if data.bar_times_1m_min is not None:
            self._times_min = data.bar_times_1m_min
        else:
            self._times_min = (idx.hour * 60 + idx.minute).to_numpy(np.int32)

        atr = _wilder_atr(
            data.high_1m, data.low_1m, data.close_1m, self.atr_len
        )

        session_opens = np.array(
            [_SESSION_MAP[s] for s in self.sessions if s in _SESSION_MAP],
            dtype=np.int32,
        )

        self._long_sig, self._short_sig, self._sl_arr, self._tp_arr = _compute_signals(
            data.high_1m, data.low_1m, data.close_1m, data.open_1m,
            self._times_min.astype(np.int32), atr,
            self.n, self.threshold, self.atr_mult,
            session_opens, self.window_mins,
            self.atr_sl_low, self.atr_sl_high,
            self.sl_ticks_low, self.sl_ticks_mid, self.sl_ticks_high,
            self.tp_multiple,
        )

        self._sig_mask = self._long_sig | self._short_sig

        # Precompute session open bar indices for runtime constraint checks
        if self.session_config:
            for s in self.sessions:
                if s in _SESSION_MAP:
                    sm = _SESSION_MAP[s]
                    self._sess_open_bars[s] = np.where(self._times_min == sm)[0]

    def _active_session(self, i: int) -> Optional[str]:
        """Return the session name active at bar i, matching Numba priority order."""
        for s in self.sessions:
            bars = self._sess_open_bars.get(s)
            if bars is None or len(bars) == 0:
                continue
            idx = int(np.searchsorted(bars, i, side="right")) - 1
            if idx >= 0 and (i - int(bars[idx])) <= self.window_mins:
                return s
        return None

    def _check_session_resets(self, i: int) -> None:
        """Reset per-session counters whenever a new session-day open is crossed."""
        for s in self.sessions:
            bars = self._sess_open_bars.get(s)
            if bars is None or len(bars) == 0:
                continue
            idx = int(np.searchsorted(bars, i, side="right")) - 1
            if idx < 0:
                continue
            open_bar = int(bars[idx])
            if self._sess_current_open_bar.get(s, -1) != open_bar:
                self._sess_current_open_bar[s] = open_bar
                self._sess_trade_count[s]       = 0
                self._sess_net_pnl[s]           = 0.0

    def _resolve_pending_pnl(self, data: MarketData, up_to: int) -> float:
        """
        Scan bars from the pending trade's entry forward to find whether
        SL or TP was hit first.  Returns points gained/lost (unsigned SL/TP dist).
        """
        p  = self._pending_trade
        eb = p["entry_bar"]
        d  = p["direction"]
        sl = p["sl_price"]
        tp = p["tp_price"]
        n  = len(data.high_1m)
        for j in range(eb + 1, min(up_to + 1, n)):
            if d == 1:
                if data.low_1m[j]  <= sl:
                    return -p["sl_dist"]
                if data.high_1m[j] >= tp:
                    return  p["tp_dist"]
            else:
                if data.high_1m[j] >= sl:
                    return -p["sl_dist"]
                if data.low_1m[j]  <= tp:
                    return  p["tp_dist"]
        return 0.0   # unresolved (position still open — shouldn't reach here when flat)

    def _session_allows_trade(self, session: str) -> bool:
        cfg      = self.session_config.get(session, {})
        max_t    = cfg.get("max_trades")
        stop_pos = cfg.get("stop_if_positive", False)
        if max_t is not None and self._sess_trade_count.get(session, 0) >= max_t:
            return False
        if stop_pos and self._sess_net_pnl.get(session, 0.0) > 0.0:
            return False
        return True

    def signal_bar_mask(self, data: MarketData) -> np.ndarray:
        self._setup(data)
        return self._sig_mask

    def generate_signals(self, data: MarketData, i: int) -> Optional[Order]:
        self._setup(data)

        if self.session_config:
            self._check_session_resets(i)
            # Resolve outcome of the last trade now that we're flat
            if self._pending_trade is not None:
                pnl  = self._resolve_pending_pnl(data, i)
                sess = self._pending_trade["session"]
                self._sess_net_pnl[sess] = self._sess_net_pnl.get(sess, 0.0) + pnl
                self._pending_trade = None

        if self._long_sig[i]:
            if self.session_config:
                sess = self._active_session(i)
                if sess is not None and not self._session_allows_trade(sess):
                    return None
            return self._build_order(1, float(data.close_1m[i]),
                                     self._sl_arr[i], self._tp_arr[i], "jj_long")
        if self._short_sig[i]:
            if self.session_config:
                sess = self._active_session(i)
                if sess is not None and not self._session_allows_trade(sess):
                    return None
            return self._build_order(-1, float(data.close_1m[i]),
                                     self._sl_arr[i], self._tp_arr[i], "jj_short")
        return None

    def _build_order(
        self, direction: int, entry: float,
        sl_dist: float, tp_dist: float, reason: str,
    ) -> Optional[Order]:
        if sl_dist <= 0.0:
            return None
        sl_price = round(entry - direction * sl_dist, 2)
        tp_price = round(entry + direction * tp_dist, 2)

        # Fixed sizing by SL tier (NQ minis)
        if sl_dist <= 16.5:    # 66 ticks
            contracts = 3
        elif sl_dist <= 25.0:  # 100 ticks
            contracts = 2
        else:                  # 200 ticks
            contracts = 1

        return Order(
            direction    = direction,
            order_type   = OrderType.MARKET,
            size_type    = SizeType.CONTRACTS,
            size_value   = float(contracts),
            sl_price     = sl_price,
            tp_price     = tp_price,
            trade_reason = reason,
        )

    def on_fill(self, position: OpenPosition, data: MarketData, bar_index: int) -> None:
        position.set_initial_sl_tp(position.sl_price, position.tp_price)
        if self.session_config:
            sess = self._active_session(bar_index)
            if sess is not None:
                self._sess_trade_count[sess] = self._sess_trade_count.get(sess, 0) + 1
                sl_dist = abs(position.entry_price - position.sl_price)
                tp_dist = abs(position.tp_price   - position.entry_price)
                self._pending_trade = {
                    "session":   sess,
                    "entry_bar": bar_index,
                    "direction": position.direction,
                    "sl_price":  position.sl_price,
                    "tp_price":  position.tp_price,
                    "sl_dist":   sl_dist,
                    "tp_dist":   tp_dist,
                }

    def manage_position(
        self, data: MarketData, i: int, position: OpenPosition,
    ) -> Optional[PositionUpdate]:
        return None
