"""
Session Open Mean Reversion & Momentum Strategy (NQ Futures)
Spec: Strategy_spec2.md v1.1+

Three sessions traded per day:
  Asia   20:00-22:00 ET  (RC = 19:59 candle)
  London 03:00-05:00 ET  (RC = 02:59 candle)
  NY     09:30-11:00 ET  (RC = 09:29 candle)

Entry: displacement candle at bar close (MARKET order), or forced entry
       after `force_entry_min` bars with no displacement signal.
Exit:  SL/TP hit, or force-close at last bar of session window.

New parameters:
  force_entry_min  (int, default 0 = disabled)
      After this many session bars with no displacement signal, force an
      entry in the direction given by force_entry_mode.
  force_entry_mode ('momentum' | 'reversion', default 'momentum')
      'momentum' : enter in the direction cl has moved vs true_price.
      'reversion': enter against the deviation (fade back to true_price).
  window_extensions (dict, default {})
      Per-session duration overrides in minutes from session start.
      e.g. {'NY': 180} extends NY from 90 min to 180 min (to 12:00 ET).
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

try:
    from numba import njit as _numba_njit
    _njit = lambda fn: _numba_njit(fn, cache=True)
except ImportError:
    _njit = lambda fn: fn


# ---------------------------------------------------------------------------
# Numba-accelerated Wilder ATR for the full bar series
# ---------------------------------------------------------------------------

@_njit
def _wilder_atr_full(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    n = len(close)
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


# ---------------------------------------------------------------------------
# Session table: (name, rc_min, win_start_min, win_end_min)
# ---------------------------------------------------------------------------
_SESSIONS: tuple = (
    ('Asia',   19 * 60 + 59, 20 * 60,      22 * 60),   # 19:59 / 20:00-22:00
    ('London', 2  * 60 + 59, 3  * 60,       5 * 60),   # 02:59 / 03:00-05:00
    ('NY',     9  * 60 + 29, 9  * 60 + 30, 11 * 60),   # 09:29 / 09:30-11:00
)
_N  = 3
_ASIA = 0

POINT_VALUE = 20.0

# Default session durations in minutes (for window_extensions reference)
_DEFAULT_DURATIONS = {
    'Asia':   22 * 60 - 20 * 60,   # 120 min
    'London':  5 * 60 -  3 * 60,   # 120 min
    'NY':     11 * 60 - (9 * 60 + 30),  # 90 min
}


def _build_sessions(window_extensions: dict) -> tuple:
    """Return session table with optional per-session duration overrides."""
    if not window_extensions:
        return _SESSIONS
    sessions = []
    for (name, rc_min, win_start, win_end) in _SESSIONS:
        if name in window_extensions:
            win_end = win_start + int(window_extensions[name])
        sessions.append((name, rc_min, win_start, win_end))
    return tuple(sessions)


def _sessions_to_trading_hours(sessions: tuple) -> list:
    """Convert session table to (start_time, end_time) list for trading_hours."""
    hours = []
    for (_, _, win_start, win_end) in sessions:
        end_min = win_end - 1
        h_s, m_s = win_start // 60, win_start % 60
        h_e, m_e = end_min // 60,   end_min % 60
        h_e = min(h_e, 23); m_e = min(m_e, 59) if h_e < 23 else 59
        hours.append((_time(h_s, m_s), _time(h_e, m_e)))
    return hours


class SessionMeanRevStrategy(BaseStrategy):
    """
    Session Open Mean Reversion & Momentum (spec v1.1+).
    """

    trading_hours = [
        (_time(20, 0), _time(21, 59)),  # Asia   20:00-22:00 ET
        (_time(3, 0),  _time(4, 59)),   # London 03:00-05:00 ET
        (_time(9, 30), _time(10, 59)),  # NY     09:30-11:00 ET
    ]
    min_lookback  = 20

    def __init__(self, params: dict = None) -> None:
        super().__init__(params)
        p = params or {}

        self.atr_period        = int(p.get('atr_period',       14))
        self.wick_threshold    = float(p.get('wick_threshold',  0.15))
        self.rr_ratio          = float(p.get('rr_ratio',        1.5))
        self.sl_atr_mult       = float(p.get('sl_atr_multiplier', 1.0))
        self.risk_per_trade    = float(p.get('risk_per_trade',  0.01))
        self.equity_mode       = str(p.get('equity_mode',       'dynamic'))
        self.starting_equity   = float(p.get('starting_equity', 100_000))
        self.point_value       = float(p.get('point_value',     POINT_VALUE))
        self.require_bos        = bool(p.get('require_bos',         False))
        self.max_trades_per_day = int(p.get('max_trades_per_day',  3))
        self.disp_min_atr_mult  = float(p.get('disp_min_atr_mult', 0.0))
        self.momentum_only      = bool(p.get('momentum_only',       False))
        self._allowed_sessions  = frozenset(p.get('allowed_sessions', ['Asia', 'London', 'NY']))

        # Force-entry parameters
        self.force_entry_min  = int(p.get('force_entry_min',  0))
        self.force_entry_mode = str(p.get('force_entry_mode', 'momentum'))

        # Window extensions: {'NY': 180} extends NY to 180 min (12:00 ET)
        _win_ext = p.get('window_extensions', {})
        self._sessions_eff = _build_sessions(_win_ext)
        if _win_ext:
            self.trading_hours = _sessions_to_trading_hours(self._sessions_eff)

        # Lazy-computed bar arrays (populated on first call to _setup)
        self._times_min: Optional[np.ndarray] = None
        self._atr_arr:   Optional[np.ndarray] = None

        # ── Per-session state (index 0=Asia, 1=London, 2=NY) ────────────────
        # Reference candle
        self._rc_o   = [0.0] * _N
        self._rc_c   = [0.0] * _N
        self._true_p = [0.0] * _N   # RC.close = true price
        self._body_h = [0.0] * _N   # max(rc_open, rc_close)
        self._body_l = [0.0] * _N   # min(rc_open, rc_close)
        self._is_doji = [True]  * _N
        self._rc_ok  = [False] * _N  # has RC been captured?

        # Window state
        self._in_win = [False] * _N

        # Swing detection: rolling 4-bar buffer per session
        self._sh = [[0.0, 0.0, 0.0, 0.0] for _ in range(_N)]  # highs
        self._sl = [[0.0, 0.0, 0.0, 0.0] for _ in range(_N)]  # lows
        self._sb = [[-1,  -1,  -1,  -1]  for _ in range(_N)]  # bar indices
        self._sess_n = [0] * _N  # bars accumulated in current session

        # BOS state
        self._swing_hi_p = [0.0]   * _N
        self._swing_hi_v = [False] * _N
        self._swing_lo_p = [0.0]   * _N
        self._swing_lo_v = [False] * _N

        # Active position session tracking
        self._pos_sidx    = -1
        self._pos_end_min = -1

        # Daily trade counter (resets at Asia open each evening)
        self._trades_today = 0

        # Per-session trade tracking (resets at each session start)
        self._sess_had_trade = [False] * _N  # any trade taken in current session
        self._force_fired    = [False] * _N  # force-entry used in current session

        # Equity cache: O(1) incremental update
        self._eq_cache   = self.starting_equity
        self._eq_cache_n = 0

    # ── Setup ─────────────────────────────────────────────────────────────────

    def _setup(self, data: MarketData) -> None:
        if self._times_min is not None:
            return
        idx = data.df_1m.index
        t = (idx.hour * 60 + idx.minute).to_numpy(np.int32)
        self._times_min = t
        if data.bar_times_1m_min is None:
            data.bar_times_1m_min = t
        self._atr_arr = _wilder_atr_full(
            data.high_1m, data.low_1m, data.close_1m, self.atr_period
        )

    # ── Current equity (O(1) incremental) ─────────────────────────────────────

    def _current_equity(self) -> float:
        if self.equity_mode == 'fixed':
            return self.starting_equity
        n = len(self.closed_trades)
        while self._eq_cache_n < n:
            self._eq_cache += self.closed_trades[self._eq_cache_n].net_pnl_dollars
            self._eq_cache_n += 1
        return self._eq_cache

    # ── Reference candle capture ───────────────────────────────────────────────

    def _capture_rc(self, data: MarketData, i: int, sidx: int, rc_min: int) -> None:
        limit = max(0, i - 5)
        j = i - 1
        while j >= limit:
            if int(self._times_min[j]) == rc_min:
                o = float(data.open_1m[j])
                c = float(data.close_1m[j])
                self._rc_o[sidx]   = o
                self._rc_c[sidx]   = c
                self._true_p[sidx] = c
                self._body_h[sidx] = max(o, c)
                self._body_l[sidx] = min(o, c)
                self._is_doji[sidx] = (o == c)
                self._rc_ok[sidx]  = True
                return
            j -= 1

    # ── Session reset ──────────────────────────────────────────────────────────

    def _reset_session(self, sidx: int) -> None:
        self._rc_ok[sidx]    = False
        self._in_win[sidx]   = False
        self._sess_n[sidx]   = 0
        self._sh[sidx][:]    = [0.0, 0.0, 0.0, 0.0]
        self._sl[sidx][:]    = [0.0, 0.0, 0.0, 0.0]
        self._sb[sidx][:]    = [-1, -1, -1, -1]
        self._swing_hi_v[sidx] = False
        self._swing_lo_v[sidx] = False
        self._sess_had_trade[sidx] = False
        self._force_fired[sidx]    = False
        if self._pos_sidx == sidx:
            self._pos_sidx    = -1
            self._pos_end_min = -1

    # ── Swing / BOS update ────────────────────────────────────────────────────

    def _update_swing_bos(
        self, sidx: int, bar_i: int, hi: float, lo: float, cl: float,
    ) -> tuple[bool, bool]:
        sh = self._sh[sidx]
        sl = self._sl[sidx]
        sb = self._sb[sidx]
        sh[3], sh[2], sh[1], sh[0] = sh[2], sh[1], sh[0], hi
        sl[3], sl[2], sl[1], sl[0] = sl[2], sl[1], sl[0], lo
        sb[3], sb[2], sb[1], sb[0] = sb[2], sb[1], sb[0], bar_i

        n = self._sess_n[sidx]
        self._sess_n[sidx] = n + 1

        if n >= 2 and sb[3] >= 0:
            cand_h, left_h, right_h = sh[2], sh[3], sh[1]
            cand_l, left_l, right_l = sl[2], sl[3], sl[1]
            if cand_h > left_h and cand_h > right_h:
                self._swing_hi_p[sidx] = cand_h
                self._swing_hi_v[sidx] = True
            if cand_l < left_l and cand_l < right_l:
                self._swing_lo_p[sidx] = cand_l
                self._swing_lo_v[sidx] = True

        bullish_bos = False
        bearish_bos = False
        if self._swing_hi_v[sidx] and cl > self._swing_hi_p[sidx]:
            bullish_bos = True
            self._swing_hi_v[sidx] = False
        if self._swing_lo_v[sidx] and cl < self._swing_lo_p[sidx]:
            bearish_bos = True
            self._swing_lo_v[sidx] = False
        return bullish_bos, bearish_bos

    # ── Displacement candle checks ─────────────────────────────────────────────

    def _bullish_disp(self, op: float, hi: float, lo: float, cl: float, prev_hi: float) -> bool:
        if cl <= prev_hi or cl <= op:
            return False
        rng = hi - lo
        if rng == 0.0:
            return False
        return (hi - cl) / rng <= self.wick_threshold

    def _bearish_disp(self, op: float, hi: float, lo: float, cl: float, prev_lo: float) -> bool:
        if cl >= prev_lo or cl >= op:
            return False
        rng = hi - lo
        if rng == 0.0:
            return False
        return (cl - lo) / rng <= self.wick_threshold

    # ── Order builder helper ───────────────────────────────────────────────────

    def _make_order(
        self, direction: int, cl: float, atr: float,
        sidx: int, win_end: int, trade_reason: str,
    ) -> Optional[Order]:
        """Build a market order with ATR-based SL/TP. Returns None if sizing fails."""
        if direction == 1:
            sl_price = cl - self.sl_atr_mult * atr
            tp_price = cl + self.rr_ratio * self.sl_atr_mult * atr
            sl_dist  = cl - sl_price
        else:
            sl_price = cl + self.sl_atr_mult * atr
            tp_price = cl - self.rr_ratio * self.sl_atr_mult * atr
            sl_dist  = sl_price - cl

        if sl_dist <= 0.0:
            return None

        equity    = self._current_equity()
        contracts = int(equity * self.risk_per_trade / (sl_dist * self.point_value))
        if contracts < 1:
            return None

        self._trades_today        += 1
        self._sess_had_trade[sidx] = True
        self._pos_sidx             = sidx
        self._pos_end_min          = win_end
        return Order(
            direction    = direction,
            order_type   = OrderType.MARKET,
            size_type    = SizeType.CONTRACTS,
            size_value   = float(contracts),
            sl_price     = sl_price,
            tp_price     = tp_price,
            trade_reason = trade_reason,
        )

    # ── Signal generation ──────────────────────────────────────────────────────

    def generate_signals(self, data: MarketData, i: int) -> Optional[Order]:
        self._setup(data)

        t_min  = int(self._times_min[i])
        hi     = float(data.high_1m[i])
        lo     = float(data.low_1m[i])
        cl     = float(data.close_1m[i])
        op_bar = float(data.open_1m[i])

        for sidx in range(_N):
            name, rc_min, win_start, win_end = self._sessions_eff[sidx]
            in_win = win_start <= t_min < win_end
            was_in = self._in_win[sidx]

            # Daily trade counter resets at Asia open
            if sidx == _ASIA and in_win and not was_in:
                self._trades_today = 0

            if name not in self._allowed_sessions:
                continue

            # ── Session transition handling ──────────────────────────────────
            if was_in and not in_win:
                self._reset_session(sidx)
                continue

            if in_win and not was_in:
                self._capture_rc(data, i, sidx, rc_min)
                self._in_win[sidx] = True

            if not in_win:
                continue

            # ── Guard: doji RC or no RC found ───────────────────────────────
            if not self._rc_ok[sidx] or self._is_doji[sidx]:
                continue

            # ── Pre-checks ───────────────────────────────────────────────────
            if self._trades_today >= self.max_trades_per_day:
                self._update_swing_bos(sidx, i, hi, lo, cl)
                continue

            if i < 1:
                continue
            prev_hi = float(data.high_1m[i - 1])
            prev_lo = float(data.low_1m[i - 1])

            # ── Swing / BOS update (runs every in-session bar) ───────────────
            bullish_bos, bearish_bos = self._update_swing_bos(sidx, i, hi, lo, cl)

            atr = float(self._atr_arr[i])
            if atr <= 0.0:
                continue

            true_p = self._true_p[sidx]

            # ── Displacement entry ────────────────────────────────────────────

            # Bullish displacement -> long signal
            if self._bullish_disp(op_bar, hi, lo, cl, prev_hi) and (hi - lo) >= self.disp_min_atr_mult * atr:
                if cl != true_p:
                    if not self.require_bos or bullish_bos:
                        trade_type = 'momentum_long' if cl > true_p else 'reversion_long'
                        skip = (trade_type == 'reversion_long' and cl + self.rr_ratio * self.sl_atr_mult * atr > true_p)
                        if self.momentum_only and trade_type == 'reversion_long':
                            skip = True
                        if not skip:
                            order = self._make_order(1, cl, atr, sidx, win_end, f'{name}_{trade_type}')
                            if order is not None:
                                return order

            # Bearish displacement -> short signal
            if self._bearish_disp(op_bar, hi, lo, cl, prev_lo) and (hi - lo) >= self.disp_min_atr_mult * atr:
                if cl != true_p:
                    if not self.require_bos or bearish_bos:
                        trade_type = 'momentum_short' if cl < true_p else 'reversion_short'
                        skip = (trade_type == 'reversion_short' and cl - self.rr_ratio * self.sl_atr_mult * atr < true_p)
                        if self.momentum_only and trade_type == 'reversion_short':
                            skip = True
                        if not skip:
                            order = self._make_order(-1, cl, atr, sidx, win_end, f'{name}_{trade_type}')
                            if order is not None:
                                return order

            # ── Forced entry after force_entry_min bars with no trade ────────
            if (self.force_entry_min > 0
                    and self._sess_n[sidx] >= self.force_entry_min
                    and not self._force_fired[sidx]
                    and not self._sess_had_trade[sidx]):

                dev = cl - true_p
                if abs(dev) >= 0.01:  # require any meaningful deviation
                    if self.force_entry_mode == 'momentum':
                        direction = 1 if dev > 0 else -1
                    else:  # reversion
                        direction = -1 if dev > 0 else 1

                    self._force_fired[sidx] = True
                    reason = f'{name}_force_{self.force_entry_mode}'
                    order = self._make_order(direction, cl, atr, sidx, win_end, reason)
                    if order is not None:
                        return order

        return None

    # ── Position management ────────────────────────────────────────────────────

    def manage_position(
        self,
        data: MarketData,
        i: int,
        position: OpenPosition,
    ) -> Optional[PositionUpdate]:
        self._setup(data)

        if self._pos_sidx < 0:
            return None

        t_min = int(self._times_min[i])

        if t_min == self._pos_end_min - 1:
            cl = float(data.close_1m[i])
            return PositionUpdate(new_tp_price=cl)

        sidx = self._pos_sidx
        _, _, win_start, win_end = self._sessions_eff[sidx]
        if win_start <= t_min < win_end:
            self._update_swing_bos(
                sidx, i,
                float(data.high_1m[i]),
                float(data.low_1m[i]),
                float(data.close_1m[i]),
            )

        return None
