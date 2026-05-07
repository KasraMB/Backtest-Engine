"""
AnchoredMeanReversion Strategy v2 (NQ Futures, 1-min bars)
Full feature parity with spec.md PineScript indicator.

Changes vs v1:
  - Direction filter: only mean-rev trades after against_fair_mins (Pine default=15)
  - BOS direction fix: long requires brokeHigh, short requires brokeLow
  - Window fix: < window_mins (was <=, off-by-one)
  - Per-session window: window_mins_per_session dict
  - displacement_style: 'Upper Wick' | 'Marubozu'
  - threshold_uw renamed from threshold; threshold_mw added for Marubozu
  - skip_first_mins, filter_tp_beyond_fair, move_sl_to_entry, link1400to930
"""
from __future__ import annotations
from typing import Optional

import numpy as np

from backtest.data.market_data import MarketData
from backtest.strategy.base import BaseStrategy
from backtest.strategy.enums import OrderType, SizeType
from backtest.strategy.order import Order
from backtest.strategy.update import OpenPosition, PositionUpdate

POINT_VALUE = 20.0
TICK        = 0.25

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
    threshold_uw:   float,   # Upper Wick close-position threshold (default 0.8)
    threshold_mw:   float,   # Marubozu wick/range ratio threshold (default 0.15)
    displacement_style: int, # 0=Upper Wick, 1=Marubozu
    atr_mult:       float,
    session_opens:  np.ndarray,   # int32, session open times in minutes
    window_mins_arr: np.ndarray,  # int32, per-session window durations
    link_s2_to_s1:  int,          # 1=link session[1]+ fair price to session[0]'s
    against_fair_mins: int,       # 0..N — within this many bars, any direction OK
    skip_first_mins:   int,       # skip first N bars of each session window
    filter_tp_beyond_fair: int,   # 1=require TP to extend past fair by pct
    tp_beyond_fair_pct:    float, # 0–100, portion of TP range that must be past fair
    require_bos:   int,          # 1=require BOS for signal (Pine default), 0=allow without BOS
    atr_sl_low:    float,
    atr_sl_high:   float,
    sl_ticks_low:  int,
    sl_ticks_mid:  int,
    sl_ticks_high: int,
    tp_multiple:   float,
) -> tuple:
    MAX_SW = 200
    N      = len(closes)
    long_sig  = np.zeros(N, dtype=np.bool_)
    short_sig = np.zeros(N, dtype=np.bool_)
    sl_arr    = np.zeros(N, dtype=np.float64)
    tp_arr    = np.zeros(N, dtype=np.float64)

    sh_prices = np.zeros(MAX_SW, dtype=np.float64)
    sh_count  = 0
    sl_prices = np.zeros(MAX_SW, dtype=np.float64)
    sl_count  = 0

    n_sess    = len(session_opens)
    last_open = np.full(n_sess, -1, dtype=np.int64)
    last_fair = np.zeros(n_sess, dtype=np.float64)
    has_fair  = np.zeros(n_sess, dtype=np.int32)   # 0=not recorded, 1=recorded

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
                if link_s2_to_s1 and k > 0 and has_fair[0]:
                    last_fair[k] = last_fair[0]   # link to first session's fair price
                else:
                    last_fair[k] = opens[i]
                has_fair[k] = 1

        # ── Within window + active fair price + elapsed ────────────────────
        within      = False
        active_fair = 0.0
        fair_valid  = 0
        elapsed     = 0
        for k in range(n_sess):
            if has_fair[k] and last_open[k] >= 0:
                diff = i - last_open[k]
                if diff < window_mins_arr[k]:        # < not <= (spec: nowHHMM < endHHMM)
                    within      = True
                    active_fair = last_fair[k]
                    fair_valid  = 1
                    elapsed     = diff
                    break

        # ── SL / TP ────────────────────────────────────────────────────────
        if av < atr_sl_low:
            sl_tks = sl_ticks_low
        elif av < atr_sl_high:
            sl_tks = sl_ticks_mid
        else:
            sl_tks = sl_ticks_high
        sl_d = sl_tks * TICK
        tp_d = sl_d * tp_multiple
        sl_arr[i] = sl_d
        tp_arr[i] = tp_d

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
                w = 0
                for j in range(sh_count):
                    if sh_prices[j] >= new_h:
                        sh_prices[w] = sh_prices[j]; w += 1
                sh_count = w
                if sh_count < MAX_SW:
                    sh_prices[sh_count] = new_h; sh_count += 1

            is_ll = True
            for k in range(1, n + 1):
                if lows[si] >= lows[si - k] or lows[si] >= lows[si + k]:
                    is_ll = False
                    break
            if is_ll:
                new_l = lows[si]
                w = 0
                for j in range(sl_count):
                    if sl_prices[j] <= new_l:
                        sl_prices[w] = sl_prices[j]; w += 1
                sl_count = w
                if sl_count < MAX_SW:
                    sl_prices[sl_count] = new_l; sl_count += 1

        # ── BOS: detect and compact away broken levels ─────────────────────
        broke_high = False
        for j in range(sh_count):
            if cl > sh_prices[j]:
                broke_high = True; break
        if broke_high:
            w = 0
            for j in range(sh_count):
                if sh_prices[j] >= cl:
                    sh_prices[w] = sh_prices[j]; w += 1
            sh_count = w

        broke_low = False
        for j in range(sl_count):
            if cl < sl_prices[j]:
                broke_low = True; break
        if broke_low:
            w = 0
            for j in range(sl_count):
                if sl_prices[j] <= cl:
                    sl_prices[w] = sl_prices[j]; w += 1
            sl_count = w

        # ── Gate: skip if not in window or ATR not warm ────────────────────
        if not within or av == 0.0:
            continue

        # ── Skip first N minutes of window ────────────────────────────────
        if elapsed < skip_first_mins:
            continue

        # ── Displacement candle ────────────────────────────────────────────
        candle_range = hi - lo
        range_safe   = candle_range if candle_range > 0.0 else 0.01
        atr_ok = (candle_range >= av * atr_mult) if atr_mult > 0.0 else True

        bull_range = hi - op
        bull_pos   = (cl - op) / bull_range if bull_range > 0.0 else 0.0
        bull_cond  = (cl >= op) and (bull_pos >= threshold_uw) and atr_ok

        bear_range = op - lo
        bear_pos   = (op - cl) / bear_range if bear_range > 0.0 else 0.0
        bear_cond  = (cl < op)  and (bear_pos >= threshold_uw) and atr_ok

        if displacement_style == 1:  # Marubozu: also require low total-wick ratio
            body_top    = cl if cl >= op else op
            body_bottom = op if op <= cl else cl
            upper_wick  = hi - body_top
            lower_wick  = body_bottom - lo
            total_wick  = upper_wick + lower_wick
            is_maru     = (total_wick / range_safe) <= threshold_mw
            bull_cond   = bull_cond and is_maru
            bear_cond   = bear_cond and is_maru

        # ── Direction filter (Pine: longDirectionOK / shortDirectionOK) ───
        in_first     = elapsed <= against_fair_mins
        long_dir_ok  = (fair_valid == 1 and cl < active_fair) or in_first
        short_dir_ok = (fair_valid == 1 and cl > active_fair) or in_first

        # ── TP-beyond-fair filter ──────────────────────────────────────────
        long_tp_ok  = True
        short_tp_ok = True
        if filter_tp_beyond_fair and fair_valid:
            if cl < active_fair and tp_d > 0.0:
                long_tp  = cl + tp_d
                if long_tp > active_fair:
                    ratio = (active_fair - cl) / tp_d
                    long_tp_ok = ratio >= (tp_beyond_fair_pct / 100.0)
            if cl > active_fair and tp_d > 0.0:
                short_tp = cl - tp_d
                if short_tp < active_fair:
                    ratio = (cl - active_fair) / tp_d
                    short_tp_ok = ratio >= (tp_beyond_fair_pct / 100.0)

        # ── Signals: BOS condition gated by require_bos ───────────────────
        bos_long  = (broke_high or not require_bos)
        bos_short = (broke_low  or not require_bos)
        if bull_cond and bos_long and long_dir_ok and long_tp_ok:
            long_sig[i]  = True
        elif bear_cond and bos_short and short_dir_ok and short_tp_ok:
            short_sig[i] = True

    return long_sig, short_sig, sl_arr, tp_arr


class AnchoredMeanReversionStrategy(BaseStrategy):
    """
    Displacement + BOS + session-window momentum/mean-rev strategy.
    Full feature parity with spec.md PineScript indicator.
    """

    trading_hours = None
    min_lookback  = 30

    def __init__(self, params: dict = None) -> None:
        super().__init__(params)
        p = params or {}

        self.n               = int(p.get("swing_periods",     1))
        self.threshold_uw    = float(p.get("threshold_uw",    0.8))
        self.threshold_mw    = float(p.get("threshold_mw",    0.15))
        self.displacement_style = str(p.get("displacement_style", "Upper Wick"))
        self.atr_len         = int(p.get("atr_len",           14))
        self.atr_mult        = float(p.get("atr_mult",        0.0))
        self.tp_multiple     = float(p.get("tp_multiple",     1.5))
        self.atr_sl_low      = float(p.get("atr_sl_low",      7.0))
        self.atr_sl_high     = float(p.get("atr_sl_high",     20.0))
        self.sl_ticks_low    = int(p.get("sl_ticks_low",      66))
        self.sl_ticks_mid    = int(p.get("sl_ticks_mid",      100))
        self.sl_ticks_high   = int(p.get("sl_ticks_high",     200))
        self.window_mins     = int(p.get("window_mins",        90))
        self.sessions        = list(p.get("sessions",         ["930", "1400"]))
        self.window_mins_per_session: dict = dict(p.get("window_mins_per_session", {}))

        # Spec-parity params
        self.require_bos          = bool(p.get("require_bos",          True))
        self.against_fair_mins    = int(p.get("against_fair_mins",    15))
        self.skip_first_mins      = int(p.get("skip_first_mins",      0))
        self.filter_tp_beyond_fair = bool(p.get("filter_tp_beyond_fair", False))
        self.tp_beyond_fair_pct   = float(p.get("tp_beyond_fair_pct",  50.0))
        self.move_sl_to_entry     = bool(p.get("move_sl_to_entry",     False))
        self.link1400to930        = bool(p.get("link1400to930",        False))

        # Per-session trade constraints (unchanged from v1)
        self.session_config: dict = dict(p.get("session_config", {}))

        self._times_min:  Optional[np.ndarray] = None
        self._long_sig:   Optional[np.ndarray] = None
        self._short_sig:  Optional[np.ndarray] = None
        self._sl_arr:     Optional[np.ndarray] = None
        self._tp_arr:     Optional[np.ndarray] = None
        self._sig_mask:   Optional[np.ndarray] = None

        self._sess_open_bars:        dict[str, np.ndarray] = {}
        self._sess_current_open_bar: dict[str, int]        = {}
        self._sess_trade_count:      dict[str, int]        = {}
        self._sess_net_pnl:          dict[str, float]      = {}
        self._pending_trade:         Optional[dict]        = None

        # move_sl_to_entry runtime state
        self._pos_mean_rev:          bool  = False
        self._pos_fair_price:        float = 0.0
        self._pos_sl_moved_to_entry: bool  = False
        self._pos_open_bar:          int   = -1

    def _setup(self, data: MarketData) -> None:
        if self._long_sig is not None:
            return

        idx = data.df_1m.index
        if data.bar_times_1m_min is not None:
            self._times_min = data.bar_times_1m_min
        else:
            self._times_min = (idx.hour * 60 + idx.minute).to_numpy(np.int32)

        atr = _wilder_atr(data.high_1m, data.low_1m, data.close_1m, self.atr_len)

        session_opens = np.array(
            [_SESSION_MAP[s] for s in self.sessions if s in _SESSION_MAP],
            dtype=np.int32,
        )
        window_mins_arr = np.array(
            [self.window_mins_per_session.get(s, self.window_mins)
             for s in self.sessions if s in _SESSION_MAP],
            dtype=np.int32,
        )

        n_sess       = len(session_opens)
        link_s2_to_s1 = 1 if (self.link1400to930 and n_sess >= 2) else 0
        disp_style   = 0 if self.displacement_style == "Upper Wick" else 1

        self._long_sig, self._short_sig, self._sl_arr, self._tp_arr = _compute_signals(
            data.high_1m, data.low_1m, data.close_1m, data.open_1m,
            self._times_min.astype(np.int32), atr,
            self.n,
            self.threshold_uw, self.threshold_mw, disp_style,
            self.atr_mult,
            session_opens, window_mins_arr,
            link_s2_to_s1,
            self.against_fair_mins, self.skip_first_mins,
            1 if self.filter_tp_beyond_fair else 0, self.tp_beyond_fair_pct,
            1 if self.require_bos else 0,
            self.atr_sl_low, self.atr_sl_high,
            self.sl_ticks_low, self.sl_ticks_mid, self.sl_ticks_high,
            self.tp_multiple,
        )
        self._sig_mask = self._long_sig | self._short_sig

        if self.session_config or self.move_sl_to_entry:
            for s in self.sessions:
                if s in _SESSION_MAP:
                    sm = _SESSION_MAP[s]
                    self._sess_open_bars[s] = np.where(self._times_min == sm)[0]

    def _active_session(self, i: int) -> Optional[str]:
        for s in self.sessions:
            bars = self._sess_open_bars.get(s)
            if bars is None or len(bars) == 0:
                continue
            idx = int(np.searchsorted(bars, i, side="right")) - 1
            if idx >= 0:
                win = self.window_mins_per_session.get(s, self.window_mins)
                if (i - int(bars[idx])) < win:    # < matches kernel fix
                    return s
        return None

    def _check_session_resets(self, i: int) -> None:
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
                self._sess_trade_count[s]      = 0
                self._sess_net_pnl[s]          = 0.0

    def _resolve_pending_pnl(self, data: MarketData, up_to: int) -> float:
        p  = self._pending_trade
        eb = p["entry_bar"]
        d  = p["direction"]
        sl = p["sl_price"]
        tp = p["tp_price"]
        n  = len(data.high_1m)
        for j in range(eb + 1, min(up_to + 1, n)):
            if d == 1:
                if data.low_1m[j]  <= sl: return -p["sl_dist"]
                if data.high_1m[j] >= tp: return  p["tp_dist"]
            else:
                if data.high_1m[j] >= sl: return -p["sl_dist"]
                if data.low_1m[j]  <= tp: return  p["tp_dist"]
        return 0.0

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

        needs_sess = bool(self.session_config) or self.move_sl_to_entry
        if needs_sess:
            self._check_session_resets(i)
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

    def _build_order(self, direction: int, entry: float,
                     sl_dist: float, tp_dist: float, reason: str) -> Optional[Order]:
        if sl_dist <= 0.0:
            return None
        sl_price = round(entry - direction * sl_dist, 2)
        tp_price = round(entry + direction * tp_dist, 2)

        if sl_dist <= 16.5:    # 66 ticks → low tier
            contracts = 3
        elif sl_dist <= 25.0:  # 100 ticks → mid tier
            contracts = 2
        else:                  # 200 ticks → high tier
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

        needs_sess = bool(self.session_config) or self.move_sl_to_entry
        if not needs_sess:
            return

        sess = self._active_session(bar_index)
        if sess is None:
            return

        if self.session_config:
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

        if self.move_sl_to_entry:
            open_bar = self._sess_current_open_bar.get(sess, -1)
            if open_bar >= 0:
                fair = float(data.open_1m[open_bar])
                is_long   = position.direction == 1
                is_mean_rev = (is_long and position.entry_price < fair) or \
                              (not is_long and position.entry_price > fair)
                self._pos_mean_rev          = is_mean_rev
                self._pos_fair_price        = fair
                self._pos_sl_moved_to_entry = False
                self._pos_open_bar          = open_bar
            else:
                self._pos_mean_rev = False

    def manage_position(
        self, data: MarketData, i: int, position: OpenPosition,
    ) -> Optional[PositionUpdate]:
        if (self.move_sl_to_entry
                and self._pos_mean_rev
                and not self._pos_sl_moved_to_entry):
            # Only move SL after the against-fair window has elapsed (Pine spec)
            if (i - self._pos_open_bar) > self.against_fair_mins:
                fair    = self._pos_fair_price
                is_long = position.direction == 1
                hit = (is_long and data.high_1m[i] >= fair) or \
                      (not is_long and data.low_1m[i] <= fair)
                if hit:
                    self._pos_sl_moved_to_entry = True
                    return PositionUpdate(new_sl_price=position.entry_price)
        return None
