"""
IVB v10 signal generators — pure functions, vectorised data prep + per-day
loop for irreducibly sequential breakout/retest events.

Two generators:
  - orb_bare_signals: first close past ORB-30 extreme in trade window
  - orb_poc_retest_signals: same trigger but with POC-retest filter on entry

Both return dicts of (N_signals,) arrays compatible with
backtest.ivb.proxy_validation.build_proxy_trades_vec:
  - signal_bars, direction
  - sl_dist_per_bar (per-bar array of same length as close; only sig-indexed values matter)
  - tp_dist_per_bar (same shape)
  - max_exit_bar_per_signal (EOD cap)

Per v10 plan (tasks/ivb_v10_plan.md):
  ORB window: 9:31-10:00 ET (close-inclusive, 30 1-min bars)
  Trade window: 10:01-15:30 ET (entries) / hold until 15:30
  Entry trigger: bar-close (first close past ORB extreme after 10:00)
  SL: opposite ORB extreme; TP: 1R AND 2R as separate runs
  EOD flat: 15:30 ET
  POC retest tolerance: FD bin width capped at 0.15 × ORB-range
  Retest timeout: 60 minutes from breakout
  Invalidation: price moves past opposite ORB extreme before retest

Holiday handling: days with <30 ORB bars are excluded entirely.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtest.indicators.value_area import _fd_bin_count

# ── Time constants (NY ET clock-minutes) ────────────────────────────────────
ORB_START_MIN = 9 * 60 + 31      # 9:31 ET, first ORB bar
ORB_END_MIN   = 10 * 60          # 10:00 ET, last ORB bar (close-inclusive)
TRADE_END_MIN = 15 * 60 + 30     # 15:30 ET, EOD flat
RETEST_TIMEOUT_MIN = 60          # 60 min after breakout for POC retest
POC_TOLERANCE_FRAC_CAP = 0.15    # cap FD bin width at 0.15 × ORB-range
TICK_SIZE = 0.25                 # NQ tick


@dataclass
class V10Signals:
    """Vectorised signal tape — all fields are (N_signals,) numpy arrays."""
    signal_bars:  np.ndarray   # int, source bar index of entry
    direction:    np.ndarray   # int, +1 / -1
    sl_dist:      np.ndarray   # float, points (entry to opposite ORB extreme)
    orb_high:     np.ndarray   # float, day's ORB high
    orb_low:      np.ndarray   # float, day's ORB low
    eod_bar:      np.ndarray   # int, source bar index of 15:30 close on that day
    # Optional diagnostics
    day_ord:      np.ndarray = None       # type: ignore[assignment]
    poc:          np.ndarray = None       # type: ignore[assignment]
    retest_bar:   np.ndarray = None       # type: ignore[assignment]  # for Prong B only


# ─────────────────────────────────────────────────────────────────────────────
# Vectorised data prep shared by both generators
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _DayPrep:
    """Per-day grouped views (vectorised one-shot). Day index = 0..N_days-1."""
    day_ord:        np.ndarray   # (N_bars,) day key per bar (e.g. normalised tz-naive ns)
    day_id:         np.ndarray   # (N_bars,) day index 0..N_days-1
    bar_min:        np.ndarray   # (N_bars,) minute-of-day
    rth_mask:       np.ndarray   # (N_bars,) bool, in 9:31-15:30 inclusive
    orb_mask:       np.ndarray   # (N_bars,) bool, in ORB window
    trade_mask:     np.ndarray   # (N_bars,) bool, 10:01-15:30 entry-eligible
    day_first_bar:  np.ndarray   # (N_days,) source idx of first bar that day
    day_last_bar:   np.ndarray   # (N_days,) source idx of last bar that day
    orb_high_per_day: np.ndarray # (N_days,)
    orb_low_per_day:  np.ndarray # (N_days,)
    orb_n_bars:     np.ndarray   # (N_days,) count of ORB bars per day (filter <30)
    eod_bar_per_day:np.ndarray   # (N_days,) source idx of 15:30 bar per day
    valid_day:      np.ndarray   # (N_days,) bool, ORB has 30 bars AND has trade-window bars


def prep_day_views(bar_idx: pd.DatetimeIndex,
                   high: np.ndarray,
                   low: np.ndarray) -> _DayPrep:
    """Vectorised one-shot data prep over the full bar array."""
    n = len(bar_idx)

    # Day ordinal (tz-naive normalised ns since epoch)
    if bar_idx.tz is not None:
        day_ord = bar_idx.tz_localize(None).normalize().asi8
    else:
        day_ord = bar_idx.normalize().asi8

    # Bar-minute of day
    bar_min = (bar_idx.hour.to_numpy(np.int32) * 60 +
               bar_idx.minute.to_numpy(np.int32))

    # Day ID 0..N_days-1 via numpy
    unique_days, day_id = np.unique(day_ord, return_inverse=True)
    n_days = len(unique_days)

    # Masks
    rth_mask   = (bar_min >= ORB_START_MIN) & (bar_min <= TRADE_END_MIN)
    orb_mask   = (bar_min >= ORB_START_MIN) & (bar_min <= ORB_END_MIN)
    trade_mask = (bar_min >  ORB_END_MIN)   & (bar_min <= TRADE_END_MIN)

    # Per-day first / last bar source index
    day_first_bar = np.full(n_days, -1, dtype=np.int64)
    day_last_bar  = np.full(n_days, -1, dtype=np.int64)
    # First occurrences of each day_id
    _, first_idx = np.unique(day_id, return_index=True)
    day_first_bar = first_idx.astype(np.int64)
    # Last occurrence: scan reversed
    rev_idx = (n - 1) - np.unique(day_id[::-1], return_index=True)[1]
    day_last_bar = rev_idx.astype(np.int64)

    # Per-day ORB high/low (NaN for days where ORB bars are absent)
    orb_high_per_day = np.full(n_days, np.nan)
    orb_low_per_day  = np.full(n_days, np.nan)
    orb_n_bars       = np.zeros(n_days, dtype=np.int64)
    if orb_mask.any():
        orb_day_id    = day_id[orb_mask]
        orb_high_vals = high[orb_mask]
        orb_low_vals  = low[orb_mask]
        # np.maximum.at / np.minimum.at — vectorised scatter-reduce
        orb_high_per_day = np.full(n_days, -np.inf)
        orb_low_per_day  = np.full(n_days,  np.inf)
        np.maximum.at(orb_high_per_day, orb_day_id, orb_high_vals)
        np.minimum.at(orb_low_per_day,  orb_day_id, orb_low_vals)
        np.add.at(orb_n_bars, orb_day_id, 1)

    # EOD bar per day: last bar with bar_min <= TRADE_END_MIN
    eod_mask = rth_mask
    eod_bar_per_day = np.full(n_days, -1, dtype=np.int64)
    if eod_mask.any():
        # For each bar with eod_mask, keep the max source idx per day
        eod_ids = day_id[eod_mask]
        eod_src = np.where(eod_mask)[0].astype(np.int64)
        np.maximum.at(eod_bar_per_day, eod_ids, eod_src)

    # Valid days: ORB has 30 bars AND has at least one trade-window bar AND EOD exists
    trade_bars_per_day = np.zeros(n_days, dtype=np.int64)
    if trade_mask.any():
        np.add.at(trade_bars_per_day, day_id[trade_mask], 1)
    valid_day = (orb_n_bars == 30) & (trade_bars_per_day > 0) & (eod_bar_per_day >= 0)

    return _DayPrep(
        day_ord=day_ord, day_id=day_id, bar_min=bar_min,
        rth_mask=rth_mask, orb_mask=orb_mask, trade_mask=trade_mask,
        day_first_bar=day_first_bar, day_last_bar=day_last_bar,
        orb_high_per_day=orb_high_per_day, orb_low_per_day=orb_low_per_day,
        orb_n_bars=orb_n_bars, eod_bar_per_day=eod_bar_per_day,
        valid_day=valid_day,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Prong A — bare ORB-30min breakout
# ─────────────────────────────────────────────────────────────────────────────

def orb_bare_signals(
    bar_idx: pd.DatetimeIndex,
    close:   np.ndarray,
    high:    np.ndarray,
    low:     np.ndarray,
) -> V10Signals:
    """For each valid day, emit the first bar in 10:01-15:30 whose CLOSE
    crosses ORB-high (long) or ORB-low (short). Whichever side fires first
    wins the day. One signal per day max.

    Vectorised data prep + per-day loop (Option R per agent-pair).
    """
    prep = prep_day_views(bar_idx, high, low)
    n_days = len(prep.valid_day)

    sig_bars  : list[int]   = []
    sig_dir   : list[int]   = []
    sig_sld   : list[float] = []
    sig_oh    : list[float] = []
    sig_ol    : list[float] = []
    sig_eod   : list[int]   = []
    sig_dord  : list[int]   = []

    for d in range(n_days):
        if not prep.valid_day[d]:
            continue
        oh = prep.orb_high_per_day[d]
        ol = prep.orb_low_per_day[d]
        if not (np.isfinite(oh) and np.isfinite(ol) and oh > ol):
            continue

        # Bars in this day's trade window
        day_mask = (prep.day_id == d) & prep.trade_mask
        if not day_mask.any():
            continue
        idxs = np.where(day_mask)[0]
        trade_closes = close[idxs]

        # First breakout long (close > oh) or short (close < ol)
        long_hits  = np.where(trade_closes > oh)[0]
        short_hits = np.where(trade_closes < ol)[0]
        first_long_pos  = int(long_hits[0])  if len(long_hits)  > 0 else -1
        first_short_pos = int(short_hits[0]) if len(short_hits) > 0 else -1

        if first_long_pos < 0 and first_short_pos < 0:
            continue
        elif first_long_pos < 0:
            chosen_pos, direction = first_short_pos, -1
        elif first_short_pos < 0:
            chosen_pos, direction = first_long_pos, +1
        else:
            # Both fire on this day; whichever first wins
            if first_long_pos <= first_short_pos:
                chosen_pos, direction = first_long_pos, +1
            else:
                chosen_pos, direction = first_short_pos, -1

        sig_bar = int(idxs[chosen_pos])
        entry_close = float(close[sig_bar])
        if direction == 1:
            sl_dist = entry_close - ol
        else:
            sl_dist = oh - entry_close
        if sl_dist <= 0:
            continue   # degenerate (entry on opposite extreme)

        sig_bars.append(sig_bar)
        sig_dir.append(direction)
        sig_sld.append(float(sl_dist))
        sig_oh.append(float(oh))
        sig_ol.append(float(ol))
        sig_eod.append(int(prep.eod_bar_per_day[d]))
        sig_dord.append(int(prep.day_ord[sig_bar]))

    return V10Signals(
        signal_bars=np.asarray(sig_bars,  dtype=np.int64),
        direction  =np.asarray(sig_dir,   dtype=np.int64),
        sl_dist    =np.asarray(sig_sld,   dtype=np.float64),
        orb_high   =np.asarray(sig_oh,    dtype=np.float64),
        orb_low    =np.asarray(sig_ol,    dtype=np.float64),
        eod_bar    =np.asarray(sig_eod,   dtype=np.int64),
        day_ord    =np.asarray(sig_dord,  dtype=np.int64),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Prong B — ORB + POC retest
# ─────────────────────────────────────────────────────────────────────────────

def orb_poc_retest_signals(
    bar_idx:    pd.DatetimeIndex,
    close:      np.ndarray,
    high:       np.ndarray,
    low:        np.ndarray,
    volume:     np.ndarray,
) -> V10Signals:
    """Same as bare ORB but adds POC-retest filter:
       1. Detect ORB break in trade window (any direction).
       2. Wait for price to touch POC ± tolerance (FD bin width cap 0.15×ORB-range).
       3. Invalidate if price moves past opposite ORB extreme before retest.
       4. Invalidate if 60 min elapse without retest.
       5. Emit on FIRST bar that closes after the retest bar.
    """
    prep = prep_day_views(bar_idx, high, low)
    n_days = len(prep.valid_day)

    sig_bars: list[int]   = []
    sig_dir : list[int]   = []
    sig_sld : list[float] = []
    sig_oh  : list[float] = []
    sig_ol  : list[float] = []
    sig_eod : list[int]   = []
    sig_dord: list[int]   = []
    sig_poc : list[float] = []
    sig_rtb : list[int]   = []

    for d in range(n_days):
        if not prep.valid_day[d]:
            continue
        oh = prep.orb_high_per_day[d]
        ol = prep.orb_low_per_day[d]
        if not (np.isfinite(oh) and np.isfinite(ol) and oh > ol):
            continue
        orb_range = oh - ol

        # ORB-window bars for POC computation
        orb_day_mask = (prep.day_id == d) & prep.orb_mask
        orb_closes   = close[orb_day_mask]
        orb_vols     = volume[orb_day_mask]
        if len(orb_closes) < 30:
            continue
        # Volume-at-price profile (vectorised)
        n_bins = _fd_bin_count(orb_closes, tick_size=TICK_SIZE)
        edges  = np.linspace(orb_closes.min(), orb_closes.max(), n_bins + 1)
        hist, _ = np.histogram(orb_closes, bins=edges, weights=orb_vols)
        if hist.sum() <= 0:
            continue
        poc_idx = int(np.argmax(hist))
        poc     = float(0.5 * (edges[poc_idx] + edges[poc_idx + 1]))
        bin_width = float(edges[1] - edges[0])
        tolerance = min(bin_width, POC_TOLERANCE_FRAC_CAP * orb_range)

        # Trade-window bars
        day_trade_mask = (prep.day_id == d) & prep.trade_mask
        if not day_trade_mask.any():
            continue
        idxs = np.where(day_trade_mask)[0]
        trade_closes = close[idxs]
        trade_highs  = high[idxs]
        trade_lows   = low[idxs]
        trade_mins   = prep.bar_min[idxs]

        # Find first breakout
        long_hits  = np.where(trade_closes > oh)[0]
        short_hits = np.where(trade_closes < ol)[0]
        first_long_pos  = int(long_hits[0])  if len(long_hits)  > 0 else -1
        first_short_pos = int(short_hits[0]) if len(short_hits) > 0 else -1

        if first_long_pos < 0 and first_short_pos < 0:
            continue
        if first_long_pos < 0:
            brk_pos, direction = first_short_pos, -1
        elif first_short_pos < 0:
            brk_pos, direction = first_long_pos, +1
        else:
            brk_pos, direction = (first_long_pos, +1) if first_long_pos <= first_short_pos else (first_short_pos, -1)

        # After breakout: scan for retest, invalidation, or timeout
        brk_min = int(trade_mins[brk_pos])
        timeout_min = brk_min + RETEST_TIMEOUT_MIN

        # Retest candidate bars: after breakout, within timeout, NOT past opposite extreme
        post_brk_slice = slice(brk_pos + 1, len(idxs))
        post_mins  = trade_mins[post_brk_slice]
        post_highs = trade_highs[post_brk_slice]
        post_lows  = trade_lows[post_brk_slice]
        post_closes_arr = trade_closes[post_brk_slice]

        within_timeout = post_mins <= timeout_min

        # Invalidation: price moves past opposite ORB extreme
        if direction == 1:
            # long break: invalidation if low <= ol (price back below ORB-low)
            invalidation = post_lows <= ol
        else:
            # short break: invalidation if high >= oh
            invalidation = post_highs >= oh
        # First invalidation bar pos (within post-break slice)
        invalid_hits = np.where(invalidation & within_timeout)[0]
        invalid_pos = int(invalid_hits[0]) if len(invalid_hits) > 0 else len(post_closes_arr)

        # POC retest: bar where high >= poc - tol AND low <= poc + tol
        # (price range covers the POC tolerance band on this bar)
        retest_touch = (post_highs >= (poc - tolerance)) & (post_lows <= (poc + tolerance))
        valid_retest = retest_touch & within_timeout
        # Only consider retests BEFORE invalidation
        retest_hits = np.where(valid_retest)[0]
        retest_hits = retest_hits[retest_hits < invalid_pos]
        if len(retest_hits) == 0:
            continue
        retest_pos = int(retest_hits[0])

        # Entry: bar AFTER retest (one bar later within the same day; if retest is last bar of day, skip)
        if retest_pos + 1 >= len(post_closes_arr):
            continue
        entry_post_pos = retest_pos + 1
        sig_bar = int(idxs[brk_pos + 1 + entry_post_pos])
        retest_bar_src = int(idxs[brk_pos + 1 + retest_pos])

        entry_close = float(close[sig_bar])
        if direction == 1:
            sl_dist = entry_close - ol
        else:
            sl_dist = oh - entry_close
        if sl_dist <= 0:
            continue

        sig_bars.append(sig_bar)
        sig_dir.append(direction)
        sig_sld.append(float(sl_dist))
        sig_oh.append(float(oh))
        sig_ol.append(float(ol))
        sig_eod.append(int(prep.eod_bar_per_day[d]))
        sig_dord.append(int(prep.day_ord[sig_bar]))
        sig_poc.append(poc)
        sig_rtb.append(retest_bar_src)

    return V10Signals(
        signal_bars=np.asarray(sig_bars, dtype=np.int64),
        direction  =np.asarray(sig_dir,  dtype=np.int64),
        sl_dist    =np.asarray(sig_sld,  dtype=np.float64),
        orb_high   =np.asarray(sig_oh,   dtype=np.float64),
        orb_low    =np.asarray(sig_ol,   dtype=np.float64),
        eod_bar    =np.asarray(sig_eod,  dtype=np.int64),
        day_ord    =np.asarray(sig_dord, dtype=np.int64),
        poc        =np.asarray(sig_poc,  dtype=np.float64),
        retest_bar =np.asarray(sig_rtb,  dtype=np.int64),
    )
