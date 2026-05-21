"""
IVB v11 fade-leg signal generator.

Spec locked at tasks/ivb_v11_spec.json. Key elements:
  - Consolidation classifier: 30-min ORB range < 0.5 × ATR_20_daily AND no
    break of ORB hi/lo by 10:00 ET
  - Volume profile: fixed ORB-window (9:31-10:00), 1-tick bins, close-only
    volume contribution, 70% VA, POC = max-volume bin (lower-priced wins ties)
  - Long fade: bar.low <= VAL + 1 tick AND bar.close > bar.open (rejection)
  - Short fade: bar.high >= VAH - 1 tick AND bar.close < bar.open (rejection)
  - One fade entry per day, first valid trigger wins
  - Entry: next bar's open
  - SL: opposite VA edge ± 1 tick
  - TP: POC
  - Time-stop: 60 min after entry
  - EOD flat: 15:30 ET

All hot paths vectorised; per-day loop for irreducibly sequential trigger scan.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtest.indicators.atr import atr_daily, broadcast_daily_to_1m
from backtest.ivb.v10_signals import prep_day_views

TICK_SIZE       = 0.25
CONS_ATR_RATIO  = 0.5             # 30-min ORB range < 0.5 × ATR_20
VA_PCT          = 0.70
ATR_WINDOW      = 20
TIME_STOP_MIN   = 60              # 60 min after entry
TRADE_END_MIN   = 15 * 60 + 30    # 15:30 ET EOD


@dataclass
class V11Signals:
    """Vectorised fade-leg signal tape — all fields are (N_signals,) arrays."""
    signal_bars:   np.ndarray   # int, entry bar (next-bar-open from trigger)
    direction:     np.ndarray   # int, +1 long fade / -1 short fade
    sl_price:      np.ndarray   # float, SL price per signal
    tp_price:      np.ndarray   # float, TP price (= POC) per signal
    eod_bar:       np.ndarray   # int, EOD-flat bar index
    time_stop_bar: np.ndarray   # int, time-stop bar index (entry + 60min)
    poc:           np.ndarray   # float, POC price
    vah:           np.ndarray   # float, value-area high
    val:           np.ndarray   # float, value-area low
    trigger_bar:   np.ndarray   # int, source bar that fired the trigger
    day_ord:       np.ndarray   # int, day key per signal


def _compute_orb_vp(
    orb_closes: np.ndarray,
    orb_volumes: np.ndarray,
    va_pct: float = VA_PCT,
) -> tuple[float, float, float] | None:
    """Build close-volume-at-price profile on ORB window. Return (POC, VAH, VAL).

    Bin width = 1 tick (0.25 pt), aligned to tick grid. POC ties → lower-priced bin.
    Returns None if profile is degenerate (single bin or all-zero volume).
    """
    if len(orb_closes) < 5 or orb_volumes.sum() <= 0:
        return None
    # Tick-aligned bin indices
    anchor = orb_closes.min() - 1e-9   # slight under-anchor to avoid edge bugs
    bin_idx = np.round((orb_closes - anchor) / TICK_SIZE).astype(np.int64)
    if bin_idx.max() == bin_idx.min():
        return None   # degenerate single-bin VP
    n_bins = int(bin_idx.max() + 1)
    vol_per_bin = np.bincount(bin_idx, weights=orb_volumes.astype(np.float64), minlength=n_bins)
    if vol_per_bin.sum() <= 0:
        return None

    # POC: first (lowest-priced) bin matching max volume
    poc_bin = int(np.argmax(vol_per_bin))
    poc_price = anchor + (poc_bin + 0.5) * TICK_SIZE

    # 70% value area expansion: outward from POC, take larger neighbor each step
    total = vol_per_bin.sum()
    target = total * va_pct
    lo, hi = poc_bin, poc_bin
    acc = vol_per_bin[poc_bin]
    while acc < target and (lo > 0 or hi < n_bins - 1):
        up_vol = vol_per_bin[hi + 1] if hi < n_bins - 1 else -1.0
        dn_vol = vol_per_bin[lo - 1] if lo > 0           else -1.0
        if up_vol < 0 and dn_vol < 0:
            break
        if up_vol >= dn_vol:
            hi += 1
            acc += up_vol
        else:
            lo -= 1
            acc += dn_vol
    vah = anchor + (hi + 0.5) * TICK_SIZE
    val = anchor + (lo + 0.5) * TICK_SIZE
    return float(poc_price), float(vah), float(val)


def fade_signals(
    bar_idx: pd.DatetimeIndex,
    open_:   np.ndarray,
    close:   np.ndarray,
    high:    np.ndarray,
    low:     np.ndarray,
    volume:  np.ndarray,
) -> V11Signals:
    """Generate IVB v11 fade signals per locked spec.

    Vectorised data prep + per-day loop for first valid VA-edge rejection.
    """
    prep = prep_day_views(bar_idx, high, low)
    n_days = len(prep.valid_day)

    # ATR_20_daily for consolidation classifier
    df_for_atr = pd.DataFrame({
        'open':  open_, 'high':  high, 'low': low, 'close': close,
    }, index=bar_idx)
    atr_d = atr_daily(df_for_atr, window=ATR_WINDOW)
    atr_per_bar = broadcast_daily_to_1m(atr_d, bar_idx)

    sig_bars:  list[int]   = []
    sig_dirs:  list[int]   = []
    sig_sls:   list[float] = []
    sig_tps:   list[float] = []
    sig_eods:  list[int]   = []
    sig_ts:    list[int]   = []
    sig_pocs:  list[float] = []
    sig_vahs:  list[float] = []
    sig_vals:  list[float] = []
    sig_trigs: list[int]   = []
    sig_dords: list[int]   = []

    for d in range(n_days):
        if not prep.valid_day[d]:
            continue
        oh = prep.orb_high_per_day[d]
        ol = prep.orb_low_per_day[d]
        if not (np.isfinite(oh) and np.isfinite(ol) and oh > ol):
            continue
        orb_range = oh - ol

        # Consolidation filter: ORB range < 0.5 × ATR_20_daily
        # ATR is broadcast per bar; use first bar of ORB window
        orb_day_mask = (prep.day_id == d) & prep.orb_mask
        orb_bar_indices = np.where(orb_day_mask)[0]
        if len(orb_bar_indices) < 30:
            continue
        atr_value = float(atr_per_bar[orb_bar_indices[0]])
        if not np.isfinite(atr_value) or atr_value <= 0:
            continue
        if orb_range >= CONS_ATR_RATIO * atr_value:
            continue   # not a consolidation day → skip fade

        # Build ORB-window VP
        orb_closes  = close[orb_day_mask]
        orb_volumes = volume[orb_day_mask]
        vp_result = _compute_orb_vp(orb_closes, orb_volumes)
        if vp_result is None:
            continue
        poc_price, vah, val = vp_result
        # Degenerate case: VAH == POC == VAL → no fade levels
        if vah - val < TICK_SIZE:
            continue

        # Trade window bars (10:01-15:30) for this day
        day_trade_mask = (prep.day_id == d) & prep.trade_mask
        idxs = np.where(day_trade_mask)[0]
        if len(idxs) == 0:
            continue
        t_open  = open_[idxs]
        t_close = close[idxs]
        t_high  = high[idxs]
        t_low   = low[idxs]
        t_min   = prep.bar_min[idxs]

        # Long fade trigger: low touches VAL ± 1 tick AND rejection (close > open)
        long_trig = (t_low <= val + TICK_SIZE) & (t_close > t_open) & (t_high < vah)
        # Short fade trigger: high touches VAH ± 1 tick AND rejection (close < open)
        short_trig = (t_high >= vah - TICK_SIZE) & (t_close < t_open) & (t_low > val)

        long_hits  = np.where(long_trig)[0]
        short_hits = np.where(short_trig)[0]
        first_long_pos  = int(long_hits[0])  if len(long_hits)  > 0 else -1
        first_short_pos = int(short_hits[0]) if len(short_hits) > 0 else -1

        if first_long_pos < 0 and first_short_pos < 0:
            continue
        if first_long_pos < 0:
            trig_pos, direction = first_short_pos, -1
        elif first_short_pos < 0:
            trig_pos, direction = first_long_pos, +1
        else:
            trig_pos, direction = (first_long_pos, +1) if first_long_pos <= first_short_pos else (first_short_pos, -1)

        # Entry: NEXT bar's open after trigger
        entry_pos = trig_pos + 1
        if entry_pos >= len(idxs):
            continue   # trigger on last bar of trade window, no next bar
        sig_bar = int(idxs[entry_pos])
        trig_bar = int(idxs[trig_pos])

        # SL/TP setup
        if direction == 1:
            sl = vah + TICK_SIZE          # SL beyond opposite VA edge
            tp = poc_price                  # TP at POC
        else:
            sl = val - TICK_SIZE
            tp = poc_price

        # Time-stop bar (60 min after entry); cap at EOD bar
        ts_min = int(t_min[entry_pos]) + TIME_STOP_MIN
        # Find first bar in this day where bar_min > ts_min; cap at EOD
        ts_candidates = np.where(t_min > ts_min)[0]
        if len(ts_candidates) > 0:
            ts_bar = int(idxs[ts_candidates[0]])
        else:
            ts_bar = int(prep.eod_bar_per_day[d])
        # Time-stop must not exceed EOD
        ts_bar = min(ts_bar, int(prep.eod_bar_per_day[d]))

        sig_bars.append(sig_bar)
        sig_dirs.append(direction)
        sig_sls.append(float(sl))
        sig_tps.append(float(tp))
        sig_eods.append(int(prep.eod_bar_per_day[d]))
        sig_ts.append(ts_bar)
        sig_pocs.append(poc_price)
        sig_vahs.append(vah)
        sig_vals.append(val)
        sig_trigs.append(trig_bar)
        sig_dords.append(int(prep.day_ord[sig_bar]))

    return V11Signals(
        signal_bars   = np.asarray(sig_bars,  dtype=np.int64),
        direction     = np.asarray(sig_dirs,  dtype=np.int64),
        sl_price      = np.asarray(sig_sls,   dtype=np.float64),
        tp_price      = np.asarray(sig_tps,   dtype=np.float64),
        eod_bar       = np.asarray(sig_eods,  dtype=np.int64),
        time_stop_bar = np.asarray(sig_ts,    dtype=np.int64),
        poc           = np.asarray(sig_pocs,  dtype=np.float64),
        vah           = np.asarray(sig_vahs,  dtype=np.float64),
        val           = np.asarray(sig_vals,  dtype=np.float64),
        trigger_bar   = np.asarray(sig_trigs, dtype=np.int64),
        day_ord       = np.asarray(sig_dords, dtype=np.int64),
    )
