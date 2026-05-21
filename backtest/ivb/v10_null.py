"""
IVB v10 time-of-day matched random-entry null.

For each real trade fired at clock-minute t with direction d:
  - Sample n_perm bootstrap bars from "post-ORB entry-window bars on any day at
    clock-minute t"
  - Use SAME direction d
  - Recompute SL/TP/EOD from the bootstrap day's ORB-range and EOD bar
  - Compute forward R for each bootstrap entry
  - Average across the n_trades real-trade buckets → 1 null mean per perm row
  - Repeat n_perm times → (n_perm,) null distribution

p-value = mean(null_means >= observed_mean).

Vectorisation: Option α from agent-pair protocol — single flattened call to
build_proxy_trades_vec per perm chunk (500 perms = 100k entries, ~144 MB).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.ivb.proxy_validation import build_proxy_trades_vec
from backtest.ivb.v10_signals import (
    ORB_END_MIN, TRADE_END_MIN, prep_day_views, V10Signals,
)


def build_null_means(
    real_signals: V10Signals,
    bar_idx:      pd.DatetimeIndex,
    close:        np.ndarray,
    high:         np.ndarray,
    low:          np.ndarray,
    tp_R_multiple: float = 1.0,
    n_perm:       int   = 1000,
    chunk_perms:  int   = 500,
    horizon_bars: int   = 360,
    seed:         int   = 2026,
) -> np.ndarray:
    """Return (n_perm,) null distribution of mean-R-across-trades under
    time-of-day-matched random entry.

    `tp_R_multiple` = 1.0 → TP at 1R; = 2.0 → TP at 2R. Call once per TP rule.

    Vectorised single-tile bootstrap; chunked along the perm axis to cap memory.
    """
    if len(real_signals.signal_bars) < 1:
        return np.array([])

    rng = np.random.default_rng(seed)
    n_trades = len(real_signals.signal_bars)

    # ── Pre-compute per-day lookups (vectorised one-shot) ────────────────────
    prep = prep_day_views(bar_idx, high, low)
    day_id_of_bar = prep.day_id                                      # (N_bars,)
    bar_min_of_bar = prep.bar_min                                    # (N_bars,)

    # Post-ORB entry-window bars (the SAMPLING POOL): bars where day is valid
    # (ORB present, EOD present) AND bar is in 10:01-15:30 (trade window).
    valid_day_per_bar = prep.valid_day[day_id_of_bar]
    pool_mask = prep.trade_mask & valid_day_per_bar                  # (N_bars,) bool
    pool_idx = np.where(pool_mask)[0]                                # source-bar indices in pool
    pool_min = bar_min_of_bar[pool_idx]                              # clock-minute per pool bar

    # Per-day arrays for SL/EOD lookup
    orb_high_per_day = prep.orb_high_per_day                         # (N_days,)
    orb_low_per_day  = prep.orb_low_per_day                          # (N_days,)
    eod_per_day      = prep.eod_bar_per_day                          # (N_days,)

    # Build bucket: clock_minute → indices into pool_idx
    # Min/max of the pool's clock-minutes (should be 10:01..15:30)
    min_lo = ORB_END_MIN + 1
    min_hi = TRADE_END_MIN
    n_buckets = min_hi - min_lo + 1
    bucket_of_pool = pool_min - min_lo                               # (n_pool,) int in [0, n_buckets-1]
    # For each bucket b, the pool-indices array bucket_indices[b] (variable length)
    bucket_indices: list[np.ndarray] = [None] * n_buckets            # type: ignore[list-item]
    for b in range(n_buckets):
        bucket_indices[b] = np.where(bucket_of_pool == b)[0]

    # ── Per-real-trade bucket lookup ─────────────────────────────────────────
    real_sig_min = bar_min_of_bar[real_signals.signal_bars]          # (n_trades,)
    real_bucket  = real_sig_min - min_lo
    # Defensive: if any real trade is outside legal window, this is a bug upstream
    if (real_bucket < 0).any() or (real_bucket >= n_buckets).any():
        raise ValueError("real trade clock-minutes outside legal entry window")
    # Pre-fetch pool-index arrays for each real trade's bucket
    real_bucket_pools = [bucket_indices[b] for b in real_bucket]

    # ── Chunked perm processing ──────────────────────────────────────────────
    null_means = np.empty(n_perm, dtype=np.float64)
    for chunk_start in range(0, n_perm, chunk_perms):
        chunk_end = min(chunk_start + chunk_perms, n_perm)
        n_c = chunk_end - chunk_start

        # Sample (n_c, n_trades) bootstrap bar indices.
        # For each trade i, draw n_c bars from real_bucket_pools[i] with replacement.
        # Implementation: per-trade rng.integers, vectorised across perms.
        boot_sigs = np.empty((n_c, n_trades), dtype=np.int64)
        for i in range(n_trades):
            pool_for_trade = real_bucket_pools[i]
            if len(pool_for_trade) == 0:
                boot_sigs[:, i] = real_signals.signal_bars[i]   # degenerate fallback
            else:
                draws = rng.integers(0, len(pool_for_trade), size=n_c)
                boot_sigs[:, i] = pool_idx[pool_for_trade[draws]]

        # Flatten to (n_c * n_trades,)
        flat_sigs = boot_sigs.reshape(-1)

        # Per-bootstrap-bar: look up day_id → ORB high/low → compute SL/TP
        boot_day_id = day_id_of_bar[flat_sigs]
        boot_oh = orb_high_per_day[boot_day_id]
        boot_ol = orb_low_per_day[boot_day_id]
        boot_eod = eod_per_day[boot_day_id]
        # Direction broadcast across perms
        boot_dir = np.broadcast_to(real_signals.direction, (n_c, n_trades)).reshape(-1)

        boot_entry_close = close[flat_sigs]
        # SL distance = entry - opposite extreme
        boot_sl_dist = np.where(
            boot_dir == 1,
            boot_entry_close - boot_ol,
            boot_oh - boot_entry_close,
        )
        # Guard non-positive SL
        boot_sl_dist = np.where(boot_sl_dist > 0, boot_sl_dist, np.nan)

        # Build full-length per-bar arrays for the builder
        # The builder reads sl_dist_per_bar[sig] and tp_dist_per_bar[sig], so we
        # need arrays of length N_bars. Strategy: build sparse-fill arrays where
        # only the bootstrap bars matter.
        # Faster: write to scratch arrays at flat_sigs positions.
        # But two bootstrap entries on the SAME bar might want different sl_dist
        # because direction differs. That can't happen in this null
        # (direction is matched to real trade, but different real trades might
        # share a bootstrap bar with different directions across perms).
        # Solution: split into long-bar and short-bar passes, OR pre-compute
        # per-signal arrays. The cleanest approach: use a different builder
        # call structure — make sl_dist a per-SIGNAL array.

        # Workaround: build per-bar arrays once per direction sign.
        # Long-direction bootstrap entries:
        long_mask = boot_dir == 1
        short_mask = ~long_mask

        n_bars_total = len(close)
        chunk_R = np.full(n_c * n_trades, np.nan, dtype=np.float64)

        for d_sign, mask in [(+1, long_mask), (-1, short_mask)]:
            if not mask.any():
                continue
            sub_sigs = flat_sigs[mask]
            sub_sl   = boot_sl_dist[mask]
            sub_eod  = boot_eod[mask]
            sub_tp   = sub_sl * tp_R_multiple

            # Build per-bar arrays (only sub_sigs positions matter; rest = 0)
            sl_per_bar = np.zeros(n_bars_total, dtype=np.float64)
            tp_per_bar = np.zeros(n_bars_total, dtype=np.float64)
            sl_per_bar[sub_sigs] = sub_sl
            tp_per_bar[sub_sigs] = sub_tp

            # max_exit_bar_per_signal is (n_sub,)
            trades = build_proxy_trades_vec(
                signal_bars=sub_sigs,
                close=close, high=high, low=low,
                sl_dist_per_bar=sl_per_bar,
                direction=d_sign,
                horizon_bars=horizon_bars,
                tp_dist_per_bar=tp_per_bar,
                max_exit_bar_per_signal=sub_eod,
            )
            # Place R values back into chunk_R at original positions
            # Note: build_proxy_trades_vec may DROP signals too close to end-of-array;
            # we need to track which ones survived.
            # Builder returns entry_bar (which is the post-valid-filter sub_sigs).
            # Map back via searchsorted on sub_sigs.
            valid_mask_in_sub = np.isin(sub_sigs, trades.entry_bar)
            chunk_R[np.where(mask)[0][valid_mask_in_sub]] = trades.R

        # Reshape to (n_c, n_trades) and average across trades
        chunk_R = chunk_R.reshape(n_c, n_trades)
        # NaN-safe mean
        with np.errstate(invalid='ignore'):
            null_means[chunk_start:chunk_end] = np.nanmean(chunk_R, axis=1)

    return null_means
