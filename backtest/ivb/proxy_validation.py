"""
Stage 3 proxy validation: forward-R trade builder, permutation null,
economic threshold check, and VXN EV-separation test.

All hot loops are vectorised:
  - build_proxy_trades_vec: pre-allocated (N_sig, horizon) fancy-indexed view,
    np.maximum.accumulate / np.minimum.accumulate per row for MFE/MAE, single
    vectorised SL-hit detection.
  - permutation_null_R: single (n_perm, N) sign-flip matrix, one mean over axis=1.
  - vxn_ev_separation_test: single (n_perm, N_days) label-shuffle matrix.

Pre-registered constants for Stage 3 are sourced from ivb/constants.py at the
caller layer; this module only takes them as args so it stays test-clean.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Trade builder
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ProxyTrades:
    """Vectorised trade tape — all fields are (N_signals,) numpy arrays."""
    entry_bar:   np.ndarray   # int, source bar index
    exit_bar:    np.ndarray   # int, bar of first SL-hit or end-of-horizon
    R:           np.ndarray   # float, signed R = pnl_pts / sl_dist
    mfe_R:       np.ndarray   # float, max favourable excursion in R units
    mae_R:       np.ndarray   # float, max adverse excursion in R units (>=0)
    hit_sl:      np.ndarray   # bool, True if SL was breached within horizon
    sl_dist_pts: np.ndarray   # float, SL distance per signal (γ × ATR_20)
    hit_tp:      np.ndarray = None   # type: ignore[assignment]  # bool, True if TP fired (NaN-equivalent / False if no TP path)
    hit_eod:     np.ndarray = None   # type: ignore[assignment]  # bool, True if exit was forced by EOD cap


def build_proxy_trades_vec(
    signal_bars: np.ndarray,        # int, (N_sig,) bar indices where signal fired
    close:       np.ndarray,        # (N_bars,)
    high:        np.ndarray,        # (N_bars,)
    low:         np.ndarray,        # (N_bars,)
    sl_dist_per_bar: np.ndarray,    # (N_bars,) γ × ATR_20_daily, per source bar
    direction:   int,               # +1 long, -1 short
    horizon_bars: int = 60,
    tp_dist_per_bar: 'np.ndarray | None' = None,    # (N_bars,) optional TP distance per signal bar
    max_exit_bar_per_signal: 'np.ndarray | None' = None,  # (N_sig,) optional EOD-cap index per signal
) -> ProxyTrades:
    """Walk forward `horizon_bars` from each signal; compute R, MFE, MAE.

    SL-first within bar (per v9-final patch 9 / v10 micro-spec 9): if SL is hit
    AND TP would also be hit on the same bar, SL wins (conservative).

    Vectorisation:
      1. Drop signals too close to end of array (no full horizon)
      2. Build (N_sig, horizon) index matrix via broadcasting
      3. Fancy-index high/low/close — produces (N_sig, horizon) panels
      4. Per-row running max/min for MFE/MAE
      5. Detect SL-hit via single elementwise compare + argmax(any) per row
      6. Detect TP-hit symmetrically (when tp_dist_per_bar provided)
      7. EOD cap (when max_exit_bar_per_signal provided): bars at/after the cap
         are excluded from SL/TP detection; exit price defaults to bar-cap close
      8. Exit precedence: SL-first (ties → SL), then TP, then EOD/horizon-end

    Optional args preserve v9 backward compat (Stage 3 calls without them).
    """
    N_bars = len(close)
    if direction not in (-1, 1):
        raise ValueError(f"direction must be +/-1, got {direction}")
    if len(signal_bars) == 0:
        empty = np.array([], dtype=np.int64)
        empty_f = np.array([], dtype=np.float64)
        empty_b = np.array([], dtype=bool)
        return ProxyTrades(empty, empty, empty_f, empty_f, empty_f, empty_b, empty_f)

    # Discard signals that don't have a full horizon
    valid = signal_bars <= (N_bars - 1 - horizon_bars)
    sig = signal_bars[valid].astype(np.int64)
    N_sig = len(sig)
    if N_sig == 0:
        empty = np.array([], dtype=np.int64); empty_f = np.array([], dtype=np.float64); empty_b = np.array([], dtype=bool)
        return ProxyTrades(empty, empty, empty_f, empty_f, empty_f, empty_b, empty_f)

    # Index matrix: (N_sig, horizon+1) covering bars [sig+1, sig+horizon] inclusive
    # (entry is FILLED next bar after signal; first valid eval bar is sig+1)
    offsets = np.arange(1, horizon_bars + 1, dtype=np.int64)        # (H,)
    idx_mat = sig[:, None] + offsets[None, :]                       # (N_sig, H)

    # Entry price = close at signal bar (model: limit/market fill on the signal bar close)
    entry_px  = close[sig].astype(np.float64)                       # (N_sig,)
    sl_dist   = sl_dist_per_bar[sig].astype(np.float64)             # (N_sig,)
    # Guard zero/NaN SL distances
    bad_sl = (~np.isfinite(sl_dist)) | (sl_dist <= 0)
    sl_dist = np.where(bad_sl, np.nan, sl_dist)

    sl_price = entry_px - direction * sl_dist                       # long: entry - dist; short: entry + dist

    # Pull forward panels — single fancy-indexing op each
    panel_high = high[idx_mat]                                      # (N_sig, H)
    panel_low  = low[idx_mat]
    panel_close = close[idx_mat]

    # ── EOD cap (optional): build a (N_sig, H) "valid bar" mask. Bars at column
    # offsets >= cap_col are excluded from SL/TP detection and forced-exit at
    # the cap bar's close.
    H = horizon_bars
    col_idx = np.arange(H)[None, :]                                 # (1, H)
    if max_exit_bar_per_signal is not None:
        max_exit = max_exit_bar_per_signal[valid].astype(np.int64)  # (N_sig,)
        # Cap column = (max_exit - sig) - 1, clamped to [0, H-1]. Bar at col
        # cap_col is the LAST eligible bar (closes at EOD); cols > cap_col masked.
        cap_col = np.clip(max_exit - sig - 1, 0, H - 1)             # (N_sig,)
    else:
        cap_col = np.full(N_sig, H - 1, dtype=np.int64)              # everything in-horizon
    valid_bar = col_idx <= cap_col[:, None]                         # (N_sig, H) bool

    # ── SL-hit detection per bar (masked by EOD cap):
    if direction == 1:
        sl_hit_panel = (panel_low <= sl_price[:, None]) & valid_bar
    else:
        sl_hit_panel = (panel_high >= sl_price[:, None]) & valid_bar

    any_sl_hit = sl_hit_panel.any(axis=1)                           # (N_sig,)
    first_sl_col = np.argmax(sl_hit_panel, axis=1)                  # 0 when no hit

    # ── TP-hit detection (optional, symmetric to SL):
    if tp_dist_per_bar is not None:
        tp_dist = tp_dist_per_bar[sig].astype(np.float64)
        bad_tp = (~np.isfinite(tp_dist)) | (tp_dist <= 0)
        tp_dist = np.where(bad_tp, np.nan, tp_dist)
        tp_price = entry_px + direction * tp_dist                   # long: entry+dist; short: entry-dist
        if direction == 1:
            tp_hit_panel = (panel_high >= tp_price[:, None]) & valid_bar
        else:
            tp_hit_panel = (panel_low  <= tp_price[:, None]) & valid_bar
        any_tp_hit = tp_hit_panel.any(axis=1)
        first_tp_col = np.argmax(tp_hit_panel, axis=1)
    else:
        tp_dist = np.zeros_like(sl_dist)
        tp_price = np.zeros_like(entry_px)
        any_tp_hit = np.zeros(N_sig, dtype=bool)
        first_tp_col = np.full(N_sig, H, dtype=np.int64)            # sentinel "never hit"

    # ── Exit precedence: SL-first ties; if neither, TP; if neither, EOD/horizon.
    # If SL hits at col S and TP hits at col T:
    #   - S < T → SL wins
    #   - S > T → TP wins
    #   - S == T → SL wins (same-bar tie, v10 micro-spec 9)
    sl_wins_priority = any_sl_hit & (~any_tp_hit | (first_sl_col <= first_tp_col))
    tp_wins_priority = any_tp_hit & ~sl_wins_priority

    # Compute exit bar offset (col within horizon) and exit price per case
    exit_col = np.where(
        sl_wins_priority, first_sl_col,
        np.where(tp_wins_priority, first_tp_col, cap_col)
    )
    exit_px = np.where(
        sl_wins_priority, sl_price,
        np.where(tp_wins_priority, tp_price,
                 panel_close[np.arange(N_sig), cap_col])
    )

    # Exit bar index in source array
    exit_bar = (sig + 1 + exit_col).astype(np.int64)

    # PnL points (signed)
    pnl_pts = (exit_px - entry_px) * direction
    R = pnl_pts / sl_dist

    # EOD-forced exit flag (no SL, no TP, horizon-cap reached)
    any_eod_exit = (~sl_wins_priority) & (~tp_wins_priority)

    # MFE / MAE over the [entry+1, exit_col] inclusive window — mask bars
    # strictly AFTER the chosen exit (whether SL, TP, or EOD).
    post_exit_mask = col_idx > exit_col[:, None]                    # bars strictly after exit

    if direction == 1:
        eff_high = np.where(post_exit_mask, -np.inf, panel_high)
        eff_low  = np.where(post_exit_mask,  np.inf, panel_low)
        mfe_pts  = eff_high.max(axis=1) - entry_px
        mae_pts  = entry_px - eff_low.min(axis=1)
    else:
        eff_high = np.where(post_exit_mask, -np.inf, panel_high)
        eff_low  = np.where(post_exit_mask,  np.inf, panel_low)
        mfe_pts  = entry_px - eff_low.min(axis=1)
        mae_pts  = eff_high.max(axis=1) - entry_px

    mfe_R = mfe_pts / sl_dist
    mae_R = mae_pts / sl_dist
    mae_R = np.maximum(mae_R, 0.0)

    return ProxyTrades(
        entry_bar   = sig,
        exit_bar    = exit_bar,
        R           = R,
        mfe_R       = mfe_R,
        mae_R       = mae_R,
        hit_sl      = sl_wins_priority,
        sl_dist_pts = sl_dist,
        hit_tp      = tp_wins_priority,
        hit_eod     = any_eod_exit,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Permutation null + economic threshold
# ─────────────────────────────────────────────────────────────────────────────

def random_entry_null_mean_R(
    full_R_pool: np.ndarray,
    k_signals: int,
    n_perm: int = 1000,
    seed: int = 2026,
) -> np.ndarray:
    """Null distribution of mean(R) under random entry bar selection.

    H0: our signal bars are not better than random entries. Under H0, picking
    `k_signals` bars at random from the same universe (all valid forward-R
    candidates in the test window) gives an mean(R) distribution centred on
    pool_mean(R) with std ≈ pool_std(R) / sqrt(k_signals).

    This is the correct null for "did the proxy pick a better subset than
    chance?" — replaces the sign-flip null which tests the unrelated question
    "is the mean of these R values different from a symmetric-around-zero
    distribution of |R|."

    Vectorised: single (n_perm, k_signals) index matrix, one fancy-index, one
    mean over axis=1. With replacement (bootstrap-style) for tractable variance
    in small pools.
    """
    R = np.asarray(full_R_pool, dtype=np.float64)
    R = R[np.isfinite(R)]
    n_pool = len(R)
    if n_pool < 50 or k_signals < 20:
        return np.array([np.nan])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_pool, size=(n_perm, k_signals))   # with replacement
    null_means = R[idx].mean(axis=1)
    return null_means


def evaluate_proxy(
    proxy_R: np.ndarray,
    full_R_pool: np.ndarray,
    direction: int,
    cost_per_RT_dollars: float,
    avg_sl_dist_pts: float,
    economic_threshold_R: float = 0.15,
    n_perm: int = 1000,
    point_value: float = 20.0,
    seed: int = 2026,
) -> dict:
    """Two-gate test: random-entry permutation null + economic threshold.

    Both gates in R units. P-value is ONE-SIDED in the direction's expected sign
    (long-biased proxy should show positive mean R; we test for mean_R BETTER
    than pool, not just different).

    `full_R_pool` is the per-bar forward-R for ALL valid bars in the test window
    for THIS direction (long-side R pool for long proxies, short-side for short),
    built once by the caller and reused across all proxies of the same direction.

    Returns dict with mean_R, mean_R_net, p_value, passes_null, passes_economic,
    passes_both, plus pool_mean_R for diagnostic transparency.
    """
    if direction not in (-1, 1):
        raise ValueError(f"direction must be +/-1, got {direction}")
    R = np.asarray(proxy_R, dtype=np.float64)
    R = R[np.isfinite(R)]
    n = len(R)
    pool = np.asarray(full_R_pool, dtype=np.float64)
    pool = pool[np.isfinite(pool)]
    if n < 20:
        return {
            'n_trades': int(n),
            'mean_R': float('nan'),
            'mean_R_net': float('nan'),
            'cost_R': float('nan'),
            'pool_mean_R': float(pool.mean()) if len(pool) else float('nan'),
            'pool_n': int(len(pool)),
            'p_value': float('nan'),
            'passes_null': False,
            'passes_economic': False,
            'passes_both': False,
            'reason': 'insufficient trades',
        }

    cost_R = cost_per_RT_dollars / (point_value * avg_sl_dist_pts)
    mean_R = float(R.mean())
    mean_R_net = mean_R - cost_R

    null_means = random_entry_null_mean_R(pool, k_signals=n, n_perm=n_perm, seed=seed)
    # One-sided p-value: are we BETTER than random entry? Both long and short
    # proxies represent "this entry has positive forward-R after sign correction
    # — the proxy direction was already baked into how R was computed (entries
    # with direction=+1 → R = (exit-entry)/sl_dist; direction=-1 → R = (entry-exit)/sl_dist).
    # So in BOTH cases, a winning proxy means HIGHER mean R than random.
    p_value = float((null_means >= mean_R).mean())

    passes_null = p_value < 0.05
    passes_economic = mean_R_net >= economic_threshold_R
    return {
        'n_trades': int(n),
        'mean_R': mean_R,
        'mean_R_net': mean_R_net,
        'cost_R': float(cost_R),
        'pool_mean_R': float(pool.mean()),
        'pool_n': int(len(pool)),
        'p_value': p_value,
        'passes_null': bool(passes_null),
        'passes_economic': bool(passes_economic),
        'passes_both': bool(passes_null and passes_economic),
    }




# ─────────────────────────────────────────────────────────────────────────────
# VXN EV-separation test
# ─────────────────────────────────────────────────────────────────────────────

def vxn_ev_separation_test(
    forward_R_per_day: np.ndarray,
    vxn_z_per_day:     np.ndarray,
    effect_size_threshold: float = 0.10,
    n_perm: int = 1000,
    seed: int = 2026,
) -> dict:
    """Split days by median(vxn_z); test if mean forward-R differs across halves.

    Pass iff |Δμ_R| > effect_size_threshold AND permutation p < 0.05.

    Vectorisation: single (n_perm, N_days) permutation matrix for label shuffles;
    vectorised group means via boolean masking inside a single broadcast.
    """
    R = np.asarray(forward_R_per_day, dtype=np.float64)
    z = np.asarray(vxn_z_per_day, dtype=np.float64)
    mask = np.isfinite(R) & np.isfinite(z)
    R = R[mask]; z = z[mask]
    n = len(R)
    if n < 40:
        return {'n_days': int(n), 'delta_mu_R': float('nan'),
                'p_value': float('nan'), 'passes': False, 'reason': 'insufficient days'}

    med = float(np.median(z))
    hi_mask = z >= med
    lo_mask = ~hi_mask
    mu_hi = R[hi_mask].mean()
    mu_lo = R[lo_mask].mean()
    obs_delta = float(mu_hi - mu_lo)

    # Permutation: shuffle the hi/lo labels n_perm times. Vectorised:
    # generate n_perm independent random rankings of [0..n-1], take top half as hi.
    rng = np.random.default_rng(seed)
    # rng.permuted with axis=1 is the fastest vectorised shuffle
    base = np.broadcast_to(np.arange(n), (n_perm, n)).copy()
    shuffled = rng.permuted(base, axis=1)
    # For each perm row, take first ceil(n/2) indices as "hi" group
    n_hi = int(np.ceil(n / 2))
    hi_idx = shuffled[:, :n_hi]                                # (n_perm, n_hi)
    # Sum R over hi-indices per row; convert to mean; then derive lo mean
    R_arr = R                                                  # (n,)
    sum_hi = R_arr[hi_idx].sum(axis=1)                         # (n_perm,)
    sum_all = R_arr.sum()
    sum_lo = sum_all - sum_hi
    n_lo = n - n_hi
    null_delta = sum_hi / n_hi - sum_lo / n_lo                 # (n_perm,)

    p_value = float((np.abs(null_delta) >= abs(obs_delta)).mean())
    passes = (abs(obs_delta) >= effect_size_threshold) and (p_value < 0.05)
    return {
        'n_days': int(n),
        'mu_R_high_vxn': float(mu_hi),
        'mu_R_low_vxn':  float(mu_lo),
        'delta_mu_R':    obs_delta,
        'p_value':       p_value,
        'passes':        bool(passes),
    }
