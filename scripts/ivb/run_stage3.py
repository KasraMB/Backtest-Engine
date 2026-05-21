"""
Stage 3 driver — orderflow proxy validation + VXN EV-separation on holdoutA.

ONE-TIME RUN. holdoutA is touched exactly once. Re-running this script after
seeing results breaks the validation discipline of v9-final.

Pipeline:
  1. Load 1m parquet, slice to train_fit + train_holdoutA
  2. Load existing HMM cache (cache/amr_hmm.pkl); broadcast labels to per-bar
  3. Compute ATR_20_daily + per-bar broadcast (prior-day shift, no look-ahead)
  4. Compute bar_delta, bar_volume, |Δprice| on the combined window
  5. On train_fit ONLY: per-state quantile thresholds for momentum (delta) +
     absorption (vol HIGH cutoff, |Δp| LOW cutoff)
     Run q-CI monitor; fall back to universal q for any state with CI > 15%
  6. Apply thresholds to train_holdoutA → 4 boolean signal arrays
     (momentum_long, momentum_short, absorption_long, absorption_short)
  7. Build proxy trades on holdoutA with γ_default=1.0 × ATR_20_daily
  8. Run evaluate_proxy (permutation + economic threshold) per proxy × direction
  9. Compute per-day forward-R for VXN EV-separation test on holdoutA
 10. Write results/ivb_stage3.json

All hot operations are vectorised. End-to-end target: <30s wall-clock.
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from backtest.indicators.atr import atr_daily, broadcast_daily_to_1m
from backtest.indicators.orderflow import bar_delta, momentum_signal, absorption_signal
from backtest.indicators.quantile import per_state_quantile, ci_width_pct_of_threshold
from backtest.ivb.proxy_validation import build_proxy_trades_vec, evaluate_proxy, vxn_ev_separation_test
from backtest.ml.splits import split_bounds
from backtest.regime.vxn import load_vxn, vxn_z_prior_day_daily, broadcast_vxn_z_to_1m

# ── Pre-registered constants (LOCKED) ───────────────────────────────────────
HORIZON_BARS          = 60
GAMMA_DEFAULT         = 1.0       # × ATR_20_daily
Q_EXTREME             = 0.90      # high quantile for delta + vol
Q_LOW_PXCHG           = 0.10      # low quantile for absorption |Δp| cutoff
N_PERM                = 1000
ECON_THRESHOLD_R_NET  = 0.15
COST_PER_RT_DOLLARS   = 13.50     # NQ stop-breakout default for stage 3 economic check
POINT_VALUE           = 20.0
VXN_EFFECT_SIZE       = 0.10
Q_CI_FALLBACK_PCT     = 0.15
SEED_BOOT             = 2026
SEED_PERM             = 2026


def _broadcast_labels_to_bars(labels_by_date: dict[date, int],
                              bar_dates: pd.DatetimeIndex,
                              fallback: int = 1) -> np.ndarray:
    """Map daily HMM labels onto per-1m-bar array.

    Bars on dates with no HMM label (e.g. days outside the cache) get `fallback`
    (default = neutral). Fully vectorised via pd.Series.reindex.
    """
    src = pd.Series(labels_by_date)
    src.index = pd.DatetimeIndex(src.index)
    bar_day = bar_dates.normalize()
    # Reindex on unique days first (sub-second), then propagate per bar
    unique_days = bar_day.unique()
    day_labels = src.reindex(unique_days).fillna(fallback).astype(int)
    return day_labels.reindex(bar_day).to_numpy(np.int64)


def main():
    t_start = time.perf_counter()

    print("Loading NQ 1m parquet...", flush=True)
    df = pd.read_parquet(REPO / 'data' / 'NQ_1m.parquet')

    # Slice to [train_fit_start ... train_holdoutA_end]
    fit_s, fit_e = split_bounds('train_fit')
    ha_s,  ha_e  = split_bounds('train_holdoutA')
    s, e = min(fit_s, ha_s), max(fit_e, ha_e)
    df = df[(df.index >= s) & (df.index <= e)].copy()
    N = len(df)
    print(f"  Sliced to {s.date()}..{e.date()}: {N:,} bars", flush=True)

    # Pre-extract arrays (zero-copy where possible)
    close = df['close'].to_numpy(np.float64)
    high  = df['high'].to_numpy(np.float64)
    low   = df['low'].to_numpy(np.float64)
    vol   = df['volume'].to_numpy(np.float64)
    bv    = df['bid_volume'].to_numpy(np.float64)
    av    = df['ask_volume'].to_numpy(np.float64)

    # ── HMM labels per bar ────────────────────────────────────────────────
    print("Loading HMM cache...", flush=True)
    with open(REPO / 'cache' / 'amr_hmm.pkl', 'rb') as f:
        hmm = pickle.load(f)
    n_states = hmm.n_states
    bar_idx = pd.DatetimeIndex(df.index)
    labels_per_bar = _broadcast_labels_to_bars(hmm.labels, bar_idx)
    print(f"  n_states={n_states}, labels broadcast in {(time.perf_counter()-t_start):.2f}s", flush=True)

    # ── ATR_20_daily + broadcast (prior-day shift) ────────────────────────
    print("Computing ATR_20_daily...", flush=True)
    t0 = time.perf_counter()
    atr_d = atr_daily(df, window=20)
    atr_per_bar = broadcast_daily_to_1m(atr_d, bar_idx)
    sl_dist_per_bar = GAMMA_DEFAULT * atr_per_bar
    print(f"  ATR built + broadcast in {(time.perf_counter()-t0):.2f}s", flush=True)

    # ── bar_delta + price change ──────────────────────────────────────────
    delta = bar_delta(bv, av)
    pxchg_abs = np.abs(close - df['open'].to_numpy(np.float64))

    # ── Train-fit mask (split source from holdoutA) ───────────────────────
    fit_mask = (bar_idx >= fit_s) & (bar_idx <= fit_e)
    ha_mask  = (bar_idx >= ha_s)  & (bar_idx <= ha_e)

    # ── Per-state thresholds on train_fit only ────────────────────────────
    print("Per-state quantile thresholds (train_fit only)...", flush=True)
    t0 = time.perf_counter()
    # Momentum: |delta| extreme → split into long (delta>0) and short (delta<0) thresholds
    # Easier: use the high quantile of |delta| as a SYMMETRIC threshold.
    abs_delta_fit = np.abs(delta[fit_mask])
    labels_fit    = labels_per_bar[fit_mask]
    vol_fit       = vol[fit_mask]
    pxchg_fit     = pxchg_abs[fit_mask]

    mom_thresh   = per_state_quantile(abs_delta_fit, labels_fit, Q_EXTREME, n_states)
    vol_thresh   = per_state_quantile(vol_fit,       labels_fit, Q_EXTREME, n_states)
    pxchg_thresh = per_state_quantile(pxchg_fit,     labels_fit, Q_LOW_PXCHG, n_states)

    # ── Q-CI monitor: fall back to universal q for any state with CI > 15% ─
    universal_mom   = float(np.quantile(abs_delta_fit, Q_EXTREME))
    universal_vol   = float(np.quantile(vol_fit, Q_EXTREME))
    universal_pxchg = float(np.quantile(pxchg_fit, Q_LOW_PXCHG))
    ci_log = {}
    for s in range(n_states):
        mask = labels_fit == s
        if mask.sum() < 30:
            continue
        for name, vals, q, thresh_dict, universal in [
            ('momentum',   abs_delta_fit[mask], Q_EXTREME,    mom_thresh,   universal_mom),
            ('volume',     vol_fit[mask],       Q_EXTREME,    vol_thresh,   universal_vol),
            ('pxchg',      pxchg_fit[mask],     Q_LOW_PXCHG,  pxchg_thresh, universal_pxchg),
        ]:
            width = ci_width_pct_of_threshold(vals, q, n_boot=500, seed=SEED_BOOT)
            ci_log[f'state{s}_{name}'] = {'ci_width_pct': width, 'fell_back': bool(width > Q_CI_FALLBACK_PCT)}
            if width > Q_CI_FALLBACK_PCT:
                thresh_dict[s] = universal
    print(f"  Thresholds set in {(time.perf_counter()-t0):.2f}s", flush=True)
    print(f"  Momentum thresholds (per state): {mom_thresh}", flush=True)
    print(f"  Volume thresholds   (per state): {vol_thresh}", flush=True)
    print(f"  PxChg thresholds    (per state): {pxchg_thresh}", flush=True)

    # ── Signals on holdoutA only ──────────────────────────────────────────
    print("Computing signals on holdoutA...", flush=True)
    ha_idx = np.where(ha_mask)[0]
    # Build full-length signal masks first (faster fancy-indexing on numpy)
    mom_long  = momentum_signal(delta, labels_per_bar, mom_thresh, +1, n_states) & ha_mask
    mom_short = momentum_signal(delta, labels_per_bar, mom_thresh, -1, n_states) & ha_mask
    abs_mask  = absorption_signal(vol, pxchg_abs, labels_per_bar, vol_thresh, pxchg_thresh, n_states) & ha_mask
    # Absorption direction inferred from sign of bar_delta (was-bid-side absorption = short bias, etc.)
    abs_long  = abs_mask & (delta > 0)
    abs_short = abs_mask & (delta < 0)
    print(f"  momentum_long={mom_long.sum()}, momentum_short={mom_short.sum()}, "
          f"absorption_long={abs_long.sum()}, absorption_short={abs_short.sum()}", flush=True)

    # ── Build proxy trades + evaluate each ────────────────────────────────
    print("Building proxy trades + permutation null (holdoutA)...", flush=True)
    t0 = time.perf_counter()

    avg_sl_dist_pts = float(np.nanmean(sl_dist_per_bar[ha_mask]))
    results = {}
    for name, mask, direction in [
        ('momentum_long',   mom_long,  +1),
        ('momentum_short',  mom_short, -1),
        ('absorption_long', abs_long,  +1),
        ('absorption_short',abs_short, -1),
    ]:
        sig_idx = np.where(mask)[0]
        trades = build_proxy_trades_vec(
            signal_bars=sig_idx, close=close, high=high, low=low,
            sl_dist_per_bar=sl_dist_per_bar, direction=direction,
            horizon_bars=HORIZON_BARS,
        )
        res = evaluate_proxy(
            trades.R, cost_per_RT_dollars=COST_PER_RT_DOLLARS,
            avg_sl_dist_pts=avg_sl_dist_pts,
            economic_threshold_R=ECON_THRESHOLD_R_NET,
            n_perm=N_PERM, point_value=POINT_VALUE, seed=SEED_PERM,
        )
        res['n_signals'] = int(len(sig_idx))
        res['hit_sl_rate'] = float(trades.hit_sl.mean()) if len(trades.hit_sl) > 0 else float('nan')
        res['mean_mfe_R']  = float(np.nanmean(trades.mfe_R)) if len(trades.mfe_R) > 0 else float('nan')
        res['mean_mae_R']  = float(np.nanmean(trades.mae_R)) if len(trades.mae_R) > 0 else float('nan')
        results[name] = res
        print(f"  {name:18s} n={res['n_signals']:>5} mean_R={res.get('mean_R', float('nan')):+.4f} "
              f"net={res.get('mean_R_net', float('nan')):+.4f} p={res.get('p_value', float('nan')):.3f} "
              f"pass={res['passes_both']}", flush=True)
    print(f"  Trades + perm in {(time.perf_counter()-t0):.2f}s", flush=True)

    # ── VXN EV-separation test (holdoutA) ─────────────────────────────────
    print("VXN EV-separation test (holdoutA)...", flush=True)
    t0 = time.perf_counter()
    vxn = load_vxn()
    vxn_z = vxn_z_prior_day_daily(vxn, window=60)
    vxn_per_bar = broadcast_vxn_z_to_1m(vxn_z, bar_idx)

    # Per-day forward-R: pool all 4 directions' trades by day, take avg-R
    # Use union of all signal masks; per-day average R as the EV proxy
    union_mask = mom_long | mom_short | abs_long | abs_short
    all_sigs = np.where(union_mask)[0]
    # Direction per signal: we need to rebuild — use bar_delta sign as proxy (matches above)
    all_dir = np.where(delta[all_sigs] > 0, 1, -1).astype(np.int64)
    # Build trades per direction (vectorised batch)
    long_sigs  = all_sigs[all_dir == 1]
    short_sigs = all_sigs[all_dir == -1]
    long_trades  = build_proxy_trades_vec(long_sigs,  close, high, low, sl_dist_per_bar, +1, HORIZON_BARS)
    short_trades = build_proxy_trades_vec(short_sigs, close, high, low, sl_dist_per_bar, -1, HORIZON_BARS)

    # Aggregate to per-day mean R
    bar_day_ord = bar_idx.normalize().asi8
    def _per_day_mean_R(entry_bars, R):
        if len(entry_bars) == 0:
            return pd.Series(dtype=np.float64)
        day_keys = bar_day_ord[entry_bars]
        return pd.Series(R).groupby(day_keys).mean()
    long_daily  = _per_day_mean_R(long_trades.entry_bar,  long_trades.R)
    short_daily = _per_day_mean_R(short_trades.entry_bar, short_trades.R)
    # Combine to single per-day score (mean across directions present)
    combined = pd.concat([long_daily, short_daily], axis=1).mean(axis=1)
    # VXN per day (take first bar of each day)
    vxn_first_bar = pd.Series(vxn_per_bar, index=bar_day_ord).groupby(level=0).first()
    vxn_aligned = vxn_first_bar.reindex(combined.index)

    vxn_res = vxn_ev_separation_test(
        combined.to_numpy(np.float64),
        vxn_aligned.to_numpy(np.float64),
        effect_size_threshold=VXN_EFFECT_SIZE,
        n_perm=N_PERM, seed=SEED_PERM,
    )
    print(f"  VXN test: {vxn_res} (in {(time.perf_counter()-t0):.2f}s)", flush=True)

    # ── Write results JSON ────────────────────────────────────────────────
    out_dir = REPO / 'results'
    out_dir.mkdir(exist_ok=True)
    payload = {
        'pre_registered': {
            'horizon_bars': HORIZON_BARS, 'gamma_default': GAMMA_DEFAULT,
            'q_extreme': Q_EXTREME, 'q_low_pxchg': Q_LOW_PXCHG,
            'n_perm': N_PERM, 'econ_threshold_R_net': ECON_THRESHOLD_R_NET,
            'cost_per_RT_dollars': COST_PER_RT_DOLLARS,
            'vxn_effect_size': VXN_EFFECT_SIZE, 'q_ci_fallback_pct': Q_CI_FALLBACK_PCT,
            'seeds': {'boot': SEED_BOOT, 'perm': SEED_PERM},
        },
        'thresholds': {
            'momentum_per_state':   {int(k): float(v) for k, v in mom_thresh.items()},
            'volume_per_state':     {int(k): float(v) for k, v in vol_thresh.items()},
            'pxchg_per_state':      {int(k): float(v) for k, v in pxchg_thresh.items()},
            'universal': {'momentum': universal_mom, 'volume': universal_vol, 'pxchg': universal_pxchg},
        },
        'q_ci_monitor': ci_log,
        'proxies': results,
        'vxn_gate': vxn_res,
        'avg_sl_dist_pts_holdoutA': avg_sl_dist_pts,
    }
    out_path = out_dir / 'ivb_stage3.json'
    with open(out_path, 'w') as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\nWrote {out_path}", flush=True)

    total = time.perf_counter() - t_start
    print(f"\n=== Stage 3 complete in {total:.1f}s ===", flush=True)


if __name__ == '__main__':
    main()
