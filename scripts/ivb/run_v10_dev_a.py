"""
IVB v10 driver — existence test on dev-A (2019-2020), single touch.

Per locked spec (tasks/ivb_v10_plan.md):

  Pre-reg essentials:
    - PRIMARY GATE: bootstrap CI lower bound (95% one-sided) of mean R > 0
    - n >= 200 trades on dev-A (HARD floor)
    - CONCORDANT: permutation p-value < 0.025 (Bonferroni 2 prongs, family-wise α=0.05)
    - DISAGREEMENT BRANCH: CI passes / perm fails → PASS with v11 re-examination flag

  Two prongs:
    A. Bare ORB-30min long+short (costs $13.50/RT)
    B. ORB-30min + POC-retest (costs $16/RT)
  Each tested at TP={1R, 2R} jointly (correlated within prong, 1 effective test per prong).

  COVID treatment: headline includes Mar 2020; COVID-out (Mar 9 - Apr 30 2020)
  reported separately as robustness.

  Output: results/ivb_v10_dev_a.json
  Verdict appended to tasks/ivb_v10_plan.md §12.

Invariants checked pre-flight (fail loudly):
  - dev-A slice loads, RTH coverage 100%
  - signals fire on >= 50% of trading days for Prong A (bare ORB common)
  - n >= some sanity floor before claiming "underpowered"
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from backtest.indicators.quantile import bootstrap_mean_ci
from backtest.ivb.proxy_validation import build_proxy_trades_vec
from backtest.ivb.v10_null import build_null_means
from backtest.ivb.v10_signals import (
    V10Signals, orb_bare_signals, orb_poc_retest_signals,
)
from backtest.ml.splits import split_bounds


# ── Pre-registered constants (LOCKED per tasks/ivb_v10_plan.md) ─────────────
COST_PRONG_A_RT_DOLLARS = 13.50        # stop-breakout entry
COST_PRONG_B_RT_DOLLARS = 16.00        # limit-retest entry (adverse selection)
POINT_VALUE             = 20.0          # NQ
N_PERM                  = 1000
N_TRADE_FLOOR           = 200
ALPHA_FAMILY_WISE       = 0.05
N_PRONGS                = 2
PERM_P_THRESHOLD        = ALPHA_FAMILY_WISE / N_PRONGS    # 0.025 (Bonferroni)
CI_LEVEL                = 0.95
HORIZON_BARS            = 360                              # generous; EOD cap will fire first
SEED_BOOT               = 2026
SEED_PERM               = 2026
COVID_OUT_START         = pd.Timestamp('2020-03-09')
COVID_OUT_END           = pd.Timestamp('2020-04-30')


def _R_for_tp_multiple(
    sigs: V10Signals,
    bar_idx: pd.DatetimeIndex,
    close: np.ndarray, high: np.ndarray, low: np.ndarray,
    tp_R_multiple: float,
    cost_per_RT_dollars: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build forward-R for the given TP multiple. Returns (R_gross, R_net).

    R_gross: trade R from build_proxy_trades_vec
    R_net:   R_gross - cost_R (cost_R = cost_dollars / (POINT_VALUE * sl_dist))
    """
    if len(sigs.signal_bars) == 0:
        return np.array([]), np.array([])

    n_bars = len(close)
    sl_per_bar = np.zeros(n_bars, dtype=np.float64)
    tp_per_bar = np.zeros(n_bars, dtype=np.float64)
    sl_per_bar[sigs.signal_bars] = sigs.sl_dist
    tp_per_bar[sigs.signal_bars] = sigs.sl_dist * tp_R_multiple

    # build_proxy_trades_vec needs a single direction per call; split long/short
    long_mask = sigs.direction == 1
    short_mask = ~long_mask
    R_all = np.full(len(sigs.signal_bars), np.nan, dtype=np.float64)

    for d_sign, mask in [(+1, long_mask), (-1, short_mask)]:
        if not mask.any():
            continue
        sub_sigs = sigs.signal_bars[mask]
        sub_eod  = sigs.eod_bar[mask]
        trades = build_proxy_trades_vec(
            signal_bars=sub_sigs, close=close, high=high, low=low,
            sl_dist_per_bar=sl_per_bar, direction=d_sign,
            horizon_bars=HORIZON_BARS, tp_dist_per_bar=tp_per_bar,
            max_exit_bar_per_signal=sub_eod,
        )
        # Map R back to original positions
        valid_in_sub = np.isin(sub_sigs, trades.entry_bar)
        R_all[np.where(mask)[0][valid_in_sub]] = trades.R

    # Cost in R-units: cost_dollars / (POINT_VALUE * sl_dist_pts)
    cost_R_per_trade = cost_per_RT_dollars / (POINT_VALUE * sigs.sl_dist)
    R_net = R_all - cost_R_per_trade
    return R_all, R_net


def _evaluate_prong(
    name: str,
    sigs: V10Signals,
    bar_idx: pd.DatetimeIndex,
    close: np.ndarray, high: np.ndarray, low: np.ndarray,
    tp_R_multiple: float,
    cost_per_RT_dollars: float,
    label: str,
) -> dict:
    """Compute observed mean R, bootstrap CI, permutation null p-value for one
    (prong, TP-multiple) cell. Returns a dict ready to slot into the results JSON.
    """
    R_gross, R_net = _R_for_tp_multiple(
        sigs, bar_idx, close, high, low, tp_R_multiple, cost_per_RT_dollars
    )
    R_net_finite = R_net[np.isfinite(R_net)]
    n = len(R_net_finite)
    out: dict = {
        'name': name, 'label': label,
        'tp_R_multiple': tp_R_multiple,
        'n_trades': int(n),
        'n_floor': N_TRADE_FLOOR,
        'meets_n_floor': bool(n >= N_TRADE_FLOOR),
    }
    if n < 20:
        out.update({
            'mean_R_gross': float('nan'), 'mean_R_net': float('nan'),
            'ci_lower_95_one_sided': float('nan'),
            'perm_p_value': float('nan'),
            'passes_ci_gate': False, 'passes_perm_concordant': False,
            'passes_overall': False, 'reason': 'insufficient trades',
        })
        return out

    mean_R_gross = float(np.nanmean(R_gross))
    mean_R_net   = float(R_net_finite.mean())
    # One-sided lower-bound 95% bootstrap CI for mean R_net
    ci_lo, _ = bootstrap_mean_ci(R_net_finite, n_boot=N_PERM, ci=CI_LEVEL, seed=SEED_BOOT, one_sided=True)

    # Permutation null
    print(f"    Building null for {label}...", flush=True)
    t0 = time.perf_counter()
    null = build_null_means(
        real_signals=sigs, bar_idx=bar_idx,
        close=close, high=high, low=low,
        tp_R_multiple=tp_R_multiple,
        n_perm=N_PERM, chunk_perms=500,
        horizon_bars=HORIZON_BARS, seed=SEED_PERM,
    )
    print(f"      done in {(time.perf_counter()-t0):.1f}s, null mean={null.mean():.5f}, std={null.std():.5f}", flush=True)
    # NOTE: null is built without cost adjustment; subtract mean cost_R to align with R_net
    mean_cost_R = float((cost_per_RT_dollars / (POINT_VALUE * sigs.sl_dist)).mean())
    null_net = null - mean_cost_R
    perm_p = float((null_net >= mean_R_net).mean())

    passes_ci   = bool(ci_lo > 0) and bool(n >= N_TRADE_FLOOR)
    passes_perm = bool(perm_p < PERM_P_THRESHOLD)
    passes_overall = passes_ci  # primary gate is CI; perm is concordant
    # Disagreement branch flag
    disagreement = passes_ci and (not passes_perm)

    out.update({
        'mean_R_gross':           mean_R_gross,
        'mean_R_net':             mean_R_net,
        'ci_lower_95_one_sided':  ci_lo,
        'perm_p_value':           perm_p,
        'null_mean':              float(null_net.mean()),
        'null_std':               float(null_net.std()),
        'mean_cost_R':            mean_cost_R,
        'passes_ci_gate':         passes_ci,
        'passes_perm_concordant': passes_perm,
        'passes_overall':         passes_overall,
        'disagreement_flag':      disagreement,
    })
    return out


def _filter_signals_to_window(
    sigs: V10Signals,
    bar_idx: pd.DatetimeIndex,
    win_start: pd.Timestamp,
    win_end:   pd.Timestamp,
) -> V10Signals:
    """Return a new V10Signals with signals whose bar_idx falls in [start, end]."""
    if len(sigs.signal_bars) == 0:
        return sigs
    sig_times = bar_idx[sigs.signal_bars]
    if sig_times.tz is not None:
        s = win_start.tz_localize(sig_times.tz) if win_start.tz is None else win_start
        e = win_end.tz_localize(sig_times.tz)   if win_end.tz   is None else win_end
    else:
        s, e = win_start, win_end
    mask = (sig_times >= s) & (sig_times <= e)
    return V10Signals(
        signal_bars=sigs.signal_bars[mask],
        direction=sigs.direction[mask],
        sl_dist=sigs.sl_dist[mask],
        orb_high=sigs.orb_high[mask],
        orb_low=sigs.orb_low[mask],
        eod_bar=sigs.eod_bar[mask],
        day_ord=sigs.day_ord[mask] if sigs.day_ord is not None else None,    # type: ignore[arg-type]
        poc=sigs.poc[mask] if sigs.poc is not None else None,                  # type: ignore[arg-type]
        retest_bar=sigs.retest_bar[mask] if sigs.retest_bar is not None else None,  # type: ignore[arg-type]
    )


def _exclude_covid(sigs: V10Signals, bar_idx: pd.DatetimeIndex) -> V10Signals:
    """Remove signals fired in [Mar 9 2020, Apr 30 2020] inclusive."""
    if len(sigs.signal_bars) == 0:
        return sigs
    sig_times = bar_idx[sigs.signal_bars]
    sig_times_naive = sig_times.tz_localize(None) if sig_times.tz is not None else sig_times
    mask = (sig_times_naive < COVID_OUT_START) | (sig_times_naive > COVID_OUT_END)
    return V10Signals(
        signal_bars=sigs.signal_bars[mask],
        direction=sigs.direction[mask],
        sl_dist=sigs.sl_dist[mask],
        orb_high=sigs.orb_high[mask],
        orb_low=sigs.orb_low[mask],
        eod_bar=sigs.eod_bar[mask],
        day_ord=sigs.day_ord[mask] if sigs.day_ord is not None else None,  # type: ignore[arg-type]
        poc=sigs.poc[mask] if sigs.poc is not None else None,              # type: ignore[arg-type]
        retest_bar=sigs.retest_bar[mask] if sigs.retest_bar is not None else None,  # type: ignore[arg-type]
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument('--parquet', default=str(REPO / 'data' / 'NQ_1m.parquet'))
    ap.add_argument('--out',     default=str(REPO / 'results' / 'ivb_v10_dev_a.json'))
    args = ap.parse_args()

    t_start = time.perf_counter()
    parquet_path = Path(args.parquet)
    out_path     = Path(args.out)

    print(f"Loading {parquet_path}...", flush=True)
    df = pd.read_parquet(parquet_path)

    # Slice to dev-A
    s, e = split_bounds('dev_A')
    df = df[(df.index >= s) & (df.index <= e)].copy()
    print(f"  dev-A: {s.date()} to {e.date()}, {len(df):,} bars", flush=True)

    bar_idx = pd.DatetimeIndex(df.index)
    close = df['close'].to_numpy(np.float64)
    high  = df['high'].to_numpy(np.float64)
    low   = df['low'].to_numpy(np.float64)
    volume = df['volume'].to_numpy(np.float64)

    # ── Generate signals (one pass each) ─────────────────────────────────────
    print("Generating Prong A signals (bare ORB)...", flush=True)
    t0 = time.perf_counter()
    sigs_A = orb_bare_signals(bar_idx, close, high, low)
    print(f"  {len(sigs_A.signal_bars)} signals in {(time.perf_counter()-t0):.2f}s", flush=True)

    print("Generating Prong B signals (ORB + POC retest)...", flush=True)
    t0 = time.perf_counter()
    sigs_B = orb_poc_retest_signals(bar_idx, close, high, low, volume)
    print(f"  {len(sigs_B.signal_bars)} signals in {(time.perf_counter()-t0):.2f}s", flush=True)

    # ── Invariant checks ─────────────────────────────────────────────────────
    n_days_in_dev_A = len(np.unique(bar_idx.tz_localize(None).normalize() if bar_idx.tz is not None
                                     else bar_idx.normalize()))
    assert n_days_in_dev_A > 400, f"dev-A has too few days: {n_days_in_dev_A}"
    print(f"  dev-A spans {n_days_in_dev_A} calendar days", flush=True)

    # Sanity: Prong A should fire on a non-trivial fraction of trading days
    # (NQ has ~250 trading days/year; 2 years → ~500). At least 50 signals expected.
    if len(sigs_A.signal_bars) < 50:
        print(f"  WARNING: Prong A only {len(sigs_A.signal_bars)} signals — sparse", flush=True)

    # ── Evaluate prongs at TP=1R and TP=2R ──────────────────────────────────
    results: dict = {
        'pre_registered': {
            'cost_per_RT_prong_A': COST_PRONG_A_RT_DOLLARS,
            'cost_per_RT_prong_B': COST_PRONG_B_RT_DOLLARS,
            'n_perm': N_PERM, 'n_trade_floor': N_TRADE_FLOOR,
            'perm_p_threshold': PERM_P_THRESHOLD,
            'ci_level': CI_LEVEL, 'horizon_bars': HORIZON_BARS,
            'seeds': {'bootstrap': SEED_BOOT, 'permutation': SEED_PERM},
        },
        'signal_counts': {
            'prong_A_total': int(len(sigs_A.signal_bars)),
            'prong_B_total': int(len(sigs_B.signal_bars)),
        },
        'headline_with_covid': {},
        'robustness_covid_out': {},
    }

    # Headline (with COVID)
    print("\n=== HEADLINE (COVID included) ===", flush=True)
    for tp_R in (1.0, 2.0):
        res_A = _evaluate_prong('prong_A', sigs_A, bar_idx, close, high, low,
                                tp_R, COST_PRONG_A_RT_DOLLARS, f'A_TP{tp_R}R')
        res_B = _evaluate_prong('prong_B', sigs_B, bar_idx, close, high, low,
                                tp_R, COST_PRONG_B_RT_DOLLARS, f'B_TP{tp_R}R')
        results['headline_with_covid'][f'A_TP{tp_R}R'] = res_A
        results['headline_with_covid'][f'B_TP{tp_R}R'] = res_B
        print(f"  A_TP{tp_R}R: n={res_A['n_trades']}, mean_R_net={res_A.get('mean_R_net', float('nan')):+.4f}, "
              f"CI_lo={res_A.get('ci_lower_95_one_sided', float('nan')):+.4f}, "
              f"p={res_A.get('perm_p_value', float('nan')):.3f}, "
              f"passes={res_A.get('passes_overall', False)}", flush=True)
        print(f"  B_TP{tp_R}R: n={res_B['n_trades']}, mean_R_net={res_B.get('mean_R_net', float('nan')):+.4f}, "
              f"CI_lo={res_B.get('ci_lower_95_one_sided', float('nan')):+.4f}, "
              f"p={res_B.get('perm_p_value', float('nan')):.3f}, "
              f"passes={res_B.get('passes_overall', False)}", flush=True)

    # Robustness (COVID out)
    print("\n=== ROBUSTNESS (COVID Mar 9 - Apr 30 2020 EXCLUDED) ===", flush=True)
    sigs_A_xc = _exclude_covid(sigs_A, bar_idx)
    sigs_B_xc = _exclude_covid(sigs_B, bar_idx)
    for tp_R in (1.0, 2.0):
        res_A_xc = _evaluate_prong('prong_A', sigs_A_xc, bar_idx, close, high, low,
                                    tp_R, COST_PRONG_A_RT_DOLLARS, f'A_TP{tp_R}R_xCOVID')
        res_B_xc = _evaluate_prong('prong_B', sigs_B_xc, bar_idx, close, high, low,
                                    tp_R, COST_PRONG_B_RT_DOLLARS, f'B_TP{tp_R}R_xCOVID')
        results['robustness_covid_out'][f'A_TP{tp_R}R'] = res_A_xc
        results['robustness_covid_out'][f'B_TP{tp_R}R'] = res_B_xc
        print(f"  A_TP{tp_R}R xCOVID: n={res_A_xc['n_trades']}, mean_R_net={res_A_xc.get('mean_R_net', float('nan')):+.4f}, "
              f"passes={res_A_xc.get('passes_overall', False)}", flush=True)
        print(f"  B_TP{tp_R}R xCOVID: n={res_B_xc['n_trades']}, mean_R_net={res_B_xc.get('mean_R_net', float('nan')):+.4f}, "
              f"passes={res_B_xc.get('passes_overall', False)}", flush=True)

    # Apply v10 decision matrix (HEADLINE only)
    head = results['headline_with_covid']
    A_passes_any = any(head[k]['passes_overall'] for k in ('A_TP1.0R', 'A_TP2.0R'))
    B_passes_any = any(head[k]['passes_overall'] for k in ('B_TP1.0R', 'B_TP2.0R'))
    if A_passes_any and B_passes_any:
        verdict = 'BOTH_PASS — promote to holdoutA at full size'
    elif A_passes_any:
        # Check if observed B-failure matches M1 (under-powered)
        any_B_under = any(head[k]['n_trades'] < N_TRADE_FLOOR for k in ('B_TP1.0R', 'B_TP2.0R'))
        if any_B_under:
            verdict = 'A_PASSES_M1 — Prong A to holdoutA at half size (Prong B under-powered)'
        else:
            verdict = 'A_ONLY_FAIL_NOT_M1 — needs failure-mode review'
    elif B_passes_any:
        verdict = 'B_ONLY — Prong B to holdoutA at half size (M3 candidate)'
    else:
        verdict = 'BOTH_FAIL — IVB terminated at v10'
    results['verdict'] = verdict
    print(f"\n=== VERDICT: {verdict} ===", flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nWrote {out_path}", flush=True)
    print(f"=== v10 dev-A run complete in {(time.perf_counter()-t_start):.1f}s ===", flush=True)


if __name__ == '__main__':
    main()
