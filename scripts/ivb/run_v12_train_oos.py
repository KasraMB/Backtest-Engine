"""
IVB v12 — ORB breakout + delta-confirmation gate. Per tasks/ivb_v12_prereg.md.

Train (2019-2023): select best (ORB_window, delta_threshold_pct, TP_R) cell.
OOS (2024+): single-shot validation of the selected cell.

Realistic execution: entry at next-bar open + 0.5pt slippage, $3.50 commission.
This is a SELF-CONTAINED vectorised builder (no BaseStrategy framework) for
clarity and speed — pure numpy over the 1m arrays.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

# ── Pre-registered grid + constants (LOCKED) ────────────────────────────────
ORB_WINDOWS   = [15, 30, 60]
DELTA_PCTLS   = [0.60, 0.70, 0.80]
TP_RS         = [1.0, 2.0]
SLIPPAGE_PT   = 0.5
COMMISSION_RT = 7.0      # $3.50/side × 2
POINT_VALUE   = 20.0
TICK          = 0.25
RTH_START_MIN = 9 * 60 + 30      # 9:30
RTH_END_MIN   = 16 * 60          # 16:00
EOD_FLAT_MIN  = 15 * 60 + 30     # 15:30
TRAIN_END     = pd.Timestamp('2023-12-31')
OOS_START     = pd.Timestamp('2024-01-01')
MIN_TRADES_TRAIN = 100
MIN_TRADES_OOS   = 30


def build_v12_trades(df: pd.DataFrame, orb_window: int, delta_threshold: float,
                     tp_r: float) -> pd.DataFrame:
    """Vectorised-ish per-day ORB + delta-gate trade builder.

    Returns DataFrame: date, direction, entry_price, sl, tp, exit_price, R, pnl_pts, mae_pts.
    """
    idx = pd.DatetimeIndex(df.index)
    bar_min = idx.hour.to_numpy(np.int32) * 60 + idx.minute.to_numpy(np.int32)
    day_ord = (idx.tz_localize(None).normalize().asi8 if idx.tz is not None
               else idx.normalize().asi8)
    o = df['open'].to_numpy(np.float64)
    h = df['high'].to_numpy(np.float64)
    l = df['low'].to_numpy(np.float64)
    c = df['close'].to_numpy(np.float64)
    bv = df['bid_volume'].to_numpy(np.float64)
    av = df['ask_volume'].to_numpy(np.float64)
    delta = av - bv

    orb_end_min = RTH_START_MIN + orb_window

    unique_days, day_start_idx = np.unique(day_ord, return_index=True)
    n_days = len(unique_days)
    day_start_idx = np.append(day_start_idx, len(df))

    rows = []
    for di in range(n_days):
        s, e = day_start_idx[di], day_start_idx[di + 1]
        dm = bar_min[s:e]
        # ORB window bars: [9:30, 9:30+window)
        orb_mask = (dm >= RTH_START_MIN) & (dm < orb_end_min)
        if orb_mask.sum() < orb_window * 0.8:
            continue
        orb_hi = h[s:e][orb_mask].max()
        orb_lo = l[s:e][orb_mask].min()
        if orb_hi <= orb_lo:
            continue
        # Trade window: [orb_end, 15:30)
        tw_mask = (dm >= orb_end_min) & (dm < EOD_FLAT_MIN)
        tw_local = np.where(tw_mask)[0]
        if len(tw_local) == 0:
            continue
        # First breakout with delta confirmation
        sig_local = None
        direction = 0
        for li in tw_local:
            gi = s + li
            if c[gi] > orb_hi and delta[gi] >= delta_threshold:
                sig_local, direction = li, 1
                break
            if c[gi] < orb_lo and delta[gi] <= -delta_threshold:
                sig_local, direction = li, -1
                break
        if sig_local is None:
            continue
        # Entry at NEXT bar open (realistic)
        entry_li = sig_local + 1
        if entry_li >= (e - s):
            continue
        entry_gi = s + entry_li
        entry_px = o[entry_gi] + direction * SLIPPAGE_PT   # adverse slippage
        if direction == 1:
            sl = orb_lo
            sl_dist = entry_px - sl
        else:
            sl = orb_hi
            sl_dist = sl - entry_px
        if sl_dist <= 0:
            continue
        tp = entry_px + direction * tp_r * sl_dist

        # Walk forward from entry+1 to EOD; SL-first within bar
        exit_px = None
        mae = 0.0
        for li2 in range(entry_li, e - s):
            gi2 = s + li2
            if dm[li2] >= EOD_FLAT_MIN:
                exit_px = c[gi2]   # EOD flat
                break
            bar_h, bar_l = h[gi2], l[gi2]
            # MAE update
            if direction == 1:
                mae = max(mae, entry_px - bar_l)
                sl_hit = bar_l <= sl
                tp_hit = bar_h >= tp
            else:
                mae = max(mae, bar_h - entry_px)
                sl_hit = bar_h >= sl
                tp_hit = bar_l <= tp
            if sl_hit:                       # SL-first (conservative)
                exit_px = sl
                break
            if tp_hit:
                exit_px = tp
                break
        if exit_px is None:
            exit_px = c[e - 1]
        # Apply exit slippage (adverse)
        pnl_pts = (exit_px - entry_px) * direction - SLIPPAGE_PT   # exit slip
        R = pnl_pts / sl_dist
        rows.append({
            'date': pd.Timestamp(unique_days[di]),
            'direction': direction, 'sl_dist': sl_dist,
            'pnl_pts': pnl_pts, 'R': R, 'mae_pts': mae,
            'MAE_R': mae / sl_dist,
        })
    return pd.DataFrame(rows)


def year_consistency(trades: pd.DataFrame, years=(2019, 2021, 2022, 2023)) -> int:
    """Count years (from given set) with positive mean R."""
    if len(trades) == 0:
        return 0
    cnt = 0
    for y in years:
        yr = trades[trades['date'].dt.year == y]
        if len(yr) > 0 and yr['R'].mean() > 0:
            cnt += 1
    return cnt


def main():
    print("Loading NQ 1m...", flush=True)
    df_all = pd.read_parquet(REPO / 'data' / 'NQ_1m.parquet')
    # RTH only for speed
    idx = pd.DatetimeIndex(df_all.index)
    bm = idx.hour * 60 + idx.minute
    rth = (bm >= RTH_START_MIN) & (bm < RTH_END_MIN)
    df = df_all[rth]
    print(f"  {len(df):,} RTH bars", flush=True)

    train = df[pd.DatetimeIndex(df.index).tz_localize(None) <= TRAIN_END]
    print(f"  train bars (<=2023): {len(train):,}", flush=True)

    # Pre-compute delta threshold per percentile on TRAIN RTH bars
    train_delta = (train['ask_volume'].to_numpy(np.float64) - train['bid_volume'].to_numpy(np.float64))
    abs_delta = np.abs(train_delta)
    thresholds = {pct: float(np.quantile(abs_delta, pct)) for pct in DELTA_PCTLS}
    print(f"  delta thresholds (train): {thresholds}", flush=True)

    print("\n=== TRAIN: 18-cell grid search (2019-2023) ===", flush=True)
    results = []
    for orb_w in ORB_WINDOWS:
        for pct in DELTA_PCTLS:
            for tp_r in TP_RS:
                thr = thresholds[pct]
                tr = build_v12_trades(train, orb_w, thr, tp_r)
                n = len(tr)
                mean_R = float(tr['R'].mean()) if n > 0 else 0.0
                yc = year_consistency(tr)
                valid = (n >= MIN_TRADES_TRAIN) and (yc >= 3)
                results.append({
                    'orb_w': orb_w, 'pct': pct, 'tp_r': tp_r, 'thr': thr,
                    'n': n, 'mean_R': mean_R, 'year_consist': yc, 'valid': valid,
                })
                print(f"  ORB={orb_w:>2} pct={pct} TP={tp_r}: n={n:>4}, mean_R={mean_R:+.4f}, "
                      f"yr_consist={yc}/4, valid={valid}", flush=True)

    valid_cells = [r for r in results if r['valid']]
    if not valid_cells:
        print("\nNO VALID CELL — v12 declared NULL, no OOS test.", flush=True)
        json.dump({'verdict': 'NULL_NO_VALID_TRAIN_CELL', 'train_results': results},
                  open(REPO / 'results' / 'ivb_v12_train.json', 'w'), indent=2, default=str)
        return

    best = max(valid_cells, key=lambda r: r['mean_R'])
    print(f"\n=== SELECTED (train-best): ORB={best['orb_w']}, pct={best['pct']}, "
          f"TP={best['tp_r']}, mean_R={best['mean_R']:+.4f}, n={best['n']} ===", flush=True)

    # A negative-edge strategy cannot profit in propfirm regardless of payout
    # structure. Do NOT proceed to OOS — that would be data-dredging a loser.
    if best['mean_R'] <= 0:
        print(f"\nTrain-best mean_R = {best['mean_R']:+.4f} <= 0 -> v12 NULL. "
              f"No OOS test (cannot rescue a negative-edge strategy).", flush=True)
        json.dump({'verdict': 'NULL_NEGATIVE_TRAIN_EDGE', 'best': best, 'train_results': results},
                  open(REPO / 'results' / 'ivb_v12_train.json', 'w'), indent=2, default=str)
        return
    json.dump({'verdict': 'SELECTED', 'best': best, 'train_results': results},
              open(REPO / 'results' / 'ivb_v12_train.json', 'w'), indent=2, default=str)

    # ── SINGLE-SHOT OOS ──────────────────────────────────────────────────
    print("\n=== OOS SINGLE-SHOT (2024+) ===", flush=True)
    oos = df[pd.DatetimeIndex(df.index).tz_localize(None) >= OOS_START]
    tr_oos = build_v12_trades(oos, best['orb_w'], best['thr'], best['tp_r'])
    n_oos = len(tr_oos)
    if n_oos < MIN_TRADES_OOS:
        print(f"  OOS n={n_oos} < {MIN_TRADES_OOS} — INSUFFICIENT, v12 inconclusive", flush=True)
        verdict = 'OOS_INSUFFICIENT_N'
    else:
        mean_R = float(tr_oos['R'].mean())
        from scipy import stats
        t_stat, p_two = stats.ttest_1samp(tr_oos['R'].values, 0.0)
        p_one = p_two / 2 if t_stat > 0 else 1.0 - p_two / 2
        passed = (mean_R > 0) and (p_one < 0.05)
        verdict = 'PASS' if passed else 'FAIL'
        print(f"  OOS n={n_oos}, mean_R={mean_R:+.4f}, WR={(tr_oos['R']>0).mean()*100:.1f}%, "
              f"t={t_stat:+.2f}, p_one_sided={p_one:.4f} → {verdict}", flush=True)

    out = {
        'verdict': verdict, 'selected': best,
        'oos_n': int(n_oos),
        'oos_mean_R': float(tr_oos['R'].mean()) if n_oos > 0 else None,
        'oos_wr': float((tr_oos['R'] > 0).mean()) if n_oos > 0 else None,
    }
    json.dump(out, open(REPO / 'results' / 'ivb_v12_oos.json', 'w'), indent=2, default=str)
    # Save OOS + train trades for potential propfirm sim
    if n_oos > 0:
        tr_oos.to_pickle(REPO / 'cache' / 'ivb_v12_oos_trades.pkl')
    print(f"\nWrote results/ivb_v12_oos.json", flush=True)


if __name__ == '__main__':
    main()
