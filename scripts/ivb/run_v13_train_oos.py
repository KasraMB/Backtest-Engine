"""
IVB v13 — expanded grid: ORB breakout + delta CONFIRMATION or DIVERGENCE (fade).

Both logics from IVB.md tested. Train 2019-2023, single-shot OOS 2024+.

Logic types:
  - 'confirm':    long if close > ORB_hi AND delta >= +thr; short mirrored
  - 'divergence': fade exhaustion. Long if close < ORB_lo AND delta >= +thr
                  (sellers exhausted, fade the short breakdown); short mirrored.

Grid: 3 ORB × 3 delta_pct × 2 logic × 2 TP × 2 skip = 72 cells.

Selection (LOCKED): max train mean R subject to
  - n_trades >= 100
  - mean R > 0 overall (NOT just least-negative)
  - positive mean R in >= 3 of 4 years {2019, 2021, 2022, 2023}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

# Constants
ORB_WINDOWS = [15, 30, 60]
DELTA_PCTLS = [0.60, 0.70, 0.80]
LOGICS      = ['confirm', 'divergence']
TP_RS       = [1.0, 2.0]
SKIPS       = [0, 5]
SLIPPAGE_PT = 0.5
POINT_VALUE = 20.0
RTH_START_MIN = 9 * 60 + 30
EOD_FLAT_MIN  = 15 * 60 + 30
RTH_END_MIN   = 16 * 60
TRAIN_END = pd.Timestamp('2023-12-31')
OOS_START = pd.Timestamp('2024-01-01')


def build_v13_trades(df, orb_window, delta_thr, logic, tp_r, skip_first):
    idx = pd.DatetimeIndex(df.index)
    bar_min = idx.hour.to_numpy(np.int32) * 60 + idx.minute.to_numpy(np.int32)
    day_ord = (idx.tz_localize(None).normalize().asi8 if idx.tz is not None
               else idx.normalize().asi8)
    o = df['open'].to_numpy(np.float64)
    h = df['high'].to_numpy(np.float64)
    l = df['low'].to_numpy(np.float64)
    c = df['close'].to_numpy(np.float64)
    delta = df['ask_volume'].to_numpy(np.float64) - df['bid_volume'].to_numpy(np.float64)

    orb_end = RTH_START_MIN + orb_window
    trade_start = orb_end + skip_first
    unique_days, day_start_idx = np.unique(day_ord, return_index=True)
    day_start_idx = np.append(day_start_idx, len(df))
    rows = []
    for di in range(len(unique_days)):
        s, e = day_start_idx[di], day_start_idx[di + 1]
        dm = bar_min[s:e]
        orb_mask = (dm >= RTH_START_MIN) & (dm < orb_end)
        if orb_mask.sum() < orb_window * 0.8:
            continue
        orb_hi = h[s:e][orb_mask].max()
        orb_lo = l[s:e][orb_mask].min()
        if orb_hi <= orb_lo:
            continue
        tw_mask = (dm >= trade_start) & (dm < EOD_FLAT_MIN)
        tw_local = np.where(tw_mask)[0]
        if len(tw_local) == 0:
            continue
        sig_local, direction = None, 0
        for li in tw_local:
            gi = s + li
            if logic == 'confirm':
                if c[gi] > orb_hi and delta[gi] >= delta_thr:
                    sig_local, direction = li, 1; break
                if c[gi] < orb_lo and delta[gi] <= -delta_thr:
                    sig_local, direction = li, -1; break
            else:  # divergence (fade)
                if c[gi] > orb_hi and delta[gi] <= -delta_thr:
                    sig_local, direction = li, -1; break   # fade the up-breakout
                if c[gi] < orb_lo and delta[gi] >= delta_thr:
                    sig_local, direction = li, 1; break    # fade the down-breakdown
        if sig_local is None:
            continue
        entry_li = sig_local + 1
        if entry_li >= (e - s):
            continue
        entry_gi = s + entry_li
        entry_px = o[entry_gi] + direction * SLIPPAGE_PT
        if direction == 1:
            sl, sl_dist = orb_lo, entry_px - orb_lo
        else:
            sl, sl_dist = orb_hi, orb_hi - entry_px
        if sl_dist <= 0:
            continue
        tp = entry_px + direction * tp_r * sl_dist
        exit_px = None
        mae = 0.0
        for li2 in range(entry_li, e - s):
            gi2 = s + li2
            if dm[li2] >= EOD_FLAT_MIN:
                exit_px = c[gi2]; break
            bar_h, bar_l = h[gi2], l[gi2]
            if direction == 1:
                mae = max(mae, entry_px - bar_l)
                if bar_l <= sl: exit_px = sl; break
                if bar_h >= tp: exit_px = tp; break
            else:
                mae = max(mae, bar_h - entry_px)
                if bar_h >= sl: exit_px = sl; break
                if bar_l <= tp: exit_px = tp; break
        if exit_px is None:
            exit_px = c[e - 1]
        pnl_pts = (exit_px - entry_px) * direction - SLIPPAGE_PT
        rows.append({
            'date': pd.Timestamp(unique_days[di]),
            'direction': direction, 'sl_dist': sl_dist,
            'pnl_pts': pnl_pts, 'R': pnl_pts / sl_dist,
            'mae_pts': mae, 'MAE_R': mae / sl_dist,
        })
    return pd.DataFrame(rows)


def year_consistency(trades, years=(2019, 2021, 2022, 2023)):
    if len(trades) == 0:
        return 0
    return sum(1 for y in years
               if len(trades[trades['date'].dt.year == y]) > 0
               and trades[trades['date'].dt.year == y]['R'].mean() > 0)


def main():
    print("Loading NQ 1m...", flush=True)
    df_all = pd.read_parquet(REPO / 'data' / 'NQ_1m.parquet')
    idx = pd.DatetimeIndex(df_all.index)
    bm = idx.hour * 60 + idx.minute
    rth = (bm >= RTH_START_MIN) & (bm < RTH_END_MIN)
    df = df_all[rth]
    train = df[pd.DatetimeIndex(df.index).tz_localize(None) <= TRAIN_END]

    # Pre-compute delta thresholds on TRAIN RTH abs(delta)
    abs_delta = np.abs(train['ask_volume'].to_numpy(np.float64)
                       - train['bid_volume'].to_numpy(np.float64))
    thresholds = {pct: float(np.quantile(abs_delta, pct)) for pct in DELTA_PCTLS}
    print(f"  delta thresholds (train): {thresholds}", flush=True)

    print("\n=== TRAIN GRID (72 cells) ===", flush=True)
    results = []
    for orb_w in ORB_WINDOWS:
        for pct in DELTA_PCTLS:
            for logic in LOGICS:
                for tp_r in TP_RS:
                    for skip in SKIPS:
                        thr = thresholds[pct]
                        tr = build_v13_trades(train, orb_w, thr, logic, tp_r, skip)
                        n = len(tr)
                        if n == 0:
                            continue
                        mean_R = float(tr['R'].mean())
                        yc = year_consistency(tr)
                        valid = (n >= 100) and (mean_R > 0) and (yc >= 3)
                        results.append({
                            'orb_w': orb_w, 'pct': pct, 'logic': logic,
                            'tp_r': tp_r, 'skip': skip, 'thr': thr,
                            'n': n, 'mean_R': mean_R, 'year_consist': yc, 'valid': valid,
                        })
                        tag = ' VALID' if valid else ''
                        if mean_R > 0:
                            print(f"  ORB={orb_w:>2} pct={pct} logic={logic:<10} TP={tp_r} skip={skip}: "
                                  f"n={n:>4}, mean_R={mean_R:+.4f}, yc={yc}/4{tag}", flush=True)
    # Print summary of POSITIVE cells (negative ones are noise)
    pos = [r for r in results if r['mean_R'] > 0]
    val = [r for r in results if r['valid']]
    print(f"\n  Total cells: {len(results)}, positive mean_R: {len(pos)}, valid (n>=100 & yr>=3 & R>0): {len(val)}",
          flush=True)

    if not val:
        print("\nNO VALID CELL — IVB v13 NULL. No OOS test.", flush=True)
        with open(REPO/'results'/'ivb_v13_train.json', 'w') as f:
            json.dump({'verdict':'NULL_NO_VALID_CELL', 'results': results}, f, indent=2, default=str)
        return

    best = max(val, key=lambda r: r['mean_R'])
    print(f"\n=== SELECTED: ORB={best['orb_w']}, pct={best['pct']}, logic={best['logic']}, "
          f"TP={best['tp_r']}, skip={best['skip']}, train_mean_R={best['mean_R']:+.4f}, n={best['n']} ===",
          flush=True)

    # ── SINGLE-SHOT OOS ──
    print("\n=== TRUE OOS (2024+) SINGLE-SHOT ===", flush=True)
    oos = df[pd.DatetimeIndex(df.index).tz_localize(None) >= OOS_START]
    tr_oos = build_v13_trades(oos, best['orb_w'], best['thr'], best['logic'], best['tp_r'], best['skip'])
    n_oos = len(tr_oos)
    if n_oos < 30:
        verdict = 'OOS_INSUFFICIENT'
        oos_mean_R = None; perm_p = None
    else:
        oos_mean_R = float(tr_oos['R'].mean())
        from scipy import stats
        t, p_two = stats.ttest_1samp(tr_oos['R'].values, 0.0)
        perm_p = float(p_two/2 if t > 0 else 1.0 - p_two/2)
        verdict = 'PASS' if (oos_mean_R > 0 and perm_p < 0.05) else 'FAIL'
        print(f"  n_oos={n_oos}, mean_R={oos_mean_R:+.4f}, p_one_sided={perm_p:.4f} -> {verdict}",
              flush=True)

    out = {'verdict': verdict, 'selected': best,
           'oos_n': int(n_oos), 'oos_mean_R': oos_mean_R, 'oos_p': perm_p}
    with open(REPO/'results'/'ivb_v13_train.json', 'w') as f:
        json.dump({'verdict':'SELECTED','best':best,'train_results':results}, f, indent=2, default=str)
    with open(REPO/'results'/'ivb_v13_oos.json', 'w') as f:
        json.dump(out, f, indent=2, default=str)
    if n_oos > 0:
        tr_oos.to_pickle(REPO/'cache'/'ivb_v13_oos_trades.pkl')
    print(f"\nWrote results/ivb_v13_train.json + ivb_v13_oos.json", flush=True)


if __name__ == '__main__':
    main()
