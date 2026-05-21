"""
Apply IVB consolidation classifier as a day filter on AMR; re-run propfirm sim.

On non-consolidation days, AMR doesn't trade (R=0, no fee). On consolidation
days, AMR trades normally. Pre-analysis showed cfg_406 OOS:
  consolidation (ratio<0.5): mean_R=+0.142, win=54.4%, n=239
  trending (ratio>=0.5):     mean_R=+0.070, win=50.7%, n=71

Filter to consolidation-only -> higher per-day edge but fewer trading days.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from backtest.indicators.atr import atr_daily
from backtest.propfirm.three_phase_mc import ThreePhaseConfig, three_phase_reinvestment_mc
from backtest.propfirm.lucidflex import LUCIDFLEX_ACCOUNTS

ACCOUNT = LUCIDFLEX_ACCOUNTS['50K']
BUDGET = 1000.0
HORIZON = 84
N_SIMS = 20_000
SEEDS = [2026, 2027, 2028]
MIN_PAYOUT = 500.0
RESET_DELTA = -3.0


def compute_day_classifier(df1m: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame keyed by date with columns [orb_range, atr_20, ratio, valid]."""
    idx = pd.DatetimeIndex(df1m.index)
    bar_min = idx.hour.to_numpy(np.int32) * 60 + idx.minute.to_numpy(np.int32)
    orb_mask = (bar_min >= (9*60+31)) & (bar_min <= 10*60)
    day_naive = idx.tz_localize(None).normalize() if idx.tz is not None else idx.normalize()
    unique_days, inv = np.unique(day_naive.asi8, return_inverse=True)
    n_days = len(unique_days)
    high = df1m['high'].to_numpy(np.float64)
    low  = df1m['low'].to_numpy(np.float64)
    orb_high = np.full(n_days, -np.inf)
    orb_low  = np.full(n_days,  np.inf)
    orb_n    = np.zeros(n_days, dtype=np.int64)
    np.maximum.at(orb_high, inv[orb_mask], high[orb_mask])
    np.minimum.at(orb_low,  inv[orb_mask], low[orb_mask])
    np.add.at(orb_n, inv[orb_mask], 1)
    orb_range = orb_high - orb_low
    atr_s = atr_daily(df1m, window=20)
    atr_idx_naive = pd.DatetimeIndex(atr_s.index).tz_localize(None).normalize() \
                    if atr_s.index.tz is not None else pd.DatetimeIndex(atr_s.index).normalize()
    atr_lookup = pd.Series(atr_s.to_numpy(), index=atr_idx_naive)
    atr_per_day = atr_lookup.reindex(pd.DatetimeIndex(unique_days)).to_numpy(np.float64)
    return pd.DataFrame({
        'date': pd.to_datetime(unique_days),
        'orb_range': orb_range,
        'atr_20': atr_per_day,
        'ratio': orb_range / atr_per_day,
        'valid_orb': (orb_n >= 25) & np.isfinite(orb_range),
    })


def build_filtered_config(cfg_id: str, amr: dict, day_classifier: pd.DataFrame,
                          ratio_threshold: float, oos_start='2023-01-01') -> ThreePhaseConfig:
    """Build per-trade arrays for sim, INCLUDING zero-R days for filtered-out days.

    Strategy: for each calendar day in OOS, if day is consolidation -> emit AMR's
    actual trades for that day; if trending -> emit a single zero-R trade (no
    PnL, no fee impact, just a placeholder so the day exists in the pool).
    """
    data = amr[cfg_id]
    trades_df = data['trades'].copy()
    trades_df['date'] = pd.to_datetime(trades_df['date'])
    days_df = data['days'].copy()
    days_df['date'] = pd.to_datetime(days_df['date'])
    # Slice to OOS
    oos_trades = trades_df[trades_df['date'] >= oos_start]
    oos_days   = days_df[days_df['date'] >= oos_start]

    # Classify each day
    merged = oos_days.merge(day_classifier[['date', 'ratio', 'valid_orb']], on='date', how='left')
    merged['is_consolidation'] = merged['valid_orb'] & (merged['ratio'] < ratio_threshold)
    print(f"  Filter: ratio<{ratio_threshold}: "
          f"{merged['is_consolidation'].sum()} of {len(merged)} OOS days qualify "
          f"({100*merged['is_consolidation'].mean():.1f}%)")

    # Keep only trades on consolidation days
    cons_dates = merged.loc[merged['is_consolidation'], 'date']
    kept_trades = oos_trades[oos_trades['date'].isin(cons_dates)]
    print(f"  Trades kept: {len(kept_trades)} of {len(oos_trades)} OOS trades "
          f"({100*len(kept_trades)/len(oos_trades):.1f}%)")
    if len(kept_trades) < 50:
        return None

    # tpd: trades per day, computed on ALL OOS days (filtered + non-filtered)
    # because in reality the strategy runs every day but only fires on some
    # = filtered trades / total OOS days
    tpd = float(len(kept_trades)) / float(len(oos_days))
    print(f"  Effective tpd: {tpd:.3f} (vs unfiltered {float(len(oos_trades))/float(len(oos_days)):.3f})")
    return ThreePhaseConfig(
        name=f"{cfg_id}_OOS_cons{ratio_threshold}",
        pnl_pts=kept_trades['pnl_pts'].to_numpy(np.float32),
        sl_dists=kept_trades['sl_dist'].to_numpy(np.float32),
        tpd=tpd,
    )


def evaluate(cfg, erp, mc, seed):
    out = three_phase_reinvestment_mc(
        eval_grind_config=cfg, payout_config=cfg,
        eval_grind_risk=erp, account=ACCOUNT,
        budget=BUDGET, horizon=HORIZON,
        max_concurrent=mc, n_sims=N_SIMS, seed=seed,
        enforce_consistency_rule=True,
    )
    cash = out['cash'].astype(np.float64)
    n_evals = out['eval_fees_paid'] / 98.0
    reopens = np.maximum(n_evals - mc, 0.0)
    cash_adj = cash - RESET_DELTA * reopens
    growth = cash_adj - BUDGET
    return {
        'p_profit': float((cash_adj >= BUDGET + MIN_PAYOUT).mean()),
        'p_zero':   float((cash_adj <= 0).mean()),
        'sharpe':   float(growth.mean() / max(growth.std(), 1e-9)),
        'sortino':  float(growth.mean() / max(growth[growth<0].std() if (growth<0).any() else 1.0, 1e-9)),
        'mean_g':   float(growth.mean()),
        'median_g': float(np.median(growth)),
    }


def main():
    print("Loading data...", flush=True)
    amr = pickle.load(open(REPO / 'cache' / 'amr_per_day_trades.pkl', 'rb'))
    df1m = pd.read_parquet(REPO / 'data' / 'NQ_1m.parquet')

    print("Computing day classifier...", flush=True)
    classifier = compute_day_classifier(df1m)

    print(f"\n{'='*100}")
    print(f"AMR x IVB consolidation filter — propfirm sim at $1K, 50K Flex")
    print(f"{'='*100}")

    for cfg_id in ['cfg_406', 'cfg_029']:
        print(f"\n--- {cfg_id}_OOS ---")
        # Baseline (unfiltered)
        data = amr[cfg_id]
        td = data['trades'].copy(); td['date'] = pd.to_datetime(td['date'])
        dd = data['days'].copy();   dd['date'] = pd.to_datetime(dd['date'])
        oos_t = td[td['date'] >= '2023-01-01']
        oos_d = dd[dd['date'] >= '2023-01-01']
        baseline = ThreePhaseConfig(name=f'{cfg_id}_baseline',
            pnl_pts=oos_t['pnl_pts'].to_numpy(np.float32),
            sl_dists=oos_t['sl_dist'].to_numpy(np.float32),
            tpd=float(len(oos_t))/float(len(oos_d)))
        print(f"  Baseline tpd={baseline.tpd:.3f}, n_trades={len(baseline.pnl_pts)}")

        # Test thresholds
        for thr in [0.3, 0.5, 0.7]:
            print(f"\n  Threshold ratio<{thr}:")
            filt_cfg = build_filtered_config(cfg_id, amr, classifier, thr)
            if filt_cfg is None:
                print("    INSUFFICIENT TRADES")
                continue
            for erp in [0.5, 0.7, 0.9]:
                for mc in [1, 2]:
                    results = [evaluate(filt_cfg, erp, mc, s) for s in SEEDS]
                    pp = np.mean([r['p_profit'] for r in results])
                    sortino = np.mean([r['sortino'] for r in results])
                    sharpe = np.mean([r['sharpe'] for r in results])
                    mean_g = np.mean([r['mean_g'] for r in results])
                    median_g = np.mean([r['median_g'] for r in results])
                    tag = " PASS>80%" if pp > 0.80 else ""
                    print(f"    erp={erp:.1f} mc={mc}: p_profit={pp:.4f}{tag}, "
                          f"Sharpe={sharpe:+.3f}, Sortino={sortino:+.3f}, "
                          f"mean=${mean_g:+,.0f}, median=${median_g:+,.0f}")

        # Same params on baseline for comparison
        print(f"\n  Baseline comparison:")
        for erp in [0.5, 0.7]:
            for mc in [1, 2]:
                results = [evaluate(baseline, erp, mc, s) for s in SEEDS]
                pp = np.mean([r['p_profit'] for r in results])
                sortino = np.mean([r['sortino'] for r in results])
                mean_g = np.mean([r['mean_g'] for r in results])
                print(f"    erp={erp:.1f} mc={mc}: p_profit={pp:.4f}, Sortino={sortino:+.3f}, mean=${mean_g:+,.0f}")


if __name__ == '__main__':
    main()
