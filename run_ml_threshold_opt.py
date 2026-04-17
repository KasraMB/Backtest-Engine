"""
Threshold optimisation for EnsembleMLModel.

Two-phase approach for <10s total runtime:

  Phase 1 — Threshold selection (analytical, instant)
    Pre-compute ensemble scores for all validation trades.
    For each threshold candidate compute:
        proxy = mean_r_filtered * sqrt(trades_per_day_filtered)
    Pick the threshold that maximises this proxy.
    Captures quality × frequency tradeoff without Monte Carlo.

  Phase 2 — Geometry/risk_pct selection at the optimal threshold
    Run ONE propfirm grid per account at the optimal threshold.
    n_sims=100, 3×3 risk grid so each grid call ≈ 1.5s max.

Outputs: models/threshold_opt.json
    {
        "25K":  {"threshold": 0.12, "geometry": "floor_aware", "risk_pct": 0.30, "ev_per_day": 12.5},
        ...
    }
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from backtest.ml.model import EnsembleMLModel
from backtest.ml.features import ALL_FEATURE_NAMES
from backtest.propfirm.lucidflex import LUCIDFLEX_ACCOUNTS, run_propfirm_grid

DATASET_PATH        = ROOT / "data"   / "ml_dataset.parquet"
ENSEMBLE_MODEL_PATH = ROOT / "models" / "ict_smc_ensemble.pkl"
OUTPUT_PATH         = ROOT / "models" / "threshold_opt.json"

THRESHOLD_CANDIDATES = np.linspace(-0.5, 2.0, 20).tolist()
N_SIMS    = 100
RISK_PCTS = [0.10, 0.25, 0.50]
ACCOUNTS  = list(LUCIDFLEX_ACCOUNTS.keys())


def _count_val_trading_days(df_val: pd.DataFrame) -> int:
    dates = pd.to_datetime(df_val['date']).dt.date
    return int(dates.nunique())


def _analytical_proxy(
    scores: np.ndarray,
    r_multiples: np.ndarray,
    n_val_days: int,
    threshold: float,
) -> float:
    """
    Proxy for propfirm EV: mean_r_filtered * sqrt(trades_per_day).
    Maximising this balances trade quality against keeping enough frequency.
    Fully vectorised — no Monte Carlo required.
    """
    mask = scores > threshold
    n    = int(mask.sum())
    if n < 5:
        return -np.inf
    trades_per_day = max(0.1, n / max(n_val_days, 1))
    mean_r = float(np.mean(r_multiples[mask]))
    return mean_r * np.sqrt(trades_per_day)


def _select_geometry_and_risk(
    df_filtered: pd.DataFrame,
    regime_labels: np.ndarray | None,
    n_val_days: int,
    account_name: str,
) -> dict:
    """Run ONE propfirm grid for this account at the already-filtered set of trades."""
    account = LUCIDFLEX_ACCOUNTS[account_name]
    n       = len(df_filtered)
    trades_per_day = max(0.1, n / max(n_val_days, 1))

    sl_proxy  = np.full(n, 4.0, dtype=np.float32)
    pnl_proxy = (df_filtered['r_multiple'].values * 4.0).astype(np.float32)

    grid = run_propfirm_grid(
        trades=None,
        account=account,
        n_sims=N_SIMS,
        eval_risk_pcts=RISK_PCTS,
        funded_risk_pcts=RISK_PCTS,
        regime_labels=regime_labels,
        _pnl_pts=pnl_proxy,
        _sl_dists=sl_proxy,
        _trades_per_day=trades_per_day,
        n_workers=1,
    )

    best_ev   = -np.inf
    best_geom = 'fixed_dollar'
    best_rp   = 0.10
    for scheme, scheme_data in grid.items():
        if not isinstance(scheme_data, dict):
            continue
        for erp_key, frp_dict in scheme_data.items():
            if not isinstance(frp_dict, dict):
                continue
            for frp_key, cell in frp_dict.items():
                if not isinstance(cell, dict):
                    continue
                ev = cell.get('ev_per_day') or -np.inf
                if ev is not None and ev > best_ev:
                    best_ev   = float(ev)
                    best_geom = scheme
                    best_rp   = float(erp_key)

    return {
        'ev_per_day': float(best_ev) if best_ev > -np.inf else 0.0,
        'geometry':   best_geom,
        'risk_pct':   float(best_rp),
    }


def main() -> None:
    print("=== ML Threshold Optimisation ===\n", flush=True)

    if not DATASET_PATH.exists():
        print(f"ERROR: {DATASET_PATH} not found. Run run_ml_collect.py first.")
        return
    if not ENSEMBLE_MODEL_PATH.exists():
        print(f"ERROR: {ENSEMBLE_MODEL_PATH} not found. Run run_ml_train.py first.")
        return

    df_full = pd.read_parquet(DATASET_PATH)
    df_full['date'] = pd.to_datetime(df_full['date'])
    df_val = df_full[
        (df_full['date'] >= '2023-01-01') & (df_full['date'] < '2024-01-01')
    ].copy()
    print(f"Validation trades: {len(df_val)}", flush=True)

    model = EnsembleMLModel.load(ENSEMBLE_MODEL_PATH)
    print(f"Ensemble model loaded. threshold={model.threshold}\n", flush=True)

    n_val_days = _count_val_trading_days(df_val)
    print(f"Validation trading days: {n_val_days}", flush=True)
    print(f"Threshold candidates: {len(THRESHOLD_CANDIDATES)}"
          f"  ({THRESHOLD_CANDIDATES[0]:.2f} -> {THRESHOLD_CANDIDATES[-1]:.2f})", flush=True)

    # Pre-compute scores once
    X      = df_val[ALL_FEATURE_NAMES]
    pred_r = model._model_a.predict_r(X)
    p_win  = model._model_b.predict_proba(X.values.astype(float))[:, 1]
    scores = pred_r * p_win
    r_mult = df_val['r_multiple'].values

    # Phase 1: analytical proxy sweep (vectorised, no Monte Carlo)
    print("\nPhase 1 — analytical threshold sweep:", flush=True)
    proxies = []
    for thr in THRESHOLD_CANDIDATES:
        px = _analytical_proxy(scores, r_mult, n_val_days, thr)
        proxies.append(px)

    best_thr_idx = int(np.argmax(proxies))
    best_thr     = float(THRESHOLD_CANDIDATES[best_thr_idx])
    print(f"  Best threshold: {best_thr:.3f}  (proxy={proxies[best_thr_idx]:.4f})", flush=True)

    # Phase 2: one propfirm grid per account at the best threshold
    mask_best    = scores > best_thr
    df_best      = df_val[mask_best]
    print(f"\nPhase 2 — propfirm grid at threshold={best_thr:.3f}  "
          f"({int(mask_best.sum())} trades, {n_val_days} days)", flush=True)

    regime_labels = None
    if 'vol_regime_p_high' in df_best.columns:
        regime_labels = (df_best['vol_regime_p_high'].values > 0.5).astype(np.int8)

    optimal: dict = {}
    for acc_name in ACCOUNTS:
        res = _select_geometry_and_risk(df_best, regime_labels, n_val_days, acc_name)
        optimal[acc_name] = {
            'threshold':  best_thr,
            'geometry':   res['geometry'],
            'risk_pct':   res['risk_pct'],
            'ev_per_day': res['ev_per_day'],
        }
        print(f"  {acc_name:<6} geometry={res['geometry']:<14}  "
              f"risk_pct={res['risk_pct']:.2f}  ev/day={res['ev_per_day']:.2f}", flush=True)

    print("\n=== Optimal Parameters ===")
    for acc_name, opt in optimal.items():
        print(f"  {acc_name:<6} threshold={opt['threshold']:>6.3f}  "
              f"geometry={opt['geometry']:<14}  risk_pct={opt['risk_pct']:.2f}  "
              f"ev/day={opt['ev_per_day']:.2f}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(optimal, f, indent=2)
    print(f"\nSaved -> {OUTPUT_PATH}")
    print("Next: python run_ml_validate.py  then  python run_ml_tearsheets.py")


if __name__ == "__main__":
    import time as _time
    _t0 = _time.perf_counter()
    main()
    print(f"\nTotal time: {_time.perf_counter() - _t0:.1f}s")
