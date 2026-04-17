"""
Threshold optimisation for EnsembleMLModel.

Evaluates all threshold candidates simultaneously using vectorised Numba kernels
that batch the threshold dimension into a single prange launch per account.

n_sims=2000, 20 threshold candidates × 5 accounts → 5 propfirm grid calls total
(one per account), each evaluating all thresholds in parallel. Runs in ~10s.

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
from backtest.propfirm.lucidflex import LUCIDFLEX_ACCOUNTS, run_propfirm_grid_threshold_sweep

DATASET_PATH        = ROOT / "data"   / "ml_dataset.parquet"
ENSEMBLE_MODEL_PATH = ROOT / "models" / "ict_smc_ensemble.pkl"
OUTPUT_PATH         = ROOT / "models" / "threshold_opt.json"

THRESHOLD_CANDIDATES = np.linspace(-0.5, 2.0, 20).tolist()
N_SIMS    = 2_000
RISK_PCTS = [0.10, 0.25, 0.50]
ACCOUNTS  = list(LUCIDFLEX_ACCOUNTS.keys())

# SL proxy in points (MNQ tick = 0.25 pt, 4 pts ≈ 16 ticks — typical tight SL)
_SL_PROXY_PTS = 4.0


def _count_val_trading_days(df_val: pd.DataFrame) -> int:
    dates = pd.to_datetime(df_val['date']).dt.date
    return int(dates.nunique())


def _build_threshold_arrays(
    df_val: pd.DataFrame,
    scores: np.ndarray,
    n_val_days: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[float], list[np.ndarray | None]]:
    """
    For each threshold candidate build (pnl_pts, sl_dists, tpd, regime_labels).
    Returns four parallel lists.
    """
    r_mult = df_val['r_multiple'].values.astype(np.float32)
    has_regime = 'vol_regime_p_high' in df_val.columns
    regime_raw = (df_val['vol_regime_p_high'].values > 0.5).astype(np.int8) if has_regime else None

    pnl_list:    list[np.ndarray]       = []
    sl_list:     list[np.ndarray]       = []
    tpd_list:    list[float]            = []
    regime_list: list[np.ndarray | None] = []

    for thr in THRESHOLD_CANDIDATES:
        mask = scores > thr
        n    = int(mask.sum())
        if n < 5:
            pnl_list.append(np.zeros(0, dtype=np.float32))
            sl_list.append(np.zeros(0, dtype=np.float32))
            tpd_list.append(0.0)
            regime_list.append(None)
        else:
            pnl_list.append((r_mult[mask] * _SL_PROXY_PTS).astype(np.float32))
            sl_list.append(np.full(n, _SL_PROXY_PTS, dtype=np.float32))
            tpd_list.append(max(0.1, n / max(n_val_days, 1)))
            regime_list.append(regime_raw[mask] if regime_raw is not None else None)

    return pnl_list, sl_list, tpd_list, regime_list


def _best_result_for_account(
    results_per_thr: list[dict],
) -> tuple[int, str, float, float]:
    """
    Scan all (threshold × scheme × erp × frp) cells and return the combo with
    the highest ev_per_day.

    Returns (best_thr_idx, best_scheme, best_erp, best_ev_per_day).
    """
    best_ev  = -np.inf
    best_idx = 0
    best_scheme = ''
    best_erp    = 0.0

    for thr_i, thr_result in enumerate(results_per_thr):
        for scheme, erp_dict in thr_result.items():
            if not isinstance(erp_dict, dict):
                continue
            for erp_key, frp_dict in erp_dict.items():
                if not isinstance(frp_dict, dict):
                    continue
                for frp_key, cell in frp_dict.items():
                    if not isinstance(cell, dict):
                        continue
                    ev = cell.get('ev_per_day')
                    if ev is not None and ev > best_ev:
                        best_ev     = float(ev)
                        best_idx    = thr_i
                        best_scheme = scheme
                        best_erp    = float(erp_key)

    return best_idx, best_scheme, best_erp, best_ev


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
    print(f"Ensemble model loaded. default threshold={model.threshold}\n", flush=True)

    n_val_days = _count_val_trading_days(df_val)
    print(f"Validation trading days: {n_val_days}", flush=True)
    print(f"Threshold candidates:    {len(THRESHOLD_CANDIDATES)}"
          f"  ({THRESHOLD_CANDIDATES[0]:.2f} -> {THRESHOLD_CANDIDATES[-1]:.2f})", flush=True)
    print(f"n_sims:                  {N_SIMS}", flush=True)

    # Pre-compute scores once
    X      = df_val[ALL_FEATURE_NAMES]
    pred_r = model._model_a.predict_r(X)
    p_win  = model._model_b.predict_proba(X.values.astype(float))[:, 1]
    scores = (pred_r * p_win).astype(np.float32)

    # Build per-threshold arrays
    pnl_list, sl_list, tpd_list, regime_list = _build_threshold_arrays(
        df_val, scores, n_val_days,
    )

    counts_str = "  ".join(
        f"thr={t:.2f}:n={int(p.shape[0])}"
        for t, p in zip(THRESHOLD_CANDIDATES[::4], pnl_list[::4])
    )
    print(f"\nSample trade counts (every 4th): {counts_str}\n", flush=True)

    import time as _t
    optimal: dict = {}

    for acc_name in ACCOUNTS:
        account = LUCIDFLEX_ACCOUNTS[acc_name]
        print(f"Account [{acc_name}]...", end=" ", flush=True)
        t0 = _t.perf_counter()

        results_per_thr = run_propfirm_grid_threshold_sweep(
            pnl_list=pnl_list,
            sl_list=sl_list,
            tpd_list=tpd_list,
            regime_labels_list=regime_list,
            account=account,
            n_sims=N_SIMS,
            eval_risk_pcts=RISK_PCTS,
            funded_risk_pcts=RISK_PCTS,
            n_workers=2,
        )

        elapsed = _t.perf_counter() - t0
        best_idx, best_scheme, best_erp, best_ev = _best_result_for_account(results_per_thr)
        best_thr = float(THRESHOLD_CANDIDATES[best_idx])

        print(f"{elapsed:.1f}s  ->  threshold={best_thr:.3f}  "
              f"geometry={best_scheme}  risk_pct={best_erp:.2f}  "
              f"ev/day={best_ev:.4f}", flush=True)

        optimal[acc_name] = {
            'threshold':  best_thr,
            'geometry':   best_scheme,
            'risk_pct':   best_erp,
            'ev_per_day': round(best_ev, 4) if best_ev > -np.inf else 0.0,
        }

    print("\n=== Optimal Parameters ===")
    for acc_name, opt in optimal.items():
        print(f"  {acc_name:<6}  threshold={opt['threshold']:>6.3f}  "
              f"geometry={opt['geometry']:<14}  risk_pct={opt['risk_pct']:.2f}  "
              f"ev/day={opt['ev_per_day']:.4f}")

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
