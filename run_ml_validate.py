"""
Evaluate the trained ML model on the validation split.

Safe to run as many times as you like — validation data is not a held-out
test set, it is used for threshold tuning and model comparison.

Usage
-----
  python run_ml_validate.py

Prints a full metric table comparing all-trades vs. ML-filtered trades
on the validation period (2023-01-01 → 2023-12-31 by default).
"""
from __future__ import annotations

import sys
from pathlib import Path
from datetime import time as dtime

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from backtest.ml.model import MLModel
from backtest.ml.splits import filter_df, SPLITS
from backtest.ml.features import ALL_FEATURE_NAMES
from backtest.ml.evaluate import sortino_r, profit_factor_r, win_rate, expectancy_r

DATASET_PATH = ROOT / "data" / "ml_dataset.parquet"
MODEL_PATH   = ROOT / "models" / "ict_smc.pkl"


def main() -> None:
    print("=== ML Validation Evaluation ===\n")

    # Load model
    if not MODEL_PATH.exists():
        print(f"ERROR: Model not found at {MODEL_PATH}. Run run_ml_train.py first.")
        return
    model = MLModel.load(MODEL_PATH)
    print(f"Model loaded from {MODEL_PATH}")
    print(f"Skip threshold: {model.threshold:.4f}\n")

    # Load dataset
    if not DATASET_PATH.exists():
        print(f"ERROR: Dataset not found at {DATASET_PATH}. "
              "Run run_ml_collect.py or run_ml_train.py first.")
        return
    df_full = pd.read_parquet(DATASET_PATH)
    df_val  = filter_df(df_full, "validation")

    print(f"Validation split: {SPLITS['validation'][0]} → {SPLITS['validation'][1]}")
    print(f"Trades:           {len(df_val)}\n")

    if len(df_val) == 0:
        print("No validation trades found. Check your dataset and split boundaries.")
        return

    # Predict
    X_val   = df_val[ALL_FEATURE_NAMES]
    y_val   = df_val["r_multiple"].values
    pred_r  = model.predict_r(X_val)
    mask    = pred_r >= model.threshold
    r_taken = y_val[mask]

    # Table
    def _row(label, r):
        if len(r) == 0:
            return f"  {label:<22} {'—':>6}  {'—':>8}  {'—':>8}  {'—':>10}  {'—':>6}"
        return (
            f"  {label:<22} {len(r):>6}  "
            f"{win_rate(r):>8.1%}  "
            f"{sortino_r(r):>8.3f}  "
            f"{profit_factor_r(r):>10.3f}  "
            f"{expectancy_r(r):>6.3f}R"
        )

    header = (f"  {'':22} {'N':>6}  {'WinRate':>8}  {'Sortino':>8}  "
              f"{'ProfitFactor':>10}  {'ExpR':>6}")
    sep    = "  " + "-" * 65
    print(header)
    print(sep)
    print(_row("All trades (baseline)", y_val))
    print(_row(f"ML filtered (th={model.threshold:.3f})", r_taken))
    print()
    print(f"  Take rate: {mask.mean():.1%}  ({mask.sum()} / {len(y_val)} trades taken)")
    print()

    # Threshold sensitivity: show a few alternative thresholds
    from backtest.ml.evaluate import search_threshold
    best_thresh, _ = search_threshold(pred_r, y_val, metric="sortino", min_take_rate=0.15)
    print(f"  Best sortino threshold on this split: {best_thresh:.4f}")
    print()
    print("NOTE: The threshold above is computed ON validation data.")
    print("      Use it for reference only — do not feed it back into training.")


if __name__ == "__main__":
    main()
