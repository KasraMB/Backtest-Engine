"""
One-shot model evaluation on held-out test data.

WARNING
-------
Running this script consumes your test set.  Once you have seen the results
and made decisions based on them, the test period is no longer blind.

  test1  2024-01-01 → 2024-12-31   (first test window — use this first)
  test2  2025-01-01 → present       (reserve for a second opinion)

Rule of thumb
-------------
Only run test1 when you are fully committed to the current model and would
deploy it regardless of the result (pass or fail).  If you retrain after
seeing test1 results, treat test1 as validation_2 going forward — it is no
longer a true held-out set.

Usage
-----
  python run_ml_test.py              # evaluates test1 (default)
  python run_ml_test.py --split test2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

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


def _print_metrics(label: str, r: np.ndarray) -> None:
    if len(r) == 0:
        print(f"  {label}: no trades")
        return
    print(f"  {label}:")
    print(f"    N:             {len(r)}")
    print(f"    Win rate:      {win_rate(r):.1%}")
    print(f"    Sortino R:     {sortino_r(r):.3f}")
    print(f"    Profit factor: {profit_factor_r(r):.3f}")
    print(f"    Expectancy R:  {expectancy_r(r):.3f}")


def main(split: str = "test1") -> None:
    if split not in ("test1", "test2"):
        print(f"ERROR: Unknown split '{split}'. Use 'test1' or 'test2'.")
        return

    start_str, end_str = SPLITS[split]
    period = f"{start_str} → {end_str or 'present'}"

    print("=" * 60)
    print(f"  ML TEST EVALUATION — {split.upper()}")
    print(f"  Period: {period}")
    print("=" * 60)
    print()
    print("  !! WARNING: Running this will consume your test set.")
    print("  !! Once you see these results, this period is no longer blind.")
    print()
    confirm = input("  Type  yes  to proceed, anything else to abort: ").strip().lower()
    if confirm != "yes":
        print("  Aborted.")
        return

    print()

    # Load model
    if not MODEL_PATH.exists():
        print(f"ERROR: Model not found at {MODEL_PATH}. Run run_ml_train.py first.")
        return
    model = MLModel.load(MODEL_PATH)
    print(f"Model:     {MODEL_PATH}")
    print(f"Threshold: {model.threshold:.4f}\n")

    # Load dataset
    if not DATASET_PATH.exists():
        print(f"ERROR: Dataset not found at {DATASET_PATH}.")
        return
    df_full = pd.read_parquet(DATASET_PATH)
    df_test = filter_df(df_full, split)

    print(f"Test trades: {len(df_test)}\n")

    if len(df_test) == 0:
        print("No test trades found for this split. Check your dataset.")
        return

    # Predict
    X_test  = df_test[ALL_FEATURE_NAMES]
    y_test  = df_test["r_multiple"].values
    pred_r  = model.predict_r(X_test)
    mask    = pred_r >= model.threshold
    r_taken = y_test[mask]

    print("--- Results ---\n")
    _print_metrics("All trades (baseline)", y_test)
    print()
    _print_metrics(f"ML-filtered (threshold={model.threshold:.4f})", r_taken)
    print()
    print(f"  Take rate: {mask.mean():.1%}  ({mask.sum()} / {len(y_test)})")
    print()

    # Verdict
    baseline_sortino = sortino_r(y_test)
    filtered_sortino = sortino_r(r_taken)
    improvement      = filtered_sortino - baseline_sortino

    print("--- Verdict ---\n")
    if len(r_taken) == 0:
        print("  FAIL — model filtered out all trades.")
    elif filtered_sortino > baseline_sortino and filtered_sortino > 0:
        print(f"  PASS — filtered Sortino ({filtered_sortino:.3f}) > "
              f"baseline ({baseline_sortino:.3f}), improvement = +{improvement:.3f}")
    elif filtered_sortino > 0:
        print(f"  MARGINAL — filtered Sortino ({filtered_sortino:.3f}) is positive "
              f"but below baseline ({baseline_sortino:.3f})")
    else:
        print(f"  FAIL — filtered Sortino is negative ({filtered_sortino:.3f})")

    print()
    print("  Reminder: if you retrain after seeing this, treat this period")
    print(f"  as validation_2 and reserve {('test2' if split == 'test1' else 'new live data')} as your next blind set.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test1", choices=["test1", "test2"])
    args = parser.parse_args()
    main(split=args.split)
