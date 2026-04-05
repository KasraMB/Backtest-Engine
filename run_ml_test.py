"""
One-shot model evaluation on held-out test data.

WARNING
-------
Running this script consumes your test set.  Once you have seen the results
and made decisions based on them, the test period is no longer blind.

  test1  2024-01-01 -> 2024-12-31   (first test window — use this first)
  test2  2025-01-01 -> present       (reserve for a second opinion)

Rule of thumb
-------------
Only run test1 when you are fully committed to the current model and would
deploy it regardless of the result (pass or fail).  If you retrain after
seeing test1 results, treat test1 as validation_2 going forward.

Modes
-----
  baseline   Metrics for all trades (no ML filter).
  single     Single joint model filter.
  ensemble   Simulate K configs with majority/unanimous/weighted vote.

Usage
-----
  python run_ml_test.py                           # test1, single mode
  python run_ml_test.py --split test2
  python run_ml_test.py --mode ensemble --configs 0,5,12 --vote majority
  python run_ml_test.py --mode ensemble --n_ensemble 3 --split test1
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
from backtest.ml.ensemble import evaluate_ensemble, per_config_metrics

DATASET_PATH = ROOT / "data" / "ml_dataset.parquet"
MODEL_PATH   = ROOT / "models" / "ict_smc.pkl"


# ---------------------------------------------------------------------------
# Formatting helpers (shared with run_ml_validate)
# ---------------------------------------------------------------------------

def _fmt_row(label: str, n: int, wr: float, sr: float, pf: float, er: float,
             take_n: int | None = None, take_total: int | None = None) -> str:
    take_str = ""
    if take_n is not None and take_total is not None:
        take_str = f"  take={take_n}/{take_total} ({take_n/max(take_total,1):.0%})"
    return (
        f"  {label:<30} {n:>6}  {wr:>8.1%}  {sr:>8.3f}  {pf:>10.3f}  {er:>7.3f}R"
        f"{take_str}"
    )


def _fmt_header() -> tuple[str, str]:
    h = f"  {'':30} {'N':>6}  {'WinRate':>8}  {'Sortino':>8}  {'ProfitFactor':>10}  {'ExpR':>7}"
    s = "  " + "-" * 80
    return h, s


# ---------------------------------------------------------------------------
# Verdict helper
# ---------------------------------------------------------------------------

def _verdict(r_filtered: np.ndarray, r_all: np.ndarray) -> str:
    if len(r_filtered) == 0:
        return "FAIL — model filtered out all trades."
    bs = sortino_r(r_all)
    fs = sortino_r(r_filtered)
    diff = fs - bs
    if fs > bs and fs > 0:
        return (f"PASS — filtered Sortino ({fs:.3f}) > baseline ({bs:.3f}), "
                f"improvement = +{diff:.3f}")
    elif fs > 0:
        return (f"MARGINAL — filtered Sortino ({fs:.3f}) is positive "
                f"but below baseline ({bs:.3f})")
    else:
        return f"FAIL — filtered Sortino is negative ({fs:.3f})"


# ---------------------------------------------------------------------------
# Mode handlers
# ---------------------------------------------------------------------------

def run_baseline(df_test: pd.DataFrame) -> None:
    y = df_test['r_multiple'].values
    print(f"Test trades: {len(y)}\n")
    print("--- Results ---\n")
    h, s = _fmt_header()
    print(h)
    print(s)
    print(_fmt_row("All trades (baseline)", len(y),
                   win_rate(y), sortino_r(y), profit_factor_r(y), expectancy_r(y)))
    print()


def run_single(df_test: pd.DataFrame, model: MLModel,
               config_indices: list[int] | None) -> None:
    if config_indices:
        if 'config_idx' not in df_test.columns:
            print("WARNING: 'config_idx' column not found — ignoring --configs filter.")
        else:
            df_test = df_test[df_test['config_idx'].isin(config_indices)]
            print(f"Filtered to config indices: {config_indices}  ({len(df_test)} trades)\n")

    y    = df_test['r_multiple'].values
    X    = df_test[ALL_FEATURE_NAMES]
    pred = model.predict_r(X)
    mask = pred >= model.threshold
    r_ml = y[mask]

    print(f"Test trades: {len(y)}  |  threshold: {model.threshold:.4f}\n")
    print("--- Results ---\n")
    h, s = _fmt_header()
    print(h)
    print(s)
    print(_fmt_row("All trades (baseline)", len(y),
                   win_rate(y), sortino_r(y), profit_factor_r(y), expectancy_r(y)))
    print(_fmt_row(f"ML filtered (th={model.threshold:.3f})", len(r_ml),
                   win_rate(r_ml), sortino_r(r_ml),
                   profit_factor_r(r_ml), expectancy_r(r_ml),
                   take_n=len(r_ml), take_total=len(y)))
    print()

    print("--- Verdict ---\n")
    print(f"  {_verdict(r_ml, y)}")
    print()


def run_ensemble(df_test: pd.DataFrame, model: MLModel,
                 config_indices: list[int] | None,
                 vote_method: str,
                 n_ensemble: int,
                 split: str) -> None:
    if 'config_idx' not in df_test.columns:
        print("ERROR: Dataset does not have a 'config_idx' column.")
        print("       Run run_ml_collect.py (multi-config) first.")
        return

    available = sorted(df_test['config_idx'].unique().tolist())

    # Resolve config indices
    if config_indices:
        missing = [c for c in config_indices if c not in available]
        if missing:
            print(f"WARNING: config indices {missing} not found in test data — skipping.")
        indices = [c for c in config_indices if c in available]
    else:
        # Auto-select using VALIDATION data (never use test data to pick configs)
        from backtest.ml.splits import filter_df as _filter_df
        import pandas as _pd
        df_full = _pd.read_parquet(DATASET_PATH)
        df_val  = _filter_df(df_full, "validation")
        if len(df_val) == 0 or 'config_idx' not in df_val.columns:
            print("WARNING: Could not load validation data for auto-selection. "
                  "Using first available config indices.")
            indices = available[:n_ensemble]
        else:
            val_available = df_val['config_idx'].isin(available)
            df_val_filtered = df_val[val_available]
            cfg_stats = per_config_metrics(df_val_filtered, model)
            cfg_stats_sorted = sorted(cfg_stats, key=lambda r: r['sortino'], reverse=True)
            indices = [r['config_idx'] for r in cfg_stats_sorted[:n_ensemble]]
        print(f"Auto-selected top {n_ensemble} configs from validation: {indices}")

    if not indices:
        print("No valid config indices. Aborting.")
        return

    print(f"\nEnsemble: {len(indices)} config(s), vote={vote_method}")
    print(f"Configs: {indices}\n")

    # Per-config metrics on test data
    df_sel = df_test[df_test['config_idx'].isin(indices)]
    cfg_stats = per_config_metrics(df_sel, model)
    print("Per-config metrics:")
    h, s = _fmt_header()
    print(h)
    print(s)
    for row in sorted(cfg_stats, key=lambda r: r['sortino'], reverse=True):
        label = f"  Config {row['config_idx']}"
        print(_fmt_row(label, row['n_taken'],
                       row['win_rate'], row['sortino'],
                       row['profit_factor'], row['expectancy_r'],
                       take_n=row['n_taken'], take_total=row['n_all']))
    print()

    # Ensemble result
    result  = evaluate_ensemble(df_test, model, indices, vote_method=vote_method)
    y_all   = df_test['r_multiple'].values
    r_ens   = result['r_multiples']

    print("--- Results ---\n")
    print(h)
    print(s)
    print(_fmt_row("All trades (baseline)", len(y_all),
                   win_rate(y_all), sortino_r(y_all),
                   profit_factor_r(y_all), expectancy_r(y_all)))
    label = f"Ensemble ({vote_method}, {len(indices)}cfg)"
    print(_fmt_row(label, result['n_taken'],
                   result['win_rate'], result['sortino'],
                   result['profit_factor'], result['expectancy_r'],
                   take_n=result['n_taken'], take_total=result['n_signals']))
    print()

    print("--- Verdict ---\n")
    print(f"  {_verdict(r_ens, y_all)}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="One-shot test evaluation (consumes held-out set).")
    parser.add_argument('--split', default='test1', choices=['test1', 'test2'])
    parser.add_argument('--mode', default='single',
                        choices=['baseline', 'single', 'ensemble'],
                        help="Evaluation mode (default: single)")
    parser.add_argument('--configs', default=None,
                        help="Comma-separated config indices, e.g. '0,5,12'")
    parser.add_argument('--vote', default='majority',
                        choices=['majority', 'unanimous', 'weighted'],
                        help="Ensemble vote method (default: majority)")
    parser.add_argument('--n_ensemble', type=int, default=3,
                        help="Top-N configs for auto-selection in ensemble mode (default: 3)")
    args = parser.parse_args()

    config_indices = None
    if args.configs:
        try:
            config_indices = [int(x.strip()) for x in args.configs.split(',')]
        except ValueError:
            print(f"ERROR: --configs must be comma-separated integers, got '{args.configs}'")
            return

    split     = args.split
    start_str, end_str = SPLITS[split]
    period    = f"{start_str} -> {end_str or 'present'}"

    print("=" * 60)
    print(f"  ML TEST EVALUATION — {split.upper()}  [{args.mode.upper()}]")
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

    # Load dataset
    if not DATASET_PATH.exists():
        print(f"ERROR: Dataset not found at {DATASET_PATH}.")
        return
    df_full = pd.read_parquet(DATASET_PATH)
    df_test = filter_df(df_full, split)

    if len(df_test) == 0:
        print("No test trades found for this split. Check your dataset.")
        return

    if args.mode == 'baseline':
        run_baseline(df_test)
        print("  Reminder: if you retrain after seeing this, treat this period")
        print(f"  as validation_2 and reserve "
              f"{'test2' if split == 'test1' else 'new live data'} as your next blind set.")
        return

    # Load model
    if not MODEL_PATH.exists():
        print(f"ERROR: Model not found at {MODEL_PATH}. Run run_ml_train.py first.")
        return
    model = MLModel.load(MODEL_PATH)
    print(f"Model:     {MODEL_PATH}")
    print(f"Threshold: {model.threshold:.4f}\n")

    if args.mode == 'single':
        run_single(df_test, model, config_indices)
    elif args.mode == 'ensemble':
        run_ensemble(df_test, model, config_indices, args.vote, args.n_ensemble, split)

    print("  Reminder: if you retrain after seeing this, treat this period")
    print(f"  as validation_2 and reserve "
          f"{'test2' if split == 'test1' else 'new live data'} as your next blind set.")


if __name__ == "__main__":
    main()
