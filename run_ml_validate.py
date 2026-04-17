"""
Evaluate the trained ML model on the validation split.

Safe to run as many times as you like — validation data is not a held-out
test set; it is used for threshold tuning and model comparison.

Modes
-----
  baseline   Print metrics for all trades with no ML filter.
  single     Filter using the ML model (single joint model, any config).
  ensemble   Simulate running K configs simultaneously; only take trades
             where a majority (or chosen vote method) agree.

Usage
-----
  # single-model filter
  python run_ml_validate.py

  # baseline (no ML)
  python run_ml_validate.py --mode baseline

  # ensemble: auto-select top 3 configs by validation sortino
  python run_ml_validate.py --mode ensemble --n_ensemble 3

  # ensemble: specify exact config indices, unanimous vote
  python run_ml_validate.py --mode ensemble --configs 0,5,12 --vote unanimous

  # single mode for a specific config only
  python run_ml_validate.py --mode single --configs 7
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import json

import numpy as np
import pandas as pd

from backtest.ml.model import MLModel, EnsembleMLModel
from backtest.ml.splits import filter_df, SPLITS
from backtest.ml.features import ALL_FEATURE_NAMES
from backtest.ml.evaluate import sortino_r, profit_factor_r, win_rate, expectancy_r, search_threshold
from backtest.ml.ensemble import evaluate_ensemble, per_config_metrics

DATASET_PATH        = ROOT / "data"   / "ml_dataset.parquet"
MODEL_PATH          = ROOT / "models" / "ict_smc_ensemble.pkl"
THRESHOLD_OPT_PATH  = ROOT / "models" / "threshold_opt.json"


# ---------------------------------------------------------------------------
# Score helper — works for both MLModel and EnsembleMLModel
# ---------------------------------------------------------------------------

def _compute_scores(model: MLModel | EnsembleMLModel, X: pd.DataFrame) -> np.ndarray:
    if isinstance(model, EnsembleMLModel):
        pred_r = model._model_a.predict_r(X)
        p_win  = model._model_b.predict_proba(X.values.astype(float))[:, 1]
        return pred_r * p_win
    return model.predict_r(X)


def _load_threshold_opt() -> float | None:
    if THRESHOLD_OPT_PATH.exists():
        with open(THRESHOLD_OPT_PATH) as f:
            opt = json.load(f)
        if opt:
            return float(next(iter(opt.values()))['threshold'])
    return None


# ---------------------------------------------------------------------------
# Formatting helpers
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


def _print_metrics_block(label: str, r: np.ndarray, r_all: np.ndarray) -> None:
    h, s = _fmt_header()
    print(h)
    print(s)
    n_all = len(r_all)
    print(_fmt_row("All trades (baseline)", n_all,
                   win_rate(r_all), sortino_r(r_all),
                   profit_factor_r(r_all), expectancy_r(r_all)))
    n = len(r)
    print(_fmt_row(label, n,
                   win_rate(r), sortino_r(r),
                   profit_factor_r(r), expectancy_r(r),
                   take_n=n, take_total=n_all))
    print()


# ---------------------------------------------------------------------------
# Mode handlers
# ---------------------------------------------------------------------------

def run_baseline(df_val: pd.DataFrame) -> None:
    y = df_val['r_multiple'].values
    print(f"Trades: {len(y)}\n")
    h, s = _fmt_header()
    print(h)
    print(s)
    print(_fmt_row("All trades (baseline)", len(y),
                   win_rate(y), sortino_r(y), profit_factor_r(y), expectancy_r(y)))
    print()


def run_single(df_val: pd.DataFrame, model: MLModel | EnsembleMLModel,
               config_indices: list[int] | None) -> None:
    if config_indices:
        if 'config_idx' not in df_val.columns:
            print("WARNING: 'config_idx' column not found — ignoring --configs filter.")
        else:
            df_val = df_val[df_val['config_idx'].isin(config_indices)]
            print(f"Filtered to config indices: {config_indices}  ({len(df_val)} trades)\n")

    y     = df_val['r_multiple'].values
    X     = df_val[ALL_FEATURE_NAMES]
    pred  = _compute_scores(model, X)
    mask  = pred >= model.threshold
    r_all = y
    r_ml  = y[mask]

    print(f"Trades: {len(y)}  |  threshold: {model.threshold:.4f}\n")
    _print_metrics_block(f"ML filtered (th={model.threshold:.3f})", r_ml, r_all)

    best_t, _ = search_threshold(pred, y, metric="sortino", min_take_rate=0.15)
    print(f"  Best sortino threshold on this split: {best_t:.4f}")
    print("  (reference only — do not feed back into training)\n")


def run_ensemble(df_val: pd.DataFrame, model: MLModel,
                 config_indices: list[int] | None,
                 vote_method: str,
                 n_ensemble: int) -> None:
    if 'config_idx' not in df_val.columns:
        print("ERROR: Dataset does not have a 'config_idx' column.")
        print("       Run run_ml_collect.py (multi-config) first.")
        return

    available = sorted(df_val['config_idx'].unique().tolist())
    print(f"Available config indices: {available[:20]}"
          f"{'...' if len(available) > 20 else ''} ({len(available)} total)")

    # Resolve which configs to use
    if config_indices:
        missing = [c for c in config_indices if c not in available]
        if missing:
            print(f"WARNING: config indices {missing} not found in dataset — skipping them.")
        indices = [c for c in config_indices if c in available]
    else:
        # Auto-select top N by per-config validation sortino
        print(f"\nAuto-selecting top {n_ensemble} configs by validation sortino...")
        cfg_stats = per_config_metrics(df_val, model)
        cfg_stats_sorted = sorted(cfg_stats, key=lambda r: r['sortino'], reverse=True)
        indices = [r['config_idx'] for r in cfg_stats_sorted[:n_ensemble]]
        print(f"Selected: {indices}")

    if not indices:
        print("No valid config indices. Aborting.")
        return

    print(f"\nEnsemble: {len(indices)} config(s), vote={vote_method}")
    print(f"Configs: {indices}\n")

    # Per-config metrics table
    cfg_stats = per_config_metrics(df_val[df_val['config_idx'].isin(indices)], model)
    print("Per-config metrics (before ensemble vote):")
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
    result = evaluate_ensemble(df_val, model, indices, vote_method=vote_method)
    y_all  = df_val['r_multiple'].values
    r_ens  = result['r_multiples']

    print("Ensemble result:")
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Validate ML model on validation split.")
    parser.add_argument('--mode', default='single',
                        choices=['baseline', 'single', 'ensemble'],
                        help="Evaluation mode (default: single)")
    parser.add_argument('--configs', default=None,
                        help="Comma-separated config indices, e.g. '0,5,12'")
    parser.add_argument('--vote', default='majority',
                        choices=['majority', 'unanimous', 'weighted'],
                        help="Ensemble vote method (default: majority)")
    parser.add_argument('--n_ensemble', type=int, default=3,
                        help="Number of top configs for auto-selection in ensemble mode (default: 3)")
    args = parser.parse_args()

    config_indices = None
    if args.configs:
        try:
            config_indices = [int(x.strip()) for x in args.configs.split(',')]
        except ValueError:
            print(f"ERROR: --configs must be comma-separated integers, got '{args.configs}'")
            return

    print(f"=== ML Validation — mode={args.mode} ===\n")

    # Load dataset
    if not DATASET_PATH.exists():
        print(f"ERROR: Dataset not found at {DATASET_PATH}. "
              "Run run_ml_collect.py or run_ml_train.py first.")
        return
    df_full = pd.read_parquet(DATASET_PATH)
    df_val  = filter_df(df_full, "validation")

    print(f"Validation split: {SPLITS['validation'][0]} -> {SPLITS['validation'][1]}")
    print(f"Trades:           {len(df_val)}\n")

    if len(df_val) == 0:
        print("No validation trades found. Check your dataset and split boundaries.")
        return

    if args.mode == 'baseline':
        run_baseline(df_val)
        return

    # Load ensemble model for single / ensemble modes
    if not MODEL_PATH.exists():
        print(f"ERROR: Model not found at {MODEL_PATH}. Run run_ml_train.py first.")
        return
    model = EnsembleMLModel.load(MODEL_PATH)
    # Apply optimal threshold from threshold_opt.json if available
    opt_thr = _load_threshold_opt()
    if opt_thr is not None:
        model.threshold = opt_thr
    print(f"Model:     {MODEL_PATH}")
    print(f"Threshold: {model.threshold:.4f}"
          f"  ({'threshold_opt.json' if opt_thr is not None else 'model default'})\n")

    if args.mode == 'single':
        run_single(df_val, model, config_indices)
    elif args.mode == 'ensemble':
        run_ensemble(df_val, model, config_indices, args.vote, args.n_ensemble)


if __name__ == "__main__":
    import time as _time
    _t0 = _time.perf_counter()
    main()
    print(f"\nTotal time: {_time.perf_counter() - _t0:.1f}s")
