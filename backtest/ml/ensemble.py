"""
Ensemble evaluation for the ICT/SMC ML pipeline.

An ensemble groups multiple Phase 1 configs and only takes a trade when
a threshold fraction (majority / unanimous) of those configs agree.

Dataset-level evaluation
------------------------
Since the training dataset contains trades from many LHS configs, we can
simulate any ensemble of config indices without re-running backtests.

For each unique entry bar in the filtered dataset we check how many of the
selected configs generate and would take that signal (model prediction >=
threshold).  The vote result determines whether the ensemble takes it.

Usage
-----
    from backtest.ml.ensemble import evaluate_ensemble, per_config_metrics

    # metrics for every config individually
    by_cfg = per_config_metrics(df_val, model)

    # pick the top 3 by sortino and run ensemble
    top3 = sorted(by_cfg, key=lambda r: r['sortino'], reverse=True)[:3]
    indices = [r['config_idx'] for r in top3]
    result = evaluate_ensemble(df_val, model, config_indices=indices, vote_method='majority')
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from backtest.ml.evaluate import sortino_r, profit_factor_r, win_rate, expectancy_r
from backtest.ml.features import ALL_FEATURE_NAMES


# ---------------------------------------------------------------------------
# Vote functions
# ---------------------------------------------------------------------------

def majority_vote(votes: Sequence[bool]) -> bool:
    """True if strictly more than half of votes are True."""
    return sum(votes) > len(votes) / 2


def unanimous_vote(votes: Sequence[bool]) -> bool:
    """True only if every vote is True."""
    return all(votes)


def weighted_vote(votes: Sequence[bool], weights: Sequence[float]) -> bool:
    """True if the weighted fraction of True votes exceeds 0.5."""
    total = sum(weights)
    if total <= 0:
        return False
    weighted_true = sum(w for v, w in zip(votes, weights) if v)
    return (weighted_true / total) > 0.5


# ---------------------------------------------------------------------------
# Per-config metrics
# ---------------------------------------------------------------------------

def per_config_metrics(
    df: pd.DataFrame,
    model,
) -> list[dict]:
    """
    Compute per-config metrics for every config_idx present in *df*.

    Columns required: ALL_FEATURE_NAMES, 'r_multiple', 'config_idx'.

    Returns
    -------
    List of dicts, one per config_idx, each containing:
        config_idx, n_all, n_taken, take_rate,
        win_rate, sortino, profit_factor, expectancy_r
    """
    if 'config_idx' not in df.columns:
        raise ValueError("DataFrame must have a 'config_idx' column. "
                         "Run run_ml_collect.py to generate the multi-config dataset.")

    rows = []
    for cfg_idx, grp in df.groupby('config_idx', sort=True):
        y   = grp['r_multiple'].values
        X   = grp[ALL_FEATURE_NAMES]
        pred = model.predict_r(X)
        mask = pred >= model.threshold
        r_taken = y[mask]

        rows.append({
            'config_idx':   int(cfg_idx),
            'n_all':        len(y),
            'n_taken':      int(mask.sum()),
            'take_rate':    float(mask.mean()) if len(y) > 0 else 0.0,
            'win_rate':     float(win_rate(r_taken)),
            'sortino':      float(sortino_r(r_taken)),
            'profit_factor': float(profit_factor_r(r_taken)),
            'expectancy_r': float(expectancy_r(r_taken)),
        })
    return rows


# ---------------------------------------------------------------------------
# Ensemble evaluation
# ---------------------------------------------------------------------------

def evaluate_ensemble(
    df: pd.DataFrame,
    model,
    config_indices: list[int],
    vote_method: str = 'majority',
    weights: list[float] | None = None,
) -> dict:
    """
    Simulate an ensemble of Phase 1 configs on a pre-collected dataset.

    For each unique entry_bar in the dataset (across all selected configs),
    we check which configs would take it (model pred >= threshold) and apply
    the vote rule.  The R-multiple attributed to an ensemble trade is the mean
    R-multiple of the configs that voted to take it.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset slice (e.g. validation or test split).
        Must contain: ALL_FEATURE_NAMES, 'r_multiple', 'config_idx', 'entry_bar'.
    model : MLModel
        Trained model with threshold set.
    config_indices : list[int]
        Which config_idx values form the ensemble.
    vote_method : str
        'majority'   — strictly more than half must take it
        'unanimous'  — all configs must take it
        'weighted'   — weighted sum > 0.5 (requires weights)
    weights : list[float] | None
        Per-config weights (same order as config_indices) for weighted vote.
        Ignored for other vote methods.  Defaults to equal weights.

    Returns
    -------
    dict with keys:
        n_signals       — unique entry bars seen across selected configs
        n_taken         — ensemble-voted trades
        take_rate       — n_taken / n_signals
        win_rate, sortino, profit_factor, expectancy_r
        r_multiples     — np.ndarray of per-trade R (for further analysis)
        per_config      — list of per-config metric dicts (from per_config_metrics)
    """
    required = {'config_idx', 'entry_bar', 'r_multiple'}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")

    if len(config_indices) == 0:
        raise ValueError("config_indices must not be empty.")

    # Filter to selected configs only
    df_ens = df[df['config_idx'].isin(config_indices)].copy()
    if len(df_ens) == 0:
        return _empty_result(config_indices)

    # Precompute predictions for the filtered slice
    X    = df_ens[ALL_FEATURE_NAMES]
    pred = model.predict_r(X)
    df_ens = df_ens.copy()
    df_ens['_pred']  = pred
    df_ens['_taken'] = pred >= model.threshold

    # Build per-config lookup: entry_bar → (voted_to_take, r_multiple)
    # For each unique entry_bar, collect votes from all configs that have it
    signals: dict[int, list] = {}   # entry_bar → list of (config_idx, taken, r)
    for _, row in df_ens.iterrows():
        eb = int(row['entry_bar'])
        signals.setdefault(eb, []).append((
            int(row['config_idx']),
            bool(row['_taken']),
            float(row['r_multiple']),
        ))

    # Build weight map (config_idx → weight)
    if weights is None:
        weight_map = {ci: 1.0 for ci in config_indices}
    else:
        weight_map = {ci: float(w) for ci, w in zip(config_indices, weights)}

    ensemble_r: list[float] = []
    n_signals = len(signals)

    for eb, entries in signals.items():
        # Only consider configs that are in our ensemble
        ensemble_entries = [(ci, taken, r) for ci, taken, r in entries
                            if ci in weight_map]
        if not ensemble_entries:
            continue

        votes    = [taken for _, taken, _ in ensemble_entries]
        cfg_ids  = [ci    for ci, _, _   in ensemble_entries]
        r_vals   = [r     for _, _, r    in ensemble_entries]
        ws       = [weight_map[ci] for ci in cfg_ids]

        if vote_method == 'majority':
            take = majority_vote(votes)
        elif vote_method == 'unanimous':
            take = unanimous_vote(votes)
        elif vote_method == 'weighted':
            take = weighted_vote(votes, ws)
        else:
            raise ValueError(f"Unknown vote_method '{vote_method}'. "
                             "Use 'majority', 'unanimous', or 'weighted'.")

        if take:
            # R-multiple = mean of configs that voted True
            taking_r = [r for v, r in zip(votes, r_vals) if v]
            ensemble_r.append(float(np.mean(taking_r)))

    r_arr = np.array(ensemble_r, dtype=float)

    # Per-config breakdown on this slice
    df_sel = df[df['config_idx'].isin(config_indices)]
    per_cfg = per_config_metrics(df_sel, model) if len(df_sel) > 0 else []

    return {
        'n_signals':     n_signals,
        'n_taken':       len(r_arr),
        'take_rate':     len(r_arr) / max(n_signals, 1),
        'win_rate':      float(win_rate(r_arr)),
        'sortino':       float(sortino_r(r_arr)),
        'profit_factor': float(profit_factor_r(r_arr)),
        'expectancy_r':  float(expectancy_r(r_arr)),
        'r_multiples':   r_arr,
        'per_config':    per_cfg,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _empty_result(config_indices: list[int]) -> dict:
    return {
        'n_signals':     0,
        'n_taken':       0,
        'take_rate':     0.0,
        'win_rate':      0.0,
        'sortino':       0.0,
        'profit_factor': 0.0,
        'expectancy_r':  0.0,
        'r_multiples':   np.array([], dtype=float),
        'per_config':    [],
    }
