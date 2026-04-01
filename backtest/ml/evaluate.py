"""
Evaluation metrics for the ML pipeline.

All metrics operate on numpy arrays of R-multiples (one value per trade).
"""
from __future__ import annotations

import numpy as np


def sortino_r(r_series: np.ndarray, target: float = 0.0) -> float:
    """
    Sortino ratio computed on per-trade R-multiples.
    Only penalises downside deviation below `target`.
    Returns 0.0 if there are no trades or no downside.
    """
    r = np.asarray(r_series, dtype=float)
    if len(r) == 0:
        return 0.0
    mean_r = float(np.mean(r))
    downside = r[r < target] - target
    if len(downside) == 0:
        return float('inf') if mean_r > 0 else 0.0
    downside_std = float(np.sqrt(np.mean(downside ** 2)))
    if downside_std < 1e-9:
        return float('inf') if mean_r > target else 0.0
    return (mean_r - target) / downside_std


def profit_factor_r(r_series: np.ndarray) -> float:
    """
    Profit factor: sum of winning R / abs(sum of losing R).
    Returns inf if no losing trades, 0.0 if no winning trades.
    """
    r = np.asarray(r_series, dtype=float)
    gross_win  = float(np.sum(r[r > 0]))
    gross_loss = float(np.abs(np.sum(r[r < 0])))
    if gross_loss < 1e-9:
        return float('inf') if gross_win > 0 else 0.0
    return gross_win / gross_loss


def expectancy_r(r_series: np.ndarray) -> float:
    r = np.asarray(r_series, dtype=float)
    return float(np.mean(r)) if len(r) > 0 else 0.0


def win_rate(r_series: np.ndarray) -> float:
    r = np.asarray(r_series, dtype=float)
    if len(r) == 0:
        return 0.0
    return float(np.mean(r > 0))


def evaluate_filter(r_taken: np.ndarray, r_all: np.ndarray) -> dict:
    """
    Compare metrics for taken trades vs. all trades.
    Useful for threshold analysis.
    """
    def _stats(r: np.ndarray) -> dict:
        return {
            'n':              len(r),
            'win_rate':       win_rate(r),
            'expectancy_r':   expectancy_r(r),
            'sortino':        sortino_r(r),
            'profit_factor':  profit_factor_r(r),
        }
    return {
        'taken': _stats(r_taken),
        'all':   _stats(r_all),
        'take_rate': len(r_taken) / max(len(r_all), 1),
    }


def search_threshold(
    predicted_r: np.ndarray,
    actual_r: np.ndarray,
    metric: str = 'sortino',
    n_steps: int = 50,
    min_take_rate: float = 0.15,
) -> tuple[float, dict]:
    """
    Scan skip thresholds and return the one that maximises `metric` on the
    provided predictions vs. actuals.

    Parameters
    ----------
    predicted_r : np.ndarray
        Model's predicted R-multiple per trade.
    actual_r : np.ndarray
        Actual R-multiple per trade.
    metric : str
        One of 'sortino', 'profit_factor', 'expectancy_r', 'win_rate'.
    n_steps : int
        How many threshold values to scan.
    min_take_rate : float
        Discard thresholds that result in fewer than this fraction of trades taken.

    Returns
    -------
    (best_threshold, metrics_at_best_threshold)
    """
    _metric_fns = {
        'sortino':       sortino_r,
        'profit_factor': profit_factor_r,
        'expectancy_r':  expectancy_r,
        'win_rate':      win_rate,
    }
    fn = _metric_fns[metric]

    thresholds = np.linspace(
        float(np.percentile(predicted_r, 5)),
        float(np.percentile(predicted_r, 95)),
        n_steps,
    )

    best_thresh = thresholds[0]
    best_score  = -np.inf
    best_stats: dict = {}

    for thresh in thresholds:
        mask = predicted_r >= thresh
        if mask.sum() < max(1, int(min_take_rate * len(actual_r))):
            continue
        r_taken = actual_r[mask]
        score   = fn(r_taken)
        if score > best_score:
            best_score  = score
            best_thresh = float(thresh)
            best_stats  = evaluate_filter(r_taken, actual_r)

    return best_thresh, best_stats
