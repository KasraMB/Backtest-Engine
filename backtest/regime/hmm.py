"""
backtest/regime/hmm.py
──────────────────────
Gaussian HMM regime detection for intraday strategy analysis.

Two modes:
  OFFLINE  — fit on full period, label all days (for reporting only, look-ahead biased)
  ROLLING  — fit on expanding window up to each day, predict next day's regime
             (forward-safe, usable as a live filter)

States are always sorted by mean return so labels are consistent:
  0 = bear  (lowest mean daily return)
  1 = neutral
  2 = bull  (highest mean daily return)

Usage
─────
  from backtest.regime.hmm import fit_regimes

  result = fit_regimes(
      daily_returns,          # np.ndarray of daily log returns
      daily_dates,            # list of date objects, same length
      n_states=3,
      train_ratio=0.5,        # first 50% used for HMM training
      mode="rolling",         # "offline" | "rolling"
  )

  result.labels              # dict[date, int]  — 0/1/2 per day
  result.label_names         # {0:"bear",1:"neutral",2:"bull"}
  result.train_end_date      # date where training ends
  result.in_sample_dates     # list[date]
  result.out_of_sample_dates # list[date]
  result.transition_matrix   # (n_states, n_states) array
  result.state_means         # (n_states,) mean return per state
  result.state_stds          # (n_states,) std dev per state
  result.avg_duration_days   # {label_name: float}
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import date
from typing import Optional

import numpy as np

LABEL_NAMES = {0: "bear", 1: "neutral", 2: "bull"}


@dataclass
class RegimeResult:
    labels:               dict[date, int]          # date → 0/1/2
    label_names:          dict[int, str]            # 0→"bear" etc.
    train_end_date:       Optional[date]
    in_sample_dates:      list[date]
    out_of_sample_dates:  list[date]
    transition_matrix:    np.ndarray                # (n_states, n_states)
    state_means:          np.ndarray                # (n_states,) sorted ascending
    state_stds:           np.ndarray                # (n_states,)
    avg_duration_days:    dict[str, float]
    n_states:             int = 3


def _sort_states(model, returns_flat: np.ndarray) -> np.ndarray:
    """
    Return a permutation array that maps model states to sorted order
    (ascending mean return: 0=bear, 1=neutral, 2=bull).
    """
    order = np.argsort(model.means_.flatten())
    return order


def _relabel(raw_labels: np.ndarray, order: np.ndarray) -> np.ndarray:
    """Remap raw HMM state indices so that 0=bear, 1=neutral, 2=bull."""
    inv = np.empty_like(order)
    for new_idx, old_idx in enumerate(order):
        inv[old_idx] = new_idx
    return inv[raw_labels]


def _avg_durations_from_trans(trans: np.ndarray, n_states: int) -> dict[int, float]:
    """
    Compute expected average run length per state from the transition matrix.
    For a Markov chain: E[duration in state s] = 1 / (1 - P(s→s))
    This is more stable than counting runs in a label sequence that may have
    label-switching artifacts from rolling HMM refits.
    """
    result = {}
    for s in range(n_states):
        p_stay = float(trans[s, s])
        # Clip to avoid division by zero or negative durations
        p_stay = min(p_stay, 0.9999)
        result[s] = 1.0 / max(1e-6, 1.0 - p_stay)
    return result


def _fit_hmm(returns: np.ndarray, n_states: int, seed: int = 42):
    """Fit a Gaussian HMM and return (model, order) with states sorted by mean."""
    from hmmlearn.hmm import GaussianHMM
    # Sanitise: remove NaN/Inf before fitting
    clean = returns[np.isfinite(returns)]
    if len(clean) < n_states * 5:
        raise ValueError(f"Not enough clean data points ({len(clean)}) to fit HMM")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = GaussianHMM(
            n_components=n_states,
            covariance_type="diag",   # "diag" is stable for 1D; "full" requires >1D
            n_iter=1000,
            random_state=seed,
        )
        model.fit(clean.reshape(-1, 1))
    order = _sort_states(model, clean)
    return model, order


def _build_transition_matrix(labels: np.ndarray, n_states: int) -> np.ndarray:
    """Empirical transition matrix from label sequence."""
    mat = np.zeros((n_states, n_states))
    for a, b in zip(labels[:-1], labels[1:]):
        mat[a, b] += 1
    row_sums = mat.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    return mat / row_sums


def fit_regimes(
    daily_returns: np.ndarray,
    daily_dates:   list[date],
    n_states:      int   = 3,
    train_ratio:   float = 0.5,
    mode:          str   = "rolling",   # "offline" | "rolling"
    seed:          int   = 42,
) -> RegimeResult:
    """
    Fit HMM and return regime labels for every day.

    Parameters
    ──────────
    daily_returns : daily log returns, shape (N,)
    daily_dates   : list of N date objects matching daily_returns
    n_states      : number of HMM states (3 = bear/neutral/bull)
    train_ratio   : fraction of days used for training in rolling mode
    mode          : "offline" (full-period fit, biased) or
                    "rolling" (expanding window, forward-safe)
    """
    n = len(daily_returns)
    if n < 20:
        raise ValueError(f"Need at least 20 daily returns, got {n}")

    train_n         = max(10, int(n * train_ratio))
    train_end_date  = daily_dates[train_n - 1]
    in_sample       = daily_dates[:train_n]
    out_of_sample   = daily_dates[train_n:]

    labels_raw: np.ndarray

    if mode == "offline":
        # Fit on full period — biased, for reporting only
        model, order = _fit_hmm(daily_returns, n_states, seed)
        labels_raw   = _relabel(model.predict(daily_returns.reshape(-1, 1)), order)

    else:  # rolling / expanding window
        # Fit on first train_ratio of data, predict the rest one day at a time
        # using expanding window (each prediction only uses data up to that point)
        labels_raw = np.full(n, -1, dtype=np.int32)

        # In-sample: fit on training data and predict training labels
        model, order = _fit_hmm(daily_returns[:train_n], n_states, seed)
        labels_raw[:train_n] = _relabel(
            model.predict(daily_returns[:train_n].reshape(-1, 1)), order
        )

        # Out-of-sample: refit each day on expanding window
        # For efficiency, refit every `refit_freq` days instead of every day
        refit_freq = max(1, (n - train_n) // 50)   # ~50 refits total
        current_model, current_order = model, order
        last_refit = train_n

        for t in range(train_n, n):
            if (t - last_refit) >= refit_freq:
                current_model, current_order = _fit_hmm(
                    daily_returns[:t], n_states, seed
                )
                last_refit = t
            # Predict current bar using current model
            raw = current_model.predict(daily_returns[:t+1].reshape(-1, 1))[-1]
            labels_raw[t] = _relabel(np.array([raw]), current_order)[0]

        # Fill any remaining -1s (shouldn't happen but safety)
        labels_raw = np.where(labels_raw == -1, 1, labels_raw)

    # ── Build output ──────────────────────────────────────────────────────────
    labels_dict = {d: int(labels_raw[i]) for i, d in enumerate(daily_dates)}

    # Transition matrix from full label sequence
    trans = _build_transition_matrix(labels_raw, n_states)

    # State statistics (from training model)
    train_model, train_order = _fit_hmm(daily_returns[:train_n], n_states, seed)
    sorted_means = train_model.means_.flatten()[train_order]
    # covars_ shape for diag: (n_states, n_features) — take sqrt of variance
    sorted_stds = np.sqrt(train_model.covars_.flatten()[train_order])

    # Average duration per named state — derived from transition matrix
    # (more stable than counting label runs which can have relabelling artifacts)
    trans       = _build_transition_matrix(labels_raw, n_states)
    dur_by_idx  = _avg_durations_from_trans(trans, n_states)
    dur_by_name = {LABEL_NAMES[s]: round(dur_by_idx[s], 1) for s in range(n_states)}

    return RegimeResult(
        labels               = labels_dict,
        label_names          = dict(LABEL_NAMES),
        train_end_date       = train_end_date,
        in_sample_dates      = list(in_sample),
        out_of_sample_dates  = list(out_of_sample),
        transition_matrix    = trans,
        state_means          = sorted_means,
        state_stds           = sorted_stds,
        avg_duration_days    = dur_by_name,
        n_states             = n_states,
    )