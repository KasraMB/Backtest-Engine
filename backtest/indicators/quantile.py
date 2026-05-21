"""
Per-HMM-state quantile + block-bootstrap CI.

Per v9-final: q-quantile for orderflow proxies is computed per HMM state.
Stage 4 monitor: if bootstrap CI width > 15% of threshold value, fall back to
universal q regardless of DSR.
"""
from __future__ import annotations

import numpy as np


def per_state_quantile(
    values: np.ndarray,
    state_labels: np.ndarray,
    q: float,
    n_states: int,
) -> dict[int, float]:
    """Quantile of `values` within each HMM state.

    States with fewer than 10 observations get NaN (caller should fall back
    to the universal quantile).
    """
    if values.shape != state_labels.shape:
        raise ValueError(f"shape mismatch: values {values.shape} vs labels {state_labels.shape}")
    # Min-obs floor: a q-quantile from n samples has variance ~ q(1-q)/(n·f(x_q)^2).
    # For q=0.90 with n=10, the 90th percentile is literally the max of the sample.
    # Require ≥30 obs (≥3 × 1/(1-q) at q=0.90) before trusting the estimate.
    MIN_OBS = 30
    out: dict[int, float] = {}
    for s in range(n_states):
        mask = state_labels == s
        if mask.sum() < MIN_OBS:
            out[s] = float('nan')
            continue
        out[s] = float(np.quantile(values[mask], q))
    return out


def bootstrap_quantile_ci(
    values: np.ndarray,
    q: float,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 2026,
) -> tuple[float, float]:
    """Percentile-bootstrap CI for the q-quantile of `values`.

    Fully vectorised — single (n_boot, n) resample matrix, one np.quantile call.
    For typical (n≈500, n_boot=1000) this is ~4 MB and <50 ms.

    Returns (lo, hi) at the requested CI level (default 95%).
    """
    n = len(values)
    if n < 10:
        return float('nan'), float('nan')
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))   # (n_boot, n) resample indices
    boot = np.asarray(values, dtype=np.float64)[idx]
    samples = np.quantile(boot, q, axis=1)        # one call → (n_boot,) quantiles
    alpha = (1.0 - ci) / 2.0
    return float(np.quantile(samples, alpha)), float(np.quantile(samples, 1.0 - alpha))


def ci_width_pct_of_threshold(
    values: np.ndarray,
    q: float,
    n_boot: int = 1000,
    seed: int = 2026,
) -> float:
    """Return CI width as a fraction of the point-estimate threshold.

    Used by the v9-final stage-4 monitor: if return > 0.15 → fall back to
    universal q for this state regardless of DSR result.
    """
    if len(values) < 10:
        return float('inf')
    lo, hi = bootstrap_quantile_ci(values, q, n_boot=n_boot, seed=seed)
    point = float(np.quantile(values, q))
    if abs(point) < 1e-12:
        return float('inf')
    return (hi - lo) / abs(point)
