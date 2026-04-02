"""
Parameter sensitivity checking for the ICT/SMC ML pipeline.

Caching
-------
`_hash_params(params)` produces a short deterministic key for any params dict.
The caller (e.g. run_ml_collect.py) is responsible for maintaining the cache
so that results can be reused across runs without re-running backtests.

A parameter config is considered stable if perturbing each continuous parameter
by ±perturbation_pct does not degrade the evaluation metric by more than
max_degradation_pct relative to the base result.

This is intentionally kept simple: it re-runs the backtest for each perturbation.
For large grids, pre-compute results offline and cache them.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


def _hash_params(params: dict) -> str:
    """Return a 16-char hex key that uniquely identifies a params dict."""
    key = json.dumps(dict(sorted(params.items())), sort_keys=True, default=str)
    return hashlib.sha256(key.encode()).hexdigest()[:16]


@dataclass
class SensitivityResult:
    is_stable:       bool
    base_metric:     float
    worst_metric:    float        # lowest metric across all perturbations
    worst_param:     str          # which param caused the worst degradation
    degradation_pct: float        # (base - worst) / |base| * 100, or 0 if base=0


def check_sensitivity(
    params: dict,
    run_fn: Callable[[dict], float],
    perturbation_pct: float = 0.15,
    max_degradation_pct: float = 30.0,
    params_to_perturb: Optional[list[str]] = None,
) -> SensitivityResult:
    """
    Check whether a parameter config is stable under small perturbations.

    Parameters
    ----------
    params : dict
        Strategy parameter dict to test.
    run_fn : Callable[[dict], float]
        Function that takes a params dict and returns a scalar metric
        (e.g. Sortino R or profit factor).
    perturbation_pct : float
        Fraction to perturb each parameter up and down (default 15%).
    max_degradation_pct : float
        Maximum allowed relative degradation in metric (default 30%).
    params_to_perturb : list[str], optional
        Which keys to perturb.  Defaults to a standard set of continuous params.

    Returns
    -------
    SensitivityResult
    """
    if params_to_perturb is None:
        params_to_perturb = [
            'confluence_tolerance_atr_mult',
            'tp_confluence_tolerance_atr_mult',
            'level_penetration_atr_mult',
            'min_rr',
            'tick_offset_atr_mult',
            'po3_atr_mult',
            'po3_band_pct',
            'po3_vol_sens',
            'po3_min_manipulation_size_atr_mult',
            'min_ote_size_atr_mult',
            'cancel_pct_to_tp',
        ]

    base_metric = run_fn(params)

    worst_metric = base_metric
    worst_param  = ''

    for key in params_to_perturb:
        if key not in params:
            continue
        val = params[key]
        if not isinstance(val, (int, float)) or val == 0:
            continue

        for sign in (+1, -1):
            perturbed = dict(params)
            perturbed[key] = val * (1.0 + sign * perturbation_pct)
            m = run_fn(perturbed)
            if m < worst_metric:
                worst_metric = m
                worst_param  = key

    if abs(base_metric) < 1e-9:
        degradation_pct = 0.0
    else:
        degradation_pct = (base_metric - worst_metric) / abs(base_metric) * 100.0

    is_stable = degradation_pct <= max_degradation_pct

    return SensitivityResult(
        is_stable=is_stable,
        base_metric=base_metric,
        worst_metric=worst_metric,
        worst_param=worst_param,
        degradation_pct=degradation_pct,
    )


def build_validated_config_grid(
    base_params: dict,
    param_grid: dict,
    run_fn: Callable[[dict], float],
    perturbation_pct: float = 0.15,
    max_degradation_pct: float = 30.0,
    min_base_metric: float = 0.1,
) -> list[dict]:
    """
    Generate a list of pre-validated parameter configs from a grid.

    Parameters
    ----------
    base_params : dict
        Default parameters.  Grid values override these.
    param_grid : dict
        {param_name: [values]} to search over.
    run_fn : Callable[[dict], float]
        Metric function (same as check_sensitivity).
    perturbation_pct, max_degradation_pct : float
        Passed to check_sensitivity.
    min_base_metric : float
        Minimum metric value for a config to be included (before sensitivity check).

    Returns
    -------
    List of validated config dicts (full params dict, not just the grid slice).
    """
    import itertools

    keys   = list(param_grid.keys())
    values = list(param_grid.values())

    validated = []
    for combo in itertools.product(*values):
        params = dict(base_params)
        params.update(dict(zip(keys, combo)))

        result = check_sensitivity(
            params, run_fn,
            perturbation_pct=perturbation_pct,
            max_degradation_pct=max_degradation_pct,
        )
        if result.base_metric >= min_base_metric and result.is_stable:
            validated.append(params)

    return validated
