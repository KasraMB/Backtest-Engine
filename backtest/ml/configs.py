"""
Parameter space definitions and LHS sampling for the ICT/SMC ML pipeline.

Rounds
------
Each round narrows the param ranges based on what the previous round revealed.
  Round 1 — broad LHS exploration across the full plausible space
  Round 2 — tighter ranges in validated high-performing regions (edit after Round 1)
  Round N — repeat

Usage
-----
    from backtest.ml.configs import sample_configs, PARAM_RANGES_V1, normalize_config

    configs = sample_configs(150, PARAM_RANGES_V1, seed=42)
    for cfg in configs:
        feat = normalize_config(cfg, PARAM_RANGES_V1)   # → dict of 0-1 values

Phase 1 vs Phase 2
------------------
PHASE1_PARAMS  — fixed at strategy __init__; determine which signals fire.
                 Cannot change per trade at inference time.
PHASE2_PARAMS  — execution params; the model picks the best combination
                 per trade at inference time from PHASE2_CANDIDATES.

CONFIG_FEATURE_NAMES — the normalised values of all LHS params used as
                       model features.  Each training row has these appended
                       so the model can learn param × signal interactions.
"""
from __future__ import annotations

from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Phase 1 — fixed at strategy init (determine what signals fire)
# ---------------------------------------------------------------------------
PHASE1_PARAMS: list[str] = [
    'confluence_tolerance_atr_mult',
    'tp_confluence_tolerance_atr_mult',
    'min_rr',
    'swing_n',
    'manip_leg_timeframe',
    'manip_leg_swing_depth',
    'level_penetration_atr_mult',
    'po3_atr_mult',
    'po3_band_pct',
    'po3_vol_sens',
    'po3_min_manipulation_size_atr_mult',
    'min_ote_size_atr_mult',
    'max_ote_per_session',
    'max_stdv_per_session',
    'max_session_ote_per_session',
    'max_trades_per_day',
]

# ---------------------------------------------------------------------------
# Phase 2 — applied dynamically per trade at inference time
# ---------------------------------------------------------------------------
PHASE2_PARAMS: list[str] = [
    'cancel_pct_to_tp',
    'tick_offset_atr_mult',
    'order_expiry_bars',
]

# ---------------------------------------------------------------------------
# Config feature names (normalised 0-1 values of all sampled params)
# Prefixed with 'cfg_' to avoid collision with signal/context features.
# Categorical params get binary encoding (e.g. cfg_manip_tf_is_1m).
# ---------------------------------------------------------------------------
CONFIG_FEATURE_NAMES: list[str] = [
    # Phase 1 continuous
    'cfg_confluence_tol',
    'cfg_tp_confluence_tol',
    'cfg_min_rr',
    'cfg_level_penetration',
    'cfg_po3_atr_mult',
    'cfg_po3_band_pct',
    'cfg_po3_vol_sens',
    'cfg_po3_min_manip_size',
    'cfg_min_ote_size',
    # Phase 1 integer
    'cfg_swing_n',
    'cfg_manip_leg_swing_depth',
    'cfg_max_ote_per_session',
    'cfg_max_stdv_per_session',
    'cfg_max_session_ote_per_session',
    'cfg_max_trades_per_day',
    # Phase 1 categorical (binary)
    'cfg_manip_tf_is_1m',         # 1 if manip_leg_timeframe == '1m', else 0
    # Phase 2 continuous
    'cfg_cancel_pct_to_tp',
    'cfg_tick_offset',
    'cfg_order_expiry_bars',
]

# ---------------------------------------------------------------------------
# Default base params (non-sampled params stay at these values)
# ---------------------------------------------------------------------------
BASE_PARAMS: dict[str, Any] = dict(
    contracts=1,
    cisd_min_series_candles=2,
    cisd_min_body_ratio=0.5,
    rb_min_wick_ratio=0.3,
    session_level_validity_days=2,
    po3_lookback=6,
    po3_atr_len=14,
    po3_max_r2=0.4,
    po3_min_dir_changes=2,
    po3_min_candles=3,
    po3_max_accum_gap_bars=10,
    allowed_setup_types=['OTE', 'STDV', 'SESSION_OTE'],
    ml_model=None,
)

# ---------------------------------------------------------------------------
# Param ranges — Round 1 (broad exploration)
# ---------------------------------------------------------------------------
# Format for continuous/integer: (low, high)
# Format for categorical: [choice1, choice2, ...]
PARAM_RANGES_V1: dict[str, Any] = {
    # Phase 1 — continuous
    'confluence_tolerance_atr_mult':    (0.08, 0.28),
    'tp_confluence_tolerance_atr_mult': (0.08, 0.28),
    'min_rr':                           (2.0,  8.0),
    'level_penetration_atr_mult':       (0.25, 0.75),
    'po3_atr_mult':                     (0.6,  1.3),
    'po3_band_pct':                     (0.15, 0.5),
    'po3_vol_sens':                     (0.5,  2.0),
    'po3_min_manipulation_size_atr_mult': (0.0, 1.0),
    'min_ote_size_atr_mult':            (0.0,  0.5),
    # Phase 1 — integer
    'swing_n':                          (1,    3),
    'manip_leg_swing_depth':            (1,    2),
    'max_ote_per_session':              (1,    3),
    'max_stdv_per_session':             (1,    3),
    'max_session_ote_per_session':      (1,    3),
    'max_trades_per_day':               (1,    4),
    # Phase 1 — categorical
    'manip_leg_timeframe':              ['1m', '5m'],
    # Phase 2 — continuous
    'cancel_pct_to_tp':                 (0.5,  1.0),
    'tick_offset_atr_mult':             (0.01, 0.08),
    'order_expiry_bars':                (5,    20),
}

# ---------------------------------------------------------------------------
# Param ranges — Round 2 (fill in after reviewing Round 1 results)
# Edit these after Round 1 feature importance + partial dependence analysis.
# ---------------------------------------------------------------------------
PARAM_RANGES_V2: dict[str, Any] = dict(PARAM_RANGES_V1)  # placeholder — copy V1 for now

# Integer params (rounded during sampling)
_INTEGER_PARAMS: set[str] = {
    'swing_n', 'manip_leg_swing_depth',
    'max_ote_per_session', 'max_stdv_per_session', 'max_session_ote_per_session',
    'max_trades_per_day', 'order_expiry_bars',
}

# Categorical params (mapped from uniform [0,1] to discrete choices)
_CATEGORICAL_PARAMS: set[str] = {'manip_leg_timeframe'}

# Map from round number to ranges dict
ROUND_RANGES: dict[int, dict] = {
    1: PARAM_RANGES_V1,
    2: PARAM_RANGES_V2,
}


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def sample_configs(
    n: int,
    ranges: dict[str, Any] | None = None,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """
    Sample n parameter configs using Latin Hypercube Sampling.

    Continuous/integer params are sampled from (low, high) ranges.
    Categorical params are sampled uniformly from their choice lists.
    Non-sampled params come from BASE_PARAMS.

    Parameters
    ----------
    n      : number of configs to sample
    ranges : param ranges dict (defaults to PARAM_RANGES_V1)
    seed   : random seed for reproducibility

    Returns
    -------
    List of n full param dicts (BASE_PARAMS overridden by sampled values).
    """
    if ranges is None:
        ranges = PARAM_RANGES_V1

    try:
        from scipy.stats.qmc import LatinHypercube, scale
    except ImportError as e:
        raise ImportError("scipy is required for LHS sampling: pip install scipy") from e

    # Separate continuous/integer from categorical
    continuous_keys = [k for k, v in ranges.items() if isinstance(v, tuple)]
    categorical_keys = [k for k, v in ranges.items() if isinstance(v, list)]

    configs: list[dict] = []

    if continuous_keys:
        sampler   = LatinHypercube(d=len(continuous_keys), seed=seed)
        samples   = sampler.random(n=n)               # shape (n, d), values in [0, 1]
        low_high  = np.array([ranges[k] for k in continuous_keys], dtype=float)
        scaled    = scale(samples, low_high[:, 0], low_high[:, 1])  # shape (n, d)
    else:
        scaled = np.empty((n, 0))

    rng = np.random.default_rng(seed)

    for i in range(n):
        cfg = dict(BASE_PARAMS)

        # Continuous / integer
        for j, key in enumerate(continuous_keys):
            val = float(scaled[i, j])
            if key in _INTEGER_PARAMS:
                lo, hi = ranges[key]
                val = int(round(val))
                val = max(int(lo), min(int(hi), val))
            cfg[key] = val

        # Categorical
        for key in categorical_keys:
            choices = ranges[key]
            idx     = int(rng.integers(0, len(choices)))
            cfg[key] = choices[idx]

        configs.append(cfg)

    return configs


# ---------------------------------------------------------------------------
# Config feature encoding (for model features)
# ---------------------------------------------------------------------------

def normalize_config(
    params: dict[str, Any],
    ranges: dict[str, Any] | None = None,
) -> dict[str, float]:
    """
    Convert a params dict into normalised config features (0–1 range).
    Returns a dict keyed by CONFIG_FEATURE_NAMES.
    """
    if ranges is None:
        ranges = PARAM_RANGES_V1

    def _norm(key: str, cfg_key: str) -> float:
        val = params.get(key)
        r   = ranges.get(key)
        if val is None or r is None:
            return 0.0
        if isinstance(r, tuple):
            lo, hi = r
            if hi == lo:
                return 0.0
            return float(np.clip((val - lo) / (hi - lo), 0.0, 1.0))
        return 0.0  # categorical handled separately

    feat: dict[str, float] = {k: 0.0 for k in CONFIG_FEATURE_NAMES}

    feat['cfg_confluence_tol']         = _norm('confluence_tolerance_atr_mult',     'cfg_confluence_tol')
    feat['cfg_tp_confluence_tol']      = _norm('tp_confluence_tolerance_atr_mult',  'cfg_tp_confluence_tol')
    feat['cfg_min_rr']                 = _norm('min_rr',                            'cfg_min_rr')
    feat['cfg_level_penetration']      = _norm('level_penetration_atr_mult',        'cfg_level_penetration')
    feat['cfg_po3_atr_mult']           = _norm('po3_atr_mult',                      'cfg_po3_atr_mult')
    feat['cfg_po3_band_pct']           = _norm('po3_band_pct',                      'cfg_po3_band_pct')
    feat['cfg_po3_vol_sens']           = _norm('po3_vol_sens',                      'cfg_po3_vol_sens')
    feat['cfg_po3_min_manip_size']     = _norm('po3_min_manipulation_size_atr_mult','cfg_po3_min_manip_size')
    feat['cfg_min_ote_size']           = _norm('min_ote_size_atr_mult',             'cfg_min_ote_size')
    feat['cfg_swing_n']                = _norm('swing_n',                           'cfg_swing_n')
    feat['cfg_manip_leg_swing_depth']  = _norm('manip_leg_swing_depth',             'cfg_manip_leg_swing_depth')
    feat['cfg_max_ote_per_session']    = _norm('max_ote_per_session',               'cfg_max_ote_per_session')
    feat['cfg_max_stdv_per_session']   = _norm('max_stdv_per_session',              'cfg_max_stdv_per_session')
    feat['cfg_max_session_ote_per_session'] = _norm('max_session_ote_per_session',  'cfg_max_session_ote_per_session')
    feat['cfg_max_trades_per_day']     = _norm('max_trades_per_day',                'cfg_max_trades_per_day')
    feat['cfg_manip_tf_is_1m']         = float(params.get('manip_leg_timeframe') == '1m')
    feat['cfg_cancel_pct_to_tp']       = _norm('cancel_pct_to_tp',                 'cfg_cancel_pct_to_tp')
    feat['cfg_tick_offset']            = _norm('tick_offset_atr_mult',              'cfg_tick_offset')
    feat['cfg_order_expiry_bars']      = _norm('order_expiry_bars',                 'cfg_order_expiry_bars')

    return feat


# ---------------------------------------------------------------------------
# Phase 2 candidate generation (used at inference time)
# ---------------------------------------------------------------------------

def get_phase2_candidates(
    ranges: dict[str, Any] | None = None,
    n: int = 8,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """
    Return a small grid of Phase 2 param combinations for inference-time
    config selection.  The model is queried for each candidate and the
    highest-predicted-R combination is used for that trade.

    Parameters
    ----------
    ranges : param ranges dict (only Phase 2 ranges are used)
    n      : number of Phase 2 candidates (keep small — queried every signal)
    seed   : for reproducibility

    Returns
    -------
    List of Phase 2 param dicts (keys: cancel_pct_to_tp, tick_offset_atr_mult,
    order_expiry_bars).
    """
    if ranges is None:
        ranges = PARAM_RANGES_V1

    phase2_ranges = {k: ranges[k] for k in PHASE2_PARAMS if k in ranges}

    try:
        from scipy.stats.qmc import LatinHypercube, scale
        keys    = list(phase2_ranges.keys())
        sampler = LatinHypercube(d=len(keys), seed=seed)
        samples = sampler.random(n=n)
        low_high = np.array([phase2_ranges[k] for k in keys], dtype=float)
        scaled   = scale(samples, low_high[:, 0], low_high[:, 1])
        candidates = []
        for i in range(n):
            c = {}
            for j, key in enumerate(keys):
                val = float(scaled[i, j])
                if key in _INTEGER_PARAMS:
                    lo, hi = phase2_ranges[key]
                    val = max(int(lo), min(int(hi), int(round(val))))
                c[key] = val
            candidates.append(c)
        return candidates
    except ImportError:
        # Fallback: simple 2-point grid per param
        from itertools import product as iproduct
        vals = {}
        for k, r in phase2_ranges.items():
            lo, hi = r
            mid    = (lo + hi) / 2
            if k in _INTEGER_PARAMS:
                vals[k] = [int(lo), int(round(mid))]
            else:
                vals[k] = [lo, mid]
        return [dict(zip(vals.keys(), combo)) for combo in iproduct(*vals.values())]
