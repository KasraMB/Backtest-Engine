"""
Orderflow proxy primitives for IVB v9-final.

Two proxies (Agent #1 spec, locked in v9-final):
  - momentum:   |bar_delta| > per-state quantile threshold, sign matches direction
  - absorption: high bar_volume + small |Δprice| within per-state thresholds

All functions are pure numpy, fully vectorised over the bar axis (~1M bars
target <100 ms each per Stage 3 plan §4).

Bar delta is `ask_volume - bid_volume`, in CONTRACTS (not tick counts).
Exchange-native bid/ask volume confirmed by Stage 1 fingerprint check.
"""
from __future__ import annotations

import numpy as np


def bar_delta(bid_vol: np.ndarray, ask_vol: np.ndarray) -> np.ndarray:
    """Per-bar net aggressor flow: ask_vol - bid_vol. O(N) elementwise."""
    return np.asarray(ask_vol, dtype=np.float64) - np.asarray(bid_vol, dtype=np.float64)


def _state_thresholds_to_array(
    thresholds_per_state: dict[int, float],
    n_states: int,
) -> np.ndarray:
    """Flatten dict {state: threshold} to a length-n_states array.

    NaN thresholds (insufficient state data — see per_state_quantile) become +inf,
    so the mask never fires for that state. Caller falls back to universal q at the
    monitor layer (q-CI check); inside hot loops we just suppress.
    """
    arr = np.full(n_states, np.inf, dtype=np.float64)
    for s, v in thresholds_per_state.items():
        if not np.isnan(v):
            arr[s] = v
    return arr


def momentum_signal(
    bar_delta_arr: np.ndarray,
    state_labels_per_bar: np.ndarray,
    thresholds_per_state: dict[int, float],
    direction: int,
    n_states: int,
) -> np.ndarray:
    """Boolean mask: bars whose signed delta exceeds the per-state threshold.

    For direction=+1 (long bias):  bar_delta >  threshold[state]
    For direction=-1 (short bias): bar_delta < -threshold[state]

    Vectorised:
      thresh = thresh_arr[state_labels]   # (N,) via fancy indexing, no loop
      mask   = (sign * bar_delta) > thresh

    Performance target: 1M bars < 50 ms.
    """
    if direction not in (-1, 1):
        raise ValueError(f"direction must be +/-1, got {direction}")
    thresh_arr = _state_thresholds_to_array(thresholds_per_state, n_states)
    per_bar_thresh = thresh_arr[state_labels_per_bar]
    signed_delta = direction * bar_delta_arr
    return signed_delta > per_bar_thresh


def absorption_signal(
    bar_volume: np.ndarray,
    price_change_abs: np.ndarray,
    state_labels_per_bar: np.ndarray,
    vol_thresholds_per_state: dict[int, float],
    pxchg_thresholds_per_state: dict[int, float],
    n_states: int,
) -> np.ndarray:
    """Boolean mask: high volume AND small absolute price change for the state.

    Absorption fires when significant volume traded without moving price — the
    classic order-flow interpretation that a passive side absorbed the aggressor
    flow. Both conditions are per-HMM-state.

      vol_thresholds_per_state:    q-quantile of bar_volume per state  (HIGH cutoff)
      pxchg_thresholds_per_state:  q-quantile of |Δprice| per state    (LOW cutoff
                                   — we want bars BELOW this, i.e. quiet price)

    Wait — for "small |Δprice|" we want bars below the LOW quantile (e.g. q=0.10).
    Caller is responsible for providing the appropriate quantile (low for the
    small-Δprice cutoff, high for the volume cutoff). Function applies cleanly
    either way: vol > vol_thresh AND |Δp| < pxchg_thresh.

    Performance target: 1M bars < 50 ms.
    """
    vol_thresh_arr   = _state_thresholds_to_array(vol_thresholds_per_state, n_states)
    pxchg_thresh_arr = _state_thresholds_to_array(pxchg_thresholds_per_state, n_states)
    per_bar_vol_thresh   = vol_thresh_arr[state_labels_per_bar]
    per_bar_pxchg_thresh = pxchg_thresh_arr[state_labels_per_bar]
    # Note: NaN-thresholds map to +inf above. For the pxchg "<" check, +inf means
    # the |Δp| condition always passes (we want to suppress, so override to -inf).
    bad = ~np.isfinite(per_bar_pxchg_thresh)
    if bad.any():
        per_bar_pxchg_thresh = np.where(bad, -np.inf, per_bar_pxchg_thresh)
    return (bar_volume > per_bar_vol_thresh) & (price_change_abs < per_bar_pxchg_thresh)
