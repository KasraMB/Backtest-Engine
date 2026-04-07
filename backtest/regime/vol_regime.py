"""
backtest/regime/vol_regime.py
─────────────────────────────
Volatility regime feature for the ML pipeline.

Computes P(today = high_vol_regime) for each trading day using a lag-1
Markov prediction from the previous day's HMM regime label.

Forward-safe: only uses price data up to cutoff_date (VALIDATION_END).
Never reads test-period price data under any circumstances.

States are sorted by mean log-range ascending:
  0 = low_vol   (smallest daily range)
  1 = mid_vol
  2 = high_vol  (largest daily range)
"""
from __future__ import annotations

from datetime import date, time
from typing import Optional

import numpy as np
import pandas as pd

from backtest.regime.hmm import fit_regimes
from backtest.ml.splits import VALIDATION_END

# Hard cutoff — derived from splits.py so it stays in sync automatically.
# Never use price data beyond this date.
REGIME_CUTOFF = date.fromisoformat(VALIDATION_END)

# HMM state index for high volatility (highest mean log-range after sorting)
HIGH_VOL_LABEL = 2


def compute_vol_regime_map(
    df_1m: pd.DataFrame,
    cutoff_date: date = REGIME_CUTOFF,
    n_states: int = 3,
    train_ratio: float = 0.5,
    seed: int = 42,
) -> dict[date, float]:
    """
    Compute P(today = high_vol) for each trading day up to cutoff_date.

    Method
    ------
    1. Compute RTH daily range (high - low, 9:30–16:00 ET) from 1m bars.
    2. Fit a rolling Gaussian HMM on log(daily_range) — forward-safe,
       expanding window, hard-stopped at cutoff_date.
    3. For each day D: P(D = high_vol) = transition_matrix[regime(D-1), 2].
       First trading day gets the uninformative prior 1/n_states.

    Parameters
    ----------
    df_1m : full 1m OHLCV DataFrame (tz-aware index)
    cutoff_date : hard stop — no data beyond this date is ever used
    n_states : HMM states (3 = low/mid/high vol)
    train_ratio : fraction of cutoff-period data used for initial HMM fit
    seed : random seed for HMM

    Returns
    -------
    dict[date, float]
        Maps each trading date to P(that day is the high-vol regime).
        Dates after cutoff_date are never included.
    """
    # ── 1. Compute RTH daily range ────────────────────────────────────────────
    rth_mask = (
        (df_1m.index.time >= time(9, 30)) &
        (df_1m.index.time <= time(16, 0))
    )
    rth = df_1m[rth_mask]

    daily = rth.groupby(rth.index.date).agg(
        high=('high', 'max'),
        low=('low',  'min'),
    )

    # Hard cutoff — never touch test period
    daily = daily[daily.index <= cutoff_date]
    daily['range'] = daily['high'] - daily['low']

    # Drop days with zero or missing range (data gaps / early closes)
    daily = daily[daily['range'] > 0].copy()

    if len(daily) < 20:
        return {}

    dates      = list(daily.index)          # list[date], ascending
    log_ranges = np.log(daily['range'].values)

    # ── 2. Fit rolling HMM ────────────────────────────────────────────────────
    regime_result = fit_regimes(
        log_ranges,
        dates,
        n_states    = n_states,
        train_ratio = train_ratio,
        mode        = 'rolling',
        seed        = seed,
    )

    labels = regime_result.labels            # {date: 0/1/2}
    trans  = regime_result.transition_matrix  # (n_states, n_states)

    # ── 3. Build date → P(high_vol) via lag-1 Markov ─────────────────────────
    sorted_dates = sorted(dates)
    regime_map: dict[date, float] = {}

    for i, d in enumerate(sorted_dates):
        if i == 0:
            regime_map[d] = 1.0 / n_states          # uninformative prior
        else:
            prev_regime   = labels.get(sorted_dates[i - 1], 1)   # default neutral
            regime_map[d] = float(trans[prev_regime, HIGH_VOL_LABEL])

    return regime_map
