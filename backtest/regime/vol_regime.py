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

import numpy as np
import pandas as pd

from backtest.regime.hmm import fit_regimes
from backtest.ml.splits import VALIDATION_END

# Hard cutoff — derived from splits.py so it stays in sync automatically.
# Never use price data beyond this date.
REGIME_CUTOFF = date.fromisoformat(VALIDATION_END)

# HMM state index for high volatility (highest mean log-range after sorting)
HIGH_VOL_LABEL = 2


# HMM warmup ends here — all data before this date is used as warmup only.
# Trades from 2019-01-01 onward get fully forward-safe regime labels.
_WARMUP_END = date(2018, 12, 31)


def compute_vol_regime_map(
    df_1m: pd.DataFrame,
    cutoff_date: date = REGIME_CUTOFF,
    n_states: int = 3,
    seed: int = 42,
) -> dict[date, float]:
    """
    Compute P(today = high_vol) for each trading day up to cutoff_date.

    Method
    ------
    1. Compute RTH daily range (high - low, 9:30–16:00 ET) from 1m bars.
    2. Fit a rolling Gaussian HMM on log(daily_range) with a pre-2019 warmup:
       - Initial HMM trained on data up to _WARMUP_END (2018-12-31).
       - From 2019-01-01 onward: expanding-window labels (forward-safe).
    3. For each day D: P(D = high_vol) is computed via an expanding-window
       transition count accumulator — no global transition matrix is used.
       First trading day gets the uninformative prior 1/n_states.

    Parameters
    ----------
    df_1m : full 1m OHLCV DataFrame (tz-aware index, should include pre-2019)
    cutoff_date : hard stop — no data beyond this date is ever used
    n_states : HMM states (3 = low/mid/high vol)
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

    # ── 2. Fit rolling HMM — warmup on pre-2019 data ─────────────────────────
    # Compute train_ratio so that the in-sample period ends at _WARMUP_END.
    # Any dates before 2019 that are labelled in-sample are used as warmup
    # only — trades in the ML dataset start from 2019 onward.
    n_warmup = sum(1 for d in dates if d <= _WARMUP_END)
    train_ratio = max(n_warmup / len(dates), 0.05)  # floor at 5% for safety

    regime_result = fit_regimes(
        log_ranges,
        dates,
        n_states    = n_states,
        train_ratio = train_ratio,
        mode        = 'rolling',
        seed        = seed,
    )

    labels = regime_result.labels   # {date: 0/1/2}

    # ── 3. Build date → P(high_vol) via expanding-window Markov ─────────────
    # Use per-step transition count accumulation so that P(D = high_vol) is
    # computed from transition frequencies observed in days 0..D-2 only.
    # Dirichlet prior (all-ones) prevents zero-probability rows.
    sorted_dates = sorted(dates)
    regime_map: dict[date, float] = {}
    trans_counts = np.ones((n_states, n_states), dtype=np.float64)

    for i, d in enumerate(sorted_dates):
        if i == 0:
            regime_map[d] = 1.0 / n_states          # uninformative prior
        else:
            prev = labels.get(sorted_dates[i - 1], 1)
            regime_map[d] = float(
                trans_counts[prev, HIGH_VOL_LABEL] / trans_counts[prev].sum()
            )
            curr = labels.get(d, 1)
            trans_counts[prev, curr] += 1.0

    return regime_map
