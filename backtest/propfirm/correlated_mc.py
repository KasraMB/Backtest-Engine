"""
backtest/propfirm/correlated_mc.py
────────────────────────────────────
Regime-aware correlated reinvestment Monte Carlo.

Simulates N investors over `horizon` trading days. Accounts within one sim
share the same daily market draw (correlated). Different sims are independent.

Public API: correlated_reinvestment_mc(...)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from backtest.propfirm.lucidflex import (
    LucidFlexAccount,
    MICRO_POINT_VALUE,
    MINI_POINT_VALUE,
    MICRO_COMM_RT,
    MINI_COMM_RT,
)


@dataclass
class RegimeModel:
    n_regimes:    int
    transition:   np.ndarray   # (n_regimes, n_regimes) row-stochastic
    initial_dist: np.ndarray   # (n_regimes,)


@dataclass
class StrategyConfig:
    name:               str
    pnl_pts_by_regime:  Dict[int, np.ndarray]   # regime → float32 array
    sl_dists_by_regime: Dict[int, np.ndarray]   # regime → float32 array
    tpd_by_regime:      Dict[int, float]         # regime → avg trades per day
    entry_time_min:     int                      # minutes into session
    eval_risk:          float                    # full-slot ERP baseline (fraction of MLL)
    fund_risk:          float                    # full-slot FRP baseline (fraction of MLL)


@dataclass
class AccountSlot:
    configs: List[StrategyConfig]   # index 0 = highest priority


@dataclass
class AccountManagementStrategy:
    trigger:         str   # "greedy"|"on_fail"|"on_pass"|"on_close"|"on_payout"|"staggered"
    max_concurrent:  int   # 1–5
    reserve_n_evals: int   # eval fees to keep as cash cushion
    stagger_days:    int = 0   # days between opens; only for trigger="staggered"


def _sample_regime_sequences(
    model: RegimeModel,
    n_sims: int,
    horizon: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Pre-generate (n_sims, horizon) int8 regime array via Markov chain."""
    seqs = np.empty((n_sims, horizon), dtype=np.int8)
    seqs[:, 0] = rng.choice(model.n_regimes, size=n_sims, p=model.initial_dist)
    for d in range(1, horizon):
        prev = seqs[:, d - 1]
        for r in range(model.n_regimes):
            mask = prev == r
            if mask.any():
                seqs[mask, d] = rng.choice(
                    model.n_regimes, size=int(mask.sum()), p=model.transition[r]
                )
    return seqs
