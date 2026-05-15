"""
Three-phase prop firm account simulator — AnchoredMeanReversion's method.

Each account has 3 phases:
    Phase 1 (eval):          high-pass-rate config, ERP risk → pass eval
    Phase 2 (funded grind):  same config + FRP risk → push funded profit to +$3K
    Phase 3 (funded payout): high-winrate config + MIN risk → 5 winning days → payout

Phase 3 risk is auto-sized via min_contracts_for_winning_day() — the smallest
contract count where the expected winning-day net PnL clears the account's
winning_day_min threshold. This preserves the buffer while collecting winning
days.

Vectorised numpy implementation: (n_sims, max_concurrent) state arrays.
No JIT — used for a focused 50-combo sweep where ~1s per combo is fine.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from backtest.propfirm.lucidflex import (
    LucidFlexAccount,
    MICRO_COMM_RT,
    MICRO_POINT_VALUE,
    MINI_COMM_RT,
    MINI_POINT_VALUE,
    min_contracts_for_winning_day,
)
from backtest.regime.hmm import RegimeResult

# A "config input" can be either a single ThreePhaseConfig or a list of them
# (e.g. multi-session combos). When a list, each session generates trades
# independently and daily PnL is summed.
ConfigInput = Union['ThreePhaseConfig', List['ThreePhaseConfig']]

# A SlotAssignment dedicates one account slot to a specific (P12, P3) pair.
# When provided as a list of length max_concurrent, each account trades only
# its own session — supports "divide accounts evenly between sessions".
SlotAssignment = Tuple['ThreePhaseConfig', 'ThreePhaseConfig']  # (P12_cfg, P3_cfg)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

@dataclass
class ThreePhaseConfig:
    """One side of a AnchoredMeanReversion 3-phase test (a single config + risk setting).

    pnl_pts / sl_dists hold the flat (regime-blind) trade pool — used when no
    regime sequence is supplied. When regime-aware MC is enabled, the per-day
    draw uses pnl_pts_by_regime[r] and sl_dists_by_regime[r] for that day's
    regime r.
    """
    name:          str
    pnl_pts:       np.ndarray   # (n_trades,) float — flat pool fallback
    sl_dists:      np.ndarray   # (n_trades,) float — corresponding SL distances
    tpd:           float        # trades per day (Poisson rate)
    # Optional regime-conditional pools.
    # When supplied AND a regime_result is passed to the simulator, daily PnL
    # is drawn from pnl_pts_by_regime[r] where r = regime on that day.
    pnl_pts_by_regime:  Optional[Dict[int, np.ndarray]] = None
    sl_dists_by_regime: Optional[Dict[int, np.ndarray]] = None


# ---------------------------------------------------------------------------
# Trade pre-generation
# ---------------------------------------------------------------------------

def _generate_day_pnl(
    cfg:        ThreePhaseConfig,
    n_minis:    int,
    n_micros:   int,
    n_sims:     int,
    horizon:    int,
    rng:        np.random.Generator,
) -> np.ndarray:
    """Returns (n_sims, horizon) float64 array of NET daily dollar PnL
    using the specified contract sizing (minis xor micros, not both).
    Assumes max 1 trade/day (Bernoulli-on-fire), drawn from cfg's pool."""
    fired = rng.poisson(cfg.tpd, size=(n_sims, horizon)) > 0
    pool_n = len(cfg.pnl_pts)
    idx = rng.integers(0, max(pool_n, 1), size=(n_sims, horizon), dtype=np.int32)
    pnl_pts = cfg.pnl_pts[idx]  # (n_sims, horizon) signed points per trade

    if n_minis > 0:
        gross = pnl_pts * MINI_POINT_VALUE * n_minis
        comm  = MINI_COMM_RT * n_minis
    else:
        gross = pnl_pts * MICRO_POINT_VALUE * n_micros
        comm  = MICRO_COMM_RT * n_micros

    net = gross - comm
    return np.where(fired, net, 0.0).astype(np.float64)


def _as_list(x: ConfigInput) -> List[ThreePhaseConfig]:
    """Normalise input to a list of configs."""
    if isinstance(x, list):
        return x
    return [x]


def build_regime_switched_config(
    name:                  str,
    sources_by_regime:     Dict[int, ThreePhaseConfig],
    regime_freq:           Optional[Dict[int, float]] = None,
) -> ThreePhaseConfig:
    """Construct a synthetic ThreePhaseConfig that uses a different source
    config per regime. Each regime's `pnl_pts_by_regime[r]` and
    `sl_dists_by_regime[r]` are taken from `sources_by_regime[r]`.

    Args:
        name: human-readable identifier for the synthesised config.
        sources_by_regime: {regime_int: source_ThreePhaseConfig}. Every regime
            present in the values' pnl_pts_by_regime is taken from the
            corresponding source. Missing regimes fall back to the first source.
        regime_freq: optional {regime: frequency} for tpd weighting. If None,
            tpd is the simple mean of source tpds.

    Returns the synthesised ThreePhaseConfig — usable directly in any
    regime-aware run (slot_assignments, payout_config, etc.)."""
    if not sources_by_regime:
        raise ValueError("sources_by_regime must contain at least one entry")

    # All regimes — union across sources
    all_regimes = set()
    for cfg in sources_by_regime.values():
        if cfg.pnl_pts_by_regime is not None:
            all_regimes.update(cfg.pnl_pts_by_regime.keys())
    all_regimes = sorted(all_regimes)
    if not all_regimes:
        raise ValueError("Source configs must have pnl_pts_by_regime populated")

    pnl_by_regime: Dict[int, np.ndarray] = {}
    sl_by_regime:  Dict[int, np.ndarray] = {}
    fallback = next(iter(sources_by_regime.values()))
    for r in all_regimes:
        src = sources_by_regime.get(r, fallback)
        if src.pnl_pts_by_regime is None or r not in src.pnl_pts_by_regime:
            # Use src's flat pool for this regime as fallback
            pnl_by_regime[r] = src.pnl_pts
            sl_by_regime[r]  = src.sl_dists
        else:
            pnl_by_regime[r] = src.pnl_pts_by_regime[r]
            sl_by_regime[r]  = src.sl_dists_by_regime[r]

    # tpd: weighted by regime_freq, else simple mean
    if regime_freq is not None:
        total_w = sum(regime_freq.get(r, 0.0) for r in sources_by_regime)
        if total_w > 0:
            tpd = sum(sources_by_regime[r].tpd * regime_freq.get(r, 0.0)
                      for r in sources_by_regime) / total_w
        else:
            tpd = np.mean([c.tpd for c in sources_by_regime.values()])
    else:
        tpd = float(np.mean([c.tpd for c in sources_by_regime.values()]))

    # Flat pool fallback = concat of all regime pools
    all_pnl = np.concatenate([pnl_by_regime[r] for r in all_regimes])
    all_sl  = np.concatenate([sl_by_regime[r]  for r in all_regimes])

    return ThreePhaseConfig(
        name=name,
        pnl_pts=all_pnl.astype(np.float32),
        sl_dists=all_sl.astype(np.float32),
        tpd=float(tpd),
        pnl_pts_by_regime=pnl_by_regime,
        sl_dists_by_regime=sl_by_regime,
    )


def _regime_initial_dist(regime_result: RegimeResult) -> np.ndarray:
    """Empirical regime distribution from historical labels (used as
    initial-day distribution when sampling regime sequences)."""
    counts = np.zeros(regime_result.n_states, dtype=np.float64)
    for r in regime_result.labels.values():
        counts[r] += 1
    total = counts.sum()
    if total == 0:
        return np.full(regime_result.n_states, 1.0 / regime_result.n_states)
    return counts / total


def _sample_regime_sequences(
    regime_result: RegimeResult,
    n_sims:        int,
    horizon:       int,
    rng:           np.random.Generator,
) -> np.ndarray:
    """Pre-generate (n_sims, horizon) int8 regime sequences via Markov chain.

    Day 0 sampled from empirical initial distribution. Subsequent days sampled
    via regime_result.transition_matrix. This mirrors correlated_mc.py's logic
    and is what makes cross-day, cross-session correlation natural — every
    session in a sim shares the SAME regime sequence."""
    n_states = regime_result.n_states
    if n_states == 1:
        return np.zeros((n_sims, horizon), dtype=np.int8)

    trans   = regime_result.transition_matrix
    initial = _regime_initial_dist(regime_result)
    seqs    = np.empty((n_sims, horizon), dtype=np.int8)
    seqs[:, 0] = rng.choice(n_states, size=n_sims, p=initial)

    # Vectorised Markov stepping via cumulative-probability lookup
    cum_trans = np.cumsum(trans, axis=1)            # (n_states, n_states)
    for d in range(1, horizon):
        u          = rng.random(n_sims)             # (n_sims,) uniform
        thresholds = cum_trans[seqs[:, d - 1].astype(np.int32)]  # (n_sims, n_states)
        seqs[:, d] = np.argmax(thresholds > u[:, None], axis=1).astype(np.int8)

    return seqs


def _generate_day_pnl_regime_aware(
    cfg:        ThreePhaseConfig,
    n_minis:    int,
    n_micros:   int,
    n_sims:     int,
    horizon:    int,
    regime_seqs: np.ndarray,    # (n_sims, horizon) int8
    rng:        np.random.Generator,
) -> np.ndarray:
    """Generate (n_sims, horizon) daily dollar PnL using regime-conditional
    pools. If cfg has no regime pools, falls back to the flat pool."""
    fired = rng.poisson(cfg.tpd, size=(n_sims, horizon)) > 0

    if cfg.pnl_pts_by_regime is None or cfg.sl_dists_by_regime is None:
        # Fallback: flat-pool draws, regime is ignored
        pool_n = len(cfg.pnl_pts)
        idx    = rng.integers(0, max(pool_n, 1), size=(n_sims, horizon),
                              dtype=np.int32)
        pnl_pts = cfg.pnl_pts[idx]
    else:
        # Regime-aware: per-cell, pick pool[regime] then sample
        # Build pnl_pts cell-by-cell via masked sampling
        pnl_pts = np.zeros((n_sims, horizon), dtype=np.float64)
        for r, pool_pnl in cfg.pnl_pts_by_regime.items():
            mask = regime_seqs == r
            if not mask.any() or len(pool_pnl) == 0:
                continue
            n_cells = int(mask.sum())
            idx_r   = rng.integers(0, len(pool_pnl), size=n_cells, dtype=np.int32)
            pnl_pts[mask] = pool_pnl[idx_r]

    if n_minis > 0:
        gross = pnl_pts * MINI_POINT_VALUE * n_minis
        comm  = MINI_COMM_RT * n_minis
    else:
        gross = pnl_pts * MICRO_POINT_VALUE * n_micros
        comm  = MICRO_COMM_RT * n_micros

    net = gross - comm
    return np.where(fired, net, 0.0).astype(np.float64)


def _generate_combined_day_pnl(
    configs:              List[ThreePhaseConfig],
    risk_dollars_per_cfg: float,
    account:              LucidFlexAccount,
    n_sims:               int,
    horizon:              int,
    rng:                  np.random.Generator,
    regime_seqs:          Optional[np.ndarray] = None,   # (n_sims, horizon)
) -> Tuple[np.ndarray, List[Tuple[int, int]], float]:
    """Sum daily $ PnL across N configs.

    When `regime_seqs` is provided AND each config carries regime-conditional
    pools, the daily draw uses pool[regime] for each (sim, day) — so sessions
    in the same sim share the same regime sequence (cross-session correlation
    via shared regime state).

    Each config sized to risk_dollars_per_cfg.
    Returns (combined_pnl, [(n_minis, n_micros) per cfg], total_max_risk_per_day)."""
    combined = np.zeros((n_sims, horizon), dtype=np.float64)
    sizings: List[Tuple[int, int]] = []
    total_max_risk = 0.0
    for cfg in configs:
        n_minis, n_micros = _size_for_risk(cfg, risk_dollars_per_cfg, account)
        if regime_seqs is not None:
            cfg_pnl = _generate_day_pnl_regime_aware(
                cfg, n_minis, n_micros, n_sims, horizon, regime_seqs, rng,
            )
        else:
            cfg_pnl = _generate_day_pnl(cfg, n_minis, n_micros, n_sims, horizon, rng)
        combined += cfg_pnl
        sizings.append((n_minis, n_micros))
        total_max_risk += risk_dollars_per_cfg
    return combined, sizings, total_max_risk


def _size_for_risk(
    cfg:         ThreePhaseConfig,
    risk_dollars: float,
    account:     LucidFlexAccount,
) -> Tuple[int, int]:
    """Pick (n_minis, n_micros) so per-trade $ risk ≈ risk_dollars.
    Mirrors nq_first sizing: minis when possible, else micros."""
    # Use median SL distance as representative
    sl_pts = float(np.median(cfg.sl_dists))
    if sl_pts <= 0:
        sl_pts = 1.0

    mini_risk_1c = sl_pts * MINI_POINT_VALUE
    max_minis    = account.max_micros // 10

    if mini_risk_1c <= risk_dollars and max_minis > 0:
        n_minis = int(min(max(np.floor(risk_dollars / mini_risk_1c), 1), max_minis))
        return (n_minis, 0)

    micro_risk_1c = sl_pts * MICRO_POINT_VALUE
    if micro_risk_1c <= 0:
        return (0, 1)
    n_micros = int(min(max(np.floor(risk_dollars / micro_risk_1c), 1), account.max_micros))
    return (0, n_micros)


# ---------------------------------------------------------------------------
# Main simulator
# ---------------------------------------------------------------------------

def _build_per_slot_phase12_pnl(
    slot_assignments: List[SlotAssignment],
    risk_dollars:     float,                  # full single-session risk
    account:          LucidFlexAccount,
    n_sims:           int,
    horizon:          int,
    rng:              np.random.Generator,
    regime_seqs:      Optional[np.ndarray],
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """Pre-generate (max_c, n_sims, horizon) daily PnL for phase 1+2 per slot.
    Each slot's PnL stream uses its own dedicated P12 config and full risk."""
    max_c   = len(slot_assignments)
    out     = np.zeros((max_c, n_sims, horizon), dtype=np.float64)
    sizings: List[Tuple[int, int]] = []
    for a, (p12_cfg, _p3_cfg) in enumerate(slot_assignments):
        n_minis, n_micros = _size_for_risk(p12_cfg, risk_dollars, account)
        if regime_seqs is not None:
            cfg_pnl = _generate_day_pnl_regime_aware(
                p12_cfg, n_minis, n_micros, n_sims, horizon, regime_seqs, rng,
            )
        else:
            cfg_pnl = _generate_day_pnl(p12_cfg, n_minis, n_micros, n_sims, horizon, rng)
        out[a] = cfg_pnl
        sizings.append((n_minis, n_micros))
    return out, sizings


# ---------------------------------------------------------------------------
# Block bootstrap (OOS evaluation) — assumption-light alternative to regime MC
# ---------------------------------------------------------------------------

def _sample_stationary_bootstrap_indices(
    n_oos_days:      int,
    n_sims:          int,
    horizon:         int,
    mean_block_len:  float,
    rng:             np.random.Generator,
) -> np.ndarray:
    """Politis-Romano stationary bootstrap. Returns (n_sims, horizon) int32
    indices into [0, n_oos_days). Block lengths are geometrically distributed
    with mean `mean_block_len`, blocks wrap circularly.

    Fully vectorised. Matches the original RNG sequence: same n_sims*horizon
    uniforms for break decisions and same n_breaks integers for starts."""
    if n_oos_days <= 0:
        raise ValueError("n_oos_days must be positive")
    p = 1.0 / max(mean_block_len, 1.0)
    breaks = rng.random((n_sims, horizon)) < p
    breaks[:, 0] = True
    breaks_flat = breaks.reshape(-1)
    n_breaks = int(breaks_flat.sum())
    starts = rng.integers(0, n_oos_days, size=n_breaks, dtype=np.int32)

    n_total = n_sims * horizon
    arange = np.arange(n_total, dtype=np.int64)
    # block_id[i] = which block does cell i belong to (0-indexed)
    block_id = (np.cumsum(breaks_flat) - 1).astype(np.int64)
    # block_start_pos[i] = global position of the first cell in i's block
    block_start_pos = np.where(breaks_flat, arange, 0)
    np.maximum.accumulate(block_start_pos, out=block_start_pos)
    offset = arange - block_start_pos
    idx_flat = ((starts[block_id].astype(np.int64) + offset) % n_oos_days).astype(np.int32)
    return idx_flat.reshape(n_sims, horizon)


def _build_per_slot_bootstrap_pnl(
    R_per_slot:        np.ndarray,    # (max_c, n_oos_days) float32 — sum_R per day
    sl_dists_per_slot: np.ndarray,    # (max_c, n_oos_days) float32 — mean SL per day (for sizing decision)
    fired_per_slot:    np.ndarray,    # (max_c, n_oos_days) bool
    n_trades_per_slot: np.ndarray,    # (max_c, n_oos_days) int — trade count per day (for commissions)
    day_indices:       np.ndarray,    # (n_sims, horizon) int32 — SHARED across slots
    risk_dollars:      float,
    account:           LucidFlexAccount,
    cap_daily_risk:    bool = False,  # if True, per-trade risk = risk_dollars / n_trades_today
) -> Tuple[np.ndarray, List[Tuple[int, int, float]]]:
    """For each slot, look up its per-day SUM_R (= Σ pnl_pts_i / sl_dist_i across trades)
    at the bootstrapped day indices. Same indices for all slots → preserves cross-session
    correlation. $PnL per day = R × risk_dollars − n_trades × commission_per_RT.
    This formulation is exact across multi-trade days (no leverage bias).

    sl_dists_per_slot is consulted only to pick mini vs micro (commission rate).
    Returns (max_c, n_sims, horizon) net $ PnL + diagnostics."""
    max_c, _n_oos_days = R_per_slot.shape
    n_sims, horizon = day_indices.shape
    sizings: List[Tuple[int, int, float]] = []
    comm_per_slot = np.zeros(max_c, dtype=np.float64)
    active_slot   = np.ones(max_c, dtype=bool)   # False if slot has no fired days

    # Per-slot sizing decision (small loop over max_c, ≤5)
    for a in range(max_c):
        fired_a = fired_per_slot[a]
        if not fired_a.any():
            sizings.append((0, 0, 0.0))
            active_slot[a] = False
            continue
        sl_pts = float(np.median(sl_dists_per_slot[a][fired_a]))
        if sl_pts <= 0:
            sl_pts = 1.0
        mini_risk_1c = sl_pts * MINI_POINT_VALUE
        max_minis = account.max_micros // 10
        use_mini = mini_risk_1c <= risk_dollars and max_minis > 0
        comm_per_slot[a] = MINI_COMM_RT if use_mini else MICRO_COMM_RT
        if use_mini:
            n_minis  = int(min(max(np.floor(risk_dollars / mini_risk_1c), 1), max_minis))
            n_micros = 0
        else:
            micro_risk_1c = sl_pts * MICRO_POINT_VALUE
            n_minis  = 0
            n_micros = int(min(max(np.floor(risk_dollars / micro_risk_1c), 1), account.max_micros))
        sizings.append((n_minis, n_micros, float(risk_dollars)))

    # Vectorised PnL across all slots: fancy-index once into (max_c, n_sims, horizon)
    R_today        = R_per_slot[:, day_indices].astype(np.float64)        # promote once
    n_trades_today = n_trades_per_slot[:, day_indices]
    fired_today    = fired_per_slot[:, day_indices]

    if cap_daily_risk:
        safe_n = np.maximum(n_trades_today, 1)
        gross = (R_today / safe_n) * risk_dollars
    else:
        gross = R_today * risk_dollars
    # comm_per_slot broadcasts (max_c,) → (max_c, n_sims, horizon) via [:, None, None]
    comm = n_trades_today * comm_per_slot[:, None, None]
    net  = gross - comm
    out  = np.where(fired_today, net, 0.0)
    # Zero out inactive slots (no fired days at all)
    if not active_slot.all():
        out[~active_slot] = 0.0

    return out, sizings


def _compute_per_slot_commission(
    sl_dists_per_slot: np.ndarray,    # (max_c, n_oos_days)
    fired_per_slot:    np.ndarray,    # (max_c, n_oos_days) bool
    target_risk:       float,
    account:           LucidFlexAccount,
) -> Tuple[np.ndarray, List[Tuple[int, int, float]]]:
    """Per-slot mini-vs-micro decision + commission rate, given a target $ risk.
    Returns (comm_per_slot, sizings) where comm_per_slot is (max_c,) and sizings is
    a list of (n_minis, n_micros, target_risk_for_diagnostics) per slot."""
    max_c = sl_dists_per_slot.shape[0]
    comm = np.zeros(max_c, dtype=np.float64)
    sizings: List[Tuple[int, int, float]] = []
    for a in range(max_c):
        fired_a = fired_per_slot[a]
        if not fired_a.any():
            sizings.append((0, 0, 0.0))
            continue
        sl_pts = float(np.median(sl_dists_per_slot[a][fired_a]))
        if sl_pts <= 0:
            sl_pts = 1.0
        mini_risk_1c = sl_pts * MINI_POINT_VALUE
        max_minis = account.max_micros // 10
        use_mini = mini_risk_1c <= target_risk and max_minis > 0
        comm[a] = MINI_COMM_RT if use_mini else MICRO_COMM_RT
        if use_mini:
            n_min = int(min(max(np.floor(target_risk / mini_risk_1c), 1), max_minis))
            n_mic = 0
        else:
            micro_risk_1c = sl_pts * MICRO_POINT_VALUE
            n_min = 0
            n_mic = int(min(max(np.floor(target_risk / micro_risk_1c), 1), account.max_micros))
        sizings.append((n_min, n_mic, float(target_risk)))
    return comm, sizings


def _build_per_slot_phase3_pnl(
    slot_assignments: List[SlotAssignment],
    account:          LucidFlexAccount,
    n_sims:           int,
    horizon:          int,
    rng:              np.random.Generator,
    regime_seqs:      Optional[np.ndarray],
    p3_risk_dollars:  Optional[float] = None,    # if set, fixed $ risk overrides min-winning-day
) -> Tuple[np.ndarray, List[Tuple[int, int, float]]]:
    """Pre-generate (max_c, n_sims, horizon) daily PnL for phase 3 per slot.

    Default sizing: min_contracts_for_winning_day (smallest that clears the
    winning_day_min threshold net of commission).

    Override: if p3_risk_dollars is provided, use _size_for_risk with that
    fixed target — letting callers sweep optimal Phase 3 risk levels."""
    max_c     = len(slot_assignments)
    out       = np.zeros((max_c, n_sims, horizon), dtype=np.float64)
    sizings:  List[Tuple[int, int, float]] = []
    for a, (_p12_cfg, p3_cfg) in enumerate(slot_assignments):
        if p3_risk_dollars is not None:
            # Fixed-risk sizing — use the same _size_for_risk as phase 1+2
            n_min, n_mic = _size_for_risk(p3_cfg, p3_risk_dollars, account)
            # Estimated actual risk: contracts × median SL × point_value
            sl_pts = float(np.median(p3_cfg.sl_dists)) if len(p3_cfg.sl_dists) > 0 else 0.0
            if n_min > 0:
                risk = n_min * sl_pts * MINI_POINT_VALUE
            else:
                risk = n_mic * sl_pts * MICRO_POINT_VALUE
        else:
            win_mask = p3_cfg.pnl_pts > 0
            avg_win_pts = float(p3_cfg.pnl_pts[win_mask].mean()) if win_mask.any() else 0.0
            avg_sl_pts  = float(np.median(p3_cfg.sl_dists)) if len(p3_cfg.sl_dists) > 0 else 0.0
            n_min, n_mic, risk = min_contracts_for_winning_day(
                avg_win_pts=avg_win_pts, avg_sl_pts=avg_sl_pts,
                account=account, n_trades_per_day=p3_cfg.tpd,
            )
        if regime_seqs is not None:
            cfg_pnl = _generate_day_pnl_regime_aware(
                p3_cfg, n_min, n_mic, n_sims, horizon, regime_seqs, rng,
            )
        else:
            cfg_pnl = _generate_day_pnl(p3_cfg, n_min, n_mic, n_sims, horizon, rng)
        out[a] = cfg_pnl
        sizings.append((n_min, n_mic, risk))
    return out, sizings


def three_phase_reinvestment_mc(
    eval_grind_config: Optional[ConfigInput] = None,
    eval_grind_risk:   float = 0.30,           # ERP/FRP as fraction of MLL
    payout_config:     Optional[ConfigInput] = None,
    account:           LucidFlexAccount = None,
    budget:            float = 3_000.0,
    horizon:           int   = 84,
    max_concurrent:    int   = 10,
    n_sims:            int   = 5_000,
    seed:              int   = 42,
    scale_risk_by_n:   bool  = True,           # 1/N risk scaling for multi-config
    regime_result:     Optional[RegimeResult] = None,  # enables regime-aware draws
    slot_assignments:  Optional[List[SlotAssignment]] = None,  # per-slot (P12, P3) configs
    p3_risk_pct:       Optional[float] = None, # if set, fixed risk pct for phase 3 (overrides min-winning-day)
    grind_risk_pct:    Optional[float] = None, # if set, separate risk for phase 2 (funded grind). If None, uses eval_grind_risk for both eval+grind (legacy).
    # ── Bootstrap mode (OOS evaluation) ──
    # When all four arrays are supplied, the simulator skips regime/poisson sampling
    # and instead samples (n_sims, horizon) day-indices via stationary block bootstrap
    # over the precomputed per-slot OOS day arrays. Cross-slot correlation is preserved
    # because the SAME day indices are used for all slots in a sim.
    bootstrap_block_len:              Optional[float] = None,
    bootstrap_oos_R_per_slot:         Optional[np.ndarray] = None,    # (max_c, n_oos_days) float — sum_R per day
    bootstrap_oos_sl_dists_per_slot:  Optional[np.ndarray] = None,    # (max_c, n_oos_days) float — mean SL (for mini/micro decision)
    bootstrap_oos_fired_per_slot:     Optional[np.ndarray] = None,    # (max_c, n_oos_days) bool
    bootstrap_oos_n_trades_per_slot:  Optional[np.ndarray] = None,    # (max_c, n_oos_days) int — for commissions
    bootstrap_cap_daily_risk:         bool = False,                    # if True, per-trade risk = risk_dollars/n_trades_today
    bootstrap_buffer_safety:          Optional[float] = None,           # if set (e.g. 0.9), per-trade risk = min(target_risk, buffer × safety)
    bootstrap_min_effective_risk:     Optional[float] = None,           # if set (e.g. 500.0), abandon accounts whose next-trade effective_risk would be below this — frees the slot for reinvestment, models a real trader giving up on a thin-buffer account
    bootstrap_intraday_blow_check:    bool  = False,                    # if True, check intraday MLL touches on losing days (M3): losing days where (bal - risk_dollars) < mll_level get blown
    bootstrap_intraday_strict:        bool  = False,                    # if True+intraday_blow_check, also flag WINNING days where intraday adverse could have touched MLL (M4): catches winning trades with deep adverse drawdown that recovered before close
    # ── Real-MAE intraday-blow inputs (preferred over the proxy check above) ──
    # When BOTH arrays are supplied AND bootstrap_intraday_blow_check=True,
    # the simulator walks each fired day's trades sequentially:
    #   floating_low_at_trade_k = bal + Σ(pnl_$_0..k-1) − $MAE_k
    #   if floating_low ≤ mll_level → account blown at trade k (subsequent
    #   trades that day don't fire; bal is forced to mll_level).
    # max_trades_per_day is whatever third dim these arrays carry (typically 3-4).
    bootstrap_oos_R_per_trade_per_slot:     Optional[np.ndarray] = None,   # (max_c, n_oos_days, max_trades) per-trade R, padded with 0 for non-existent trades
    bootstrap_oos_MAE_R_per_trade_per_slot: Optional[np.ndarray] = None,   # (max_c, n_oos_days, max_trades) per-trade MAE_R = mae_pts / sl_dist, padded with 0
    enforce_consistency_rule:         bool  = False,                    # if True, eval pass requires (largest_day_pnl <= total_profit * 0.5) AND (n_profitable_days >= 2). LucidFlex eval-only rule.
    # ── Adaptive max-concurrent: cap starts at max_concurrent_initial and grows
    # by max_concurrent_grow_per_payout for each payout collected, never exceeding
    # max_concurrent. Set initial == max_concurrent and grow == 0 (defaults) for
    # legacy fixed-cap behaviour.
    max_concurrent_initial:           Optional[int] = None,             # if None, use max_concurrent (legacy)
    max_concurrent_grow_per_payout:   int = 0,                          # +N per payout (per-sim)
) -> dict:
    """
    Simulate AnchoredMeanReversion's 3-phase method: buy 10 evals upfront (or up to budget),
    grind funded to +$3K, then switch to payout-farming config with min risk.

    Returns:
        dict with arrays of length n_sims:
            cash:           final cash per sim
            n_payouts:      total payouts collected
            n_passed_eval:  accounts that passed eval
            n_hit_target:   accounts that reached funded profit target
            withdrawn:      gross withdrawn (sum of net payouts)
            eval_fees_paid: total $ spent on eval fees
    """
    rng = np.random.default_rng(seed)
    SB        = float(account.starting_balance)
    MLL_AMT   = float(account.mll_amount)
    PT        = float(account.profit_target)
    EVAL_FEE  = float(account.eval_fee)
    MAX_PAY   = int(account.max_payouts)
    PAY_CAP   = float(account.payout_cap)
    SPLIT     = float(account.split)
    WD_MIN    = float(account.winning_day_min)
    LOCKED_MLL = SB + 100.0
    INITIAL_TRAIL = SB + MLL_AMT + 100.0

    # Bootstrap mode flag — overrides regime/per-slot/legacy modes
    bootstrap_mode = (
        bootstrap_block_len is not None
        and bootstrap_oos_R_per_slot is not None
        and bootstrap_oos_sl_dists_per_slot is not None
        and bootstrap_oos_fired_per_slot is not None
        and bootstrap_oos_n_trades_per_slot is not None
    )

    # Per-slot mode flag — when set, each account slot has its own dedicated
    # (P12, P3) configs and PnL streams (single-session-per-account).
    per_slot_mode = slot_assignments is not None or bootstrap_mode
    if bootstrap_mode:
        if bootstrap_oos_R_per_slot.shape[0] != max_concurrent:
            raise ValueError(
                f"bootstrap_oos_R_per_slot first dim "
                f"{bootstrap_oos_R_per_slot.shape[0]} must equal "
                f"max_concurrent {max_concurrent}"
            )
        eg_list, p3_list = [], []
        n_eg, n_p3 = 0, 0
    elif per_slot_mode:
        if len(slot_assignments) != max_concurrent:
            raise ValueError(
                f"slot_assignments length {len(slot_assignments)} must equal "
                f"max_concurrent {max_concurrent}"
            )
        # In per-slot mode the legacy eval_grind_config / payout_config aren't
        # used for PnL generation. We still need placeholders for diagnostics.
        eg_list, p3_list = [], []
        n_eg, n_p3 = 0, 0
    else:
        if eval_grind_config is None or payout_config is None:
            raise ValueError(
                "Either provide slot_assignments OR both eval_grind_config and payout_config"
            )
        # ── Normalise to lists (multi-config support) ───────────────────────
        eg_list = _as_list(eval_grind_config)
        p3_list = _as_list(payout_config)
        n_eg    = len(eg_list)
        n_p3    = len(p3_list)

    # ── Pre-generate shared regime sequence (used by all configs in this sim)
    # — Markov-chain sampled via regime_result.transition_matrix.  Cross-
    # session correlation comes from EVERY session in a sim drawing from its
    # own pool indexed by the SAME regime[sim, day]. ─────────────────────────
    regime_seqs: Optional[np.ndarray] = None
    if regime_result is not None and not bootstrap_mode:
        regime_seqs = _sample_regime_sequences(regime_result, n_sims, horizon, rng)

    # ── Phase 1+2 sizing: total risk = eval_grind_risk × MLL ───────────────
    eg_total_risk = eval_grind_risk * MLL_AMT
    # Optional separate phase-2 (funded grind) risk; defaults to eval_grind_risk
    grind_total_risk = (grind_risk_pct * MLL_AMT
                        if grind_risk_pct is not None else eg_total_risk)

    if bootstrap_mode:
        # Stationary block bootstrap path: shared (n_sims, horizon) day indices,
        # per-slot $ PnL via R_total × risk_dollars formulation.
        n_oos_days = bootstrap_oos_R_per_slot.shape[1]
        day_indices = _sample_stationary_bootstrap_indices(
            n_oos_days, n_sims, horizon, float(bootstrap_block_len), rng,
        )
        # Phase 3 uses p3_risk_pct override (or falls back to ERP if unspecified)
        p3_risk_dollars = (p3_risk_pct if p3_risk_pct is not None else eval_grind_risk) * MLL_AMT

        if bootstrap_buffer_safety is not None:
            # Buffer-aware: per-day PnL computed dynamically in the day loop
            # using current per-sim balance vs MLL. Precompute only commission rates.
            eg_comm_per_slot, eg_sizings = _compute_per_slot_commission(
                bootstrap_oos_sl_dists_per_slot, bootstrap_oos_fired_per_slot,
                eg_total_risk, account,
            )
            p3_comm_per_slot, p3_sizings_full = _compute_per_slot_commission(
                bootstrap_oos_sl_dists_per_slot, bootstrap_oos_fired_per_slot,
                p3_risk_dollars, account,
            )
            pnl_eg_per_slot = None
            pnl_grind_per_slot = None
            pnl_p3_per_slot = None
        else:
            eg_comm_per_slot = None
            p3_comm_per_slot = None
            pnl_eg_per_slot, eg_sizings = _build_per_slot_bootstrap_pnl(
                bootstrap_oos_R_per_slot,
                bootstrap_oos_sl_dists_per_slot,
                bootstrap_oos_fired_per_slot,
                bootstrap_oos_n_trades_per_slot,
                day_indices, eg_total_risk, account,
                cap_daily_risk=bootstrap_cap_daily_risk,
            )
            # Phase 2 (funded grind) uses grind_total_risk if separate;
            # else reuses the eval stream (legacy behaviour).
            if grind_risk_pct is not None and grind_total_risk != eg_total_risk:
                # FAST PATH: when both phases use the same contract type (mini),
                # the R-formulation lets us scale gross exactly: $PnL ∝ risk.
                # Commission stays the same (n_trades × comm_per_trade), so we
                # back out gross_eg, scale it, re-add commission.
                # `eg_sizings` rows are (n_minis, n_micros, actual_risk_$).
                same_contract_type = all(
                    (n_min > 0) == (eg_sizings[a][0] > 0)
                    for a, (n_min, n_mic, _) in enumerate(eg_sizings)
                ) if False else True  # both phases inherit per-slot mini decision
                # Always safe path: scale gross. Commission unaffected (rate
                # identical because mini-vs-micro decision is per-slot SL-driven,
                # which doesn't depend on risk_$ as long as the choice doesn't flip).
                # Verify by recomputing the per-slot mini-vs-micro under grind_$:
                from backtest.propfirm.lucidflex import (
                    MINI_POINT_VALUE, MICRO_POINT_VALUE,
                )
                use_mini_eg = np.array([s[0] > 0 for s in eg_sizings], dtype=bool)
                # Mini-vs-micro flip happens only if mini_risk_1c > grind_risk
                # AND <= eg_risk. Detect.
                sl_pts_per_slot = np.array([
                    float(np.median(bootstrap_oos_sl_dists_per_slot[a][bootstrap_oos_fired_per_slot[a]]))
                    if bootstrap_oos_fired_per_slot[a].any() else 1.0
                    for a in range(bootstrap_oos_R_per_slot.shape[0])
                ])
                mini_risk_1c_per_slot = sl_pts_per_slot * MINI_POINT_VALUE
                use_mini_grind = mini_risk_1c_per_slot <= grind_total_risk
                if np.array_equal(use_mini_eg, use_mini_grind):
                    # SAFE to scale: contract type unchanged → commission unchanged.
                    # pnl_eg = R*risk - n*comm  →  R*risk = pnl_eg + n*comm.
                    # pnl_grind = R*grind_risk - n*comm
                    #           = (pnl_eg + n*comm) * (grind_risk/eg_risk) - n*comm
                    # But comm depends on n_trades_today which we don't have here.
                    # Simpler: compute gross_eg = pnl_eg + comm, scale to grind, sub comm.
                    # comm shape = (max_c, n_sims, horizon) = comm_per_slot × n_today
                    # We can reconstruct n_today × comm_per_slot:
                    n_today = bootstrap_oos_n_trades_per_slot[:, day_indices].astype(np.float64)
                    from backtest.propfirm.lucidflex import MINI_COMM_RT, MICRO_COMM_RT
                    comm_per_slot = np.where(use_mini_eg, MINI_COMM_RT, MICRO_COMM_RT)
                    comm_3d = comm_per_slot[:, None, None] * n_today
                    gross_eg = pnl_eg_per_slot + comm_3d
                    pnl_grind_per_slot = (gross_eg * (grind_total_risk / eg_total_risk)
                                          - comm_3d)
                else:
                    # FALLBACK: contract type changes between phases → full rebuild
                    pnl_grind_per_slot, _ = _build_per_slot_bootstrap_pnl(
                        bootstrap_oos_R_per_slot,
                        bootstrap_oos_sl_dists_per_slot,
                        bootstrap_oos_fired_per_slot,
                        bootstrap_oos_n_trades_per_slot,
                        day_indices, grind_total_risk, account,
                        cap_daily_risk=bootstrap_cap_daily_risk,
                    )
            else:
                pnl_grind_per_slot = pnl_eg_per_slot
            pnl_p3_per_slot, p3_sizings_full = _build_per_slot_bootstrap_pnl(
                bootstrap_oos_R_per_slot,
                bootstrap_oos_sl_dists_per_slot,
                bootstrap_oos_fired_per_slot,
                bootstrap_oos_n_trades_per_slot,
                day_indices, p3_risk_dollars, account,
            )
        eg_n_minis  = sum(s[0] for s in eg_sizings)
        eg_n_micros = sum(s[1] for s in eg_sizings)
        eg_risk_dollars = sum(s[2] for s in eg_sizings)
        p3_n_minis  = sum(s[0] for s in p3_sizings_full)
        p3_n_micros = sum(s[1] for s in p3_sizings_full)
        p3_actual_risk = sum(s[2] for s in p3_sizings_full)
        pnl_eg = None
        pnl_p3 = None
    elif per_slot_mode:
        # Per-slot mode: each slot gets its own (max_c, n_sims, horizon) PnL.
        # No multi-config-per-slot — single full-risk per slot.
        pnl_eg_per_slot, eg_sizings = _build_per_slot_phase12_pnl(
            slot_assignments, eg_total_risk, account, n_sims, horizon, rng, regime_seqs,
        )
        p3_risk_dollars = p3_risk_pct * MLL_AMT if p3_risk_pct is not None else None
        pnl_p3_per_slot, p3_sizings_full = _build_per_slot_phase3_pnl(
            slot_assignments, account, n_sims, horizon, rng, regime_seqs,
            p3_risk_dollars=p3_risk_dollars,
        )
        # Diagnostics: aggregate across slots for summary
        eg_n_minis = sum(s[0] for s in eg_sizings)
        eg_n_micros = sum(s[1] for s in eg_sizings)
        eg_risk_dollars = eg_total_risk * max_concurrent   # total daily capacity across slots
        p3_n_minis  = sum(s[0] for s in p3_sizings_full)
        p3_n_micros = sum(s[1] for s in p3_sizings_full)
        p3_actual_risk = sum(s[2] for s in p3_sizings_full)
        # Placeholders for legacy code paths that read pnl_eg / pnl_p3
        pnl_eg = None
        pnl_p3 = None
    else:
        # Legacy mode: all slots share the same daily PnL stream.
        # If multi-config and scale_risk_by_n=True: divide risk across configs so
        # the SUM of per-session risk stays at total_risk. If False: each session
        # uses the full risk (total daily risk could be N × total).
        eg_risk_per   = eg_total_risk / n_eg if scale_risk_by_n else eg_total_risk

        pnl_eg, eg_sizings, eg_actual_risk = _generate_combined_day_pnl(
            eg_list, eg_risk_per, account, n_sims, horizon, rng, regime_seqs,
        )

        p3_threshold_per = WD_MIN / n_p3 if scale_risk_by_n else WD_MIN
        p3_sizings: List[Tuple[int, int]] = []
        pnl_p3 = np.zeros((n_sims, horizon), dtype=np.float64)
        p3_total_risk = 0.0
        from dataclasses import replace
        acct_per_session = replace(account, winning_day_min=p3_threshold_per)
        for cfg in p3_list:
            win_mask = cfg.pnl_pts > 0
            avg_win_pts = float(cfg.pnl_pts[win_mask].mean()) if win_mask.any() else 0.0
            avg_sl_pts  = float(np.median(cfg.sl_dists)) if len(cfg.sl_dists) > 0 else 0.0
            n_min, n_mic, risk = min_contracts_for_winning_day(
                avg_win_pts=avg_win_pts, avg_sl_pts=avg_sl_pts,
                account=acct_per_session, n_trades_per_day=cfg.tpd,
            )
            if regime_seqs is not None:
                cfg_pnl = _generate_day_pnl_regime_aware(
                    cfg, n_min, n_mic, n_sims, horizon, regime_seqs, rng,
                )
            else:
                cfg_pnl = _generate_day_pnl(cfg, n_min, n_mic, n_sims, horizon, rng)
            pnl_p3 += cfg_pnl
            p3_sizings.append((n_min, n_mic))
            p3_total_risk += risk
        p3_n_minis  = p3_sizings[0][0] if n_p3 == 1 else sum(s[0] for s in p3_sizings)
        p3_n_micros = p3_sizings[0][1] if n_p3 == 1 else sum(s[1] for s in p3_sizings)
        p3_actual_risk = p3_total_risk
        eg_n_minis  = eg_sizings[0][0] if n_eg == 1 else sum(s[0] for s in eg_sizings)
        eg_n_micros = eg_sizings[0][1] if n_eg == 1 else sum(s[1] for s in eg_sizings)
        eg_risk_dollars = eg_actual_risk
        pnl_eg_per_slot = None
        pnl_p3_per_slot = None

    # ── State arrays (n_sims, max_c) ───────────────────────────────────────
    max_c = max_concurrent
    # phase: 0=inactive, 1=eval, 2=funded_grind, 3=funded_payout
    phase     = np.zeros((n_sims, max_c), dtype=np.int8)
    bal       = np.full ((n_sims, max_c), SB)
    mll_level = np.full ((n_sims, max_c), SB - MLL_AMT)
    peak      = np.full ((n_sims, max_c), SB)
    mll_locked = np.zeros((n_sims, max_c), dtype=bool)
    fu_total_profit = np.zeros((n_sims, max_c))  # total funded profit (for grind→payout trigger)
    fu_win_days = np.zeros((n_sims, max_c), dtype=np.int32)
    fu_n_pay  = np.zeros((n_sims, max_c), dtype=np.int32)
    # Eval-only consistency tracking (LucidFlex rule):
    # max single-day profit and count of profitable days per eval account.
    eval_max_day_pnl = np.zeros((n_sims, max_c), dtype=np.float64)
    eval_n_prof_days = np.zeros((n_sims, max_c), dtype=np.int32)

    cash = np.full(n_sims, float(budget))
    total_withdrawn = np.zeros(n_sims)
    total_eval_fees = np.zeros(n_sims)
    total_payouts   = np.zeros(n_sims, dtype=np.int32)
    n_passed_eval   = np.zeros(n_sims, dtype=np.int32)
    n_hit_target    = np.zeros(n_sims, dtype=np.int32)
    cash_trajectory = np.empty((n_sims, horizon + 1), dtype=np.float64)
    cash_trajectory[:, 0] = cash

    # Per-sim active concurrency cap (adaptive scaling). Initialised to
    # max_concurrent_initial (or max_concurrent if not set), grows by
    # max_concurrent_grow_per_payout per payout up to max_concurrent.
    init_cap = int(max_concurrent_initial if max_concurrent_initial is not None
                   else max_concurrent)
    init_cap = min(init_cap, max_concurrent)
    current_max_c = np.full(n_sims, init_cap, dtype=np.int32)

    ac_range = np.arange(max_c)

    # ── Account opener ─────────────────────────────────────────────────────
    def _open_accounts(should_open: np.ndarray) -> None:
        """Open accounts in inactive slots for sims where should_open is True
        and cash >= eval_fee. Per-sim concurrency cap = current_max_c[sim]."""
        nonlocal cash, total_eval_fees
        for _ in range(max_c):
            n_act = (phase > 0).sum(axis=1)
            can   = should_open & (n_act < current_max_c) & (cash >= EVAL_FEE)
            if not can.any():
                break
            free       = phase == 0
            first_free = np.argmax(free, axis=1)
            open_m     = can[:, None] & free & (ac_range == first_free[:, None])

            phase[open_m]      = 1
            bal[open_m]        = SB
            mll_level[open_m]  = SB - MLL_AMT
            peak[open_m]       = SB
            mll_locked[open_m] = False
            fu_total_profit[open_m] = 0.0
            fu_win_days[open_m] = 0
            fu_n_pay[open_m]    = 0
            eval_max_day_pnl[open_m] = 0.0
            eval_n_prof_days[open_m] = 0

            opened = can & open_m.any(axis=1)
            cash[opened]            -= EVAL_FEE
            total_eval_fees[opened] += EVAL_FEE

    # ── Day-0: open as many as budget allows (greedy buy-in) ───────────────
    _open_accounts(np.ones(n_sims, dtype=bool))

    # ── Pre-fetch day-indexed slices outside the day loop (fast path) ─────
    # day_indices is fixed at sample time; doing the fancy-index ONCE up
    # front avoids repeating the same lookup 84× during the day loop.
    # Saves ~0.5-1.5s per MC run when MAE arrays are supplied.
    if (bootstrap_mode
        and bootstrap_oos_R_per_trade_per_slot is not None
        and bootstrap_oos_MAE_R_per_trade_per_slot is not None):
        # (max_c, n_sims, horizon, max_trades) — cast once to float64
        prefetched_R_per_trade   = bootstrap_oos_R_per_trade_per_slot[:, day_indices, :].astype(np.float64)
        prefetched_MAE_per_trade = bootstrap_oos_MAE_R_per_trade_per_slot[:, day_indices, :].astype(np.float64)
        # (max_c, n_sims, horizon) — n_trades for this day per (slot, sim)
        prefetched_n_trades = bootstrap_oos_n_trades_per_slot[:, day_indices]
        # Pre-compute valid-trade mask (max_c, n_sims, horizon, max_trades)
        max_trades_dim = prefetched_R_per_trade.shape[3]
        trade_idx_arr = np.arange(max_trades_dim)[None, None, None, :]
        prefetched_valid = trade_idx_arr < prefetched_n_trades[:, :, :, None]
    else:
        prefetched_R_per_trade   = None
        prefetched_MAE_per_trade = None
        prefetched_valid         = None

    # ── Main day loop — fully vectorised across (n_sims, max_c) ───────────
    for day in range(horizon):
        # ── Initial phase masks (snapshot BEFORE state updates this day) ──
        is_eval_i  = phase == 1      # (n_sims, max_c)
        is_grind_i = phase == 2
        is_pay_i   = phase == 3
        active_i   = phase > 0

        # ── Pick today's PnL per (sim, slot) ──────────────────────────────
        if bootstrap_mode and bootstrap_buffer_safety is not None:
            # Buffer-aware: compute today's PnL dynamically. Per-trade risk is
            # capped at current_buffer × safety so intraday adverse can't touch MLL.
            di = day_indices[:, day]                                          # (n_sims,)
            R_today_per_slot = bootstrap_oos_R_per_slot[:, di].astype(np.float64)  # (max_c, n_sims)
            n_today_per_slot = bootstrap_oos_n_trades_per_slot[:, di]
            fired_today_per_slot = bootstrap_oos_fired_per_slot[:, di]
            R_today_t     = R_today_per_slot.T                                # (n_sims, max_c)
            n_today_t     = n_today_per_slot.T.astype(np.float64)
            fired_today_t = fired_today_per_slot.T

            safe_buffer = np.maximum(bal - mll_level, 0.0)                    # (n_sims, max_c)
            eg_eff_risk = np.minimum(eg_total_risk,   safe_buffer * bootstrap_buffer_safety)
            p3_eff_risk = np.minimum(p3_risk_dollars, safe_buffer * bootstrap_buffer_safety)

            if bootstrap_cap_daily_risk:
                safe_n = np.maximum(n_today_t, 1.0)
                gross_eg = (R_today_t / safe_n) * eg_eff_risk
                gross_p3 = (R_today_t / safe_n) * p3_eff_risk
            else:
                gross_eg = R_today_t * eg_eff_risk
                gross_p3 = R_today_t * p3_eff_risk

            comm_eg_today = n_today_t * eg_comm_per_slot[None, :]
            comm_p3_today = n_today_t * p3_comm_per_slot[None, :]

            pnl_eg_today = np.where(fired_today_t, gross_eg - comm_eg_today, 0.0)
            pnl_p3_today = np.where(fired_today_t, gross_p3 - comm_p3_today, 0.0)
            # buf_safety path doesn't yet separate grind from eval risk;
            # treat them the same (legacy behaviour).
            pnl_grind_today = pnl_eg_today
        elif per_slot_mode:
            pnl_eg_today = pnl_eg_per_slot[:, :, day].T   # (n_sims, max_c)
            pnl_p3_today = pnl_p3_per_slot[:, :, day].T
            pnl_grind_today = (pnl_grind_per_slot[:, :, day].T
                               if pnl_grind_per_slot is not None
                               else pnl_eg_today)
        else:
            pnl_eg_today = np.broadcast_to(pnl_eg[:, day:day+1], (n_sims, max_c))
            pnl_p3_today = np.broadcast_to(pnl_p3[:, day:day+1], (n_sims, max_c))
            pnl_grind_today = pnl_eg_today   # legacy mode shares stream

        # Phase-aware PnL routing: eval→eg, grind→grind (may differ), payout→p3
        d_pnl = np.where(is_pay_i, pnl_p3_today,
                np.where(is_grind_i, pnl_grind_today, pnl_eg_today))
        d_pnl = np.where(active_i, d_pnl, 0.0)

        # ── Apply PnL, detect blown, update bal/peak ──────────────────────
        new_bal = bal + d_pnl
        blown   = active_i & (new_bal <= mll_level)
        # Voluntary abandon: if next trade's effective_risk would be too small
        # to be useful, treat as blown so the slot can be reopened. Models a
        # real trader giving up on a thin-buffer account rather than grinding
        # at vanishingly small position size.
        if (bootstrap_buffer_safety is not None
            and bootstrap_min_effective_risk is not None):
            next_buffer  = np.maximum(new_bal - mll_level, 0.0)
            next_eff_risk = next_buffer * bootstrap_buffer_safety
            abandon = active_i & (next_eff_risk < bootstrap_min_effective_risk) & ~blown
            blown   = blown | abandon
        # Intraday-blow check: on a losing day, intraday adverse went at least
        # to the day's net loss; if buffer was thinner than the per-trade risk
        # at trade time, the position likely touched MLL intraday before the
        # EOD close. Conservatively assume losing days with thin buffer always
        # touch MLL → blown. (Phase-aware: uses EG risk for phase 1+2, P3 risk
        # for phase 3.)
        if bootstrap_intraday_blow_check and bootstrap_mode:
            # Per-trade risk: with cap_daily_risk on, individual trade risk is
            # phase_risk / n_trades_today. The intraday-blow test checks whether
            # buffer was below this per-trade max-adverse — single-trade granularity.
            # Phase-aware per-trade risk:
            #   eval     (phase=1) → eg_total_risk
            #   grind    (phase=2) → grind_total_risk (may differ from eval)
            #   payout   (phase=3) → p3_risk_dollars
            phase_risk = np.where(
                is_pay_i, p3_risk_dollars,
                np.where(is_grind_i, grind_total_risk, eg_total_risk),
            )
            n_today_for_risk = bootstrap_oos_n_trades_per_slot[:, day_indices[:, day]].T  # (n_sims, max_c)
            if bootstrap_cap_daily_risk:
                per_trade_risk = phase_risk / np.maximum(n_today_for_risk, 1)
            else:
                per_trade_risk = phase_risk   # full risk per trade with no cap
            # M4 (strict): any day where a trade fired AND buffer < per-trade risk.
            # M3 (default): only losing days.
            if bootstrap_intraday_strict:
                fired_proxy = (n_today_for_risk > 0)
            else:
                fired_proxy = (d_pnl < 0)
            intraday_blow = active_i & fired_proxy & ((bal - per_trade_risk) < mll_level) & ~blown
            blown = blown | intraday_blow

            # ── Real-MAE sequential walk (preferred over the proxy above) ──
            # When per-trade R + MAE_R arrays are supplied, replace the proxy
            # with an exact intraday walk: trade k's floating low equals
            #   bal + Σ pnl_$_(0..k-1)  −  $MAE_k.
            # If the floating low touches MLL at any trade, the account is
            # blown at that trade and subsequent trades that day don't fire.
            if prefetched_R_per_trade is not None:
                # Pre-fetched slices are (max_c, n_sims, horizon, max_trades).
                # Pull this day's view — much cheaper than re-running fancy index.
                R_today_3d   = prefetched_R_per_trade[:, :, day, :]      # (max_c, n_sims, max_trades)
                MAE_today_3d = prefetched_MAE_per_trade[:, :, day, :]
                valid_mask   = prefetched_valid[:, :, day, :]
                # per_trade_risk shape: scalar OR (n_sims, max_c). Cast to
                # (max_c, n_sims, 1) for per-trade broadcasting.
                if np.ndim(per_trade_risk) == 0:
                    ptr_3d = float(per_trade_risk)
                    pnl_per_trade = R_today_3d * ptr_3d
                    mae_per_trade = MAE_today_3d * ptr_3d
                else:
                    ptr_3d = per_trade_risk.T[:, :, None]
                    pnl_per_trade = R_today_3d * ptr_3d
                    mae_per_trade = MAE_today_3d * ptr_3d
                # Cumulative pnl BEFORE trade k (exclusive cumsum)
                cumsum_before = np.cumsum(pnl_per_trade, axis=2) - pnl_per_trade
                # bal broadcast to (max_c, n_sims, 1); mll_level same
                bal_3d = bal.T[:, :, None]
                mll_3d = mll_level.T[:, :, None]
                floating_low = bal_3d + cumsum_before - mae_per_trade
                blow_per_trade = valid_mask & (floating_low <= mll_3d)
                # Account blown if any of its day's trades hit MLL intraday
                intraday_blow_real = (active_i.T & blow_per_trade.any(axis=2)).T
                blown = blown | intraday_blow_real
        survives = active_i & ~blown
        bal  = np.where(survives, new_bal, bal)
        peak = np.where(survives, np.maximum(peak, new_bal), peak)

        # ── MLL tracking (eval vs funded) ─────────────────────────────────
        on_funded = survives & (is_grind_i | is_pay_i)
        on_eval   = survives & is_eval_i

        # Funded: detect lock event, then update level
        just_lock = on_funded & ~mll_locked & (peak >= INITIAL_TRAIL)
        mll_locked = mll_locked | just_lock
        new_mll_funded = np.where(
            mll_locked,
            LOCKED_MLL,
            np.maximum(mll_level, peak - MLL_AMT),
        )
        mll_level = np.where(on_funded, new_mll_funded, mll_level)

        # Eval: trailing MLL, capped at LOCKED_MLL
        new_mll_eval = np.minimum(peak - MLL_AMT, LOCKED_MLL)
        new_mll_eval = np.maximum(mll_level, new_mll_eval)
        mll_level = np.where(on_eval, new_mll_eval, mll_level)

        # ── Update eval consistency tracking ───────────────────────────────
        if enforce_consistency_rule:
            on_eval_today = on_eval                 # accounts that survived & still in eval
            profitable_today = on_eval_today & (d_pnl > 0)
            eval_max_day_pnl = np.where(on_eval_today,
                                         np.maximum(eval_max_day_pnl, d_pnl),
                                         eval_max_day_pnl)
            eval_n_prof_days = eval_n_prof_days + profitable_today.astype(np.int32)

        # ── Eval pass → grind transition (resets balance to SB) ───────────
        ev_passed = on_eval & (bal >= SB + PT)
        if enforce_consistency_rule:
            total_eval_profit = bal - SB
            consistent = (
                (eval_max_day_pnl <= total_eval_profit * 0.5)
                & (eval_n_prof_days >= 2)
            )
            ev_passed = ev_passed & consistent
        if ev_passed.any():
            phase = np.where(ev_passed, np.int8(2), phase)
            bal        = np.where(ev_passed, SB, bal)
            mll_level  = np.where(ev_passed, SB - MLL_AMT, mll_level)
            peak       = np.where(ev_passed, SB, peak)
            mll_locked = np.where(ev_passed, False, mll_locked)
            fu_total_profit = np.where(ev_passed, 0.0, fu_total_profit)
            n_passed_eval += ev_passed.sum(axis=1).astype(np.int32)

        # ── Accumulate funded profit (only for accounts ALREADY in grind/pay before this day) ──
        fu_gain_today = np.where(
            survives & (is_grind_i | is_pay_i),  # exclude just-transitioned eval-passers
            d_pnl, 0.0,
        )
        fu_total_profit = fu_total_profit + fu_gain_today

        # ── Grind → payout transition ─────────────────────────────────────
        hit_target = survives & is_grind_i & (fu_total_profit >= PT)
        if hit_target.any():
            phase = np.where(hit_target, np.int8(3), phase)
            fu_win_days = np.where(hit_target, np.int32(0), fu_win_days)
            n_hit_target += hit_target.sum(axis=1).astype(np.int32)

        # ── Winning-day count (phase 3 only, after both transitions) ──────
        is_pay_after = phase == 3
        won_today    = survives & is_pay_after & (d_pnl >= WD_MIN)
        fu_win_days  = np.where(won_today, fu_win_days + 1, fu_win_days)

        # ── Payouts ───────────────────────────────────────────────────────
        can_payout = is_pay_after & survives & (fu_win_days >= 5) & (fu_n_pay < MAX_PAY)
        if can_payout.any():
            profits = np.maximum(0.0, bal - SB)
            gross   = np.minimum(profits * 0.5, PAY_CAP)
            do_pay  = can_payout & (gross >= 500.0)
            net     = gross * SPLIT
            # Withdraw: per-slot bal adjustment, per-sim cash sum across slots
            bal             = np.where(do_pay, bal - gross, bal)
            cash           += (net * do_pay).sum(axis=1)
            total_withdrawn += (net * do_pay).sum(axis=1)
            total_payouts   += do_pay.sum(axis=1).astype(np.int32)
            fu_n_pay        = np.where(do_pay, fu_n_pay + 1, fu_n_pay)
            fu_win_days     = np.where(do_pay, np.int32(0), fu_win_days)
            mll_locked      = mll_locked | do_pay
            mll_level       = np.where(do_pay, LOCKED_MLL, mll_level)
            # Adaptive scaling: each payout unlocks more concurrent slots
            if max_concurrent_grow_per_payout > 0:
                payouts_today = do_pay.sum(axis=1).astype(np.int32)
                current_max_c = np.minimum(
                    current_max_c + payouts_today * max_concurrent_grow_per_payout,
                    max_c,
                )

        # ── Close accounts (blown or max payouts reached) ─────────────────
        phase = np.where(blown, np.int8(0), phase)
        max_reached = is_pay_after & survives & (fu_n_pay >= MAX_PAY)
        if max_reached.any():
            phase = np.where(max_reached, np.int8(0), phase)

        # ── Greedy reinvestment: open new accounts if cash allows ─────────
        _open_accounts(np.ones(n_sims, dtype=bool))

        # Record EOD cash for this day (after payouts and reinvestment)
        cash_trajectory[:, day + 1] = cash

    return {
        'cash':           cash,
        'cash_trajectory': cash_trajectory,
        'n_payouts':      total_payouts,
        'n_passed_eval':  n_passed_eval,
        'n_hit_target':   n_hit_target,
        'withdrawn':      total_withdrawn,
        'eval_fees_paid': total_eval_fees,
        # Sizing diagnostics — useful when debugging
        '_p3_n_minis':    p3_n_minis,
        '_p3_n_micros':   p3_n_micros,
        '_p3_risk':       p3_actual_risk,
        '_eg_n_minis':    eg_n_minis,
        '_eg_n_micros':   eg_n_micros,
        '_eg_risk':       eg_risk_dollars,
    }
