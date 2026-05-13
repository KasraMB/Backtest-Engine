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


def _build_per_slot_phase3_pnl(
    slot_assignments: List[SlotAssignment],
    account:          LucidFlexAccount,
    n_sims:           int,
    horizon:          int,
    rng:              np.random.Generator,
    regime_seqs:      Optional[np.ndarray],
) -> Tuple[np.ndarray, List[Tuple[int, int, float]]]:
    """Pre-generate (max_c, n_sims, horizon) daily PnL for phase 3 per slot.
    Each slot uses min_contracts_for_winning_day sizing on its own P3 config —
    single-session, so the full winning_day_min threshold applies."""
    max_c     = len(slot_assignments)
    out       = np.zeros((max_c, n_sims, horizon), dtype=np.float64)
    sizings:  List[Tuple[int, int, float]] = []
    for a, (_p12_cfg, p3_cfg) in enumerate(slot_assignments):
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

    # Per-slot mode flag — when set, each account slot has its own dedicated
    # (P12, P3) configs and PnL streams (single-session-per-account).
    per_slot_mode = slot_assignments is not None
    if per_slot_mode:
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
    if regime_result is not None:
        regime_seqs = _sample_regime_sequences(regime_result, n_sims, horizon, rng)

    # ── Phase 1+2 sizing: total risk = eval_grind_risk × MLL ───────────────
    eg_total_risk = eval_grind_risk * MLL_AMT

    if per_slot_mode:
        # Per-slot mode: each slot gets its own (max_c, n_sims, horizon) PnL.
        # No multi-config-per-slot — single full-risk per slot.
        pnl_eg_per_slot, eg_sizings = _build_per_slot_phase12_pnl(
            slot_assignments, eg_total_risk, account, n_sims, horizon, rng, regime_seqs,
        )
        pnl_p3_per_slot, p3_sizings_full = _build_per_slot_phase3_pnl(
            slot_assignments, account, n_sims, horizon, rng, regime_seqs,
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

    cash = np.full(n_sims, float(budget))
    total_withdrawn = np.zeros(n_sims)
    total_eval_fees = np.zeros(n_sims)
    total_payouts   = np.zeros(n_sims, dtype=np.int32)
    n_passed_eval   = np.zeros(n_sims, dtype=np.int32)
    n_hit_target    = np.zeros(n_sims, dtype=np.int32)

    ac_range = np.arange(max_c)

    # ── Account opener ─────────────────────────────────────────────────────
    def _open_accounts(should_open: np.ndarray) -> None:
        """Open accounts in inactive slots for sims where should_open is True
        and cash >= eval_fee. Loops up to max_c times (each loop fills 1 slot per sim)."""
        nonlocal cash, total_eval_fees
        for _ in range(max_c):
            n_act = (phase > 0).sum(axis=1)
            can   = should_open & (n_act < max_c) & (cash >= EVAL_FEE)
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

            opened = can & open_m.any(axis=1)
            cash[opened]            -= EVAL_FEE
            total_eval_fees[opened] += EVAL_FEE

    # ── Day-0: open as many as budget allows (greedy buy-in) ───────────────
    _open_accounts(np.ones(n_sims, dtype=bool))

    # ── Main day loop ──────────────────────────────────────────────────────
    for day in range(horizon):
        for a in range(max_c):
            is_eval  = phase[:, a] == 1
            is_grind = phase[:, a] == 2
            is_pay   = phase[:, a] == 3
            active   = is_eval | is_grind | is_pay
            if not active.any():
                continue

            # Pick PnL: eval+grind use eval_grind config; payout uses payout config.
            # In per-slot mode each slot has its OWN dedicated PnL stream
            # (single-session-per-account); otherwise all slots share the same stream.
            if per_slot_mode:
                pnl_eg_day = pnl_eg_per_slot[a, :, day]
                pnl_p3_day = pnl_p3_per_slot[a, :, day]
            else:
                pnl_eg_day = pnl_eg[:, day]
                pnl_p3_day = pnl_p3[:, day]
            d_pnl = np.where(is_pay, pnl_p3_day, pnl_eg_day)
            d_pnl = np.where(active, d_pnl, 0.0)

            # Apply PnL
            new_bal = bal[:, a] + d_pnl

            # MLL breach (account dies)
            blown = active & (new_bal <= mll_level[:, a])

            # Update balance / peak for survivors
            survives = active & ~blown
            bal[:, a]  = np.where(survives, new_bal, bal[:, a])
            peak[:, a] = np.where(survives, np.maximum(peak[:, a], new_bal), peak[:, a])

            # MLL tracking
            # — In eval phase: trailing MLL = peak − mll_amount, capped at LOCKED_MLL
            # — In funded: if peak >= INITIAL_TRAIL and not locked → lock at LOCKED_MLL
            on_funded = survives & (is_grind | is_pay)
            on_eval   = survives & is_eval

            # Funded: detect lock event
            just_lock = on_funded & ~mll_locked[:, a] & (peak[:, a] >= INITIAL_TRAIL)
            mll_locked[:, a] = mll_locked[:, a] | just_lock
            new_mll_funded = np.where(
                mll_locked[:, a],
                LOCKED_MLL,
                np.maximum(mll_level[:, a], peak[:, a] - MLL_AMT),
            )
            mll_level[:, a] = np.where(on_funded, new_mll_funded, mll_level[:, a])

            # Eval: trailing MLL, capped at LOCKED_MLL (per Lucid rules)
            new_mll_eval = np.minimum(peak[:, a] - MLL_AMT, LOCKED_MLL)
            new_mll_eval = np.maximum(mll_level[:, a], new_mll_eval)
            mll_level[:, a] = np.where(on_eval, new_mll_eval, mll_level[:, a])

            # ── Phase transitions ─────────────────────────────────────────
            # Eval pass → funded grind (eval profit target reached)
            ev_passed = on_eval & (bal[:, a] >= SB + PT)
            if ev_passed.any():
                phase[ev_passed, a] = 2
                # Reset for funded: balance starts back at SB (eval profit becomes the "buffer")
                # In Lucid Flex, funded account starts at SB regardless of eval profit
                bal[ev_passed, a]        = SB
                mll_level[ev_passed, a]  = SB - MLL_AMT
                peak[ev_passed, a]       = SB
                mll_locked[ev_passed, a] = False
                fu_total_profit[ev_passed, a] = 0.0
                n_passed_eval[ev_passed] += 1

            # Track funded profit for grind→payout transition
            # Note: must come AFTER MLL update but reflect the bal change THIS day
            fu_gain_today = np.where(survives & (is_grind | is_pay), d_pnl, 0.0)
            fu_total_profit[:, a] += fu_gain_today

            # Grind → payout transition (hit +$3K funded profit)
            hit_target = survives & is_grind & (fu_total_profit[:, a] >= PT)
            if hit_target.any():
                phase[hit_target, a] = 3
                # Reset winning-day counter (start fresh in payout phase)
                fu_win_days[hit_target, a] = 0
                n_hit_target[hit_target] += 1

            # ── Winning-day count & payouts (phase 3 only) ────────────────
            is_pay_now = phase[:, a] == 3
            won_today  = is_pay_now & (d_pnl >= WD_MIN) & survives
            fu_win_days[:, a] = np.where(won_today, fu_win_days[:, a] + 1, fu_win_days[:, a])

            can_payout = is_pay_now & (fu_win_days[:, a] >= 5) & (fu_n_pay[:, a] < MAX_PAY) & survives
            if can_payout.any():
                profits = np.maximum(0.0, bal[:, a] - SB)
                gross   = np.minimum(profits * 0.5, PAY_CAP)
                do_pay  = can_payout & (gross >= 500.0)

                net = gross * SPLIT
                # Withdraw
                bal[do_pay, a]            -= gross[do_pay]
                cash[do_pay]              += net[do_pay]
                total_withdrawn[do_pay]   += net[do_pay]
                total_payouts[do_pay]     += 1
                fu_n_pay[do_pay, a]       += 1
                fu_win_days[do_pay, a]    = 0
                # Payout forces MLL lock
                mll_locked[do_pay, a]     = True
                mll_level[do_pay, a]      = LOCKED_MLL

            # ── Close accounts ────────────────────────────────────────────
            phase[blown, a] = 0

            # Max payouts reached → close
            max_reached = is_pay_now & (fu_n_pay[:, a] >= MAX_PAY) & survives
            if max_reached.any():
                phase[max_reached, a] = 0

        # ── Greedy reinvestment: open new accounts if cash allows ─────────
        _open_accounts(np.ones(n_sims, dtype=bool))

    return {
        'cash':           cash,
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
