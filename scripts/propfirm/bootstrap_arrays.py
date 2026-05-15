"""
Vectorised builder for per-slot OOS bootstrap arrays.

Replaces the previous Python double-loop (over slots × days × cfg lookups)
with a fancy-indexed table lookup. Speedup is roughly 50-100x for typical
shapes (5 slots × 526 days × 4 cfgs/regime).

Usage
─────
  from scripts.propfirm.bootstrap_arrays import build_per_slot_arrays_vec

  Rs, Ss, Fs, Ns, Rts, Mts = build_per_slot_arrays_vec(
      session_key='930+1400',
      n_slots=5,
      regime_seq=regime_lag2,        # (n_oos_days,) int8
      selection=train_selection,
      per_day=per_day_cache,
      all_oos_dates=all_oos_dates,    # list of date objs
      max_trades_per_day=4,
  )
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np


def build_per_slot_arrays_vec(
    session_key:        str,
    n_slots:            int,
    regime_seq:         np.ndarray,
    selection:          dict,
    per_day:            dict,
    all_oos_dates:      Sequence,
    max_trades_per_day: int = 4,
):
    """Vectorised per-slot per-OOS-day array builder.

    Returns (Rs, Ss, Fs, Ns, Rts, Mts) all shape (n_slots, n_oos_days [, max_trades]).
    """
    n_oos_days = len(all_oos_dates)
    date_to_idx = {d: i for i, d in enumerate(all_oos_dates)}

    # ── Resolve per-(regime, slot) cfg_key from selection ───────────────
    sel = selection[session_key]
    slot_alloc = sel['slot_alloc_per_regime']
    top_k      = sel['top_k_per_regime']
    per_regime_cfgs: Dict[int, List[str]] = {}
    for r in range(3):
        order = [k for k, *_ in top_k[r]]
        seq: List[str] = []
        for cfg_key in order:
            seq.extend([cfg_key] * slot_alloc[r].get(cfg_key, 0))
        while len(seq) < n_slots:
            seq.append(seq[-1] if seq else order[0])
        per_regime_cfgs[r] = seq[:n_slots]

    # ── Discover all unique cfg_keys we need, assign integer IDs ────────
    needed: List[str] = sorted({ck for r in range(3) for ck in per_regime_cfgs[r]})
    cfg_id = {ck: i for i, ck in enumerate(needed)}
    n_cfgs = len(needed)

    # ── Stack per-cfg arrays for VECTORISED day lookup ──────────────────
    # All shape: (n_cfgs, n_oos_days[, max_trades_per_day])
    R_stack  = np.zeros((n_cfgs, n_oos_days), dtype=np.float32)
    S_stack  = np.zeros((n_cfgs, n_oos_days), dtype=np.float32)
    N_stack  = np.zeros((n_cfgs, n_oos_days), dtype=np.int16)
    F_stack  = np.zeros((n_cfgs, n_oos_days), dtype=bool)
    Rt_stack = np.zeros((n_cfgs, n_oos_days, max_trades_per_day), dtype=np.float32)
    Mt_stack = np.zeros((n_cfgs, n_oos_days, max_trades_per_day), dtype=np.float32)

    # The only non-vectorised part: per-cfg DataFrame -> per-day arrays.
    # This is unavoidable because the source is a pandas DataFrame keyed
    # by date, but it's O(n_cfgs * n_days) which is small.
    for ck in needed:
        ci = cfg_id[ck]
        days = per_day[ck]['days']
        m = days.date.isin(all_oos_dates)
        sub = days.loc[m]
        # Vectorised assignment via index lookup
        if len(sub):
            idx = np.array([date_to_idx[d] for d in sub.date.values], dtype=np.int32)
            R_stack[ci, idx] = sub.R.values
            S_stack[ci, idx] = sub.sl_dist.values
            N_stack[ci, idx] = sub.n_trades.values
            F_stack[ci, idx] = True

        trades = per_day[ck]['trades']
        m_t = trades.date.isin(all_oos_dates)
        sub_t = trades.loc[m_t]
        if len(sub_t):
            # Build per-trade index within day
            sub_t = sub_t.assign(_within_day=sub_t.groupby('date').cumcount())
            keep = sub_t['_within_day'] < max_trades_per_day
            sub_t = sub_t.loc[keep]
            di = np.array([date_to_idx[d] for d in sub_t.date.values], dtype=np.int32)
            ti = sub_t['_within_day'].values.astype(np.int32)
            Rt_stack[ci, di, ti] = sub_t.R.values
            Mt_stack[ci, di, ti] = sub_t.MAE_R.values

    # ── Build (n_states=3, n_slots) cfg_id table ────────────────────────
    slot_cfg_table = np.zeros((3, n_slots), dtype=np.int32)
    for r in range(3):
        for i in range(n_slots):
            slot_cfg_table[r, i] = cfg_id[per_regime_cfgs[r][i]]

    # ── Per-(slot, day) cfg_id via fancy indexing — fully vectorised ───
    regime_seq_int = regime_seq.astype(np.int32)
    # cfg_per_day[slot, day] = slot_cfg_table[regime_seq[day], slot]
    cfg_per_day = slot_cfg_table[regime_seq_int[None, :],
                                  np.arange(n_slots, dtype=np.int32)[:, None]]
    # shape (n_slots, n_oos_days)

    # ── Fancy-index the stacked arrays ─────────────────────────────────
    day_idx_2d = np.arange(n_oos_days, dtype=np.int32)[None, :]   # (1, n_oos_days)
    Rs = R_stack[cfg_per_day, day_idx_2d]                           # (n_slots, n_oos_days)
    Ss = S_stack[cfg_per_day, day_idx_2d]
    Ns = N_stack[cfg_per_day, day_idx_2d]
    Fs = F_stack[cfg_per_day, day_idx_2d]
    Rts = Rt_stack[cfg_per_day, day_idx_2d, :]                      # (n_slots, n_oos_days, max_trades)
    Mts = Mt_stack[cfg_per_day, day_idx_2d, :]

    # Replace SL=0 with 1.0 to match legacy convention (avoid div-by-zero)
    Ss = np.where(Fs, Ss, 1.0).astype(np.float32)

    return Rs, Ss, Fs, Ns, Rts, Mts


def build_alloc_arrays_vec(
    n_9301400:       int,
    n_2000:          int,
    selection:       dict,
    per_day:         dict,
    all_oos_dates:   Sequence,
    regime_for_9301400: np.ndarray,
    regime_for_2000:    np.ndarray,
    max_trades_per_day: int = 4,
):
    """Stack per-session vectorised arrays for a (n_9301400, n_2000) split."""
    parts = []
    if n_9301400 > 0:
        parts.append(build_per_slot_arrays_vec(
            '930+1400', n_9301400, regime_for_9301400, selection, per_day,
            all_oos_dates, max_trades_per_day,
        ))
    if n_2000 > 0:
        parts.append(build_per_slot_arrays_vec(
            '2000', n_2000, regime_for_2000, selection, per_day,
            all_oos_dates, max_trades_per_day,
        ))
    Rs  = np.concatenate([p[0] for p in parts], axis=0)
    Ss  = np.concatenate([p[1] for p in parts], axis=0)
    Fs  = np.concatenate([p[2] for p in parts], axis=0)
    Ns  = np.concatenate([p[3] for p in parts], axis=0)
    Rts = np.concatenate([p[4] for p in parts], axis=0)
    Mts = np.concatenate([p[5] for p in parts], axis=0)
    return Rs, Ss, Fs, Ns, Rts, Mts


def shifted_regime_seq_vec(regime_base: np.ndarray, shift_td: int) -> np.ndarray:
    """Vectorised shift: out[i] = regime_base[max(0, i - shift_td)]."""
    n = len(regime_base)
    out = np.empty(n, dtype=regime_base.dtype)
    if shift_td <= 0:
        out[:] = regime_base[max(0, -shift_td):n + max(0, -shift_td)]
        return out
    out[:shift_td] = regime_base[0]
    out[shift_td:] = regime_base[:n - shift_td]
    return out


def predicted_regime_from_lag1_vec(regime_base: np.ndarray, transition_matrix: np.ndarray) -> np.ndarray:
    """Build 'predicted-tomorrow' regime sequence from lag-1: argmax of transition row."""
    pred_table = np.argmax(transition_matrix, axis=1).astype(regime_base.dtype)
    lag1 = shifted_regime_seq_vec(regime_base, 1)
    return pred_table[lag1.astype(np.int32)]
