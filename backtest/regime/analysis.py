"""
backtest/regime/analysis.py
───────────────────────────
Regime breakdown statistics, permutation significance test, and trade filtering.

Main entry point:
  from backtest.regime.analysis import run_regime_analysis

  result = run_regime_analysis(
      trades,           # list[Trade] from RunResult
      regime_result,    # RegimeResult from hmm.fit_regimes
      data,             # MarketData
      allowed_regimes,  # e.g. ["bull","neutral"] or None for no filter
      n_permutations=5_000,
  )
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Per-regime stats
# ---------------------------------------------------------------------------

@dataclass
class RegimeStats:
    name:          str
    n_trades:      int
    win_rate:      float
    avg_pnl:       float
    total_pnl:     float
    sharpe:        float     # annualised, from daily PnL
    expectancy_r:  float     # avg PnL in units of avg loss
    label:         int       # 0/1/2


@dataclass
class RegimeAnalysisResult:
    # Per-regime breakdown (all trades)
    breakdown:             list[RegimeStats]

    # Filtered trade set (only allowed regimes, OOS only)
    filtered_trades:       list          # list[Trade]
    unfiltered_oos_trades: list          # OOS trades regardless of regime

    # Permutation test
    perm_pvalue:           float         # p-value: P(random filter ≥ actual improvement)
    perm_n:                int
    actual_sharpe_gain:    float         # sharpe(filtered) - sharpe(unfiltered OOS)
    perm_sharpe_gains:     np.ndarray    # distribution of random gains

    # Filter config
    allowed_regimes:       Optional[list[str]]
    train_end_date:        Optional[date]

    # Regime stability
    avg_duration_days:     dict[str, float]
    transition_matrix:     np.ndarray

    # In-sample vs out-of-sample comparison
    is_filtered_sharpe:    float   # filtered, in-sample
    oos_filtered_sharpe:   float   # filtered, out-of-sample
    is_unfiltered_sharpe:  float
    oos_unfiltered_sharpe: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _trade_date(trade, data) -> date:
    """Get the calendar date of a trade's entry bar."""
    return data.df_1m.index[trade.entry_bar].date()


def _daily_sharpe(pnls: np.ndarray) -> float:
    """Annualised Sharpe from a list of per-trade dollar PnLs."""
    if len(pnls) < 2:
        return 0.0
    # Group by day: sum per day then compute daily Sharpe
    std = float(np.std(pnls))
    if std == 0:
        return 0.0
    return float(np.mean(pnls) / std * np.sqrt(252))


def _regime_stats(trades: list, name: str, label: int, data) -> RegimeStats:
    if not trades:
        return RegimeStats(name=name, n_trades=0, win_rate=0.0, avg_pnl=0.0,
                           total_pnl=0.0, sharpe=0.0, expectancy_r=0.0, label=label)
    pnls  = np.array([t.net_pnl_dollars for t in trades])
    wins  = pnls[pnls > 0]
    losses= pnls[pnls < 0]
    avg_loss = float(np.mean(np.abs(losses))) if len(losses) > 0 else 1.0
    exp_r    = float(np.mean(pnls)) / avg_loss if avg_loss > 0 else 0.0
    return RegimeStats(
        name         = name,
        n_trades     = len(trades),
        win_rate     = float((pnls > 0).mean()),
        avg_pnl      = float(pnls.mean()),
        total_pnl    = float(pnls.sum()),
        sharpe       = _daily_sharpe(pnls),
        expectancy_r = exp_r,
        label        = label,
    )


def _permutation_test(
    oos_trades:      list,
    regime_labels:   dict,
    allowed_regimes: list[str],
    label_names:     dict[int, str],
    data,
    n_perm:          int,
    rng:             np.random.Generator,
) -> tuple[float, np.ndarray, float]:
    """
    Permutation test: randomly shuffle regime labels across OOS days and
    recompute the Sharpe gain each time.

    Returns (p_value, perm_sharpe_gains, actual_gain).
    """
    if not oos_trades or not allowed_regimes:
        return 1.0, np.array([0.0]), 0.0

    oos_dates = sorted(set(_trade_date(t, data) for t in oos_trades))
    n_days    = len(oos_dates)
    pnls_all  = np.array([t.net_pnl_dollars for t in oos_trades])
    base_sharpe = _daily_sharpe(pnls_all)

    # Map each OOS trade to its regime
    allowed_labels = {k for k, v in label_names.items() if v in allowed_regimes}
    trade_dates    = [_trade_date(t, data) for t in oos_trades]
    actual_mask    = np.array([
        regime_labels.get(d, 1) in allowed_labels for d in trade_dates
    ])
    filtered_pnls  = pnls_all[actual_mask]
    actual_gain    = _daily_sharpe(filtered_pnls) - base_sharpe if filtered_pnls.any() else 0.0

    # Build a day-level label array we can shuffle
    day_labels_arr = np.array([regime_labels.get(d, 1) for d in oos_dates])
    day_to_idx     = {d: i for i, d in enumerate(oos_dates)}
    trade_day_idx  = np.array([day_to_idx.get(d, 0) for d in trade_dates])

    gains = np.empty(n_perm)
    for p in range(n_perm):
        shuffled = rng.permutation(day_labels_arr)
        mask     = np.array([shuffled[trade_day_idx[i]] in allowed_labels
                              for i in range(len(oos_trades))])
        fp       = pnls_all[mask]
        gains[p] = _daily_sharpe(fp) - base_sharpe if fp.any() else 0.0

    p_value = float((gains >= actual_gain).mean())
    return p_value, gains, actual_gain


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_regime_analysis(
    trades:          list,
    regime_result,                       # RegimeResult
    data,                                # MarketData
    allowed_regimes: Optional[list[str]] = None,
    n_permutations:  int                 = 5_000,
    seed:            int                 = 42,
) -> RegimeAnalysisResult:
    """
    Full regime analysis:
      1. Tag each trade with its regime
      2. Compute per-regime stats
      3. Split IS/OOS trades
      4. Permutation test on filtered vs unfiltered Sharpe (OOS only)
    """
    rng = np.random.default_rng(seed)

    labels     = regime_result.labels
    label_names= regime_result.label_names
    train_end  = regime_result.train_end_date

    # ── Tag every trade with a regime ─────────────────────────────────────────
    def get_regime(trade) -> int:
        d = _trade_date(trade, data)
        return labels.get(d, 1)   # default neutral

    # ── Per-regime breakdown (all trades) ─────────────────────────────────────
    by_label: dict[int, list] = {0: [], 1: [], 2: []}
    for t in trades:
        by_label[get_regime(t)].append(t)

    breakdown = [
        _regime_stats(by_label[lbl], name, lbl, data)
        for lbl, name in label_names.items()
    ]

    # ── IS / OOS split ────────────────────────────────────────────────────────
    is_trades  = [t for t in trades if train_end and _trade_date(t, data) <= train_end]
    oos_trades = [t for t in trades if not train_end or _trade_date(t, data) > train_end]

    # ── Apply regime filter ───────────────────────────────────────────────────
    if allowed_regimes:
        allowed_labels = {k for k, v in label_names.items() if v in allowed_regimes}
        filtered_trades     = [t for t in trades if get_regime(t) in allowed_labels]
        filtered_oos        = [t for t in oos_trades if get_regime(t) in allowed_labels]
        filtered_is         = [t for t in is_trades  if get_regime(t) in allowed_labels]
    else:
        filtered_trades  = trades
        filtered_oos     = oos_trades
        filtered_is      = is_trades

    # ── Sharpe comparisons ────────────────────────────────────────────────────
    def sharpe(t_list):
        if not t_list:
            return 0.0
        return _daily_sharpe(np.array([t.net_pnl_dollars for t in t_list]))

    is_filt_sharpe  = sharpe(filtered_is)
    oos_filt_sharpe = sharpe(filtered_oos)
    is_unfilt_sharpe  = sharpe(is_trades)
    oos_unfilt_sharpe = sharpe(oos_trades)

    # ── Permutation test ──────────────────────────────────────────────────────
    pvalue, perm_gains, actual_gain = _permutation_test(
        oos_trades, labels, allowed_regimes or [],
        label_names, data, n_permutations, rng
    )

    return RegimeAnalysisResult(
        breakdown              = breakdown,
        filtered_trades        = filtered_trades,
        unfiltered_oos_trades  = oos_trades,
        perm_pvalue            = pvalue,
        perm_n                 = n_permutations,
        actual_sharpe_gain     = actual_gain,
        perm_sharpe_gains      = perm_gains,
        allowed_regimes        = allowed_regimes,
        train_end_date         = train_end,
        avg_duration_days      = regime_result.avg_duration_days,
        transition_matrix      = regime_result.transition_matrix,
        is_filtered_sharpe     = is_filt_sharpe,
        oos_filtered_sharpe    = oos_filt_sharpe,
        is_unfiltered_sharpe   = is_unfilt_sharpe,
        oos_unfiltered_sharpe  = oos_unfilt_sharpe,
    )