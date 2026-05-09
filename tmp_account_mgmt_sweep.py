"""
Account management strategy sweep — correlated reinvestment MC.

Usage:
  1. Build your StrategyConfig objects below (load pnl_pts_by_regime from
     backtest results split by HMM regime label).
  2. Build REGIME_RESULT from a fitted RegimeResult (backtest/regime/hmm.py).
  3. Run: python tmp_account_mgmt_sweep.py
  4. Results ranked by P($10K) printed to stdout and saved to
     account_mgmt_results.csv.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from backtest.propfirm.lucidflex import LUCIDFLEX_ACCOUNTS
from backtest.propfirm.correlated_mc import (
    AccountManagementStrategy,
    AccountSlot,
    StrategyConfig,
    correlated_reinvestment_mc,
)
from backtest.regime.hmm import RegimeResult

# ── CONFIG ────────────────────────────────────────────────────────────────

ACCOUNT  = LUCIDFLEX_ACCOUNTS['25K']
BUDGET   = 300.0
HORIZON  = 84
GOAL     = 10_000.0
N_SIMS   = 1_000
SEED     = 42

# ── POPULATE THESE ────────────────────────────────────────────────────────
# Build StrategyConfig from a backtest run's trade log:
#
#   from backtest.regime.hmm import fit_regimes
#   from collections import Counter
#   import numpy as np
#
#   regime_result = fit_regimes(daily_returns, daily_dates, n_states=3)
#   labels = regime_result.labels  # dict[date, int]
#
#   pnl_by_regime  = {r: [] for r in range(regime_result.n_states)}
#   sl_by_regime   = {r: [] for r in range(regime_result.n_states)}
#   days_by_regime = Counter(labels.values())
#
#   for trade in trades:
#       r = labels.get(trade.entry_date, 0)
#       pnl_by_regime[r].append(trade.pnl_pts)
#       sl_by_regime[r].append(trade.sl_dist_pts)
#
#   pnl_by_regime_arr = {
#       r: np.array(v, dtype=np.float32) if v else np.array([0.0], dtype=np.float32)
#       for r, v in pnl_by_regime.items()
#   }
#   sl_by_regime_arr = {
#       r: np.array(v, dtype=np.float32) if v else np.array([4.0], dtype=np.float32)
#       for r, v in sl_by_regime.items()
#   }
#   tpd_by_regime = {
#       r: len(pnl_by_regime[r]) / max(1, days_by_regime[r])
#       for r in range(regime_result.n_states)
#   }
#
#   cfg_A = StrategyConfig(
#       name="my_strategy_best",
#       pnl_pts_by_regime=pnl_by_regime_arr,
#       sl_dists_by_regime=sl_by_regime_arr,
#       tpd_by_regime=tpd_by_regime,
#       entry_time_min=30,
#       eval_risk=0.20,   # best ERP from lucidflex sweep
#       fund_risk=0.40,   # best FRP from lucidflex sweep
#   )

# ── PLACEHOLDER — replace with real configs ───────────────────────────────
from datetime import date, timedelta

_base = date(2020, 1, 1)
_dummy_labels = {_base + timedelta(days=i): i % 3 for i in range(252)}
_dummy_trans = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.1, 0.2, 0.7]])
_dummy_pnl = {r: np.array([5.0, -3.0, 2.0, -1.0], dtype=np.float32) for r in range(3)}
_dummy_sl  = {r: np.array([4.0,  4.0, 4.0,  4.0], dtype=np.float32) for r in range(3)}
_dummy_tpd = {0: 0.3, 1: 0.5, 2: 0.4}

cfg_A = StrategyConfig("config_A", _dummy_pnl, _dummy_sl, _dummy_tpd, 30, 0.20, 0.40)
cfg_B = StrategyConfig("config_B", _dummy_pnl, _dummy_sl, _dummy_tpd, 60, 0.15, 0.30)

from backtest.regime.hmm import RegimeResult
REGIME_RESULT = RegimeResult(
    labels=_dummy_labels,
    label_names={0: "bear", 1: "neutral", 2: "bull"},
    train_end_date=None,
    in_sample_dates=[],
    out_of_sample_dates=[],
    transition_matrix=_dummy_trans,
    state_means=np.zeros(3),
    state_stds=np.ones(3),
    avg_duration_days={"bear": 5.0, "neutral": 5.0, "bull": 5.0},
    n_states=3,
)
# ─────────────────────────────────────────────────────────────────────────

# ── SLOT TEMPLATES ────────────────────────────────────────────────────────

SLOT_TEMPLATES: dict[str, list[AccountSlot]] = {
    "single_A":    [AccountSlot([cfg_A])],
    "single_B":    [AccountSlot([cfg_B])],
    "both_A_prio": [AccountSlot([cfg_A, cfg_B])],
    "both_B_prio": [AccountSlot([cfg_B, cfg_A])],
    "alternating": [AccountSlot([cfg_A]), AccountSlot([cfg_B])],
}

# ── SWEEP GRID ────────────────────────────────────────────────────────────

TRIGGERS        = ["greedy", "on_fail", "on_pass", "on_close", "on_payout", "staggered"]
MAX_CONCURRENTS = [1, 2, 3, 4, 5]
RESERVE_EVALS   = [0, 1, 2]
STAGGER_DAYS    = [7, 14]   # only used when trigger="staggered"

# ── RUN ──────────────────────────────────────────────────────────────────

combos = [
    (tmpl_name, trigger, max_c, reserve, stagger)
    for tmpl_name in SLOT_TEMPLATES
    for trigger in TRIGGERS
    for max_c in MAX_CONCURRENTS
    for reserve in RESERVE_EVALS
    for stagger in (STAGGER_DAYS if trigger == "staggered" else [0])
]
total = len(combos)
print(f"Running {total} configs × {N_SIMS} sims each…")

rows = []
t0   = time.time()

for i, (tmpl_name, trigger, max_c, reserve, stagger) in enumerate(combos):
    strat = AccountManagementStrategy(trigger, max_c, reserve, stagger)
    cash  = correlated_reinvestment_mc(
        slot_template=SLOT_TEMPLATES[tmpl_name],
        regime_result=REGIME_RESULT,
        account=ACCOUNT,
        eval_fee=ACCOUNT.eval_fee,
        strategy=strat,
        budget=BUDGET,
        horizon=HORIZON,
        n_sims=N_SIMS,
        seed=SEED,
    )
    rows.append({
        "slot_template":   tmpl_name,
        "trigger":         trigger,
        "max_concurrent":  max_c,
        "reserve_n_evals": reserve,
        "stagger_days":    stagger,
        "p_goal":          float((cash >= GOAL).mean()),
        "median_cash":     float(np.median(cash)),
        "mean_cash":       float(np.mean(cash)),
        "p_double":        float((cash >= BUDGET * 2).mean()),
    })
    if (i + 1) % 50 == 0:
        elapsed = time.time() - t0
        rate    = (i + 1) / elapsed
        eta     = (total - i - 1) / rate
        print(f"  {i+1}/{total}  ETA {eta:.0f}s")

df = pd.DataFrame(rows).sort_values("p_goal", ascending=False)
print(f"\n-- Top 20 by P(${GOAL:,.0f}) --")
print(df.head(20).to_string(index=False))
df.to_csv("account_mgmt_results.csv", index=False)
print(f"\nSaved to account_mgmt_results.csv  ({total} rows)")
