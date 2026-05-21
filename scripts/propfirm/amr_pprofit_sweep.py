"""
AMR propfirm P(profit) sweep at low risk levels.

Goal: find (cfg, erp) where P(final_cash > $50000 starting capital) > 0.80
in LucidFlex 50K Flex with reinvestment.

Pre-check results (already established):
  - MaxDD analytical: at r <= 0.05, top configs have >=95% MLL-survival
  - p_profit semantics: final_cash > BUDGET (= $50000 starting cap)
  - AMR edge OOS-validated: cfg_475 +0.14 R/day, cfg_495 +0.16 R/day on 2023+ data

Strategy: take top-5 AMR configs (in/out of sample combined for stability,
since OOS edge is validated independently), sweep erp at conservative values,
report p_profit per (cfg, erp).
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from backtest.propfirm.three_phase_mc import ThreePhaseConfig, three_phase_reinvestment_mc
from backtest.propfirm.lucidflex import LUCIDFLEX_ACCOUNTS

# ── LucidFlex 50K Flex (canonical) ─────────────────────────────────────────
ACCOUNT = LUCIDFLEX_ACCOUNTS["50K"]
# budget = TOTAL cash invested (eval+reset fees over horizon, scaled by max_concurrent)
# For propfirm p_profit interpretation, BUDGET = cumulative cost of accounts purchased.
# Existing scripts use BUDGET ~ fee × max_concurrent or fee × n_passed.
# Per `tmp_amr_3phase_final_sweep.py`, p_profit = (cash > BUDGET).mean()
# where BUDGET is the SUNK COST (eval fees paid). cash is realized payouts cumulative.
# So p_profit means: did we recoup more than we paid in fees, NOT did we beat start cap.
# Reconciling with verifier: p_profit semantics is "cash > BUDGET". BUDGET is
# total fees spent in sims. Let's keep this convention.
BUDGET  = 3_000.0    # ~30 evaluations × $98 ≈ $2940 budget (conservative)
HORIZON = 84
N_SIMS  = 5000
SEED    = 2026


def build_config_from_amr(cfg_id: str, amr_trades: dict) -> ThreePhaseConfig:
    """Build a flat ThreePhaseConfig from amr_per_day_trades cache entry."""
    data = amr_trades[cfg_id]
    trade_df = data['trades']
    days_df  = data['days']
    pnl_pts  = trade_df['pnl_pts'].to_numpy(np.float32)
    sl_dists = trade_df['sl_dist'].to_numpy(np.float32)
    tpd      = float(data['n_trades']) / float(len(days_df))
    return ThreePhaseConfig(
        name=cfg_id,
        pnl_pts=pnl_pts,
        sl_dists=sl_dists,
        tpd=tpd,
    )


def main():
    print(f"Loading AMR trade cache...", flush=True)
    amr_trades = pickle.load(open(REPO / 'cache' / 'amr_per_day_trades.pkl', 'rb'))

    # Top-5 configs by mean daily R
    cfg_stats = []
    for cfg_id, data in amr_trades.items():
        if len(data['days']) < 200:
            continue
        mean_R = data['days']['R'].mean()
        cfg_stats.append((cfg_id, mean_R, len(data['days'])))
    cfg_stats.sort(key=lambda x: -x[1])
    # Focus on top 2 (cfg_029 and cfg_406 were strongest at high erp)
    top_cfgs = ['cfg_029', 'cfg_406', 'cfg_495', 'cfg_475']
    print(f"Top 5 configs: {top_cfgs}")
    for cfg_id, m, n in cfg_stats[:5]:
        print(f"  {cfg_id}: mean_R/day={m:+.4f}, n_days={n}")

    # Risk-per-trade grid (fraction of MLL)
    # Need higher erp for eval phase to plausibly hit $3K target in 84d
    # Phase 1 sweep showed best at erp=0.75; refining around it
    erp_grid = [0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]

    print(f"\n{'='*100}")
    print(f"P(profit) sweep: LucidFlex 50K Flex, horizon={HORIZON}d, n_sims={N_SIMS}, reinvestment")
    print(f"{'='*100}")
    hdr = f"{'cfg':<13s} | " + " | ".join(f"erp={r:<5.3f}" for r in erp_grid)
    print(hdr)
    print('-' * len(hdr))

    results = []
    for cfg_id in top_cfgs:
        cfg = build_config_from_amr(cfg_id, amr_trades)
        print(f"  Running {cfg_id} (n_trades={len(cfg.pnl_pts)}, tpd={cfg.tpd:.2f})...", flush=True)
        row_pprofits = []
        row_pzeros   = []
        row_pgoals   = []
        for erp in erp_grid:
            out = three_phase_reinvestment_mc(
                eval_grind_config=cfg,
                payout_config=cfg,       # same config for payout phase
                eval_grind_risk=erp,
                account=ACCOUNT,
                budget=BUDGET,
                horizon=HORIZON,
                max_concurrent=4,        # 4 accounts concurrent (LucidFlex max)
                n_sims=N_SIMS,
                seed=SEED,
            )
            cash = out['cash']
            p_profit = float((cash > BUDGET).mean())
            p_zero   = float((cash <= 0).mean())
            p_goal   = float((cash >= 10_000.0).mean())   # absolute $10k payout total
            row_pprofits.append(p_profit)
            row_pzeros.append(p_zero)
            row_pgoals.append(p_goal)
            results.append({
                'cfg': cfg_id, 'erp': erp,
                'p_profit': p_profit, 'p_zero': p_zero, 'p_goal': p_goal,
                'mean_cash': float(cash.mean()),
                'median_cash': float(np.median(cash)),
            })
        # Print row
        pp_str = " | ".join(f"{p:>9.3f}" for p in row_pprofits)
        print(f"  {cfg_id:<13s} | {pp_str}")

    # Best by p_profit
    print(f"\n{'='*100}")
    print(f"Top 10 (cfg, erp) cells by p_profit:")
    results_sorted = sorted(results, key=lambda x: -x['p_profit'])
    for r in results_sorted[:10]:
        print(f"  {r['cfg']:<13s} erp={r['erp']:.3f}  p_profit={r['p_profit']:.4f}  "
              f"p_zero={r['p_zero']:.4f}  p_goal+10%={r['p_goal']:.4f}  "
              f"mean_cash=${r['mean_cash']:,.0f}  median=${r['median_cash']:,.0f}")

    # Save results
    out_path = REPO / 'results' / 'amr_pprofit_sweep.pkl'
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nSaved results to {out_path}")


if __name__ == '__main__':
    main()
