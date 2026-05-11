"""
ORB parameter sweep + propfirm grid.
Run from project root: python tmp_orb_sweep.py
"""
import pickle, itertools, traceback
import numpy as np

# ── Load cached market data ───────────────────────────────────────────────────
with open('md_cache.pkl', 'rb') as f:
    cache = pickle.load(f)
md = cache['md']
n_days = cache['n_days']
trading_dates = cache['trading_dates']

from backtest.runner.runner import run_backtest
from backtest.runner.config import RunConfig
from backtest.propfirm.lucidflex import (
    run_propfirm_grid, LUCIDFLEX_ACCOUNTS, extract_normalised_trades
)
from strategies.profitable.orb_atr_breakout_strategy import ORBATRBreakoutStrategy

account = LUCIDFLEX_ACCOUNTS['25K']

# ── Grid params ───────────────────────────────────────────────────────────────
rr_ratios      = [1.0, 1.5, 2.0, 2.5]
atr_space_mults= [0.2, 0.3, 0.5]
max_trades_pd  = [2, 3]

schemes         = ['fixed_dollar', 'pct_balance', 'martingale']
eval_risk_pcts  = [0.2, 0.3, 0.4, 0.5]
funded_risk_pcts= [0.5, 0.7, 0.9, 1.0]
n_sims          = 2000
n_workers       = 2

base_config = RunConfig(
    starting_capital=100_000,
    slippage_points=0.25,
    commission_per_contract=4.50,
    track_equity_curve=False,
)

results = []
combos = list(itertools.product(rr_ratios, atr_space_mults, max_trades_pd))
print(f"Running {len(combos)} ORB combos...")

for ci, (rr, atr_m, mtpd) in enumerate(combos):
    params = {
        'rr_ratio': rr,
        'atr_space_multiplier': atr_m,
        'max_trades_per_day': mtpd,
        'sl_pct': 0.005,
        'use_200sma_filter': False,
        'risk_per_trade': 0.01,
        'starting_equity': 100_000,
        'equity_mode': 'dynamic',
    }
    try:
        from backtest.runner.config import RunConfig as RC
        cfg = RC(
            starting_capital=100_000,
            slippage_points=0.25,
            commission_per_contract=4.50,
            track_equity_curve=False,
            params=params,
        )
        result = run_backtest(ORBATRBreakoutStrategy, cfg, data=md, validate=False)
        trades = result.trades
        n_t = len(trades)
        if n_t == 0:
            print(f"  [{ci+1}/{len(combos)}] rr={rr} atr_m={atr_m} mtpd={mtpd}: 0 trades, skip")
            continue
        tpd = n_t / n_days
        wins = sum(1 for t in trades if t.net_pnl_dollars > 0)
        wr = wins / n_t
        pnl_pts, sl_dists = extract_normalised_trades(trades)
        avgR_vals = pnl_pts / sl_dists
        avgR = float(np.mean(avgR_vals))

        # Run propfirm grid
        grid = run_propfirm_grid(
            trades=trades,
            account=account,
            n_sims=n_sims,
            schemes=schemes,
            eval_risk_pcts=eval_risk_pcts,
            funded_risk_pcts=funded_risk_pcts,
            n_workers=n_workers,
            n_trading_days=n_days,
            seed=42,
        )

        # Find best ev/day cell
        best_ev = -1e9
        best_cell = None
        for scheme_name, scheme_data in grid.items():
            if not isinstance(scheme_data, dict):
                continue
            for erp_key, erp_data in scheme_data.items():
                if not isinstance(erp_data, dict):
                    continue
                for frp_key, cell in erp_data.items():
                    if not isinstance(cell, dict):
                        continue
                    ev = cell.get('ev_per_day', -1e9)
                    if ev > best_ev:
                        best_ev = ev
                        payout_days = cell.get('median_payout_days') or [None]
                        best_cell = {
                            'scheme': scheme_name,
                            'erp': erp_key,
                            'frp': frp_key,
                            'pass_rate': cell.get('pass_rate', 0),
                            'median_days_to_pass': cell.get('median_days_to_pass', 0),
                            'median_payout_days_0': payout_days[0] if payout_days else None,
                            'mean_withdrawal': cell.get('mean_withdrawal', 0),
                        }

        row = {
            'strategy': 'ORB',
            'rr': rr, 'atr_m': atr_m, 'mtpd': mtpd,
            'n_trades': n_t, 'tpd': tpd, 'wr': wr, 'avgR': avgR,
            'ev_per_day': best_ev,
        }
        row.update(best_cell)
        results.append(row)
        print(f"  [{ci+1}/{len(combos)}] rr={rr} atr_m={atr_m} mtpd={mtpd}: "
              f"n={n_t} tpd={tpd:.3f} WR={wr:.1%} avgR={avgR:.3f} ev/day=${best_ev:.2f} "
              f"({best_cell['scheme']} erp={best_cell['erp']} frp={best_cell['frp']})")
    except Exception as e:
        print(f"  [{ci+1}/{len(combos)}] rr={rr} atr_m={atr_m} mtpd={mtpd}: ERROR {e}")
        traceback.print_exc()

print(f"\nDone ORB sweep. {len(results)} results.")
with open('tmp_orb_results.pkl', 'wb') as f:
    pickle.dump(results, f)
print("Saved to tmp_orb_results.pkl")
