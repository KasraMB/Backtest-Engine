"""
SMR disp=0 test + propfirm grid.
"""
import pickle, traceback
import numpy as np

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
from strategies.profitable.session_mean_rev import SessionMeanRevStrategy

account = LUCIDFLEX_ACCOUNTS['25K']

schemes         = ['fixed_dollar', 'pct_balance', 'martingale']
eval_risk_pcts  = [0.2, 0.3, 0.4, 0.5]
funded_risk_pcts= [0.5, 0.7, 0.9, 1.0]
n_sims          = 2000
n_workers       = 2

configs_to_test = [
    {
        'label': 'SMR_disp0',
        'params': {
            'allowed_sessions': ['NY'],
            'require_bos': True,
            'momentum_only': True,
            'rr_ratio': 0.5,
            'sl_atr_multiplier': 1.0,
            'wick_threshold': 0.15,
            'atr_period': 10,
            'disp_min_atr_mult': 0.0,
        }
    },
    # Also test the known best SMR config for comparison
    {
        'label': 'SMR_best',
        'params': {
            'allowed_sessions': ['NY'],
            'require_bos': True,
            'momentum_only': True,
            'rr_ratio': 0.5,
            'sl_atr_multiplier': 1.0,
            'wick_threshold': 0.15,
            'atr_period': 10,
            'disp_min_atr_mult': 2.0,
        }
    },
]

smr_results = []
for cfg_info in configs_to_test:
    label = cfg_info['label']
    params = cfg_info['params']
    try:
        cfg = RunConfig(
            starting_capital=100_000,
            slippage_points=0.25,
            commission_per_contract=4.50,
            track_equity_curve=False,
            params=params,
        )
        result = run_backtest(SessionMeanRevStrategy, cfg, data=md, validate=False)
        trades = result.trades
        n_t = len(trades)
        print(f"\n{label}: {n_t} trades")
        if n_t == 0:
            print("  0 trades, skip")
            continue
        tpd = n_t / n_days
        wins = sum(1 for t in trades if t.net_pnl_dollars > 0)
        wr = wins / n_t
        pnl_pts, sl_dists = extract_normalised_trades(trades)
        avgR_vals = pnl_pts / sl_dists
        avgR = float(np.mean(avgR_vals))
        print(f"  tpd={tpd:.3f} WR={wr:.1%} avgR={avgR:.4f}")

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
            'strategy': label,
            'rr': params.get('rr_ratio'), 'atr_m': params.get('disp_min_atr_mult'),
            'mtpd': params.get('max_trades_per_day', 3),
            'n_trades': n_t, 'tpd': tpd, 'wr': wr, 'avgR': avgR,
            'ev_per_day': best_ev,
        }
        row.update(best_cell)
        smr_results.append(row)
        print(f"  ev/day=${best_ev:.2f} ({best_cell['scheme']} erp={best_cell['erp']} frp={best_cell['frp']})")
    except Exception as e:
        print(f"  {label}: ERROR {e}")
        traceback.print_exc()

with open('tmp_smr_results.pkl', 'wb') as f:
    pickle.dump(smr_results, f)
print("\nSaved to tmp_smr_results.pkl")
