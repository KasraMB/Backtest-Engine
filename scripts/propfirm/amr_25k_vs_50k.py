"""
Test AMR on 25K Flex vs 50K Flex at $1K budget.

User-confirmed pricing:
- 25K Flex: $70 eval, $60 reset; SB=$25K, target=$1250, MLL=$1000
- 50K Flex: $98 eval, $95 reset; SB=$50K, target=$3000, MLL=$2000

25K hypothesis:
- Cheaper fees → more attempts ($1K/$70 = 14 attempts vs $1K/$98 = 10)
- Smaller target ($1250 vs $3K) → faster eval pass
- BUT smaller MLL buffer ($1000 vs $2000) → easier blow

Test both and report p_profit + cash Sharpe (max risk-adjusted returns).
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from backtest.propfirm.three_phase_mc import ThreePhaseConfig, three_phase_reinvestment_mc
from backtest.propfirm.lucidflex import LUCIDFLEX_ACCOUNTS, LucidFlexAccount

# Override sim's eval_fee for 25K test (sim defaults to 98). Need to wrap or pass.
# Actually `account.eval_fee` is read; verify.
HORIZON = 84
N_SIMS  = 20_000
SEEDS   = [2026, 2027, 2028]
BUDGET  = 1000.0
MIN_PAYOUT = 500.0

# Reset deltas per real LucidFlex pricing
RESET_DELTAS = {
    '25K': -10.0,   # $60 reset - $70 eval = -$10
    '50K': -3.0,    # $95 reset - $98 eval = -$3
}


def build_oos(cfg_id, amr):
    data = amr[cfg_id]
    td = data['trades'].copy()
    td['date'] = pd.to_datetime(td['date'])
    dd = data['days'].copy()
    dd['date'] = pd.to_datetime(dd['date'])
    oos_t = td[td['date'] >= '2023-01-01']
    oos_d = dd[dd['date'] >= '2023-01-01']
    return ThreePhaseConfig(
        name=f"{cfg_id}_OOS",
        pnl_pts=oos_t['pnl_pts'].to_numpy(np.float32),
        sl_dists=oos_t['sl_dist'].to_numpy(np.float32),
        tpd=float(len(oos_t)) / float(len(oos_d)),
    )


def evaluate(cfg, erp, mc, account_size, seed):
    account = LUCIDFLEX_ACCOUNTS[account_size]
    eval_fee = account.eval_fee
    out = three_phase_reinvestment_mc(
        eval_grind_config=cfg, payout_config=cfg,
        eval_grind_risk=erp, account=account,
        budget=BUDGET, horizon=HORIZON,
        max_concurrent=mc, n_sims=N_SIMS, seed=seed,
        enforce_consistency_rule=True,
    )
    cash = out['cash'].astype(np.float64)
    n_evals = out['eval_fees_paid'] / eval_fee
    reopens = np.maximum(n_evals - mc, 0.0)
    cash_adj = cash - RESET_DELTAS[account_size] * reopens

    # Metrics
    p_profit = float((cash_adj >= BUDGET + MIN_PAYOUT).mean())
    p_zero = float((cash_adj <= 0).mean())
    # Risk-adjusted return on cash (Sharpe of cash_growth)
    cash_growth = cash_adj - BUDGET
    sharpe = float(cash_growth.mean() / max(cash_growth.std(), 1e-9))
    # Sortino: downside std only
    downside = cash_growth[cash_growth < 0]
    sortino = float(cash_growth.mean() / max(downside.std() if len(downside) else 1.0, 1e-9))
    return {
        'p_profit': p_profit,
        'p_zero':   p_zero,
        'mean_cash_growth': float(cash_growth.mean()),
        'median_cash_growth': float(np.median(cash_growth)),
        'sharpe': sharpe,
        'sortino': sortino,
    }


def main():
    amr = pickle.load(open(REPO / 'cache' / 'amr_per_day_trades.pkl', 'rb'))

    cfg_406 = build_oos('cfg_406', amr)
    cfg_029 = build_oos('cfg_029', amr)

    print(f"\n{'='*120}")
    print(f"AMR on 25K vs 50K Flex at BUDGET=$1K, OOS 2023+, n_sims={N_SIMS}, 3 seeds")
    print(f"User pricing: 25K=$70 eval/$60 reset; 50K=$98 eval/$95 reset")
    print(f"Metric: p_profit (cash >= $1500) | Sharpe(growth) | Sortino(growth) | mean_growth | median_growth")
    print(f"{'='*120}")

    for cfg, name in [(cfg_406, 'cfg_406_OOS'), (cfg_029, 'cfg_029_OOS')]:
        print(f"\n--- {name} ---")
        print(f"{'Account':<7s} | {'mc':<4s} | {'erp':<5s} | {'p_profit':<10s} | {'Sharpe':<8s} | {'Sortino':<8s} | {'mean_$':<10s} | {'median_$':<10s}")
        print('-' * 100)
        for acct in ['25K', '50K']:
            for mc in [1, 2, 3]:
                for erp in [0.5, 0.7, 0.9]:
                    results = [evaluate(cfg, erp, mc, acct, s) for s in SEEDS]
                    mean_pp = np.mean([r['p_profit'] for r in results])
                    mean_sharpe = np.mean([r['sharpe'] for r in results])
                    mean_sortino = np.mean([r['sortino'] for r in results])
                    mean_g = np.mean([r['mean_cash_growth'] for r in results])
                    median_g = np.mean([r['median_cash_growth'] for r in results])
                    tag = " <-PASS" if mean_pp > 0.80 else ""
                    print(f"{acct:<7s} | {mc:<4d} | {erp:<5.2f} | {mean_pp:.4f}{tag:<6s} | "
                          f"{mean_sharpe:>+7.3f} | {mean_sortino:>+7.3f} | "
                          f"${mean_g:>+8,.0f} | ${median_g:>+8,.0f}")


if __name__ == '__main__':
    main()
