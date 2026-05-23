"""Phase 3: apply the pre-registered inclusion filter to Phase 2 cell results.

Reads cache/portfolio/cell_results.pkl, runs inclusion_filter on every cell, and
reports the survivors. Also prints the ICT sl_dist scrutiny (pre-reg v3) and the
landscape of cells rejected only for the proxy-SL (invalid-R) reason.

Pure reporting — makes NO selection decision beyond the locked filter.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from backtest.portfolio.cell import inclusion_filter

CACHE = REPO / 'cache' / 'portfolio'


def main():
    results = pickle.load(open(CACHE / 'cell_results.pkl', 'rb'))
    rows = []
    for name, r in results.items():
        passed, reason = inclusion_filter(r)
        rows.append({
            'cell': name, 'instrument': r.cell.instrument,
            'n': r.n_trades, 'mean_R': round(r.mean_R, 4),
            'yrs_pos': r.years_positive, 'yrs_tot': len(r.by_year),
            'proxy_sl': round(r.proxy_sl_fraction, 3),
            'min_sl': round(r.min_sl_dist, 3),
            'p5': round(r.bootstrap_p5_mean_R, 4) if r.n_trades >= 30 else None,
            'pass': passed, 'reason': reason,
        })
    df = pd.DataFrame(rows).sort_values(['pass', 'mean_R'], ascending=[False, False])

    survivors = df[df['pass']]
    print(f"=== PHASE 3 INCLUSION FILTER — {len(results)} cells ===")
    print(f"\nSURVIVORS: {len(survivors)}")
    if len(survivors):
        print(survivors[['cell', 'n', 'mean_R', 'yrs_pos', 'proxy_sl', 'min_sl', 'p5']].to_string(index=False))

    # Rejected ONLY because of proxy-SL (invalid R) — for transparency
    proxy_rej = df[(~df['pass']) & (df['proxy_sl'] > 0)]
    print(f"\nREJECTED — no real stop (proxy-SL, invalid R): {len(proxy_rej)}")
    if len(proxy_rej):
        print(proxy_rej[['cell', 'n', 'mean_R', 'proxy_sl']].to_string(index=False))

    # ICT scrutiny (pre-reg v3)
    ict = [r for n, r in results.items() if 'ICTSMC' in n]
    if ict:
        r = ict[0]
        sld = r.trades['sl_dist'].to_numpy(np.float64)
        print(f"\n=== ICT scrutiny: {r.cell.name} mean_R={r.mean_R:.3f} ===")
        print(f"  sl_dist: min={sld.min():.3f} p5={np.percentile(sld,5):.3f} "
              f"median={np.median(sld):.3f} | frac at 0.25 floor={np.mean(np.abs(sld-0.25)<1e-6):.3f}")

    df.to_csv(REPO / 'results' / 'portfolio_phase3_filter.csv', index=False)
    print(f"\nFull table -> results/portfolio_phase3_filter.csv")
    print(f"\nDeploy gate (pre-reg): need >= 3 survivors, else NULL. -> "
          f"{'PROCEED' if len(survivors) >= 3 else 'NULL / thin'}")


if __name__ == '__main__':
    main()
