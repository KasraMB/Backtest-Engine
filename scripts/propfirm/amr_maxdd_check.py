"""
Analytical MaxDD feasibility check for AMR + LucidFlex 50K propfirm goal.

LucidFlex 50K trailing MLL = $2000 = 4% of starting balance.
For per-trade-risk r (fraction of MLL = $2000):
  - 4% drawdown allowance = MLL / start_capital = $2000 / $50000 = 0.04
  - Per-trade risk in dollars = r * $2000 (since r is fraction of MLL_AMT)
  - In R-units: MaxDD_R × r * $2000 must stay < $2000 from peak
  - Therefore: MaxDD_R < 1/r  (e.g., r=0.5 -> MaxDD < 2R; r=1.0 -> MaxDD < 1R)

Wait — that's not right either. Let me redo:
  - Account starts $50000. MLL_AMT = $2000. So trail = $48000 initially.
  - Per trade r = risk fraction OF the trailing buffer (or fixed-fractional-of-balance?)
  - The propfirm code in three_phase_mc uses `risk_dollars = MLL_AMT * eval_grind_risk_pct`
    so eval_risk_pct = 0.5 -> $1000 risk per trade.
  - At r = 0.5 (50% of MLL), $1000 risk per trade. R-units: drawdown_dollars = MaxDD_R × $1000.
  - Trailing MLL = $2000 from peak.
  - So MaxDD_R × $1000 must stay < $2000 -> MaxDD_R < 2.0 R from peak.
  - At r = 1.0 (100% of MLL = $2000 risk per trade): MaxDD_R < 1.0 R (trivially broken on any losing day).
  - At r = 0.25 (25% of MLL = $500 risk per trade): MaxDD_R < 4.0 R from peak.

To survive: P(MaxDD_R-from-peak over 84 days < threshold) must be ≥ 0.80.

This script:
1. Loads AMR's cached per-day R arrays for the top configs
2. For each config: bootstrap 84-day sub-sequences (block-bootstrap, block=5 days for serial correlation)
3. Compute max running drawdown in R from each sequence's peak
4. Per (config, r): fraction of bootstrap paths surviving MLL
"""
from __future__ import annotations

import pickle
import numpy as np
import pandas as pd
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# LucidFlex 50K Flex constants
START_BAL = 50_000.0
MLL_AMT   = 2_000.0           # 4% trailing drawdown
SIM_DAYS  = 84                # propfirm period
N_BOOT    = 5000              # bootstrap paths per config × r cell
BLOCK_LEN = 5                 # block bootstrap (serial correlation)
SEED      = 2026


def stationary_block_bootstrap(R_days: np.ndarray, sim_days: int, n_paths: int,
                                 block_len: int, seed: int) -> np.ndarray:
    """Stationary block bootstrap: sample sim_days × n_paths paths from R_days.

    Returns (n_paths, sim_days) array of R-per-day.
    """
    n_src = len(R_days)
    rng = np.random.default_rng(seed)
    # Stationary bootstrap: at each step, with prob 1/block_len start a new block
    out = np.empty((n_paths, sim_days), dtype=np.float64)
    # Vectorise: build start indices for each block-start, copy block contents
    for p in range(n_paths):
        i = 0
        while i < sim_days:
            start = rng.integers(0, n_src)
            block = rng.geometric(1.0 / block_len)  # random block length
            end = min(i + block, sim_days)
            for k in range(end - i):
                out[p, i + k] = R_days[(start + k) % n_src]
            i = end
    return out


def max_drawdown_from_peak(paths_R: np.ndarray, risk_per_trade_R_to_dollars: float) -> np.ndarray:
    """For each (n_paths, sim_days) path of per-day R, compute the maximum
    drawdown from running peak in DOLLARS, given risk_per_trade_R_to_dollars
    is the dollar value of 1 R (= eval_risk_pct * MLL_AMT).

    Returns (n_paths,) max-DD-from-peak in dollars.
    """
    pnl_per_day_dollars = paths_R * risk_per_trade_R_to_dollars
    cum = np.cumsum(pnl_per_day_dollars, axis=1)               # cumulative dollar PnL
    running_peak = np.maximum.accumulate(cum, axis=1)
    drawdown_from_peak = running_peak - cum                    # always >= 0
    return drawdown_from_peak.max(axis=1)                       # (n_paths,)


def main():
    trades = pickle.load(open(REPO / 'cache' / 'amr_per_day_trades.pkl', 'rb'))

    # Sort configs by mean daily R
    cfg_stats = []
    for cfg_id, data in trades.items():
        R_days = data['days']['R'].to_numpy()
        if len(R_days) < 100:
            continue
        cfg_stats.append((cfg_id, R_days.mean(), R_days.std(), len(R_days), R_days))
    cfg_stats.sort(key=lambda x: -x[1])

    print(f"Top 5 AMR configs by mean daily R:")
    for cfg, m, s, n, _ in cfg_stats[:5]:
        print(f"  {cfg}: mean={m:+.4f} std={s:.3f} n_days={n}")

    # Risk-per-trade-pct grid (fraction of MLL_AMT)
    r_grid = [0.005, 0.01, 0.02, 0.03, 0.05, 0.10, 0.15, 0.25, 0.50, 1.00]

    print(f"\n{'='*88}")
    print(f"P(MaxDD from peak < MLL=$2000) over 84-day block-bootstrap sims (n={N_BOOT}/cell)")
    print(f"Per-trade-risk r expressed as fraction of MLL_AMT ($2000)")
    print(f"At r=0.005: $10 risk per R -> MaxDD must stay < 200 R to survive")
    print(f"At r=0.10:  $200 risk per R -> MaxDD must stay < 10 R to survive")
    print(f"At r=1.00:  $2000 risk per R -> MaxDD must stay < 1 R (impossible)")
    print(f"{'='*88}")

    # Header
    hdr = f"{'cfg':<13s} | {'mean_R':>7s} | {'std':>5s} | " + " ".join(f"{r:>5.3f}" for r in r_grid)
    print(hdr)
    print('-' * len(hdr))

    rng_master = np.random.default_rng(SEED)
    for cfg, m, s, n_days, R_days in cfg_stats[:8]:   # top 8 configs
        # Generate one set of 5000 bootstrap paths
        boot_seed = int(rng_master.integers(0, 2**31))
        paths = stationary_block_bootstrap(R_days, SIM_DAYS, N_BOOT, BLOCK_LEN, boot_seed)
        survival_per_r = []
        for r in r_grid:
            risk_per_R_dollars = r * MLL_AMT       # e.g. r=0.01 -> $20 per R
            maxdd_dollars = max_drawdown_from_peak(paths, risk_per_R_dollars)
            p_survive = float((maxdd_dollars < MLL_AMT).mean())
            survival_per_r.append(p_survive)
        row = f"{cfg:<13s} | {m:+.4f} | {s:.3f} | " + " ".join(f"{p:>5.2f}" for p in survival_per_r)
        print(row)

    print(f"\nInterpretation:")
    print(f"  cells >= 0.80 -> potentially propfirm-viable at that r")
    print(f"  cells <  0.50 -> guaranteed >50% blow-up rate -> drop")


if __name__ == '__main__':
    main()
