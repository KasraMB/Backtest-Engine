# -*- coding: utf-8 -*-
"""
sweep_enhanced_orb_propfirm.py
------------------------
Sweeps Enhanced ORB strategy on NQ IS data and runs propfirm Monte Carlo grid.
Focused on maximizing propfirm EV/day, not raw PnL.
"""
from __future__ import annotations
import sys
import os

# Add project root to path so imports work when running from sweeps/scripts/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import csv, hashlib, itertools, json, os, time
from datetime import time as dtime
import numpy as np
import pandas as pd

from backtest.data.loader import DataLoader
from backtest.data.market_data import MarketData
from backtest.propfirm.lucidflex import (
    LUCIDFLEX_ACCOUNTS, extract_normalised_trades, run_propfirm_grid,
)
from backtest.runner.config import RunConfig
from backtest.runner.runner import run_backtest
from strategies.untested.enhanced_orb import EnhancedORBStrategy

DATE_FROM = "2022-01-01"
DATE_TO = "2024-12-31"
MIN_TRADES = 50
SCREEN_N_SIMS = 1000
SCREEN_SCHEMES = ["fixed_dollar", "pct_balance", "floor_aware"]
SCREEN_EVAL_RISKS = [0.20, 0.40, 0.60, 0.70]
SCREEN_FND_RISKS = [0.40, 0.60, 0.80, 1.00]

OUT_DIR = "sweep_results_enhanced_orb"
CSV_PATH = os.path.join(OUT_DIR, "results.csv")

CSV_COLS = ["combo_id", "strategy", "params_json", "n_trades", "win_rate", "avg_r",
            "trades_per_day", "best_ev_per_day", "best_account", "best_scheme",
            "best_eval_risk", "best_funded_risk", "best_pass_rate", "screen_time_s"]


def _combo_hash(strategy_name: str, params: dict) -> str:
    key = strategy_name + json.dumps(params, sort_keys=True, default=str)
    return hashlib.md5(key.encode()).hexdigest()[:12]


def _load_data() -> MarketData:
    loader = DataLoader()
    df_1m = pd.read_parquet("data/NQ_1m.parquet")
    df_5m = pd.read_parquet("data/NQ_5m.parquet")
    start = pd.Timestamp(DATE_FROM, tz="America/New_York")
    end = (pd.Timestamp(DATE_TO, tz="America/New_York")
           + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))
    m1 = df_1m[(df_1m.index >= start) & (df_1m.index <= end)]
    m5 = df_5m[(df_5m.index >= start) & (df_5m.index <= end)]
    rth = (m1.index.time >= dtime(9, 30)) & (m1.index.time <= dtime(16, 0))
    td = sorted(set(m1[rth].index.date))
    a1 = {c: m1[c].to_numpy("float64") for c in ["open","high","low","close","volume"]}
    a5 = {c: m5[c].to_numpy("float64") for c in ["open","high","low","close","volume"]}
    bm = loader._build_bar_map(m1, m5)
    data = MarketData(
        df_1m=m1, df_5m=m5,
        open_1m=a1["open"], high_1m=a1["high"],
        low_1m=a1["low"], close_1m=a1["close"],
        volume_1m=a1["volume"],
        open_5m=a5["open"], high_5m=a5["high"],
        low_5m=a5["low"], close_5m=a5["close"],
        volume_5m=a5["volume"],
        bar_map=bm, trading_dates=td,
    )
    return data


def _screen_combo(params: dict, data: MarketData) -> dict | None:
    """Run one param combo through backtest + propfirm screen."""
    t0 = time.perf_counter()

    eod = dtime(15, 0)  # Close all positions at 15:00

    config = RunConfig(
        starting_capital=100_000,
        slippage_points=0.25,
        commission_per_contract=4.50,
        eod_exit_time=eod,
        params=params,
    )

    result = run_backtest(EnhancedORBStrategy, config, data)

    if result.n_trades < MIN_TRADES:
        return None

    # Extract normalized trades for propfirm simulation
    pnl_pts, sl_dists = extract_normalised_trades(result.trades)
    trades_per_day = result.n_trades / len(data.trading_dates)

    # Find best account by EV/day across all accounts
    best_ev = -1e9
    best_record = None

    for acct_name, acct in LUCIDFLEX_ACCOUNTS.items():
        acct_size = acct_name  # Already has the account size name like "25K"

        grid = run_propfirm_grid(
            trades=None, account=acct,
            n_sims=SCREEN_N_SIMS, sizing_mode="micros",
            schemes=SCREEN_SCHEMES,
            eval_risk_pcts=SCREEN_EVAL_RISKS,
            funded_risk_pcts=SCREEN_FND_RISKS,
            _pnl_pts=pnl_pts, _sl_dists=sl_dists, _trades_per_day=trades_per_day,
        )

        # Find best combination in the grid
        for scheme in SCREEN_SCHEMES:
            for erp in SCREEN_EVAL_RISKS:
                for frp in SCREEN_FND_RISKS:
                    cell = grid.get(scheme, {}).get(erp, {}).get(frp)
                    if not isinstance(cell, dict):
                        continue
                    ev_day = cell.get("ev_per_day")
                    if ev_day is not None and ev_day > best_ev:
                        best_ev = ev_day
                        best_record = {
                            "account": acct_size,
                            "scheme": scheme,
                            "eval_risk": erp,
                            "funded_risk": frp,
                            "ev_per_day": ev_day,
                            "ev_84d": ev_day * 84,
                            "pass_rate_eval": cell.get("pass_rate", 0),
                        }

    screen_time = time.perf_counter() - t0

    return {
        "n_trades": result.n_trades,
        "win_rate": result.summary.get("win_rate", 0) if hasattr(result, "summary") else 0,
        "avg_r": result.summary.get("avg_r", 0) if hasattr(result, "summary") else 0,
        "trades_per_day": trades_per_day,
        "best": best_record,
        "screen_time": screen_time,
    }


def main():
    print(f"Loading data ({DATE_FROM} to {DATE_TO})...")
    data = _load_data()
    print(f"Loaded {len(data.df_1m):,} bars, {len(data.trading_dates)} trading days\n")

    # Parameter grid - using actual EnhancedORB params
    param_grid = {
        "or_minutes": [15, 30, 60],  # Opening range duration
        "breakout_candles": [1, 2, 3],  # Consecutive closes needed
        "sl_type": ["atr_multiple", "fixed_pct"],  # SL type
        "sl_value": [1.5, 2.0, 2.5, 3.0],  # SL value
        "tp_type": ["risk_reward", "atr_multiple"],  # TP type
        "tp_value": [1.5, 2.0],  # TP value
        "atr_length": [10, 14],  # ATR period
    }

    combos = list(itertools.product(
        param_grid["or_minutes"],
        param_grid["breakout_candles"],
        param_grid["sl_type"],
        param_grid["sl_value"],
        param_grid["tp_type"],
        param_grid["tp_value"],
        param_grid["atr_length"],
    ))

    print(f"Testing {len(combos)} Enhanced ORB combinations...\n")

    os.makedirs(OUT_DIR, exist_ok=True)

    results = []
    for or_min, bc, sl_type, sl_val, tp_type, tp_val, atr_len in combos:
        params = {
            "or_minutes": or_min,
            "breakout_candles": bc,
            "reverse_logic": False,
            "sl_type": sl_type,
            "sl_value": sl_val,
            "tp_type": tp_type,
            "tp_value": tp_val,
            "atr_length": atr_len,
            "enable_second_chance": False,
            "contracts": 1,
        }

        print(f"Testing: or_min={or_min}, bc={bc}, sl_type={sl_type}, sl_val={sl_val}, tp_type={tp_type}, tp_val={tp_val}, atr_len={atr_len}")
        screen = _screen_combo(params, data)

        if screen is None:
            print(f"  -> Skipped (n < {MIN_TRADES})")
            continue

        combo_id = _combo_hash("EnhancedORB", params)
        row = {
            "combo_id": combo_id,
            "strategy": "EnhancedORB",
            "params_json": json.dumps(params),
            "n_trades": screen["n_trades"],
            "win_rate": screen["win_rate"],
            "avg_r": screen["avg_r"],
            "trades_per_day": screen["trades_per_day"],
            "best_ev_per_day": screen["best"]["ev_per_day"] if screen["best"] else 0,
            "best_account": screen["best"]["account"] if screen["best"] else "",
            "best_scheme": screen["best"]["scheme"] if screen["best"] else "",
            "best_eval_risk": screen["best"]["eval_risk"] if screen["best"] else 0,
            "best_funded_risk": screen["best"]["funded_risk"] if screen["best"] else 0,
            "best_pass_rate": screen["best"]["pass_rate_eval"] if screen["best"] else 0,
            "screen_time_s": screen["screen_time"],
        }
        results.append(row)

        if screen["best"]:
            print(f"  -> EV/day: ${screen['best']['ev_per_day']:.2f} | "
                  f"{screen['best']['account']} {screen['best']['scheme']} "
                  f"eval_risk={screen['best']['eval_risk']} funded_risk={screen['best']['funded_risk']} "
                  f"Pass rate: {screen['best']['pass_rate_eval']*100:.0f}%")
        else:
            print(f"  -> No positive EV found")

    # Save CSV (append mode to preserve results if interrupted)
    existing_results = []
    if os.path.exists(CSV_PATH):
        try:
            with open(CSV_PATH, "r") as f:
                reader = csv.DictReader(f)
                existing_results = list(reader)
        except:
            existing_results = []

    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLS)
        writer.writeheader()
        # Write existing results first (skip duplicates by combo_id)
        existing_ids = {r["combo_id"] for r in existing_results}
        for r in existing_results:
            if r["combo_id"] not in {row["combo_id"] for row in results}:
                writer.writerow(r)
        writer.writerows(results)

    print(f"\nSaved {len(results)} results to {CSV_PATH}")

    # Summary
    df = pd.read_csv(CSV_PATH)
    if len(df) > 0:
        df = df.sort_values("best_ev_per_day", ascending=False)
        print("\n=== TOP 5 RESULTS ===")
        print(df[["combo_id", "n_trades", "best_ev_per_day", "best_account", "best_pass_rate"]].head(5).to_string())

        best = df.iloc[0]
        print(f"\n=== BEST CONFIG ===")
        print(f"EV/day: ${best['best_ev_per_day']:.2f}")
        print(f"EV/84d: ${best['best_ev_per_day']*84:.2f}")
        print(f"Account: {best['best_account']}")
        print(f"Pass rate: {best['best_pass_rate']*100:.1f}%")

        # Calculate accounts needed for $10K
        ev_84d = best['best_ev_per_day'] * 84
        accounts_needed = 10000 / ev_84d if ev_84d > 0 else float('inf')
        print(f"\nAccounts needed for $10K in 84 days: {accounts_needed:.1f}")

        # Check budget constraint (40% discount: 25K=$70, 50K=$98, 100K=$157.50, 150K=$294)
        account_costs = {"25K": 70, "50K": 98, "100K": 157.50, "150K": 294}
        best_acct = best['best_account']
        cost_per_acct = account_costs.get(best_acct, 70)
        total_cost = accounts_needed * cost_per_acct
        print(f"Cost per {best_acct} eval: ${cost_per_acct:.2f}")
        print(f"Total cost for {accounts_needed:.1f} accounts: ${total_cost:.2f}")
        if total_cost <= 300:
            print("✓ BUDGET FEASIBLE!")
        else:
            print(f"✗ Over budget by ${total_cost - 300:.2f}")


if __name__ == "__main__":
    main()
