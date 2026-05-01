# -*- coding: utf-8 -*-
"""
sweep3_structural.py
--------------------
Conservative sweep: fix the best IS params from sweep2, vary structural params.
Goal: find BE/volatility filter combos that push IS EV/day toward $25-30.

Fixed base: NY, BOS=True, momentum_only=True, disp=2.0
            rr=1.0, sl_mult=1.25, wick=0.15, atr=10  (sweep2 rank1, IS $16.30)

Grid:
  breakeven_r      : [0.0, 0.3, 0.5, 0.75, 1.0]
  atr_vol_filter   : [0.0, 0.85, 1.0, 1.15]
  require_daily_mom: [False, True]
  Total: 5 × 4 × 2 = 40 combos
"""
from __future__ import annotations

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import csv, hashlib, itertools, json, os, pickle, time
from datetime import time as dtime

import numpy as np
import pandas as pd

from backtest.data.loader import DataLoader
from backtest.data.market_data import MarketData
from backtest.propfirm.lucidflex import (
    LUCIDFLEX_ACCOUNTS, _estimate_trading_days,
    extract_normalised_trades, run_propfirm_grid,
)
from backtest.runner.config import RunConfig
from backtest.runner.runner import run_backtest
from strategies.session_mean_rev import SessionMeanRevStrategy

DATE_FROM = "2019-01-01"
DATE_TO   = "2024-12-31"

FIXED_BASE = {
    "allowed_sessions": ["NY"], "require_bos": True, "momentum_only": True,
    "disp_min_atr_mult": 2.0, "rr_ratio": 1.0, "sl_atr_multiplier": 1.25,
    "wick_threshold": 0.15, "atr_period": 10,
    "risk_per_trade": 0.01, "equity_mode": "dynamic",
    "starting_equity": 100_000, "point_value": 20.0, "max_trades_per_day": 3,
}

GRID = {
    "breakeven_r":            [0.0, 0.3, 0.5, 0.75, 1.0],
    "atr_vol_filter":         [0.0, 0.85, 1.0, 1.15],
    "require_daily_momentum": [False, True],
}

SCREEN_ACCOUNTS = ["50K", "100K", "150K"]
SCREEN_SCHEMES  = ["fixed_dollar", "pct_balance", "floor_aware"]
SCREEN_EVAL_RISKS = [0.20, 0.40, 0.60]
SCREEN_FND_RISKS  = [0.20, 0.40, 0.60]
SCREEN_N_SIMS  = 1_000
FULL_N_SIMS    = 5_000
MIN_TRADES     = 50
TOP_N_FULL     = 10

OUT_DIR  = "sweep_results3"
PKL_DIR  = os.path.join(OUT_DIR, "pkl")
CSV_PATH = os.path.join(OUT_DIR, "results.csv")
SUMM_PATH= os.path.join(OUT_DIR, "summary.txt")

CSV_COLS = ["combo_id","breakeven_r","atr_vol_filter","require_daily_momentum",
            "n_trades","win_rate","avg_r","trades_per_day",
            "best_ev_per_day","best_account","best_scheme",
            "best_eval_risk","best_funded_risk","best_pass_rate","screen_time_s"]


def _combo_hash(params: dict) -> str:
    key = json.dumps({k: params[k] for k in sorted(GRID.keys())}, sort_keys=True)
    return hashlib.md5(key.encode()).hexdigest()[:12]


def _load_data() -> MarketData:
    loader = DataLoader()
    df_1m = pd.read_parquet("data/NQ_1m.parquet")
    df_5m = pd.read_parquet("data/NQ_5m.parquet")
    start = pd.Timestamp(DATE_FROM, tz="America/New_York")
    end   = (pd.Timestamp(DATE_TO, tz="America/New_York")
             + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))
    m1 = df_1m[(df_1m.index >= start) & (df_1m.index <= end)]
    m5 = df_5m[(df_5m.index >= start) & (df_5m.index <= end)]
    rth = (m1.index.time >= dtime(9,30)) & (m1.index.time <= dtime(16,0))
    trading_dates = sorted(set(m1[rth].index.date))
    a1 = {c: m1[c].to_numpy("float64") for c in ["open","high","low","close","volume"]}
    a5 = {c: m5[c].to_numpy("float64") for c in ["open","high","low","close","volume"]}
    bm = loader._build_bar_map(m1, m5)
    return MarketData(df_1m=m1, df_5m=m5,
                      open_1m=a1["open"], high_1m=a1["high"], low_1m=a1["low"],
                      close_1m=a1["close"], volume_1m=a1["volume"],
                      open_5m=a5["open"], high_5m=a5["high"], low_5m=a5["low"],
                      close_5m=a5["close"], volume_5m=a5["volume"],
                      bar_map=bm, trading_dates=trading_dates)


def _screen(pnl_pts, sl_dists, tpd):
    best_ev, best_info = -np.inf, {}
    for acc_name in SCREEN_ACCOUNTS:
        grid = run_propfirm_grid(
            trades=None, account=LUCIDFLEX_ACCOUNTS[acc_name],
            n_sims=SCREEN_N_SIMS, sizing_mode="micros",
            schemes=SCREEN_SCHEMES,
            eval_risk_pcts=SCREEN_EVAL_RISKS, funded_risk_pcts=SCREEN_FND_RISKS,
            _pnl_pts=pnl_pts, _sl_dists=sl_dists, _trades_per_day=tpd,
        )
        for scheme in SCREEN_SCHEMES:
            for erp in SCREEN_EVAL_RISKS:
                for frp in SCREEN_FND_RISKS:
                    cell = grid.get(scheme, {}).get(erp, {}).get(frp)
                    if not isinstance(cell, dict):
                        continue
                    ev = cell.get("ev_per_day")
                    if ev is not None and ev > best_ev:
                        best_ev = ev
                        best_info = {
                            "best_ev_per_day":  round(ev, 4),
                            "best_account":     acc_name,
                            "best_scheme":      scheme,
                            "best_eval_risk":   erp,
                            "best_funded_risk": frp,
                            "best_pass_rate":   round(cell.get("pass_rate", 0.0), 4),
                        }
    if not best_info:
        best_info = {k: None for k in ["best_ev_per_day","best_account","best_scheme",
                                        "best_eval_risk","best_funded_risk","best_pass_rate"]}
    return best_info


def _full_grid(pnl_pts, sl_dists, tpd):
    results = {}
    for acc_name, account in LUCIDFLEX_ACCOUNTS.items():
        results[acc_name] = run_propfirm_grid(
            trades=None, account=account, n_sims=FULL_N_SIMS, sizing_mode="micros",
            _pnl_pts=pnl_pts, _sl_dists=sl_dists, _trades_per_day=tpd,
        )
    return results


def _load_existing(csv_path):
    if not os.path.exists(csv_path):
        return set()
    with open(csv_path, newline="") as f:
        return {r["combo_id"] for r in csv.DictReader(f)}


def main():
    os.makedirs(OUT_DIR, exist_ok=True); os.makedirs(PKL_DIR, exist_ok=True)

    print(f"Loading data {DATE_FROM} -> {DATE_TO}...", flush=True)
    t0 = time.perf_counter()
    data = _load_data()
    print(f"  {len(data.df_1m):,} bars | {len(data.trading_dates):,} days  ({time.perf_counter()-t0:.1f}s)\n")

    keys   = list(GRID.keys())
    combos = [dict(zip(keys, v)) for v in itertools.product(*[GRID[k] for k in keys])]
    total  = len(combos)
    done_ids = _load_existing(CSV_PATH)
    print(f"Total combos: {total}  |  Already done: {len(done_ids)}\n")

    write_header = not os.path.exists(CSV_PATH) or not done_ids
    csv_file = open(CSV_PATH, "a", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=CSV_COLS)
    if write_header:
        writer.writeheader()

    n_proc = 0; t_total = 0.0
    cfg = RunConfig(starting_capital=100_000, slippage_points=0.25,
                    commission_per_contract=4.50, eod_exit_time=dtime(23,59), params={})

    for idx, grid_params in enumerate(combos, start=1):
        params = {**FIXED_BASE, **grid_params}
        cid = _combo_hash(params)
        if cid in done_ids:
            continue

        print(f"[{idx:3d}/{total}] "
              f"be={grid_params['breakeven_r']:.2f}  "
              f"vol={grid_params['atr_vol_filter']:.2f}  "
              f"dm={'Y' if grid_params['require_daily_momentum'] else 'N'}",
              end="  ", flush=True)

        cfg.params = params
        t_bt = time.perf_counter()
        try:
            result = run_backtest(SessionMeanRevStrategy, cfg, data)
        except Exception as e:
            print(f"ERROR: {e}"); continue
        bt_time = time.perf_counter() - t_bt
        n_trades = len(result.trades)

        if n_trades < MIN_TRADES:
            row = {"combo_id": cid, **{k: grid_params[k] for k in keys},
                   "n_trades": n_trades, **{k: None for k in CSV_COLS[5:-1]},
                   "screen_time_s": round(bt_time, 2)}
            writer.writerow(row); csv_file.flush(); done_ids.add(cid)
            n_proc += 1; t_total += bt_time
            print(f"SKIP ({n_trades} trades < {MIN_TRADES})  [{bt_time:.1f}s]")
            remaining = total - idx
            if n_proc > 0 and remaining > 0:
                print(f"  ETA: {(t_total/n_proc)*remaining/60:.1f}m  ({n_proc} done, {remaining} left)")
            continue

        wins = sum(1 for t in result.trades if t.net_pnl_dollars > 0)
        wr   = wins / n_trades
        pnl_pts, sl_dists = extract_normalised_trades(result.trades)
        avg_r = float((pnl_pts / np.where(sl_dists > 0, sl_dists, 1.0)).mean())
        tpd   = max(0.5, n_trades / max(1, _estimate_trading_days(result.trades)))

        t_pf = time.perf_counter()
        screen = _screen(pnl_pts, sl_dists, tpd)
        pf_time = time.perf_counter() - t_pf

        ev = screen.get("best_ev_per_day")
        pr = screen.get("best_pass_rate")
        print(f"n={n_trades:4d} WR={wr:.1%} avgR={avg_r:+.3f} "
              f"ev/d={f'${ev:.2f}' if ev else 'N/A':>9} "
              f"pass={f'{pr:.0%}' if pr else 'N/A':>5}  [{bt_time:.1f}s + {pf_time:.1f}s]")

        row = {"combo_id": cid, **{k: grid_params[k] for k in keys},
               "n_trades": n_trades, "win_rate": round(wr, 4), "avg_r": round(avg_r, 4),
               "trades_per_day": round(tpd, 4), **screen,
               "screen_time_s": round(bt_time + pf_time, 2)}
        writer.writerow(row); csv_file.flush(); done_ids.add(cid)
        n_proc += 1; t_total += bt_time + pf_time
        remaining = total - idx
        if n_proc > 0 and remaining > 0:
            print(f"  ETA: {(t_total/n_proc)*remaining/60:.1f}m  ({n_proc} done, {remaining} left)")

    csv_file.close()

    # Rank and summarise
    print("\n" + "="*80)
    df = pd.read_csv(CSV_PATH)
    df_v = (df.dropna(subset=["best_ev_per_day"])
              .sort_values("best_ev_per_day", ascending=False)
              .reset_index(drop=True))
    print(f"Valid: {len(df_v)}/{len(df)}  |  Positive EV: {(df_v['best_ev_per_day'] > 0).sum()}")
    top = df_v.head(20).copy(); top.index = range(1, len(top)+1)
    pd.set_option("display.width", 200); pd.set_option("display.max_colwidth", 20)
    print("\nTOP 20:\n" + top[["breakeven_r","atr_vol_filter","require_daily_momentum",
                                "n_trades","win_rate","avg_r",
                                "best_ev_per_day","best_account","best_pass_rate"]].to_string())
    with open(SUMM_PATH, "w") as f:
        f.write(f"Sweep3 (structural): {DATE_FROM} -> {DATE_TO}\n")
        f.write(f"Fixed: {json.dumps(FIXED_BASE)}\n")
        f.write(f"Valid: {len(df_v)}/{len(df)}  Positive EV: {(df_v['best_ev_per_day']>0).sum()}\n\n")
        f.write("TOP 20:\n" + top[["breakeven_r","atr_vol_filter","require_daily_momentum",
                                    "n_trades","win_rate","avg_r",
                                    "best_ev_per_day","best_account","best_pass_rate"]].to_string())

    # Full grid on top-N
    print(f"\nRunning full propfirm grid (n_sims={FULL_N_SIMS}) on top {TOP_N_FULL}...")
    combo_lookup = {_combo_hash({**FIXED_BASE, **dict(zip(keys, v))}): dict(zip(keys, v))
                    for v in itertools.product(*[GRID[k] for k in keys])}
    for rank, (_, row) in enumerate(df_v.head(TOP_N_FULL).iterrows(), start=1):
        cid = row["combo_id"]
        gp  = combo_lookup.get(cid)
        if gp is None:
            continue
        pkl_path = os.path.join(PKL_DIR, f"rank{rank:02d}_{cid}.pkl")
        if os.path.exists(pkl_path):
            print(f"  [{rank:2d}] already saved, skipping"); continue
        print(f"  [{rank:2d}] be={gp['breakeven_r']:.2f} vol={gp['atr_vol_filter']:.2f} "
              f"dm={'Y' if gp['require_daily_momentum'] else 'N'}  "
              f"(screen=${row['best_ev_per_day']:.2f})", end="  ", flush=True)
        params = {**FIXED_BASE, **gp}
        cfg.params = params
        t0 = time.perf_counter()
        result = run_backtest(SessionMeanRevStrategy, cfg, data, validate=False)
        pnl_pts, sl_dists = extract_normalised_trades(result.trades)
        tpd = max(0.5, len(result.trades) / max(1, _estimate_trading_days(result.trades)))
        full = _full_grid(pnl_pts, sl_dists, tpd)
        best_ev, best_info = -np.inf, {}
        for acc_name, acc_grid in full.items():
            for scheme in acc_grid:
                if scheme == "optimal_funded_rp":
                    continue
                for erp, erp_data in acc_grid[scheme].items():
                    if not isinstance(erp_data, dict):
                        continue
                    for frp, cell in erp_data.items():
                        if not isinstance(cell, dict):
                            continue
                        ev = cell.get("ev_per_day")
                        if ev is not None and ev > best_ev:
                            best_ev = ev
                            best_info = {"account": acc_name, "scheme": scheme,
                                         "eval_risk": erp, "funded_risk": frp,
                                         "ev_per_day": ev, "pass_rate": cell.get("pass_rate"),
                                         "net_ev": cell.get("net_ev"),
                                         "total_cost": cell.get("total_cost"),
                                         "roi": cell.get("roi")}
        with open(pkl_path, "wb") as f:
            pickle.dump({"rank": rank, "combo_id": cid, "params": params,
                         "n_trades": len(result.trades), "full_ev_per_day": best_ev,
                         "best_info": best_info, "full_grid": full}, f)
        print(f"full_ev=${best_ev:.2f}  [{time.perf_counter()-t0:.1f}s]")

    print("\nDone. Results in", OUT_DIR)


if __name__ == "__main__":
    main()
