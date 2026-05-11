# -*- coding: utf-8 -*-
"""
sweep6_orb_vwap.py
------------------
Sweeps ORBATRBreakoutStrategy and VWAPMeanRevStrategy on NQ 2019-2024 IS data.
Results saved to sweep_results6/.
"""
from __future__ import annotations
import sys
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

from strategies.orb_atr_breakout_strategy import ORBATRBreakoutStrategy
from strategies.vwap_mean_rev_strategy import VWAPMeanRevStrategy

DATE_FROM = "2019-01-01"
DATE_TO   = "2024-12-31"
MIN_TRADES = 30
SCREEN_N_SIMS = 500
SCREEN_ACCOUNTS = ["25K", "50K", "100K", "150K"]
SCREEN_SCHEMES = ["fixed_dollar", "pct_balance", "martingale", "floor_aware"]
SCREEN_EVAL_RISKS = [0.20, 0.40, 0.60, 0.70]
SCREEN_FND_RISKS  = [0.40, 0.60, 0.80, 1.00]

OUT_DIR   = "sweep_results6"
CSV_PATH  = os.path.join(OUT_DIR, "results.csv")
SUMM_PATH = os.path.join(OUT_DIR, "summary.txt")

CSV_COLS = ["combo_id", "strategy", "params_json", "n_trades", "win_rate", "avg_r",
            "trades_per_day", "best_ev_per_day", "best_account", "best_scheme",
            "best_eval_risk", "best_funded_risk", "best_pass_rate", "screen_time_s"]

STRATEGIES = [
    {
        "name": "ORBATRBreakout",
        "cls":  ORBATRBreakoutStrategy,
        "fixed": {
            "risk_per_trade":  0.01,
            "starting_equity": 100_000,
            "equity_mode":     "dynamic",
        },
        "grid": {
            "atr_space_multiplier": [0.15, 0.20, 0.25, 0.30, 0.40, 0.50],
            "sl_pct":               [0.003, 0.005, 0.007, 0.010],
            "rr_ratio":             [1.0, 1.5, 2.0, 2.5],
            "use_200sma_filter":    [False, True],
            "max_trades_per_day":   [1, 2],
        },
    },
    {
        "name": "VWAPMeanRev",
        "cls":  VWAPMeanRevStrategy,
        "fixed": {
            "risk_per_trade":  0.01,
            "starting_equity": 100_000,
            "equity_mode":     "dynamic",
            "atr_period":      14,
        },
        "grid": {
            "vwap_deviation_atr":  [1.0, 1.5, 2.0, 2.5, 3.0],
            "sl_atr_multiplier":   [0.5, 1.0, 1.5],
            "rr_ratio":            [0.75, 1.0, 1.5, 2.0],
            "use_200sma_filter":   [False, True],
            "require_confirmation": [False, True],
            "entry_delay_min":     [0, 30, 60],
            "max_trades_per_day":  [1, 2, 3],
        },
    },
]


def _combo_hash(strategy_name: str, params: dict) -> str:
    key = strategy_name + json.dumps(params, sort_keys=True, default=str)
    return hashlib.md5(key.encode()).hexdigest()[:12]


def _load_data() -> tuple[MarketData, list]:
    loader = DataLoader()
    df_1m = pd.read_parquet("data/NQ_1m.parquet")
    df_5m = pd.read_parquet("data/NQ_5m.parquet")
    start = pd.Timestamp(DATE_FROM, tz="America/New_York")
    end   = (pd.Timestamp(DATE_TO, tz="America/New_York")
             + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))
    m1 = df_1m[(df_1m.index >= start) & (df_1m.index <= end)]
    m5 = df_5m[(df_5m.index >= start) & (df_5m.index <= end)]
    rth = (m1.index.time >= dtime(9, 30)) & (m1.index.time <= dtime(16, 0))
    td  = sorted(set(m1[rth].index.date))
    a1  = {c: m1[c].to_numpy("float64") for c in ["open","high","low","close","volume"]}
    a5  = {c: m5[c].to_numpy("float64") for c in ["open","high","low","close","volume"]}
    bm  = loader._build_bar_map(m1, m5)
    m1_idx = m1.index
    times_min = (m1_idx.hour * 60 + m1_idx.minute).to_numpy(np.int32)
    bar_day_int = m1_idx.normalize().view(np.int64) // (86_400 * 10**9)
    data = MarketData(df_1m=m1, df_5m=m5,
                      open_1m=a1["open"],  high_1m=a1["high"],  low_1m=a1["low"],
                      close_1m=a1["close"], volume_1m=a1["volume"],
                      open_5m=a5["open"],  high_5m=a5["high"],  low_5m=a5["low"],
                      close_5m=a5["close"], volume_5m=a5["volume"],
                      bar_map=bm, trading_dates=td,
                      bar_times_1m_min=times_min,
                      bar_day_int_1m=bar_day_int)
    return data, td


def _screen(pnl_pts: np.ndarray, sl_dists: np.ndarray, tpd: float) -> dict:
    best_ev, best_info = -np.inf, {}
    for acc_name in SCREEN_ACCOUNTS:
        grid = run_propfirm_grid(
            trades=None, account=LUCIDFLEX_ACCOUNTS[acc_name],
            n_sims=SCREEN_N_SIMS, sizing_mode="micros",
            schemes=SCREEN_SCHEMES,
            eval_risk_pcts=SCREEN_EVAL_RISKS,
            funded_risk_pcts=SCREEN_FND_RISKS,
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


def _load_existing() -> set:
    if not os.path.exists(CSV_PATH):
        return set()
    with open(CSV_PATH, newline="") as f:
        return {r["combo_id"] for r in csv.DictReader(f)}


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Loading data {DATE_FROM} -> {DATE_TO}...", flush=True)
    t0 = time.perf_counter()
    data, td = _load_data()
    n_td = len(td)
    print(f"  {len(data.df_1m):,} bars | {n_td} trading days  ({time.perf_counter()-t0:.1f}s)\n")

    done_ids  = _load_existing()
    write_hdr = not os.path.exists(CSV_PATH) or not done_ids
    csv_file  = open(CSV_PATH, "a", newline="")
    writer    = csv.DictWriter(csv_file, fieldnames=CSV_COLS)
    if write_hdr:
        writer.writeheader()

    total = sum(
        len(list(itertools.product(*[s["grid"][k] for k in s["grid"]])))
        for s in STRATEGIES
    )
    print(f"Total combos: {total}\n")

    cfg = RunConfig(starting_capital=100_000, slippage_points=0.25,
                    commission_per_contract=4.50, eod_exit_time=dtime(23, 59),
                    track_equity_curve=False, params={})

    global_idx = 0
    n_proc = 0
    t_total = 0.0

    for strat_def in STRATEGIES:
        sname = strat_def["name"]
        cls   = strat_def["cls"]
        fixed = strat_def["fixed"]
        grid  = strat_def["grid"]

        keys   = list(grid.keys())
        combos = list(itertools.product(*[grid[k] for k in keys]))
        print(f"\n{'='*60}")
        print(f"Strategy: {sname}  ({len(combos)} combos)")
        print(f"{'='*60}")

        for combo_vals in combos:
            global_idx += 1
            grid_params = dict(zip(keys, combo_vals))
            params      = {**fixed, **grid_params}
            cid         = _combo_hash(sname, params)

            if cid in done_ids:
                continue

            param_str = "  ".join(f"{k}={v}" for k, v in grid_params.items())
            print(f"[{global_idx:4d}/{total}] {sname} | {param_str}", end="  ", flush=True)

            cfg.params = params
            t_bt = time.perf_counter()
            try:
                result = run_backtest(cls, cfg, data, validate=False)
            except Exception as e:
                print(f"ERROR: {e}")
                continue
            bt_time = time.perf_counter() - t_bt
            n_trades = len(result.trades)

            if n_trades < MIN_TRADES:
                row = {"combo_id": cid, "strategy": sname,
                       "params_json": json.dumps(grid_params, default=str),
                       "n_trades": n_trades,
                       **{k: None for k in CSV_COLS[4:-1]},
                       "screen_time_s": round(bt_time, 2)}
                writer.writerow(row); csv_file.flush(); done_ids.add(cid)
                n_proc += 1; t_total += bt_time
                print(f"SKIP ({n_trades} < {MIN_TRADES})  [{bt_time:.1f}s]")
                remaining = total - global_idx
                if n_proc > 0 and remaining > 0:
                    print(f"  ETA: {(t_total/n_proc)*remaining/60:.1f}m")
                continue

            wins  = sum(1 for t in result.trades if t.net_pnl_dollars > 0)
            wr    = wins / n_trades
            pnl_pts, sl_dists = extract_normalised_trades(result.trades)
            avg_r = float((pnl_pts / np.where(sl_dists > 0, sl_dists, 1.0)).mean())
            tpd   = n_trades / max(1, n_td)

            t_pf = time.perf_counter()
            screen = _screen(pnl_pts, sl_dists, tpd)
            pf_time = time.perf_counter() - t_pf

            ev = screen.get("best_ev_per_day")
            pr = screen.get("best_pass_rate")
            print(f"n={n_trades:5d} WR={wr:.1%} avgR={avg_r:+.3f} "
                  f"ev/d={f'${ev:.2f}' if ev is not None else 'N/A':>10} "
                  f"pass={f'{pr:.0%}' if pr is not None else 'N/A':>5}  [{bt_time:.1f}s+{pf_time:.1f}s]")

            row = {"combo_id": cid, "strategy": sname,
                   "params_json": json.dumps(grid_params, default=str),
                   "n_trades": n_trades, "win_rate": round(wr, 4),
                   "avg_r": round(avg_r, 4), "trades_per_day": round(tpd, 4),
                   **screen, "screen_time_s": round(bt_time + pf_time, 2)}
            writer.writerow(row); csv_file.flush(); done_ids.add(cid)
            n_proc += 1; t_total += bt_time + pf_time
            remaining = total - global_idx
            if n_proc > 0 and remaining > 0:
                eta_min = (t_total / n_proc) * remaining / 60
                print(f"  ETA: {eta_min:.1f}m  ({n_proc} done, {remaining} left)")

    csv_file.close()

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "="*80)
    df = pd.read_csv(CSV_PATH)
    df_v = (df.dropna(subset=["best_ev_per_day"])
              .sort_values("best_ev_per_day", ascending=False)
              .reset_index(drop=True))

    print(f"Valid combos: {len(df_v)}/{len(df)}  |  Positive EV: {(df_v['best_ev_per_day'] > 0).sum()}")
    top = df_v.head(30).copy(); top.index = range(1, len(top)+1)
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 80)
    print("\nTOP 30:")
    print(top[["strategy","params_json","n_trades","win_rate","avg_r",
               "best_ev_per_day","best_account","best_scheme","best_pass_rate"]].to_string())

    print("\n\nBEST PER STRATEGY:")
    for strat_def in STRATEGIES:
        sname = strat_def["name"]
        sub = df_v[df_v["strategy"] == sname]
        if len(sub) == 0:
            print(f"  {sname}: no valid results")
        else:
            best = sub.iloc[0]
            print(f"  {sname}: ev=${best['best_ev_per_day']:.2f}/day  "
                  f"n={int(best['n_trades'])}  WR={best['win_rate']:.1%}  "
                  f"avgR={best['avg_r']:+.3f}  {best['params_json']}")

    with open(SUMM_PATH, "w") as f:
        f.write(f"Sweep6 ORB+VWAP: {DATE_FROM} -> {DATE_TO}\n")
        f.write(f"Valid: {len(df_v)}/{len(df)}  Positive EV: {(df_v['best_ev_per_day']>0).sum()}\n\n")
        f.write("TOP 30:\n")
        f.write(top[["strategy","params_json","n_trades","win_rate","avg_r",
                     "best_ev_per_day","best_account","best_scheme"]].to_string())


if __name__ == "__main__":
    main()
