# -*- coding: utf-8 -*-
"""
sweep_propfirm2.py
------------------
Secondary sweep: fix the winning config from sweep 1
(NY, rr=1.25, bos=True, mom=True, disp=2.0)
and vary sl_atr_mult, wick_threshold, atr_period.

Goal: find fine-tuning that pushes EV/day higher.

Output:
  sweep_results2/results.csv
  sweep_results2/pkl/<hash>.pkl  (top-20 full grid)
  sweep_results2/summary.txt
"""
from __future__ import annotations

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import csv
import hashlib
import itertools
import json
import os
import pickle
import time
from datetime import time as dtime

import numpy as np
import pandas as pd

from backtest.data.loader import DataLoader
from backtest.data.market_data import MarketData
from backtest.propfirm.lucidflex import (
    LUCIDFLEX_ACCOUNTS,
    _estimate_trading_days,
    extract_normalised_trades,
    run_propfirm_grid,
)
from backtest.runner.config import RunConfig
from backtest.runner.runner import run_backtest
from strategies.session_mean_rev import SessionMeanRevStrategy

DATE_FROM  = "2019-01-01"
DATE_TO    = "2024-12-31"
MIN_TRADES = 100   # lower threshold — these are selective combos

SCREEN_ACCOUNTS    = ["50K", "100K", "150K"]
SCREEN_SCHEMES     = ["fixed_dollar", "pct_balance", "floor_aware"]
SCREEN_EVAL_RISKS  = [0.20, 0.40, 0.60]
SCREEN_FND_RISKS   = [0.20, 0.40, 0.60]
SCREEN_N_SIMS      = 1_000   # higher sims for better accuracy

FULL_N_SIMS    = 5_000
TOP_N_FULL     = 20

OUT_DIR   = "sweep_results2"
CSV_PATH  = os.path.join(OUT_DIR, "results.csv")
PKL_DIR   = os.path.join(OUT_DIR, "pkl")
SUMM_PATH = os.path.join(OUT_DIR, "summary.txt")

# ── Fixed: winning config from sweep 1 ────────────────────────────────────────
FIXED_BASE = {
    "allowed_sessions":  ["NY"],
    "require_bos":       True,
    "momentum_only":     True,
    "disp_min_atr_mult": 2.0,
    "risk_per_trade":    0.01,
    "equity_mode":       "dynamic",
    "starting_equity":   100_000,
    "point_value":       20.0,
    "max_trades_per_day": 3,
}

# ── Secondary sweep grid ───────────────────────────────────────────────────────
GRID: dict[str, list] = {
    "rr_ratio":          [1.0, 1.25, 1.5],
    "sl_atr_multiplier": [0.5, 0.75, 1.0, 1.25, 1.5],
    "wick_threshold":    [0.10, 0.15, 0.20, 0.25],
    "atr_period":        [7, 10, 14],
}

CSV_COLS = [
    "combo_id", "rr_ratio", "sl_atr_multiplier", "wick_threshold", "atr_period",
    "n_trades", "win_rate", "avg_r", "trades_per_day",
    "best_ev_per_day", "best_account", "best_scheme",
    "best_eval_risk", "best_funded_risk", "best_pass_rate",
    "screen_time_s",
]


def _combo_hash(params: dict) -> str:
    key = json.dumps(params, sort_keys=True, default=str)
    return hashlib.md5(key.encode()).hexdigest()[:12]


def _load_data() -> MarketData:
    loader = DataLoader()
    df_1m   = pd.read_parquet("data/NQ_1m.parquet")
    df_5m   = pd.read_parquet("data/NQ_5m.parquet")
    bar_map = np.load("data/NQ_bar_map.npy")

    start_ts = pd.Timestamp(DATE_FROM, tz="America/New_York")
    end_ts   = (pd.Timestamp(DATE_TO, tz="America/New_York")
                + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))

    mask_1m = (df_1m.index >= start_ts) & (df_1m.index <= end_ts)
    mask_5m = (df_5m.index >= start_ts) & (df_5m.index <= end_ts)
    df_1m_f, df_5m_f = df_1m[mask_1m], df_5m[mask_5m]

    rth_f           = (df_1m_f.index.time >= dtime(9, 30)) & (df_1m_f.index.time <= dtime(16, 0))
    trading_dates_f = sorted(set(df_1m_f[rth_f].index.date))
    arrays_1m_f     = {c: df_1m_f[c].to_numpy(dtype="float64") for c in ["open","high","low","close","volume"]}
    arrays_5m_f     = {c: df_5m_f[c].to_numpy(dtype="float64") for c in ["open","high","low","close","volume"]}
    bar_map_f       = loader._build_bar_map(df_1m_f, df_5m_f)

    return MarketData(
        df_1m=df_1m_f, df_5m=df_5m_f,
        open_1m=arrays_1m_f["open"], high_1m=arrays_1m_f["high"],
        low_1m=arrays_1m_f["low"],   close_1m=arrays_1m_f["close"],
        volume_1m=arrays_1m_f["volume"],
        open_5m=arrays_5m_f["open"], high_5m=arrays_5m_f["high"],
        low_5m=arrays_5m_f["low"],   close_5m=arrays_5m_f["close"],
        volume_5m=arrays_5m_f["volume"],
        bar_map=bar_map_f, trading_dates=trading_dates_f,
    )


def _run_screen(pnl_pts: np.ndarray, sl_dists: np.ndarray, tpd: float) -> dict:
    best_ev, best_info = -np.inf, {}
    for acc_name in SCREEN_ACCOUNTS:
        account = LUCIDFLEX_ACCOUNTS[acc_name]
        grid = run_propfirm_grid(
            trades=None, account=account,
            n_sims=SCREEN_N_SIMS, sizing_mode="micros",
            schemes=SCREEN_SCHEMES,
            eval_risk_pcts=SCREEN_EVAL_RISKS,
            funded_risk_pcts=SCREEN_FND_RISKS,
            _pnl_pts=pnl_pts, _sl_dists=sl_dists, _trades_per_day=tpd,
        )
        for scheme in SCREEN_SCHEMES:
            if scheme not in grid:
                continue
            for erp in SCREEN_EVAL_RISKS:
                for frp in SCREEN_FND_RISKS:
                    cell = grid.get(scheme, {}).get(erp, {}).get(frp)
                    if cell is None:
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


def _run_full_grid(pnl_pts: np.ndarray, sl_dists: np.ndarray, tpd: float) -> dict:
    results = {}
    for acc_name, account in LUCIDFLEX_ACCOUNTS.items():
        results[acc_name] = run_propfirm_grid(
            trades=None, account=account,
            n_sims=FULL_N_SIMS, sizing_mode="micros",
            _pnl_pts=pnl_pts, _sl_dists=sl_dists, _trades_per_day=tpd,
        )
    return results


def _build_combos() -> list[dict]:
    keys   = list(GRID.keys())
    values = [GRID[k] for k in keys]
    combos = []
    for combo in itertools.product(*values):
        params = {**FIXED_BASE, **dict(zip(keys, combo))}
        combos.append(params)
    return combos


def _load_existing(csv_path: str) -> set[str]:
    if not os.path.exists(csv_path):
        return set()
    done = set()
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            done.add(row["combo_id"])
    return done


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(PKL_DIR, exist_ok=True)

    print(f"Loading data {DATE_FROM} -> {DATE_TO}...", flush=True)
    t0   = time.perf_counter()
    data = _load_data()
    print(f"  {len(data.df_1m):,} bars | {len(data.trading_dates):,} days  ({time.perf_counter()-t0:.1f}s)\n")

    combos   = _build_combos()
    done_ids = _load_existing(CSV_PATH)
    total    = len(combos)
    print(f"Total combos: {total}  |  Already done: {len(done_ids)}\n")

    write_header = not os.path.exists(CSV_PATH) or len(done_ids) == 0
    csv_file = open(CSV_PATH, "a", newline="")
    writer   = csv.DictWriter(csv_file, fieldnames=CSV_COLS)
    if write_header:
        writer.writeheader()

    n_processed   = 0
    t_total_proc  = 0.0
    config_base   = RunConfig(
        starting_capital=100_000,
        slippage_points=0.25,
        commission_per_contract=4.50,
        eod_exit_time=dtime(23, 59),
        params={},
    )

    for idx, params in enumerate(combos, start=1):
        combo_id = _combo_hash(params)
        if combo_id in done_ids:
            continue

        print(
            f"[{idx:3d}/{total}] rr={params['rr_ratio']:.2f}  "
            f"sl_mult={params['sl_atr_multiplier']:.2f}  "
            f"wick={params['wick_threshold']:.2f}  "
            f"atr={params['atr_period']:2d}",
            end="  ", flush=True,
        )

        config_base.params = params
        t_bt = time.perf_counter()
        try:
            result = run_backtest(SessionMeanRevStrategy, config_base, data)
        except Exception as e:
            print(f"ERROR: {e}")
            continue
        bt_time = time.perf_counter() - t_bt

        n_trades = len(result.trades)

        if n_trades < MIN_TRADES:
            row = {
                "combo_id": combo_id,
                "rr_ratio": params["rr_ratio"],
                "sl_atr_multiplier": params["sl_atr_multiplier"],
                "wick_threshold": params["wick_threshold"],
                "atr_period": params["atr_period"],
                "n_trades": n_trades,
                **{k: None for k in ["win_rate","avg_r","trades_per_day","best_ev_per_day",
                                      "best_account","best_scheme","best_eval_risk",
                                      "best_funded_risk","best_pass_rate"]},
                "screen_time_s": round(bt_time, 2),
            }
            writer.writerow(row); csv_file.flush()
            done_ids.add(combo_id)
            n_processed += 1; t_total_proc += bt_time
            print(f"SKIP ({n_trades} trades < {MIN_TRADES})  [{bt_time:.1f}s]")
            continue

        wins      = sum(1 for t in result.trades if t.net_pnl_dollars > 0)
        win_rate  = wins / n_trades
        pnl_pts, sl_dists = extract_normalised_trades(result.trades)
        avg_r     = float((pnl_pts / np.where(sl_dists > 0, sl_dists, 1.0)).mean())
        tpd       = max(0.5, n_trades / max(1, _estimate_trading_days(result.trades)))

        t_pf  = time.perf_counter()
        screen = _run_screen(pnl_pts, sl_dists, tpd)
        pf_time = time.perf_counter() - t_pf

        ev = screen.get("best_ev_per_day")
        pr = screen.get("best_pass_rate")
        print(
            f"n={n_trades:4d} WR={win_rate:.1%} avgR={avg_r:+.3f} "
            f"ev/d={f'${ev:.2f}' if ev else 'N/A':>9} "
            f"pass={f'{pr:.0%}' if pr else 'N/A':>5}  "
            f"[{bt_time:.1f}s + {pf_time:.1f}s]"
        )

        row = {
            "combo_id":          combo_id,
            "rr_ratio":          params["rr_ratio"],
            "sl_atr_multiplier": params["sl_atr_multiplier"],
            "wick_threshold":    params["wick_threshold"],
            "atr_period":        params["atr_period"],
            "n_trades":          n_trades,
            "win_rate":          round(win_rate, 4),
            "avg_r":             round(avg_r, 4),
            "trades_per_day":    round(tpd, 4),
            **screen,
            "screen_time_s":     round(bt_time + pf_time, 2),
        }
        writer.writerow(row); csv_file.flush()
        done_ids.add(combo_id)
        n_processed += 1
        t_total_proc += bt_time + pf_time

        remaining = total - idx
        if n_processed > 0 and remaining > 0:
            eta_s = (t_total_proc / n_processed) * remaining
            print(f"  ETA: {eta_s/60:.1f}m  ({n_processed} done, {remaining} left)")

    csv_file.close()

    # ── Rank results ───────────────────────────────────────────────────────────
    print("\n" + "="*80)
    df = pd.read_csv(CSV_PATH)
    df_v = df.dropna(subset=["best_ev_per_day"]).sort_values("best_ev_per_day", ascending=False).reset_index(drop=True)

    print(f"Valid combos: {len(df_v)} / {len(df)}  |  Positive EV: {(df_v['best_ev_per_day'] > 0).sum()}")
    print("\nTOP 20 BY SCREENING EV/DAY:")
    top = df_v.head(20).copy(); top.index = range(1, len(top)+1)
    pd.set_option("display.width", 200); pd.set_option("display.max_colwidth", 20)
    print(top[["rr_ratio","sl_atr_multiplier","wick_threshold","atr_period",
               "n_trades","win_rate","avg_r","best_ev_per_day",
               "best_account","best_pass_rate"]].to_string())

    with open(SUMM_PATH, "w") as f:
        f.write(f"Sweep2: {DATE_FROM} -> {DATE_TO}\n")
        f.write(f"Fixed: {json.dumps(FIXED_BASE)}\n")
        f.write(f"Valid: {len(df_v)}/{len(df)}  Positive EV: {(df_v['best_ev_per_day'] > 0).sum()}\n\n")
        f.write("TOP 20:\n" + top[["rr_ratio","sl_atr_multiplier","wick_threshold","atr_period",
                                    "n_trades","win_rate","avg_r","best_ev_per_day",
                                    "best_account","best_pass_rate"]].to_string())

    # ── Full grid on top-20 ────────────────────────────────────────────────────
    print(f"\nRunning full propfirm grid (n_sims={FULL_N_SIMS}) on top {TOP_N_FULL}...")
    combo_lookup = {_combo_hash(p): p for p in _build_combos()}

    for rank, (_, row) in enumerate(df_v.head(TOP_N_FULL).iterrows(), start=1):
        cid    = row["combo_id"]
        params = combo_lookup.get(cid)
        if params is None:
            continue

        pkl_path = os.path.join(PKL_DIR, f"rank{rank:02d}_{cid}.pkl")
        if os.path.exists(pkl_path):
            print(f"  [{rank:2d}] already saved, skipping")
            continue

        print(
            f"  [{rank:2d}] rr={params['rr_ratio']:.2f} sl={params['sl_atr_multiplier']:.2f} "
            f"wick={params['wick_threshold']:.2f} atr={params['atr_period']}  "
            f"(screen=${row['best_ev_per_day']:.2f})",
            end="  ", flush=True,
        )
        config_base.params = params
        result = run_backtest(SessionMeanRevStrategy, config_base, data)
        if len(result.trades) < MIN_TRADES:
            print("SKIP")
            continue

        pnl_pts, sl_dists = extract_normalised_trades(result.trades)
        tpd = max(0.5, len(result.trades) / max(1, _estimate_trading_days(result.trades)))

        t_full = time.perf_counter()
        full_grid = _run_full_grid(pnl_pts, sl_dists, tpd)
        elapsed_full = time.perf_counter() - t_full

        best_ev_full, best_full_info = -np.inf, {}
        for acc_name, acc_grid in full_grid.items():
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
                        if ev is not None and ev > best_ev_full:
                            best_ev_full = ev
                            best_full_info = {
                                "account": acc_name, "scheme": scheme,
                                "eval_risk": erp, "funded_risk": frp,
                                "ev_per_day": ev,
                                "pass_rate": cell.get("pass_rate"),
                                "net_ev": cell.get("net_ev"),
                                "optimal_k": cell.get("optimal_k"),
                            }

        print(f"full EV/day=${best_ev_full:.2f}  [{elapsed_full:.1f}s]")

        payload = {
            "rank": rank, "combo_id": cid, "params": params,
            "n_trades": len(result.trades),
            "win_rate": float(sum(1 for t in result.trades if t.net_pnl_dollars > 0) / len(result.trades)),
            "trades_per_day": tpd,
            "pnl_pts": pnl_pts, "sl_dists": sl_dists,
            "full_grid": full_grid, "best_full": best_full_info,
            "screen_ev_per_day": row["best_ev_per_day"],
            "full_ev_per_day": best_ev_full,
            "sweep": "sweep2",
        }
        with open(pkl_path, "wb") as f:
            pickle.dump(payload, f)
        print(f"    -> saved {pkl_path}")

    # ── Final table ────────────────────────────────────────────────────────────
    print("\nFULL GRID RESULTS (sweep2 top combos):")
    full_rows = []
    for rank in range(1, TOP_N_FULL + 1):
        for fname in os.listdir(PKL_DIR):
            if fname.startswith(f"rank{rank:02d}_"):
                with open(os.path.join(PKL_DIR, fname), "rb") as f:
                    p = pickle.load(f)
                full_rows.append({
                    "rank":  rank,
                    "rr":    p["params"]["rr_ratio"],
                    "sl":    p["params"]["sl_atr_multiplier"],
                    "wick":  p["params"]["wick_threshold"],
                    "atr":   p["params"]["atr_period"],
                    "n":     p["n_trades"],
                    "WR":    f"{p['win_rate']:.1%}",
                    "scr":   f"${p['screen_ev_per_day']:.2f}",
                    "full":  f"${p['full_ev_per_day']:.2f}",
                    "acct":  p["best_full"].get("account",""),
                    "pass":  f"{p['best_full'].get('pass_rate',0):.1%}" if p["best_full"].get("pass_rate") else "",
                })
                break
    if full_rows:
        df_f = pd.DataFrame(full_rows).set_index("rank")
        print(df_f.to_string())
        with open(SUMM_PATH, "a") as f:
            f.write("\n\nFULL GRID RESULTS:\n" + df_f.to_string())

    print(f"\nDone. Results in: {OUT_DIR}/")


if __name__ == "__main__":
    main()
