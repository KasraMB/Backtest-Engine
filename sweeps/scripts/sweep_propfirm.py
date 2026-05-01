# -*- coding: utf-8 -*-
"""
sweep_propfirm.py
─────────────────
Sweeps SessionMeanRevStrategy parameter combos on 2019-2024 data.
Goal: maximise propfirm EV/day (LucidFlex).

Output:
  sweep_results/results.csv          — one row per evaluated combo
  sweep_results/pkl/<hash>.pkl       — full propfirm grid for top-N combos
  sweep_results/summary.txt          — human-readable top-20 table

Run: python sweep_propfirm.py
"""
from __future__ import annotations

import sys
import io
# Force UTF-8 stdout/stderr on Windows to avoid UnicodeEncodeError
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

# ── Date range ─────────────────────────────────────────────────────────────────
DATE_FROM  = "2019-01-01"
DATE_TO    = "2024-12-31"

# ── Quality filter ─────────────────────────────────────────────────────────────
MIN_TRADES = 150   # skip combos with fewer trades (MC noise dominates)

# ── Screening propfirm grid (fast — full grid only on top-N) ──────────────────
SCREEN_ACCOUNTS    = ["50K", "100K"]
SCREEN_SCHEMES     = ["fixed_dollar", "floor_aware"]
SCREEN_EVAL_RISKS  = [0.20, 0.40, 0.60]
SCREEN_FND_RISKS   = [0.20, 0.40, 0.60]
SCREEN_N_SIMS      = 500

# ── Full propfirm grid (top-N combos after sweep) ─────────────────────────────
FULL_N_SIMS    = 5_000
TOP_N_FULL     = 20     # how many combos get the full grid treatment
TOP_N_SAVE_PKL = 20     # how many combos get individual pickles

# ── Output paths ──────────────────────────────────────────────────────────────
OUT_DIR   = "sweep_results"
CSV_PATH  = os.path.join(OUT_DIR, "results.csv")
PKL_DIR   = os.path.join(OUT_DIR, "pkl")
SUMM_PATH = os.path.join(OUT_DIR, "summary.txt")

# ── Parameter grid ─────────────────────────────────────────────────────────────
# Core sweep: 3×5×2×2×3 = 180 combos
GRID: dict[str, list] = {
    "allowed_sessions":  [["NY"], ["London", "NY"], ["Asia", "London", "NY"]],
    "rr_ratio":          [0.5, 0.75, 1.0, 1.25, 1.5],
    "require_bos":       [True, False],
    "momentum_only":     [True, False],
    "disp_min_atr_mult": [0.0, 1.0, 2.0],
}

# Fixed parameters (not swept)
FIXED_PARAMS: dict = {
    "atr_period":        10,
    "wick_threshold":    0.15,
    "sl_atr_multiplier": 1.0,
    "risk_per_trade":    0.01,
    "equity_mode":       "dynamic",
    "starting_equity":   100_000,
    "point_value":       20.0,
    "max_trades_per_day": 3,
}

# CSV columns
CSV_COLS = [
    "combo_id", "allowed_sessions", "rr_ratio", "require_bos",
    "momentum_only", "disp_min_atr_mult",
    "n_trades", "win_rate", "avg_r", "trades_per_day",
    "best_ev_per_day", "best_account", "best_scheme",
    "best_eval_risk", "best_funded_risk", "best_pass_rate",
    "screen_time_s",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _combo_hash(params: dict) -> str:
    key = json.dumps(params, sort_keys=True, default=str)
    return hashlib.md5(key.encode()).hexdigest()[:12]


def _load_data() -> MarketData:
    CACHE_1M      = "data/NQ_1m.parquet"
    CACHE_5M      = "data/NQ_5m.parquet"
    CACHE_BAR_MAP = "data/NQ_bar_map.npy"

    loader = DataLoader()

    df_1m   = pd.read_parquet(CACHE_1M)
    df_5m   = pd.read_parquet(CACHE_5M)
    bar_map = np.load(CACHE_BAR_MAP)

    arrays_1m = {c: df_1m[c].to_numpy(dtype="float64")
                 for c in ["open", "high", "low", "close", "volume"]}
    arrays_5m = {c: df_5m[c].to_numpy(dtype="float64")
                 for c in ["open", "high", "low", "close", "volume"]}
    rth_mask      = (df_1m.index.time >= dtime(9, 30)) & (df_1m.index.time <= dtime(16, 0))
    trading_dates = sorted(set(df_1m[rth_mask].index.date))

    full = MarketData(
        df_1m=df_1m, df_5m=df_5m,
        open_1m=arrays_1m["open"], high_1m=arrays_1m["high"],
        low_1m=arrays_1m["low"],   close_1m=arrays_1m["close"],
        volume_1m=arrays_1m["volume"],
        open_5m=arrays_5m["open"], high_5m=arrays_5m["high"],
        low_5m=arrays_5m["low"],   close_5m=arrays_5m["close"],
        volume_5m=arrays_5m["volume"],
        bar_map=bar_map, trading_dates=trading_dates,
    )

    # Filter to DATE_FROM–DATE_TO
    start_ts = pd.Timestamp(DATE_FROM, tz="America/New_York")
    end_ts   = (pd.Timestamp(DATE_TO, tz="America/New_York")
                + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))

    mask_1m = (df_1m.index >= start_ts) & (df_1m.index <= end_ts)
    mask_5m = (df_5m.index >= start_ts) & (df_5m.index <= end_ts)
    df_1m_f = df_1m[mask_1m]
    df_5m_f = df_5m[mask_5m]

    rth_f         = (df_1m_f.index.time >= dtime(9, 30)) & (df_1m_f.index.time <= dtime(16, 0))
    trading_dates_f = sorted(set(df_1m_f[rth_f].index.date))

    arrays_1m_f = {c: df_1m_f[c].to_numpy(dtype="float64")
                   for c in ["open", "high", "low", "close", "volume"]}
    arrays_5m_f = {c: df_5m_f[c].to_numpy(dtype="float64")
                   for c in ["open", "high", "low", "close", "volume"]}
    bar_map_f = loader._build_bar_map(df_1m_f, df_5m_f)

    data = MarketData(
        df_1m=df_1m_f, df_5m=df_5m_f,
        open_1m=arrays_1m_f["open"], high_1m=arrays_1m_f["high"],
        low_1m=arrays_1m_f["low"],   close_1m=arrays_1m_f["close"],
        volume_1m=arrays_1m_f["volume"],
        open_5m=arrays_5m_f["open"], high_5m=arrays_5m_f["high"],
        low_5m=arrays_5m_f["low"],   close_5m=arrays_5m_f["close"],
        volume_5m=arrays_5m_f["volume"],
        bar_map=bar_map_f,
        trading_dates=trading_dates_f,
    )
    return data


def _run_screen(trades: list, pnl_pts: np.ndarray, sl_dists: np.ndarray,
                trades_per_day: float) -> dict:
    """Run the screening propfirm grid. Returns best_ev_per_day + supporting info."""
    best_ev   = -np.inf
    best_info = {}

    for acc_name in SCREEN_ACCOUNTS:
        account = LUCIDFLEX_ACCOUNTS[acc_name]
        grid = run_propfirm_grid(
            trades=trades,
            account=account,
            n_sims=SCREEN_N_SIMS,
            sizing_mode="micros",
            schemes=SCREEN_SCHEMES,
            eval_risk_pcts=SCREEN_EVAL_RISKS,
            funded_risk_pcts=SCREEN_FND_RISKS,
            _pnl_pts=pnl_pts,
            _sl_dists=sl_dists,
            _trades_per_day=trades_per_day,
        )
        for scheme in SCREEN_SCHEMES:
            if scheme not in grid:
                continue
            for erp in SCREEN_EVAL_RISKS:
                if erp not in grid[scheme]:
                    continue
                for frp in SCREEN_FND_RISKS:
                    cell = grid[scheme].get(erp, {}).get(frp)
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
        best_info = {
            "best_ev_per_day": None, "best_account": None,
            "best_scheme": None, "best_eval_risk": None,
            "best_funded_risk": None, "best_pass_rate": None,
        }
    return best_info


def _run_full_grid(pnl_pts: np.ndarray, sl_dists: np.ndarray,
                   trades_per_day: float) -> dict:
    """Run the full propfirm grid for all 4 accounts."""
    results = {}
    for acc_name, account in LUCIDFLEX_ACCOUNTS.items():
        results[acc_name] = run_propfirm_grid(
            trades=None,
            account=account,
            n_sims=FULL_N_SIMS,
            sizing_mode="micros",
            _pnl_pts=pnl_pts,
            _sl_dists=sl_dists,
            _trades_per_day=trades_per_day,
        )
    return results


def _build_combos() -> list[dict]:
    keys   = list(GRID.keys())
    values = [GRID[k] for k in keys]
    combos = []
    for combo in itertools.product(*values):
        params = dict(zip(keys, combo))
        params.update(FIXED_PARAMS)
        combos.append(params)
    return combos


def _load_existing_results(csv_path: str) -> set[str]:
    """Return set of combo_ids already in the CSV."""
    if not os.path.exists(csv_path):
        return set()
    done = set()
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            done.add(row["combo_id"])
    return done


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(PKL_DIR, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────────
    print(f"Loading data {DATE_FROM} -> {DATE_TO}...", flush=True)
    t0 = time.perf_counter()
    data = _load_data()
    print(f"  {len(data.df_1m):,} bars | {len(data.trading_dates):,} trading days "
          f"({time.perf_counter()-t0:.1f}s)\n")

    # ── Build combo list ───────────────────────────────────────────────────────
    combos = _build_combos()
    print(f"Total combos: {len(combos)}")

    # ── Resume support ─────────────────────────────────────────────────────────
    done_ids = _load_existing_results(CSV_PATH)
    print(f"Already done: {len(done_ids)}/{len(combos)}\n")

    # ── CSV writer (append mode) ───────────────────────────────────────────────
    write_header = not os.path.exists(CSV_PATH) or len(done_ids) == 0
    csv_file = open(CSV_PATH, "a", newline="")
    writer   = csv.DictWriter(csv_file, fieldnames=CSV_COLS)
    if write_header:
        writer.writeheader()

    # ── Main sweep loop ────────────────────────────────────────────────────────
    total             = len(combos)
    n_skip            = 0
    n_processed       = 0
    t_process_total   = 0.0
    t_start           = time.perf_counter()

    config_base = RunConfig(
        starting_capital=100_000,
        slippage_points=0.25,
        commission_per_contract=4.50,
        eod_exit_time=dtime(23, 59),
        params={},
    )

    for idx, params in enumerate(combos, start=1):
        combo_id = _combo_hash(params)
        if combo_id in done_ids:
            print(f"[{idx:3d}/{total}] {combo_id} — already done, skipping")
            continue

        sessions_label = "+".join(params["allowed_sessions"])
        print(
            f"[{idx:3d}/{total}] sessions={sessions_label:<20} rr={params['rr_ratio']:.2f} "
            f"bos={str(params['require_bos']):<5} mom={str(params['momentum_only']):<5} "
            f"disp={params['disp_min_atr_mult']:.1f}",
            end="  ", flush=True,
        )

        # Backtest
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
            n_skip += 1
            row = {
                "combo_id": combo_id,
                "allowed_sessions": sessions_label,
                "rr_ratio": params["rr_ratio"],
                "require_bos": params["require_bos"],
                "momentum_only": params["momentum_only"],
                "disp_min_atr_mult": params["disp_min_atr_mult"],
                "n_trades": n_trades,
                "win_rate": None, "avg_r": None, "trades_per_day": None,
                "best_ev_per_day": None, "best_account": None, "best_scheme": None,
                "best_eval_risk": None, "best_funded_risk": None, "best_pass_rate": None,
                "screen_time_s": round(bt_time, 2),
            }
            writer.writerow(row)
            csv_file.flush()
            done_ids.add(combo_id)
            n_processed += 1
            t_process_total += bt_time
            print(f"SKIP ({n_trades} trades < {MIN_TRADES})  [{bt_time:.1f}s]")
            continue

        # Basic stats
        wins      = sum(1 for t in result.trades if t.net_pnl_dollars > 0)
        win_rate  = wins / n_trades
        pnl_pts, sl_dists = extract_normalised_trades(result.trades)
        r_multiples = pnl_pts / np.where(sl_dists > 0, sl_dists, 1.0)
        avg_r     = float(r_multiples.mean())
        tpd       = max(0.5, n_trades / max(1, _estimate_trading_days(result.trades)))

        # Screening propfirm grid
        t_pf = time.perf_counter()
        screen = _run_screen(result.trades, pnl_pts, sl_dists, tpd)
        pf_time = time.perf_counter() - t_pf

        ev = screen.get("best_ev_per_day")
        pr = screen.get("best_pass_rate")
        print(
            f"n={n_trades:4d} WR={win_rate:.1%} avgR={avg_r:+.3f} "
            f"ev/d={f'${ev:.2f}' if ev else 'N/A':>9} "
            f"pass={f'{pr:.0%}' if pr else 'N/A':>5} "
            f"[bt={bt_time:.1f}s pf={pf_time:.1f}s]"
        )

        row = {
            "combo_id":          combo_id,
            "allowed_sessions":  sessions_label,
            "rr_ratio":          params["rr_ratio"],
            "require_bos":       params["require_bos"],
            "momentum_only":     params["momentum_only"],
            "disp_min_atr_mult": params["disp_min_atr_mult"],
            "n_trades":          n_trades,
            "win_rate":          round(win_rate, 4),
            "avg_r":             round(avg_r, 4),
            "trades_per_day":    round(tpd, 4),
            **screen,
            "screen_time_s":     round(bt_time + pf_time, 2),
        }
        writer.writerow(row)
        csv_file.flush()
        done_ids.add(combo_id)
        n_processed += 1
        t_process_total += bt_time + pf_time

        # ETA
        remaining = total - idx
        if n_processed > 0 and remaining > 0:
            avg_s = t_process_total / n_processed
            eta_s = remaining * avg_s
            print(f"  ETA: {eta_s/60:.1f}m  |  {n_processed} done, {remaining} remaining")

    csv_file.close()

    # ── Load all results and rank ──────────────────────────────────────────────
    print("\n" + "="*80)
    print("SWEEP COMPLETE — ranking results...")

    df = pd.read_csv(CSV_PATH)
    df_valid = df.dropna(subset=["best_ev_per_day"]).copy()
    df_valid = df_valid.sort_values("best_ev_per_day", ascending=False).reset_index(drop=True)

    print(f"\nTotal combos evaluated: {len(df)}")
    print(f"With enough trades (≥{MIN_TRADES}): {len(df_valid)}")
    print(f"Positive EV/day: {(df_valid['best_ev_per_day'] > 0).sum()}")

    if df_valid.empty:
        print("No valid results found. Check strategy and data.")
        return

    # ── Top-20 display ─────────────────────────────────────────────────────────
    print("\n" + "="*80)
    print("TOP 20 COMBOS BY SCREENING EV/DAY")
    print("="*80)

    top = df_valid.head(20).copy()
    top.index = range(1, len(top)+1)
    pd.set_option("display.max_colwidth", 25)
    pd.set_option("display.width", 200)
    print(top[[
        "allowed_sessions", "rr_ratio", "require_bos", "momentum_only",
        "disp_min_atr_mult", "n_trades", "win_rate", "avg_r",
        "best_ev_per_day", "best_account", "best_scheme",
        "best_eval_risk", "best_funded_risk", "best_pass_rate",
    ]].to_string())

    # ── Save summary ───────────────────────────────────────────────────────────
    summary_lines = [
        f"Sweep: {DATE_FROM} -> {DATE_TO}",
        f"Total combos: {len(df)}  |  Valid: {len(df_valid)}  |  Positive EV: {(df_valid['best_ev_per_day'] > 0).sum()}",
        "",
        "TOP 20 BY SCREENING EV/DAY:",
        top[[
            "allowed_sessions", "rr_ratio", "require_bos", "momentum_only",
            "disp_min_atr_mult", "n_trades", "win_rate", "avg_r",
            "best_ev_per_day", "best_account", "best_scheme",
            "best_eval_risk", "best_funded_risk", "best_pass_rate",
        ]].to_string(),
    ]
    with open(SUMM_PATH, "w") as f:
        f.write("\n".join(summary_lines))
    print(f"\nSummary saved -> {SUMM_PATH}")

    # ── Run full propfirm grid on top-N combos ─────────────────────────────────
    print(f"\n{'='*80}")
    print(f"Running FULL propfirm grid (n_sims={FULL_N_SIMS}) on top {TOP_N_FULL} combos...")
    print("="*80)

    top_combos = df_valid.head(TOP_N_FULL)

    # Rebuild params for each top combo
    combo_lookup: dict[str, dict] = {_combo_hash(p): p for p in _build_combos()}

    for rank, (_, row) in enumerate(top_combos.iterrows(), start=1):
        cid    = row["combo_id"]
        params = combo_lookup.get(cid)
        if params is None:
            print(f"  [{rank}] {cid} — params not found, skipping")
            continue

        pkl_path = os.path.join(PKL_DIR, f"rank{rank:02d}_{cid}.pkl")
        if os.path.exists(pkl_path):
            print(f"  [{rank}] {cid} — already saved, skipping full grid")
            continue

        sessions_label = "+".join(params["allowed_sessions"])
        print(
            f"  [{rank:2d}] sessions={sessions_label} rr={params['rr_ratio']:.2f} "
            f"bos={params['require_bos']} mom={params['momentum_only']} "
            f"disp={params['disp_min_atr_mult']:.1f}  "
            f"(screen EV/day=${row['best_ev_per_day']:.2f})",
            end="  ", flush=True,
        )

        config_base.params = params
        result = run_backtest(SessionMeanRevStrategy, config_base, data)

        if len(result.trades) < MIN_TRADES:
            print("SKIP (trade count changed?)")
            continue

        pnl_pts, sl_dists = extract_normalised_trades(result.trades)
        tpd = max(0.5, len(result.trades) / max(1, _estimate_trading_days(result.trades)))

        t_full = time.perf_counter()
        full_grid = _run_full_grid(pnl_pts, sl_dists, tpd)
        elapsed_full = time.perf_counter() - t_full

        # Find best EV/day across full grid
        best_ev_full = -np.inf
        best_full_info: dict = {}
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
            "rank":          rank,
            "combo_id":      cid,
            "params":        params,
            "n_trades":      len(result.trades),
            "win_rate":      float(sum(1 for t in result.trades if t.net_pnl_dollars > 0) / len(result.trades)),
            "trades_per_day": tpd,
            "pnl_pts":       pnl_pts,
            "sl_dists":      sl_dists,
            "full_grid":     full_grid,
            "best_full":     best_full_info,
            "screen_ev_per_day": row["best_ev_per_day"],
            "full_ev_per_day":   best_ev_full,
        }
        with open(pkl_path, "wb") as f:
            pickle.dump(payload, f)
        print(f"    -> saved {pkl_path}")

    # ── Final summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("FULL GRID RESULTS (top combos)")
    print("="*80)

    full_rows = []
    for rank in range(1, TOP_N_FULL + 1):
        # Find pkl for this rank
        for fname in os.listdir(PKL_DIR):
            if fname.startswith(f"rank{rank:02d}_"):
                with open(os.path.join(PKL_DIR, fname), "rb") as f:
                    p = pickle.load(f)
                full_rows.append({
                    "rank":       rank,
                    "sessions":   "+".join(p["params"]["allowed_sessions"]),
                    "rr":         p["params"]["rr_ratio"],
                    "bos":        p["params"]["require_bos"],
                    "mom":        p["params"]["momentum_only"],
                    "disp":       p["params"]["disp_min_atr_mult"],
                    "n_trades":   p["n_trades"],
                    "WR":         f"{p['win_rate']:.1%}",
                    "scr_ev/d":   f"${p['screen_ev_per_day']:.2f}",
                    "full_ev/d":  f"${p['full_ev_per_day']:.2f}",
                    "account":    p["best_full"].get("account", ""),
                    "scheme":     p["best_full"].get("scheme", ""),
                    "pass_rate":  f"{p['best_full'].get('pass_rate', 0):.1%}" if p["best_full"].get("pass_rate") else "",
                })
                break

    if full_rows:
        df_full = pd.DataFrame(full_rows).set_index("rank")
        print(df_full.to_string())

        # Append to summary
        with open(SUMM_PATH, "a") as f:
            f.write("\n\nFULL GRID RESULTS:\n")
            f.write(df_full.to_string())

    print(f"\nAll done. Results in: {OUT_DIR}/")


if __name__ == "__main__":
    main()
