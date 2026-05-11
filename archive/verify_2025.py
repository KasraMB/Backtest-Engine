# -*- coding: utf-8 -*-
"""
verify_2025.py
--------------
Runs the top configs from sweep1 + sweep2 on 2025 OOS data.
Checks whether propfirm EV/day holds out-of-sample.

Output: printed table + verify_2025_results.csv
"""
from __future__ import annotations

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

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

DATE_FROM  = "2025-01-01"
DATE_TO    = "2025-12-31"    # full 2025 (whatever data exists)
N_SIMS     = 5_000

# Configs to verify: top results from sweep1 + sweep2
# Format: (label, params_dict)
_BASE = {"allowed_sessions": ["NY"], "require_bos": True, "momentum_only": True,
         "disp_min_atr_mult": 2.0, "atr_period": 10, "wick_threshold": 0.15,
         "risk_per_trade": 0.01, "equity_mode": "dynamic",
         "starting_equity": 100_000, "point_value": 20.0, "max_trades_per_day": 3,
         "breakeven_r": 0.0, "atr_vol_filter": 0.0, "require_daily_momentum": False}

CONFIGS_TO_VERIFY = [
    # Stable baseline (best IS/OOS from sweep2, no structural filters)
    ("S2-stable: rr1.00 sl1.25 baseline",
     {**_BASE, "rr_ratio": 1.00, "sl_atr_multiplier": 1.25}),

    # Sweep3 winner: vol filter + daily momentum (IS $23.87, 108 trades/6yr)
    ("S3-dm+vol: rr1.00 sl1.25 be0.75 vol1.15 dm=Y",
     {**_BASE, "rr_ratio": 1.00, "sl_atr_multiplier": 1.25,
      "breakeven_r": 0.75, "atr_vol_filter": 1.15, "require_daily_momentum": True}),

    # Sweep3 vol-only (no daily momentum, IS $18.04, 237 trades/6yr)
    ("S3-vol: rr1.00 sl1.25 vol1.15 dm=N",
     {**_BASE, "rr_ratio": 1.00, "sl_atr_multiplier": 1.25,
      "atr_vol_filter": 1.15, "require_daily_momentum": False}),

    # Sweep4 winner: rr=0.75 wide sl with vol filter (IS $19.92, 342 trades/6yr)
    ("S4-rr0.75: sl1.50 vol1.0",
     {**_BASE, "rr_ratio": 0.75, "sl_atr_multiplier": 1.50, "atr_vol_filter": 1.0}),

    # Sweep4 alt: rr=0.75 wide sl no filter (IS $18.84, 364 trades/6yr)
    ("S4-rr0.75: sl1.50 no-filter",
     {**_BASE, "rr_ratio": 0.75, "sl_atr_multiplier": 1.50}),

    # Previous OOS star (IS $12.53 -> OOS $47.72, for reference)
    ("S2-rank7: rr1.50 sl0.75 (OOS star)",
     {**_BASE, "rr_ratio": 1.50, "sl_atr_multiplier": 0.75}),

    # Sweep3 daily-momentum + vol, no breakeven (IS $22.18)
    ("S3-dm+vol-nobe: rr1.00 sl1.25 vol1.15 dm=Y",
     {**_BASE, "rr_ratio": 1.00, "sl_atr_multiplier": 1.25,
      "atr_vol_filter": 1.15, "require_daily_momentum": True}),
]

# IS EV/day reference: S2-stable=$16.30, S3-dm+vol=$23.87, S3-vol=$18.04,
#                      S4-rr0.75-vol=$19.92, S4-rr0.75=$18.84, S2-rank7=$12.53


def _load_data() -> MarketData:
    loader = DataLoader()
    df_1m   = pd.read_parquet("data/NQ_1m.parquet")
    df_5m   = pd.read_parquet("data/NQ_5m.parquet")

    start_ts = pd.Timestamp(DATE_FROM, tz="America/New_York")
    end_ts   = (pd.Timestamp(DATE_TO, tz="America/New_York")
                + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))

    mask_1m = (df_1m.index >= start_ts) & (df_1m.index <= end_ts)
    mask_5m = (df_5m.index >= start_ts) & (df_5m.index <= end_ts)
    df_1m_f, df_5m_f = df_1m[mask_1m], df_5m[mask_5m]

    if df_1m_f.empty:
        raise ValueError("No 2025 data found in cache. Check data/NQ_1m.parquet coverage.")

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


def _run_propfirm(pnl_pts: np.ndarray, sl_dists: np.ndarray, tpd: float) -> tuple[float, dict]:
    """Run full propfirm grid, return (best_ev_per_day, best_info)."""
    best_ev, best_info = -np.inf, {}
    for acc_name, account in LUCIDFLEX_ACCOUNTS.items():
        grid = run_propfirm_grid(
            trades=None, account=account, n_sims=N_SIMS, sizing_mode="micros",
            _pnl_pts=pnl_pts, _sl_dists=sl_dists, _trades_per_day=tpd,
        )
        for scheme in grid:
            if scheme == "optimal_funded_rp":
                continue
            for erp, erp_data in grid[scheme].items():
                if not isinstance(erp_data, dict):
                    continue
                for frp, cell in erp_data.items():
                    if not isinstance(cell, dict):
                        continue
                    ev = cell.get("ev_per_day")
                    if ev is not None and ev > best_ev:
                        best_ev = ev
                        best_info = {
                            "account": acc_name, "scheme": scheme,
                            "eval_risk": erp, "funded_risk": frp,
                            "ev_per_day": ev,
                            "pass_rate": cell.get("pass_rate"),
                            "net_ev": cell.get("net_ev"),
                            "total_cost": cell.get("total_cost"),
                            "roi": cell.get("roi"),
                        }
    return best_ev, best_info


def main() -> None:
    configs = list(CONFIGS_TO_VERIFY)

    print(f"Loading 2025 OOS data ({DATE_FROM} -> {DATE_TO})...", flush=True)
    t0   = time.perf_counter()
    data = _load_data()
    print(f"  {len(data.df_1m):,} bars | {len(data.trading_dates):,} trading days  ({time.perf_counter()-t0:.1f}s)\n")
    print(f"  Period: {data.trading_dates[0]} -> {data.trading_dates[-1]}\n")

    config_base = RunConfig(
        starting_capital=100_000, slippage_points=0.25,
        commission_per_contract=4.50, eod_exit_time=dtime(23, 59), params={},
    )

    rows = []
    for label, params in configs:
        print(f"\n{'='*70}")
        print(f"Config: {label}")
        config_base.params = params
        t_bt = time.perf_counter()
        result = run_backtest(SessionMeanRevStrategy, config_base, data)
        bt_time = time.perf_counter() - t_bt

        n_trades = len(result.trades)
        print(f"  Trades: {n_trades}  [{bt_time:.1f}s]")

        if n_trades < 5:
            print("  SKIP: too few trades for meaningful propfirm analysis")
            rows.append({
                "label": label, "n_trades": n_trades,
                "win_rate": None, "avg_r": None,
                "ev_per_day_2025": None, "pass_rate": None,
                "net_ev": None, "total_cost": None, "roi": None,
                "account": None, "scheme": None,
            })
            continue

        wins = sum(1 for t in result.trades if t.net_pnl_dollars > 0)
        win_rate = wins / n_trades
        net_pnl  = sum(t.net_pnl_dollars for t in result.trades)
        pnl_pts, sl_dists = extract_normalised_trades(result.trades)
        avg_r    = float((pnl_pts / np.where(sl_dists > 0, sl_dists, 1.0)).mean())
        tpd      = max(0.5, n_trades / max(1, _estimate_trading_days(result.trades)))

        print(f"  WR={win_rate:.1%}  avgR={avg_r:+.3f}  net_pnl=${net_pnl:,.0f}  trades/day={tpd:.3f}")

        t_pf = time.perf_counter()
        best_ev, best_info = _run_propfirm(pnl_pts, sl_dists, tpd)
        pf_time = time.perf_counter() - t_pf

        total_cost = best_info.get("total_cost") or 0
        net_ev     = best_info.get("net_ev") or 0
        roi        = best_info.get("roi")
        print(f"  EV/day(2025)=${best_ev:.2f}  pass={best_info.get('pass_rate',0):.1%}  "
              f"net_ev=${net_ev:.0f}  cost=${total_cost:.0f}  ROI={roi:.2f}x  "
              f"account={best_info.get('account','')}  scheme={best_info.get('scheme','')}  [{pf_time:.1f}s]")

        rows.append({
            "label": label,
            "n_trades": n_trades,
            "win_rate": round(win_rate, 4),
            "avg_r": round(avg_r, 4),
            "ev_per_day_2025": round(best_ev, 4),
            "pass_rate": round(best_info.get("pass_rate", 0), 4),
            "net_ev": round(net_ev, 2),
            "total_cost": round(total_cost, 2),
            "roi": round(roi, 4) if roi is not None else None,
            "account": best_info.get("account", ""),
            "scheme": best_info.get("scheme", ""),
        })

    # Summary table
    print("\n\n" + "="*80)
    print("2025 OOS VERIFICATION SUMMARY")
    print("="*80)
    df = pd.DataFrame(rows)
    df = df.sort_values("ev_per_day_2025", ascending=False, na_position="last").reset_index(drop=True)
    df.index = range(1, len(df)+1)
    pd.set_option("display.width", 200); pd.set_option("display.max_colwidth", 45)
    print(df.to_string())

    df.to_csv("verify_2025_results.csv", index_label="rank")
    print("\nSaved -> verify_2025_results.csv")


if __name__ == "__main__":
    main()
