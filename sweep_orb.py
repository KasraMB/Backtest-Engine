"""
ORB Stop Sweep -- ORBStopStrategy parameter sweep for NQ propfirm optimization.

Sweeps combos on IS (2019-2022) then validates top configs on OOS (2023-2024).
Sorts by avgR (mean R-multiple across all trades). Reports trades/day.
"""
import os
import sys
import time
import itertools
import numpy as np
import pandas as pd
from datetime import time as dtime

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from backtest.data.loader import DataLoader
from backtest.runner.runner import run_backtest
from backtest.runner.config import RunConfig
from backtest.data.market_data import MarketData
from strategies.orb_stop_strategy import ORBStopStrategy

CACHE_1M = "data/NQ_1m.parquet"
CACHE_5M = "data/NQ_5m.parquet"

IS_FROM  = "2019-01-01"
IS_TO    = "2022-12-31"
OOS_FROM = "2023-01-01"
OOS_TO   = "2024-12-31"

MIN_TRADES = 100   # IS minimum
TOP_N_OOS  = 15


def load_data(df_1m, df_5m, loader, date_from, date_to):
    start_ts = pd.Timestamp(date_from, tz="America/New_York")
    end_ts   = pd.Timestamp(date_to,   tz="America/New_York") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    m1 = (df_1m.index >= start_ts) & (df_1m.index <= end_ts)
    m5 = (df_5m.index >= start_ts) & (df_5m.index <= end_ts)
    d1 = df_1m[m1]; d5 = df_5m[m5]
    rth = (d1.index.time >= dtime(9, 30)) & (d1.index.time <= dtime(16, 0))
    tdates = sorted(set(d1[rth].index.date))
    a1 = {c: d1[c].to_numpy(dtype="float64") for c in ["open","high","low","close","volume"]}
    a5 = {c: d5[c].to_numpy(dtype="float64") for c in ["open","high","low","close","volume"]}
    bm = loader._build_bar_map(d1, d5)
    data = MarketData(
        df_1m=d1, df_5m=d5,
        open_1m=a1["open"], high_1m=a1["high"], low_1m=a1["low"],
        close_1m=a1["close"], volume_1m=a1["volume"],
        open_5m=a5["open"], high_5m=a5["high"], low_5m=a5["low"],
        close_5m=a5["close"], volume_5m=a5["volume"],
        bar_map=bm, trading_dates=tdates,
    )
    return data, len(tdates)


def run_combo(data, n_days, params):
    cfg = RunConfig(
        starting_capital=100_000,
        slippage_points=0.25,
        commission_per_contract=4.50,
        eod_exit_time=dtime(16, 0),
        params=params,
    )
    try:
        result = run_backtest(ORBStopStrategy, cfg, data)
    except Exception as e:
        return None
    trades = result.trades
    if len(trades) < 5:
        return None

    r_mults = []
    for t in trades:
        if t.initial_sl_price is None:
            continue
        risk_pts = abs(t.entry_price - t.initial_sl_price)
        if risk_pts < 0.001:
            continue
        pnl_pts = (t.exit_price - t.entry_price) * t.direction
        r_mults.append(pnl_pts / risk_pts)

    if not r_mults:
        return None

    n = len(trades)
    avg_r  = float(np.mean(r_mults))
    sl_cov = len(r_mults) / n
    wins   = sum(1 for r in r_mults if r > 0)
    wr     = wins / len(r_mults)

    net_pnl = sum(
        (t.exit_price - t.entry_price) * t.direction * t.contracts * 20.0
        - t.commission_per_contract * 2 * t.contracts
        for t in trades
    )
    tpd = n / n_days

    return {
        "n": n, "wr": wr, "avg_r": avg_r, "sl_cov": sl_cov,
        "net_pnl": net_pnl, "tpd": tpd,
    }


def main():
    t0 = time.perf_counter()

    loader = DataLoader()
    print("Loading data...", flush=True)
    df_1m = pd.read_parquet(CACHE_1M)
    df_5m = pd.read_parquet(CACHE_5M)
    print(f"  {len(df_1m):,} 1m bars ({time.perf_counter()-t0:.1f}s)\n")

    data_is,  n_is  = load_data(df_1m, df_5m, loader, IS_FROM,  IS_TO)
    data_oos, n_oos = load_data(df_1m, df_5m, loader, OOS_FROM, OOS_TO)

    print(f"IS:  {IS_FROM} to {IS_TO}   ({n_is} trading days)")
    print(f"OOS: {OOS_FROM} to {OOS_TO}  ({n_oos} trading days)\n")

    # Build combos
    or_minutes_list  = [5, 10, 15, 30]
    rr_list          = [1.0, 1.5, 2.0]
    sl_atr_list      = [0.75, 1.0, 1.5, 2.0]
    max_range_list   = [2.0, 3.0, 4.0]

    combos = []
    for or_min, rr, sl_atr, max_rng in itertools.product(
        or_minutes_list, rr_list, sl_atr_list, max_range_list
    ):
        combos.append({
            "or_minutes":       or_min,
            "rr_ratio":         rr,
            "sl_atr_multiplier": sl_atr,
            "atr_period":       14,
            "max_range_atr":    max_rng,
            "min_range_atr":    0.1,
            "direction_mode":   "momentum",
            "risk_per_trade":   0.01,
        })

    print(f"Sweeping {len(combos)} IS combos...\n", flush=True)

    is_rows = []
    for i, params in enumerate(combos):
        r = run_combo(data_is, n_is, params)
        if r is None or r["n"] < MIN_TRADES or r["sl_cov"] < 0.80:
            pass
        else:
            row = {**params, **r}
            is_rows.append(row)
        if (i + 1) % 24 == 0:
            print(f"  {i+1}/{len(combos)} done ...", flush=True)

    print(f"\n  {len(is_rows)} combos passed IS filter (>={MIN_TRADES} trades, SL>=80%)\n")

    if not is_rows:
        print("No IS combos passed. Exiting.")
        return

    df_is = pd.DataFrame(is_rows).sort_values("avg_r", ascending=False)

    print("=" * 110)
    print("TOP IS RESULTS (sorted by avg_r = mean R-multiple per trade)")
    print("=" * 110)
    cols = ["or_minutes","rr_ratio","sl_atr_multiplier","max_range_atr",
            "n","tpd","wr","avg_r","net_pnl","sl_cov"]
    top = df_is.head(20).copy()
    top["wr"]      = top["wr"].map(lambda x: f"{x:.1%}")
    top["tpd"]     = top["tpd"].map(lambda x: f"{x:.3f}")
    top["avg_r"]   = top["avg_r"].map(lambda x: f"{x:+.4f}")
    top["net_pnl"] = top["net_pnl"].map(lambda x: f"${x:,.0f}")
    top["sl_cov"]  = top["sl_cov"].map(lambda x: f"{x:.0%}")
    top.index = range(1, len(top)+1)
    print(top[cols].to_string())
    print()

    # OOS validation
    top_combos = df_is.head(TOP_N_OOS).to_dict("records")
    print(f"\nValidating top {TOP_N_OOS} on OOS ({OOS_FROM} to {OOS_TO})...\n")

    oos_rows = []
    for row in top_combos:
        params = {k: row[k] for k in ["or_minutes","rr_ratio","sl_atr_multiplier",
                                       "atr_period","max_range_atr","min_range_atr",
                                       "direction_mode","risk_per_trade"]}
        r_oos = run_combo(data_oos, n_oos, params)
        key = (f"or={row['or_minutes']} rr={row['rr_ratio']:.1f} "
               f"sl={row['sl_atr_multiplier']:.2f} mxr={row['max_range_atr']:.1f}")
        if r_oos is None:
            print(f"  {key}: NO OOS TRADES")
            continue
        oos_rows.append({
            "combo": key,
            "is_n": row["n"], "is_tpd": row["tpd"], "is_wr": row["wr"],
            "is_avg_r": row["avg_r"],
            "oos_n": r_oos["n"], "oos_tpd": r_oos["tpd"], "oos_wr": r_oos["wr"],
            "oos_avg_r": r_oos["avg_r"], "oos_net_pnl": r_oos["net_pnl"],
        })

    if not oos_rows:
        print("No OOS results.")
        return

    df_oos = pd.DataFrame(oos_rows).sort_values("oos_avg_r", ascending=False)

    print("=" * 120)
    print("IS vs OOS COMPARISON (sorted by OOS avg_r)")
    print("=" * 120)
    for _, row in df_oos.iterrows():
        print(
            f"  {row['combo']:45s} | "
            f"IS n={row['is_n']:4.0f} tpd={row['is_tpd']:.3f} WR={row['is_wr']:.0%} avgR={row['is_avg_r']:+.4f} | "
            f"OOS n={row['oos_n']:4.0f} tpd={row['oos_tpd']:.3f} WR={row['oos_wr']:.0%} avgR={row['oos_avg_r']:+.4f} pnl=${row['oos_net_pnl']:+,.0f}"
        )

    print(f"\nTotal elapsed: {time.perf_counter()-t0:.1f}s")

    # Propfirm cycle math for best OOS config
    best = df_oos.iloc[0]
    print(f"\n{'='*70}")
    print("BEST OOS CONFIG -- PROPFIRM CYCLE MATH (25K LucidFlex)")
    print(f"{'='*70}")
    tpd   = best["oos_tpd"]
    avg_r = best["oos_avg_r"]
    wr    = best["oos_wr"]
    print(f"  Config: {best['combo']}")
    print(f"  OOS: trades/day={tpd:.3f}  WR={wr:.1%}  avgR={avg_r:+.4f}")
    if avg_r > 0:
        # 25K: eval target $1,250, eval_risk ~$200 (20% of $1,000 MLL)
        ev_per_trade = avg_r * 200
        trades_to_pass = 1250 / ev_per_trade
        days_to_pass   = trades_to_pass / tpd
        funded_days    = max(0, 84 - days_to_pass)
        print(f"  EV per trade (@$200 risk): ${ev_per_trade:.2f}")
        print(f"  Trades to pass eval:       {trades_to_pass:.0f}")
        print(f"  Days to pass eval:         {days_to_pass:.0f}")
        print(f"  Remaining funded days:     {funded_days:.0f} / 84")
    else:
        print("  WARNING: negative avgR -- this config loses money")


if __name__ == "__main__":
    main()
