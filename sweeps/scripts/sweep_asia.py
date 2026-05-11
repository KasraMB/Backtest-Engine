"""
Asia Session Range Strategy — comprehensive multi-angle sweep.

Angles tested:
  entry_mode  : breakout (stop at range edge), fade (counter-range), ny_confirm (9:30 market)
  direction   : long, short, momentum (follow overnight), contrarian (fade overnight)
  asia_start  : 18 (6 PM ET), 20 (8 PM ET)
  sl_type     : atr, range
  rr_ratio    : 1.0, 1.5, 2.0
  sl_atr_mult : 0.5, 0.75, 1.0
  min_range_atr: 0.0, 0.3, 0.5

IS: 2019-2022  OOS: 2023-2024
Sorts by IS avgR, validates top 25 on OOS, reports propfirm math.
Max 2 workers.
"""
import sys, time, itertools
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import time as dtime

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from backtest.data.loader import DataLoader
from backtest.runner.runner import run_backtest
from backtest.runner.config import RunConfig
from backtest.data.market_data import MarketData
from strategies.asia_breakout_strategy import AsiaBreakoutStrategy

CACHE_1M = "data/NQ_1m.parquet"
CACHE_5M = "data/NQ_5m.parquet"

IS_FROM  = "2019-01-01"
IS_TO    = "2022-12-31"
OOS_FROM = "2023-01-01"
OOS_TO   = "2024-12-31"

MIN_TRADES = 30
TOP_N_OOS  = 25
MAX_WORKERS = 2


def load_data(df_1m, df_5m, loader, date_from, date_to):
    start_ts = pd.Timestamp(date_from, tz="America/New_York")
    end_ts   = pd.Timestamp(date_to,   tz="America/New_York") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    m1 = (df_1m.index >= start_ts) & (df_1m.index <= end_ts)
    m5 = (df_5m.index >= start_ts) & (df_5m.index <= end_ts)
    d1, d5 = df_1m[m1], df_5m[m5]
    # Include extended hours for Asia session (starts 6 PM or 8 PM ET)
    rth    = (d1.index.time >= dtime(3, 0)) & (d1.index.time <= dtime(16, 0))
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
    # EOD exit time depends on entry_mode
    eod = dtime(16, 0) if params.get('entry_mode') == 'ny_confirm' else dtime(9, 25)
    cfg = RunConfig(
        starting_capital=100_000,
        slippage_points=0.25,
        commission_per_contract=4.50,
        eod_exit_time=eod,
        params=params,
    )
    try:
        result = run_backtest(AsiaBreakoutStrategy, cfg, data)
    except Exception:
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

    n     = len(trades)
    avg_r = float(np.mean(r_mults))
    wins  = sum(1 for r in r_mults if r > 0)
    wr    = wins / len(r_mults)
    sl_cov = len(r_mults) / n
    tpd   = n / n_days
    net_pnl = sum(
        (t.exit_price - t.entry_price) * t.direction * t.contracts * 20.0
        - t.commission_per_contract * 2 * t.contracts
        for t in trades
    )
    return {"n": n, "wr": wr, "avg_r": avg_r, "sl_cov": sl_cov, "tpd": tpd, "net_pnl": net_pnl}


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

    # ── Parameter grid ──────────────────────────────────────────────────────
    combos = []

    # min_range_atr values scaled to 1-minute ATR (Asia range ÷ 1min-ATR ≈ 15–35x typically)
    # 0.0 = no lower filter, 10 = filter quiet nights, 15/20 = filter only active nights
    MIN_RNG_VALUES = [0.0, 10.0, 20.0]

    # Angle 1: Standard breakout at London open (3 AM)
    # sl_type='range': sl_atr_mult irrelevant — deduplicate by using only one mult value
    # sl_type='atr': wide mult values since 1-min ATR is small (2-4 pts on NQ)
    for asia_start, direction, rr, min_rng in itertools.product(
        [18, 20], ['long', 'short', 'momentum', 'contrarian'],
        [1.0, 1.5, 2.0], MIN_RNG_VALUES,
    ):
        combos.append({
            'entry_mode': 'breakout', 'asia_start_hour': asia_start, 'entry_hour': 3,
            'direction': direction, 'sl_type': 'range',
            'rr_ratio': rr, 'sl_atr_multiplier': 1.0, 'atr_period': 14,
            'min_range_atr': min_rng, 'max_range_atr': 200.0, 'risk_per_trade': 0.01,
        })

    for asia_start, direction, rr, sl_mult, min_rng in itertools.product(
        [18, 20], ['long', 'short', 'momentum', 'contrarian'],
        [1.0, 1.5, 2.0], [1.0, 3.0, 5.0], MIN_RNG_VALUES,
    ):
        combos.append({
            'entry_mode': 'breakout', 'asia_start_hour': asia_start, 'entry_hour': 3,
            'direction': direction, 'sl_type': 'atr',
            'rr_ratio': rr, 'sl_atr_multiplier': sl_mult, 'atr_period': 14,
            'min_range_atr': min_rng, 'max_range_atr': 200.0, 'risk_per_trade': 0.01,
        })

    # Angle 2: NY-open confirmation (9:30 AM market order if price already outside range)
    for asia_start, direction, rr, sl_mult, min_rng in itertools.product(
        [18, 20], ['long', 'short', 'momentum', 'contrarian'],
        [1.0, 1.5, 2.0], [1.0, 3.0, 5.0], MIN_RNG_VALUES,
    ):
        combos.append({
            'entry_mode': 'ny_confirm', 'asia_start_hour': asia_start, 'entry_hour': 3,
            'direction': direction, 'sl_type': 'atr',
            'rr_ratio': rr, 'sl_atr_multiplier': sl_mult, 'atr_period': 14,
            'min_range_atr': min_rng, 'max_range_atr': 200.0, 'risk_per_trade': 0.01,
        })

    # Angle 3: Fade (enter counter-trend relative to overnight momentum)
    for asia_start, direction, rr, sl_mult, min_rng in itertools.product(
        [18, 20], ['momentum', 'contrarian'],
        [1.0, 1.5, 2.0], [1.0, 3.0, 5.0], MIN_RNG_VALUES,
    ):
        combos.append({
            'entry_mode': 'fade', 'asia_start_hour': asia_start, 'entry_hour': 3,
            'direction': direction, 'sl_type': 'atr',
            'rr_ratio': rr, 'sl_atr_multiplier': sl_mult, 'atr_period': 14,
            'min_range_atr': min_rng, 'max_range_atr': 200.0, 'risk_per_trade': 0.01,
        })

    print(f"Sweeping {len(combos)} IS combos (max {MAX_WORKERS} workers)...\n", flush=True)

    # ── IS sweep ────────────────────────────────────────────────────────────
    is_rows = []
    done = 0

    def _run_is(params):
        return params, run_combo(data_is, n_is, params)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(_run_is, p): p for p in combos}
        for fut in as_completed(futures):
            params, r = fut.result()
            done += 1
            if r is not None and r["n"] >= MIN_TRADES and r["sl_cov"] >= 0.75:
                is_rows.append({**params, **r})
            if done % 100 == 0:
                elapsed = time.perf_counter() - t0
                eta = elapsed / done * (len(combos) - done) / 60
                print(f"  {done}/{len(combos)} done ... ETA {eta:.0f}min", flush=True)

    print(f"\n  {len(is_rows)} combos passed IS filter\n")

    if not is_rows:
        print("No IS combos passed. Exiting.")
        return

    df_is = pd.DataFrame(is_rows).sort_values("avg_r", ascending=False)

    print("=" * 120)
    print("TOP IS RESULTS (sorted by avg_r)")
    print("=" * 120)
    cols = ["entry_mode","direction","asia_start_hour","sl_type","rr_ratio","sl_atr_multiplier",
            "min_range_atr","n","tpd","wr","avg_r","net_pnl"]
    top = df_is.head(30).copy()
    top["wr"]      = top["wr"].map(lambda x: f"{x:.1%}")
    top["tpd"]     = top["tpd"].map(lambda x: f"{x:.3f}")
    top["avg_r"]   = top["avg_r"].map(lambda x: f"{x:+.4f}")
    top["net_pnl"] = top["net_pnl"].map(lambda x: f"${x:,.0f}")
    top.index = range(1, len(top) + 1)
    print(top[cols].to_string())
    print()

    # ── OOS validation ──────────────────────────────────────────────────────
    top_combos = df_is.head(TOP_N_OOS).to_dict("records")
    print(f"Validating top {TOP_N_OOS} on OOS ({OOS_FROM} to {OOS_TO})...\n")

    oos_rows = []

    def _run_oos(row):
        params = {k: row[k] for k in [
            'entry_mode','asia_start_hour','entry_hour','direction','sl_type',
            'rr_ratio','sl_atr_multiplier','atr_period','min_range_atr','max_range_atr','risk_per_trade'
        ]}
        r_oos = run_combo(data_oos, n_oos, params)
        key = (f"{row['entry_mode']:10s} dir={row['direction']:11s} "
               f"asia={row['asia_start_hour']:2.0f}h sl={row['sl_type']:5s} "
               f"rr={row['rr_ratio']:.1f} slm={row['sl_atr_multiplier']:.2f} "
               f"minR={row['min_range_atr']:.1f}")
        return row, r_oos, key

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        oos_futures = [ex.submit(_run_oos, row) for row in top_combos]
        for fut in as_completed(oos_futures):
            row, r_oos, key = fut.result()
            if r_oos is None:
                print(f"  {key}: NO OOS TRADES")
                continue
            oos_rows.append({
                "combo": key,
                "is_n": row["n"], "is_tpd": row["tpd"], "is_wr": row["wr"], "is_avg_r": row["avg_r"],
                "oos_n": r_oos["n"], "oos_tpd": r_oos["tpd"], "oos_wr": r_oos["wr"],
                "oos_avg_r": r_oos["avg_r"], "oos_net_pnl": r_oos["net_pnl"],
            })

    if not oos_rows:
        print("No OOS results.")
        return

    df_oos = pd.DataFrame(oos_rows).sort_values("oos_avg_r", ascending=False)

    print()
    print("=" * 135)
    print("IS vs OOS COMPARISON (sorted by OOS avg_r)")
    print("=" * 135)
    for _, row in df_oos.iterrows():
        print(
            f"  {row['combo']:65s} | "
            f"IS  n={row['is_n']:4.0f} tpd={row['is_tpd']:.3f} WR={row['is_wr']:.0%} avgR={row['is_avg_r']:+.4f} | "
            f"OOS n={row['oos_n']:4.0f} tpd={row['oos_tpd']:.3f} WR={row['oos_wr']:.0%} avgR={row['oos_avg_r']:+.4f} pnl=${row['oos_net_pnl']:+,.0f}"
        )

    print(f"\nTotal elapsed: {time.perf_counter()-t0:.1f}s")

    # ── Best OOS config → propfirm math ─────────────────────────────────────
    best = df_oos.iloc[0]
    if best["oos_avg_r"] > 0:
        print(f"\n{'='*70}")
        print("BEST OOS CONFIG — 25K LucidFlex × 4 ACCOUNTS")
        print(f"{'='*70}")
        tpd   = best["oos_tpd"]
        avg_r = best["oos_avg_r"]
        wr    = best["oos_wr"]
        print(f"  Config: {best['combo']}")
        print(f"  OOS: trades/day={tpd:.3f}  WR={wr:.1%}  avgR={avg_r:+.4f}")
        risk_per_trade = 250.0   # 1% of 25K
        ev_per_trade   = avg_r * risk_per_trade
        ev_per_day     = tpd * ev_per_trade
        days_to_pass   = 1250.0 / ev_per_day if ev_per_day > 0 else float("inf")
        funded_days    = max(0, 84 - days_to_pass)
        print(f"  EV/trade: ${ev_per_trade:.2f}  EV/day: ${ev_per_day:.2f}")
        print(f"  Days to pass eval: {days_to_pass:.0f}  |  Funded days: {funded_days:.0f}/84")
        if funded_days > 0:
            ev_funded = ev_per_day * funded_days
            print(f"  Expected funded PnL (1 acct): ${ev_funded:,.0f}")
            print(f"  4x accounts ($280 eval cost): net ${ev_funded*4 - 280:,.0f}")
    else:
        print("\n  WARNING: best OOS avg_r is negative.")


if __name__ == "__main__":
    main()
