"""
Generate baseline trade records for regression testing.

Usage:
    python make_baseline.py          # generate baselines
    python make_baseline.py --check  # compare current output against baselines
"""
import json, hashlib, sys, time
import numpy as np
import pandas as pd
from datetime import time as dtime
from backtest.data.market_data import MarketData
from strategies.ict_smc import ICTSMCStrategy
from backtest.runner.runner import run_backtest
from backtest.runner.config import RunConfig

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _load_raw():
    df_1m   = pd.read_parquet("data/NQ_1m.parquet")
    df_5m   = pd.read_parquet("data/NQ_5m.parquet")
    bar_map = np.load("data/NQ_bar_map.npy")
    a1 = {c: df_1m[c].to_numpy(dtype="float64") for c in ["open","high","low","close","volume"]}
    a5 = {c: df_5m[c].to_numpy(dtype="float64") for c in ["open","high","low","close","volume"]}
    rth = (df_1m.index.time >= dtime(9, 30)) & (df_1m.index.time <= dtime(16, 0))
    tds = sorted(set(df_1m[rth].index.date))
    return df_1m, df_5m, bar_map, a1, a5, tds


def _make_data(df_1m, df_5m, bar_map, a1, a5, tds, d_from, d_to):
    s  = pd.Timestamp(d_from, tz="America/New_York")
    e  = pd.Timestamp(d_to,   tz="America/New_York") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    m1 = (df_1m.index >= s) & (df_1m.index <= e)
    m5 = (df_5m.index >= s) & (df_5m.index <= e)
    i1s = int(np.argmax(m1));  i1e = int(len(m1) - np.argmax(m1[::-1]))
    i5s = int(np.argmax(m5));  i5e = int(len(m5) - np.argmax(m5[::-1]))
    bm  = bar_map[i1s:i1e] - i5s
    td  = [d for d in tds if pd.Timestamp(d_from).date() <= d <= pd.Timestamp(d_to).date()]
    return MarketData(
        df_1m=df_1m.iloc[i1s:i1e], df_5m=df_5m.iloc[i5s:i5e],
        open_1m=a1["open"][i1s:i1e], high_1m=a1["high"][i1s:i1e],
        low_1m =a1["low"] [i1s:i1e], close_1m=a1["close"][i1s:i1e],
        volume_1m=a1["volume"][i1s:i1e],
        open_5m=a5["open"][i5s:i5e], high_5m=a5["high"][i5s:i5e],
        low_5m =a5["low"] [i5s:i5e], close_5m=a5["close"][i5s:i5e],
        volume_5m=a5["volume"][i5s:i5e],
        bar_map=bm, trading_dates=td,
    )


def _trades_to_records(trades, df_1m):
    out = []
    for t in trades:
        out.append({
            "entry_bar":   int(t.entry_bar),
            "entry_time":  str(df_1m.index[t.entry_bar]),
            "direction":   int(t.direction),
            "entry_price": float(t.entry_price),
            "exit_price":  float(t.exit_price),
            "exit_reason": str(t.exit_reason),
            "sl_price":    float(t.sl_price) if t.sl_price is not None else None,
            "tp_price":    float(t.tp_price) if t.tp_price is not None else None,
            "pnl_pts":     round(float((t.exit_price - t.entry_price) * t.direction), 4),
        })
    return out


def _hash(records):
    s = json.dumps(records, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(s.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Baseline run definitions
# ---------------------------------------------------------------------------

_POI_TYPES_ALL = [
    "OB","BB","FVG","IFVG","RB",
    "PDH","PDL",
    "Asia_H","Asia_L","London_H","London_L",
    "NYPre_H","NYPre_L","NYAM_H","NYAM_L",
    "NYLunch_H","NYLunch_L","NYPM_H","NYPM_L",
    "Daily_H","Daily_L","NDOG","NWOG",
]

RUNS = [
    {
        "label":       "train_5m_2019_2022",
        "description": "ML collect training period — 5m manip, STDV+SESSION_OTE, default params",
        "date_from":   "2019-01-01",
        "date_to":     "2022-12-31",
        "params": {
            "contracts": 1, "swing_n": 1,
            "cisd_min_series_candles": 2, "cisd_min_body_ratio": 0.5,
            "rb_min_wick_ratio": 0.3,
            "confluence_tolerance_atr_mult": 0.18,
            "level_penetration_atr_mult": 0.5,
            "min_rr": 5.0, "tick_offset_atr_mult": 0.035,
            "order_expiry_bars": 10, "session_level_validity_days": 2,
            "po3_lookback": 6, "po3_atr_mult": 0.95, "po3_atr_len": 14,
            "po3_band_pct": 0.3, "po3_vol_sens": 1.0,
            "po3_max_r2": 0.4, "po3_min_dir_changes": 2, "po3_min_candles": 3,
            "po3_max_accum_gap_bars": 10,
            "po3_min_manipulation_size_atr_mult": 0.0,
            "max_trades_per_day": 2,
            "manip_leg_timeframe": "5m", "manip_leg_swing_depth": 1,
            "validation_timeframes": {
                "OTE":         ["1m","5m","15m","30m"],
                "STDV":        ["5m","15m","30m"],
                "SESSION_OTE": ["1m","5m","15m","30m"],
            },
            "validation_poi_types": {
                "OTE":         _POI_TYPES_ALL,
                "STDV":        _POI_TYPES_ALL,
                "SESSION_OTE": _POI_TYPES_ALL,
            },
            "session_ote_anchors": [
                "PDH","PDL","Asia_H","Asia_L",
                "London_H","London_L","NYPre_H","NYPre_L","NYAM_H","NYAM_L",
            ],
            "cancel_pct_to_tp": 0.75, "min_ote_size_atr_mult": 10,
            "allowed_setup_types": ["STDV","SESSION_OTE"],
            "stdv_reverse": False,
        },
    },
    {
        "label":       "val_1m_2023",
        "description": "Tearsheets val period — 1m manip, OTE+STDV+SESSION_OTE, permissive params (no ML model)",
        "date_from":   "2023-01-01",
        "date_to":     "2023-12-31",
        "params": {
            "contracts": 1,
            "cisd_min_series_candles": 2, "cisd_min_body_ratio": 0.5,
            "rb_min_wick_ratio": 0.3,
            "confluence_tolerance_atr_mult": 0.28,
            "tp_confluence_tolerance_atr_mult": 0.28,
            "level_penetration_atr_mult": 0.75,
            "min_rr": 2.0, "po3_atr_mult": 1.3, "po3_band_pct": 0.5,
            "po3_vol_sens": 0.5,
            "po3_min_manipulation_size_atr_mult": 0.0,
            "min_ote_size_atr_mult": 0.0,
            "swing_n": 1, "manip_leg_timeframe": "1m", "manip_leg_swing_depth": 1,
            "max_ote_per_session": 3, "max_stdv_per_session": 3,
            "max_session_ote_per_session": 3,
            "max_trades_per_day": 10, "entry_end_min": 11 * 60,
            "tick_offset_atr_mult": 0.035, "order_expiry_bars": 10,
            "session_level_validity_days": 2,
            "po3_lookback": 6, "po3_atr_len": 14,
            "po3_max_r2": 0.4, "po3_min_dir_changes": 2, "po3_min_candles": 3,
            "po3_max_accum_gap_bars": 10,
            "validation_timeframes": {
                "OTE":         ["1m","5m","15m","30m"],
                "STDV":        ["5m","15m","30m"],
                "SESSION_OTE": ["1m","5m","15m","30m"],
            },
            "validation_poi_types": {
                "OTE":         _POI_TYPES_ALL,
                "STDV":        _POI_TYPES_ALL,
                "SESSION_OTE": _POI_TYPES_ALL,
            },
            "session_ote_anchors": [
                "PDH","PDL","Asia_H","Asia_L",
                "London_H","London_L","NYPre_H","NYPre_L","NYAM_H","NYAM_L",
            ],
            "cancel_pct_to_tp": 1.0,
            "allowed_setup_types": ["OTE","STDV","SESSION_OTE"],
            "stdv_reverse": False,
        },
    },
]


# ---------------------------------------------------------------------------
# Generate
# ---------------------------------------------------------------------------

def generate():
    import os
    os.makedirs("baseline", exist_ok=True)
    print("Loading data...")
    df_1m, df_5m, bar_map, a1, a5, tds = _load_raw()

    import subprocess
    commit = subprocess.check_output(
        ["git","rev-parse","--short","HEAD"], text=True
    ).strip()

    for run in RUNS:
        print(f"\nRunning {run['label']} ({run['date_from']} -> {run['date_to']})...")
        data   = _make_data(df_1m, df_5m, bar_map, a1, a5, tds, run["date_from"], run["date_to"])
        config = RunConfig(starting_capital=100_000, slippage_points=0.5,
                           commission_per_contract=4.50, params=run["params"])
        t0     = time.perf_counter()
        result = run_backtest(ICTSMCStrategy, config, data)
        elapsed = time.perf_counter() - t0

        records = _trades_to_records(result.trades, data.df_1m)
        h       = _hash(records)

        doc = {
            "label":       run["label"],
            "description": run["description"],
            "date_from":   run["date_from"],
            "date_to":     run["date_to"],
            "git_commit":  commit,
            "n_trades":    len(records),
            "trade_hash":  h,
            "elapsed_s":   round(elapsed, 1),
            "trades":      records,
        }
        path = f"baseline/{run['label']}.json"
        with open(path, "w") as f:
            json.dump(doc, f, indent=2)
        print(f"  {len(records)} trades | hash {h[:16]}... | {elapsed:.1f}s -> {path}")

    print("\nBaseline files written to baseline/")


# ---------------------------------------------------------------------------
# Check
# ---------------------------------------------------------------------------

def check():
    print("Loading data...")
    df_1m, df_5m, bar_map, a1, a5, tds = _load_raw()
    all_ok = True

    for run in RUNS:
        path = f"baseline/{run['label']}.json"
        if not __import__("os").path.exists(path):
            print(f"[MISSING] {path} — run without --check to generate")
            all_ok = False
            continue

        with open(path) as f:
            baseline = json.load(f)

        print(f"\nChecking {run['label']}...")
        data   = _make_data(df_1m, df_5m, bar_map, a1, a5, tds, run["date_from"], run["date_to"])
        config = RunConfig(starting_capital=100_000, slippage_points=0.5,
                           commission_per_contract=4.50, params=run["params"])
        t0     = time.perf_counter()
        result = run_backtest(ICTSMCStrategy, config, data)
        elapsed = time.perf_counter() - t0

        records = _trades_to_records(result.trades, data.df_1m)
        h       = _hash(records)

        if h == baseline["trade_hash"]:
            print(f"  [OK] {len(records)} trades | hash {h[:16]}... | {elapsed:.1f}s")
        else:
            all_ok = False
            print("  [FAIL] hash mismatch!")
            print(f"    baseline : {baseline['n_trades']} trades | {baseline['trade_hash'][:16]}...")
            print(f"    current  : {len(records)} trades | {h[:16]}...")
            # Show first differing trade
            bt = baseline["trades"]
            ct = records
            for i, (b, c) in enumerate(zip(bt, ct)):
                if b != c:
                    print(f"    First diff at trade {i}:")
                    for k in b:
                        if b[k] != c.get(k):
                            print(f"      {k}: baseline={b[k]}  current={c.get(k)}")
                    break
            if len(bt) != len(ct):
                print(f"    Trade count: baseline={len(bt)}  current={len(ct)}")

    print()
    if all_ok:
        print("All baselines match.")
    else:
        print("BASELINE MISMATCH — trades have changed.")
        sys.exit(1)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if "--check" in sys.argv:
        check()
    else:
        generate()
