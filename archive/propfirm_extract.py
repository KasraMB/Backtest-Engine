"""
propfirm_extract.py
───────────────────
Runs SessionMeanRevStrategy backtest, then sweeps all 4 account sizes through
run_propfirm_grid (n_sims=5000) and reports TOP 20 combinations by ev_per_day.
Results are saved to propfirm_analysis.pkl for later inspection.
"""
import os
import time
import pickle
import numpy as np
import pandas as pd
from datetime import time as dtime

from backtest.data.loader import DataLoader
from backtest.runner.runner import run_backtest
from backtest.runner.config import RunConfig
from backtest.performance.engine import PerformanceEngine
from backtest.propfirm.lucidflex import (
    LUCIDFLEX_ACCOUNTS,
    run_propfirm_grid,
    extract_normalised_trades,
    EVAL_RISK_PCT_OF_MLL,
    FUNDED_RISK_PCT_OF_MLL,
    RISK_GEOMETRIES,
)
from strategies.session_mean_rev import SessionMeanRevStrategy
from backtest.data.market_data import MarketData

# ── Config ─────────────────────────────────────────────────────────────────────
DATE_FROM = "2022-01-01"
DATE_TO   = "2025-06-01"

N_SIMS     = 5_000
SIZING_MODE = "micros"

def _step(label: str):
    import contextlib
    @contextlib.contextmanager
    def _ctx():
        print(f"{label}...", flush=True)
        t0 = time.perf_counter()
        yield
        print(f"  done  ({time.perf_counter() - t0:.2f}s)\n", flush=True)
    return _ctx()


def main():
    t_total = time.perf_counter()

    # ── Load data ───────────────────────────────────────────────────────────────
    CACHE_1M      = "data/NQ_1m.parquet"
    CACHE_5M      = "data/NQ_5m.parquet"
    CACHE_BAR_MAP = "data/NQ_bar_map.npy"

    loader = DataLoader()

    if os.path.exists(CACHE_1M) and os.path.exists(CACHE_5M) and os.path.exists(CACHE_BAR_MAP):
        with _step("Loading data from cache"):
            df_1m   = pd.read_parquet(CACHE_1M)
            df_5m   = pd.read_parquet(CACHE_5M)
            bar_map = np.load(CACHE_BAR_MAP)

            arrays_1m = {col: df_1m[col].to_numpy(dtype="float64") for col in ["open","high","low","close","volume"]}
            arrays_5m = {col: df_5m[col].to_numpy(dtype="float64") for col in ["open","high","low","close","volume"]}
            rth_mask      = (df_1m.index.time >= dtime(9, 30)) & (df_1m.index.time <= dtime(16, 0))
            trading_dates = sorted(set(df_1m[rth_mask].index.date))

            data = MarketData(
                df_1m=df_1m, df_5m=df_5m,
                open_1m=arrays_1m["open"], high_1m=arrays_1m["high"],
                low_1m=arrays_1m["low"],  close_1m=arrays_1m["close"],
                volume_1m=arrays_1m["volume"],
                open_5m=arrays_5m["open"], high_5m=arrays_5m["high"],
                low_5m=arrays_5m["low"],  close_5m=arrays_5m["close"],
                volume_5m=arrays_5m["volume"],
                bar_map=bar_map, trading_dates=trading_dates,
            )
    else:
        raise FileNotFoundError("Data cache not found. Run run.py first.")

    print(f"  {len(data.df_1m):,} 1m bars | {len(data.df_5m):,} 5m bars")
    print(f"  {data.trading_dates[0]}  ->  {data.trading_dates[-1]}")
    print(f"  {len(data.trading_dates):,} trading days\n")

    # ── Apply date range filter ─────────────────────────────────────────────────
    start_ts = pd.Timestamp(DATE_FROM, tz="America/New_York")
    end_ts   = pd.Timestamp(DATE_TO,   tz="America/New_York") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    mask_1m = (data.df_1m.index >= start_ts) & (data.df_1m.index <= end_ts)
    mask_5m = (data.df_5m.index >= start_ts) & (data.df_5m.index <= end_ts)

    df_1m_f = data.df_1m[mask_1m]
    df_5m_f = data.df_5m[mask_5m]

    rth_mask_f      = (df_1m_f.index.time >= dtime(9, 30)) & (df_1m_f.index.time <= dtime(16, 0))
    trading_dates_f = sorted(set(df_1m_f[rth_mask_f].index.date))

    arrays_1m_f = {col: df_1m_f[col].to_numpy(dtype="float64") for col in ["open","high","low","close","volume"]}
    arrays_5m_f = {col: df_5m_f[col].to_numpy(dtype="float64") for col in ["open","high","low","close","volume"]}
    bar_map_f   = loader._build_bar_map(df_1m_f, df_5m_f)

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
    print(f"  Date filter: {DATE_FROM} -> {DATE_TO}")
    print(f"  {len(data.df_1m):,} 1m bars | {len(trading_dates_f):,} trading days\n")

    # ── Run backtest ────────────────────────────────────────────────────────────
    config = RunConfig(
        starting_capital=100_000,
        slippage_points=0.25,
        commission_per_contract=4.50,
        eod_exit_time=dtime(23, 59),
        params={
            "atr_period":          10,
            "wick_threshold":      0.15,
            "rr_ratio":            1.25,
            "sl_atr_multiplier":   1.0,
            "risk_per_trade":      0.01,
            "equity_mode":         "dynamic",
            "starting_equity":     100_000,
            "point_value":         20.0,
            "require_bos":         True,
            "max_trades_per_day":  3,
            "disp_min_atr_mult":   2.0,
            "momentum_only":       True,
            "allowed_sessions":    ['NY'],
        },
    )

    with _step("Running SessionMeanRevStrategy backtest"):
        result = run_backtest(SessionMeanRevStrategy, config, data)

    print(f"  Trades: {len(result.trades)}")

    # ── trades_per_day estimate ─────────────────────────────────────────────────
    n_tdays    = len(trading_dates_f)
    trades_per_day = len(result.trades) / max(1, n_tdays)
    print(f"\n  trades_per_day estimate: {trades_per_day:.4f}")
    print(f"  (from {len(result.trades)} trades over {n_tdays} trading days)\n")

    # ── Check SL coverage ──────────────────────────────────────────────────────
    trades_with_sl = sum(1 for t in result.trades if t.sl_price is not None)
    sl_coverage    = trades_with_sl / max(1, len(result.trades))
    print(f"  SL coverage: {sl_coverage:.1%} ({trades_with_sl}/{len(result.trades)} trades have SL)\n")

    if sl_coverage < 0.80:
        print(f"WARNING: SL coverage {sl_coverage:.1%} < 80% — propfirm sim may be inaccurate!")

    # ── Run propfirm grid for all 4 account sizes ───────────────────────────────
    all_results: dict[str, dict] = {}

    for acc_name, account in LUCIDFLEX_ACCOUNTS.items():
        print(f"[{acc_name}] Running propfirm grid (n_sims={N_SIMS})...", flush=True)
        t0 = time.perf_counter()
        grid = run_propfirm_grid(
            trades=result.trades,
            account=account,
            n_sims=N_SIMS,
            sizing_mode=SIZING_MODE,
            n_trading_days=len(trading_dates_f),
        )
        elapsed = time.perf_counter() - t0
        print(f"  done ({elapsed:.1f}s)\n")
        all_results[acc_name] = grid

    # ── Save results ────────────────────────────────────────────────────────────
    save_payload = {
        "all_results": all_results,
        "trades_per_day": trades_per_day,
        "n_trades": len(result.trades),
        "n_trading_days": n_tdays,
        "sl_coverage": sl_coverage,
        "date_from": DATE_FROM,
        "date_to": DATE_TO,
        "n_sims": N_SIMS,
        "sizing_mode": SIZING_MODE,
        "eval_risk_pcts": EVAL_RISK_PCT_OF_MLL,
        "funded_risk_pcts": FUNDED_RISK_PCT_OF_MLL,
        "schemes": RISK_GEOMETRIES,
    }
    with open("propfirm_analysis.pkl", "wb") as f:
        pickle.dump(save_payload, f)
    print("Results saved to propfirm_analysis.pkl\n")

    # ── Flatten all combinations into a list ────────────────────────────────────
    rows = []
    for acc_name, grid in all_results.items():
        for scheme in RISK_GEOMETRIES:
            if scheme not in grid:
                continue
            scheme_data = grid[scheme]
            for erp in EVAL_RISK_PCT_OF_MLL:
                if erp not in scheme_data:
                    continue
                for frp in FUNDED_RISK_PCT_OF_MLL:
                    if frp not in scheme_data[erp]:
                        continue
                    cell = scheme_data[erp][frp]
                    if cell is None:
                        continue
                    ev_pd = cell.get("ev_per_day")
                    if ev_pd is None:
                        continue
                    rows.append({
                        "account":          acc_name,
                        "scheme":           scheme,
                        "eval_risk":        erp,
                        "funded_risk":      frp,
                        "ev_per_day":       ev_pd,
                        "pass_rate":        cell.get("pass_rate"),
                        "mean_withdrawal":  cell.get("mean_withdrawal"),
                        "optimal_k":        cell.get("optimal_k"),
                        "net_ev":           cell.get("net_ev"),
                    })

    if not rows:
        print("ERROR: No valid results found!")
        return

    df = pd.DataFrame(rows).sort_values("ev_per_day", ascending=False).reset_index(drop=True)

    # ── TOP 20 ──────────────────────────────────────────────────────────────────
    print("=" * 90)
    print("TOP 20 COMBINATIONS BY EV/DAY")
    print("=" * 90)
    top20 = df.head(20).copy()
    top20["ev_per_day"]      = top20["ev_per_day"].map(lambda x: f"${x:.2f}")
    top20["pass_rate"]       = top20["pass_rate"].map(lambda x: f"{x:.1%}" if x is not None else "N/A")
    top20["mean_withdrawal"] = top20["mean_withdrawal"].map(lambda x: f"${x:,.0f}" if x is not None else "N/A")
    top20["net_ev"]          = top20["net_ev"].map(lambda x: f"${x:,.0f}" if x is not None else "N/A")
    top20["eval_risk"]       = top20["eval_risk"].map(lambda x: f"{x:.0%}")
    top20["funded_risk"]     = top20["funded_risk"].map(lambda x: f"{x:.0%}")
    top20.index = range(1, len(top20) + 1)
    print(top20.to_string())

    # ── Best combination ─────────────────────────────────────────────────────────
    best = df.iloc[0]
    print(f"\n{'=' * 90}")
    print(f"BEST COMBINATION: {best['account']} | {best['scheme']} | "
          f"eval_risk={best['eval_risk']:.0%} | funded_risk={best['funded_risk']:.0%}")
    print(f"  ev_per_day=${best['ev_per_day']:.2f}  |  pass_rate={best['pass_rate']:.1%}  |  "
          f"mean_withdrawal=${best['mean_withdrawal']:,.0f}  |  optimal_k={best['optimal_k']}  |  net_ev=${best['net_ev']:,.0f}")

    # ── 3x3 neighborhood of best combination ────────────────────────────────────
    best_acc     = best["account"]
    best_scheme  = best["scheme"]
    best_erp     = best["eval_risk"]
    best_frp     = best["funded_risk"]

    erp_idx = EVAL_RISK_PCT_OF_MLL.index(best_erp)
    frp_idx = FUNDED_RISK_PCT_OF_MLL.index(best_frp)

    neighbor_erps = [EVAL_RISK_PCT_OF_MLL[i] for i in range(
        max(0, erp_idx - 1), min(len(EVAL_RISK_PCT_OF_MLL), erp_idx + 2)
    )]
    neighbor_frps = [FUNDED_RISK_PCT_OF_MLL[i] for i in range(
        max(0, frp_idx - 1), min(len(FUNDED_RISK_PCT_OF_MLL), frp_idx + 2)
    )]

    print(f"\n{'=' * 90}")
    print(f"3x3 NEIGHBORHOOD (account={best_acc}, scheme={best_scheme})")
    print(f"  Rows = eval_risk, Cols = funded_risk")
    print(f"  Values: ev_per_day (pass_rate)")
    print("=" * 90)

    grid_data = all_results[best_acc][best_scheme]

    # Header row
    col_header = "eval \\ funded"
    print(f"{'':20}", end="")
    for frp in neighbor_frps:
        print(f"  {frp:.0%}        ", end="")
    print()

    for erp in neighbor_erps:
        marker_e = " *" if erp == best_erp else "  "
        print(f"  {erp:.0%}{marker_e}              ", end="")
        for frp in neighbor_frps:
            cell = grid_data.get(erp, {}).get(frp)
            if cell:
                ev  = cell.get("ev_per_day", 0)
                pr  = cell.get("pass_rate", 0)
                marker_f = "*" if (erp == best_erp and frp == best_frp) else " "
                print(f"  ${ev:6.2f}({pr:.0%}){marker_f}", end="")
            else:
                print(f"  {'N/A':13}", end="")
        print()

    print(f"\nTotal elapsed: {time.perf_counter() - t_total:.1f}s")


if __name__ == "__main__":
    main()
