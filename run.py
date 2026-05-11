# import subprocess
# import sys

# _DEPS = ["pandas", "numpy", "pytest", "plotly", "hmmlearn", "pyarrow", "fastparquet"]

# def _ensure_deps():
#     import importlib
#     _pkg_map = {"pytest": "pytest", "plotly": "plotly", "hmmlearn": "hmmlearn",
#                 "pyarrow": "pyarrow", "fastparquet": "fastparquet",
#                 "pandas": "pandas", "numpy": "numpy"}
#     missing = [pkg for mod, pkg in _pkg_map.items()
#                if importlib.util.find_spec(mod) is None]
#     if missing:
#         print(f"Installing missing dependencies: {', '.join(missing)}")
#         subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)

# _ensure_deps()

import os
import time
import numpy as np
import pandas as pd
from backtest.data.loader import DataLoader
from backtest.runner.runner import run_backtest, reverse_trades
from backtest.runner.config import RunConfig
from backtest.performance.engine import PerformanceEngine
from backtest.propfirm.lucidflex import (
    LUCIDFLEX_ACCOUNTS, run_propfirm_grid,
)
from backtest.performance.tearsheet import TearsheetRenderer
from backtest.regime.hmm      import fit_regimes
from backtest.regime.analysis import run_regime_analysis
from datetime import time as dtime

# ── Strategies — import all, pick one below ────────────────────────────────
from strategies.asia_breakout_strategy          import AsiaBreakoutStrategy

# ── Active strategy ────────────────────────────────────────────────────────
# Change this one line to switch strategies:
STRATEGY = AsiaBreakoutStrategy

def _step(label):
    """Context manager that prints step name and elapsed time."""
    import contextlib
    @contextlib.contextmanager
    def _ctx():
        print(f"{label}...", flush=True)
        t0 = time.perf_counter()
        yield
        print(f"  done  ({time.perf_counter() - t0:.2f}s)\n", flush=True)
    return _ctx()


if __name__ == "__main__":
    _t_total = time.perf_counter()
    # ── Config ─────────────────────────────────────────────────────────────────
    AUTO_OPEN_TEARSHEET = True

    # ── Date range filter ──────────────────────────────────────────────────────
    # Set to None to use all available data, or "YYYY-MM-DD" to restrict the range.
    DATE_FROM = "2019-01-01"   # IS period
    DATE_TO   = "2022-12-31"

    # ── Regime / HMM config ────────────────────────────────────────────────────
    # HMM_ENABLED      : run regime detection and add regime section to tearsheet
    # HMM_TRAIN_RATIO  : fraction of backtest period used to train the HMM
    #                    e.g. 0.5 = first 50% trains, second 50% is out-of-sample
    #                    Adjusts automatically to whatever DATE_FROM/DATE_TO is set.
    # HMM_N_STATES     : number of hidden states (3 = bear/neutral/bull)
    # REGIME_FILTER    : None (no filter) or list of regimes to trade in
    #                    e.g. ["bull"] or ["bull", "neutral"]
    #                    When set, a second filtered backtest result is computed.
    HMM_ENABLED     = True
    HMM_TRAIN_RATIO = 0.5
    HMM_N_STATES    = 2
    REGIME_FILTER   = None   # e.g. ["bull", "neutral"]

    # ── Load Data ──────────────────────────────────────────────────────────────
    CACHE_1M      = "data/NQ_1m.parquet"
    CACHE_5M      = "data/NQ_5m.parquet"
    CACHE_BAR_MAP = "data/NQ_bar_map.npy"

    loader = DataLoader()

    if os.path.exists(CACHE_1M) and os.path.exists(CACHE_5M) and os.path.exists(CACHE_BAR_MAP):
        with _step("Loading data from cache"):
            df_1m   = pd.read_parquet(CACHE_1M)
            df_5m   = pd.read_parquet(CACHE_5M)
            bar_map = np.load(CACHE_BAR_MAP)

            from backtest.data.market_data import MarketData
            arrays_1m = {col: df_1m[col].to_numpy(dtype="float64") for col in ["open","high","low","close","volume"]}
            arrays_5m = {col: df_5m[col].to_numpy(dtype="float64") for col in ["open","high","low","close","volume"]}
            # Filter to RTH dates only (bars between 09:30–17:00 ET) so overnight
            # globex sessions don't inflate the trading day count.
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
        with _step("Loading from raw files (first time — building cache)"):
            data = loader.load(path_1m="NQ_1m.txt", path_5m="NQ_5m.txt")
            data.df_1m.to_parquet(CACHE_1M)
            data.df_5m.to_parquet(CACHE_5M)
            np.save(CACHE_BAR_MAP, data.bar_map)
            print("  Cache saved — future loads will be near-instant.")

    print(f"  {len(data.df_1m):,} 1m bars | {len(data.df_5m):,} 5m bars")
    print(f"  {data.trading_dates[0]}  ->  {data.trading_dates[-1]}")
    print(f"  {len(data.trading_dates):,} trading days\n")

    # ── Apply date range filter ────────────────────────────────────────────────
    if DATE_FROM is not None or DATE_TO is not None:
        from backtest.data.market_data import MarketData
        import pandas as pd

        start_ts = pd.Timestamp(DATE_FROM, tz="America/New_York") if DATE_FROM else data.df_1m.index[0]
        end_ts   = pd.Timestamp(DATE_TO,   tz="America/New_York") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1) \
                   if DATE_TO else data.df_1m.index[-1]

        mask_1m = (data.df_1m.index >= start_ts) & (data.df_1m.index <= end_ts)
        mask_5m = (data.df_5m.index >= start_ts) & (data.df_5m.index <= end_ts)

        df_1m_f = data.df_1m[mask_1m]
        df_5m_f = data.df_5m[mask_5m]

        rth_mask_f    = (df_1m_f.index.time >= dtime(9, 30)) & (df_1m_f.index.time <= dtime(16, 0))
        trading_dates_f = sorted(set(df_1m_f[rth_mask_f].index.date))

        arrays_1m_f = {col: df_1m_f[col].to_numpy(dtype="float64") for col in ["open","high","low","close","volume"]}
        arrays_5m_f = {col: df_5m_f[col].to_numpy(dtype="float64") for col in ["open","high","low","close","volume"]}

        # Rebuild bar_map for the filtered date range
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
        print(f"  Date filter applied: {DATE_FROM or 'start'} -> {DATE_TO or 'end'}")
        print(f"  {len(data.df_1m):,} 1m bars | {len(data.trading_dates):,} trading days\n")

    # ── Run Config ─────────────────────────────────────────────────────────────
    # Parameters match the PineScript "Double Session Sweep Strategy" defaults:
    #   atrLength = 14, atrMultiplier = 1.5
    #   useFixedContracts = true, fixedContracts = 5
    #   useFixedRR = false (TP = asiaFinalHigh/Low)
    #   forceCloseNY = true, nyCloseMinutes = 30 -> force close at 15:30 ET
    #
    # Rule-based filters (set to None to disable):
    #   atr_filter_mult   = 1.5 -> skip if today's ATR > 1.5× 20-day avg ATR
    #   range_filter_mult = 2.0 -> skip if Asia range > 2.0× 30-day median range
    config = RunConfig(
        starting_capital=100_000,
        slippage_points=0.25,
        commission_per_contract=4.50,
        eod_exit_time=dtime(23, 59),   # strategy handles its own session exits
        params={
            # Tuned: BOS + displacement≥2×ATR + ATR-10 + momentum-only + dynamic equity
            # NOTE: params were tuned on 2022-2025 (contaminated); IS=2019-2022, OOS=2023-2024
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

    # ── Backtest ───────────────────────────────────────────────────────────────
    with _step("Running backtest"):
        result = run_backtest(STRATEGY, config, data)
    result.print_summary()

    # ── Save trade log ─────────────────────────────────────────────────────────
    from backtest.performance.trade_log import save_trade_log
    _log_path = f"trade_logs/{STRATEGY.__name__}.csv"
    save_trade_log(result, data, _log_path)
    print(f"Trade log saved -> {_log_path}\n")

    # Reversed: same trades, direction + SL/TP exit reasons flipped — instant
    # Skip if the strategy uses trailing stops (reversal semantics break down)
    if result.uses_trailing_stop:
        print("Skipping reversed analysis — strategy uses trailing stops.")
        result_rev = None
    else:
        result_rev = reverse_trades(result)

    # ── Performance ────────────────────────────────────────────────────────────
    print("Computing performance metrics (normal)...")
    perf = PerformanceEngine().compute(result, data)

    if result_rev is not None:
        print("Computing performance metrics (reversed)...")
        perf_rev = PerformanceEngine().compute(result_rev, data)
    else:
        perf_rev = None

    # ── Prop Firm Analysis ─────────────────────────────────────────────────────
    import time as _time
    prop_firm_results     = None
    prop_firm_results_rev = None

    # Only run if the strategy has SLs on at least 80% of trades.
    _trades_with_sl = sum(1 for t in result.trades if t.sl_price is not None)
    _sl_coverage    = _trades_with_sl / max(1, len(result.trades))

    if _sl_coverage >= 0.80:
        print(f"Computing prop firm analysis (LucidFlex) — {_sl_coverage:.0%} of trades have SL...")
        prop_firm_results = {}
        for acc_name, account in LUCIDFLEX_ACCOUNTS.items():
            print(f"  [{acc_name}] normal...", end=" ", flush=True)
            t0 = _time.perf_counter()
            prop_firm_results[acc_name] = run_propfirm_grid(
                trades=result.trades,
                account=account,
                n_sims=2_000,
                sizing_mode="micros",
            )
            print(f"{_time.perf_counter()-t0:.1f}s", end="")
            if result_rev is not None:
                print("  |  reversed...", end=" ", flush=True)
                t0 = _time.perf_counter()
                if prop_firm_results_rev is None:
                    prop_firm_results_rev = {}
                prop_firm_results_rev[acc_name] = run_propfirm_grid(
                    trades=result_rev.trades,
                    account=account,
                    n_sims=2_000,
                    sizing_mode="micros",
                )
                print(f"{_time.perf_counter()-t0:.1f}s")
            else:
                print()
    else:
        print(f"Skipping prop firm analysis — only {_sl_coverage:.0%} of trades have an SL "
              f"(need ≥80%). Use a strategy with a stop loss (e.g. EnhancedORBStrategy).")

    # ── Regime Analysis ────────────────────────────────────────────────────────
    regime_analysis = None
    if HMM_ENABLED and len(result.trades) >= 20:
        print("Computing regime analysis (HMM)...")
        t0 = _time.perf_counter()

        # Build daily returns from the 1m data over the backtest period
        rth_mask = (
            (data.df_1m.index.time >= dtime(9, 30)) &
            (data.df_1m.index.time <= dtime(16, 0))
        )
        df_rth       = data.df_1m[rth_mask]
        daily_close  = df_rth["close"].resample("D").last().dropna()
        daily_log_ret= np.log(daily_close / daily_close.shift(1)).dropna()
        daily_dates  = [d.date() for d in daily_log_ret.index]
        daily_rets   = daily_log_ret.values.astype(np.float64)

        if len(daily_rets) >= 20:
            regime_result = fit_regimes(
                daily_returns = daily_rets,
                daily_dates   = daily_dates,
                n_states      = HMM_N_STATES,
                train_ratio   = HMM_TRAIN_RATIO,
                mode          = "rolling",
            )
            regime_analysis = run_regime_analysis(
                trades          = result.trades,
                regime_result   = regime_result,
                data            = data,
                allowed_regimes = REGIME_FILTER,
                n_permutations  = 5_000,
            )
            train_end = regime_result.train_end_date
            n_is  = sum(1 for t in result.trades
                        if data.df_1m.index[t.entry_bar].date() <= train_end)
            n_oos = len(result.trades) - n_is
            print(f"  Train end: {train_end}  |  IS trades: {n_is}  OOS trades: {n_oos}")
            if REGIME_FILTER:
                n_filt = len(regime_analysis.filtered_trades)
                print(f"  Filter {REGIME_FILTER}: {n_filt} trades kept "
                      f"({n_filt/max(1,len(result.trades)):.0%})")
            if regime_analysis.allowed_regimes:
                sig = "✓ significant" if regime_analysis.perm_pvalue < 0.05 else "✗ not significant"
                print(f"  Permutation p-value: {regime_analysis.perm_pvalue:.3f}  {sig}")
            print(f"  done  ({_time.perf_counter()-t0:.1f}s)\n")
        else:
            print("  Skipping HMM — not enough daily data\n")
    elif HMM_ENABLED:
        print("Skipping HMM — not enough trades\n")

    # ── Tearsheet ──────────────────────────────────────────────────────────────
    with _step("Rendering tearsheet"):
        TearsheetRenderer().render(
            perf,
            output_path="tearsheet.html",
            auto_open=AUTO_OPEN_TEARSHEET,
            reversed_results=perf_rev,
            prop_firm_results=prop_firm_results,
            prop_firm_results_rev=prop_firm_results_rev,
            regime_analysis=regime_analysis,
            run_result=result,
            run_result_rev=result_rev,
            market_data=data,
        )
    print(f"Total time: {time.perf_counter() - _t_total:.1f}s")
