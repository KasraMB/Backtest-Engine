"""
Generate tearsheets for ML-filtered backtest runs.

Outputs:
  tearsheet_val.html           — validation period only (2023)
  tearsheet_train_val.html     — train + validation period (2019–2023)
"""
import os
import time
import numpy as np
import pandas as pd
from datetime import time as dtime

from backtest.data.loader import DataLoader
from backtest.data.market_data import MarketData
from backtest.runner.runner import run_backtest, reverse_trades
from backtest.runner.config import RunConfig
from backtest.performance.engine import PerformanceEngine
from backtest.performance.tearsheet import TearsheetRenderer
from backtest.performance.trade_log import save_trade_log
from backtest.ml.model import MLModel
from backtest.propfirm.lucidflex import LUCIDFLEX_ACCOUNTS, run_propfirm_grid
from backtest.regime.hmm import fit_regimes
from backtest.regime.analysis import run_regime_analysis
from strategies.ict_smc import ICTSMCStrategy


CACHE_1M      = "data/NQ_1m.parquet"
CACHE_5M      = "data/NQ_5m.parquet"
CACHE_BAR_MAP = "data/NQ_bar_map.npy"

RUNS = [
    {"label": "val",       "date_from": "2023-01-01", "date_to": "2023-12-31",  "out": "tearsheet_val.html"},
    {"label": "train_val", "date_from": "2019-01-01", "date_to": "2023-12-31",  "out": "tearsheet_train_val.html"},
]


def _step(label):
    import contextlib
    @contextlib.contextmanager
    def _ctx():
        print(f"{label}...", flush=True)
        t0 = time.perf_counter()
        yield
        print(f"  done  ({time.perf_counter() - t0:.2f}s)\n", flush=True)
    return _ctx()


def load_full_data():
    loader = DataLoader()
    if os.path.exists(CACHE_1M) and os.path.exists(CACHE_5M) and os.path.exists(CACHE_BAR_MAP):
        with _step("Loading data from cache"):
            df_1m   = pd.read_parquet(CACHE_1M)
            df_5m   = pd.read_parquet(CACHE_5M)
            bar_map = np.load(CACHE_BAR_MAP)
            arrays_1m = {c: df_1m[c].to_numpy(dtype="float64") for c in ["open","high","low","close","volume"]}
            arrays_5m = {c: df_5m[c].to_numpy(dtype="float64") for c in ["open","high","low","close","volume"]}
            rth_mask      = (df_1m.index.time >= dtime(9,30)) & (df_1m.index.time <= dtime(16,0))
            trading_dates = sorted(set(df_1m[rth_mask].index.date))
            data = MarketData(
                df_1m=df_1m, df_5m=df_5m,
                open_1m=arrays_1m["open"], high_1m=arrays_1m["high"],
                low_1m=arrays_1m["low"],   close_1m=arrays_1m["close"],
                volume_1m=arrays_1m["volume"],
                open_5m=arrays_5m["open"], high_5m=arrays_5m["high"],
                low_5m=arrays_5m["low"],   close_5m=arrays_5m["close"],
                volume_5m=arrays_5m["volume"],
                bar_map=bar_map, trading_dates=trading_dates,
            )
    else:
        with _step("Loading from raw files (building cache)"):
            data = loader.load(path_1m="NQ_1m.txt", path_5m="NQ_5m.txt")
            data.df_1m.to_parquet(CACHE_1M)
            data.df_5m.to_parquet(CACHE_5M)
            np.save(CACHE_BAR_MAP, data.bar_map)
    return data, loader


def filter_data(data, loader, date_from, date_to):
    start_ts = pd.Timestamp(date_from, tz="America/New_York")
    end_ts   = pd.Timestamp(date_to,   tz="America/New_York") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    mask_1m = (data.df_1m.index >= start_ts) & (data.df_1m.index <= end_ts)
    mask_5m = (data.df_5m.index >= start_ts) & (data.df_5m.index <= end_ts)

    df_1m_f = data.df_1m[mask_1m]
    df_5m_f = data.df_5m[mask_5m]

    rth_mask_f    = (df_1m_f.index.time >= dtime(9,30)) & (df_1m_f.index.time <= dtime(16,0))
    trading_dates_f = sorted(set(df_1m_f[rth_mask_f].index.date))
    arrays_1m_f = {c: df_1m_f[c].to_numpy(dtype="float64") for c in ["open","high","low","close","volume"]}
    arrays_5m_f = {c: df_5m_f[c].to_numpy(dtype="float64") for c in ["open","high","low","close","volume"]}
    bar_map_f = loader._build_bar_map(df_1m_f, df_5m_f)

    return MarketData(
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


def make_config(ml_model):
    return RunConfig(
        starting_capital=100_000,
        slippage_points=0.5,
        commission_per_contract=4.50,
        eod_exit_time=dtime(17, 0),
        params={
            "contracts":                     1,
            "swing_n":                       1,
            "cisd_min_series_candles":       2,
            "cisd_min_body_ratio":           0.5,
            "rb_min_wick_ratio":             0.3,
            "confluence_tolerance_atr_mult":  0.18,
            "level_penetration_atr_mult":    0.5,
            "min_rr":                        5.0,
            "tick_offset_atr_mult":          0.035,
            "order_expiry_bars":             10,
            "session_level_validity_days":   2,
            "po3_lookback":                  6,
            "po3_atr_mult":                  0.95,
            "po3_atr_len":                   14,
            "po3_band_pct":                  0.3,
            "po3_vol_sens":                  1.0,
            "po3_max_r2":                    0.4,
            "po3_min_dir_changes":           2,
            "po3_min_candles":               3,
            "po3_max_accum_gap_bars":        10,
            "po3_min_manipulation_size_atr_mult": 0.0,
            "max_trades_per_day":            10,
            "manip_leg_timeframe":           '5m',
            "manip_leg_swing_depth":         1,
            "validation_timeframes": {
                "OTE":         ['1m', '5m', '15m', '30m'],
                "STDV":        ['5m', '15m', '30m'],
                "SESSION_OTE": ['1m', '5m', '15m', '30m'],
            },
            "validation_poi_types": {
                "OTE":         ['OB', 'BB', 'FVG', 'IFVG', 'RB',
                                 'PDH', 'PDL',
                                 'Asia_H', 'Asia_L', 'London_H', 'London_L',
                                 'NYPre_H', 'NYPre_L', 'NYAM_H', 'NYAM_L',
                                 'NYLunch_H', 'NYLunch_L', 'NYPM_H', 'NYPM_L',
                                 'Daily_H', 'Daily_L', 'NDOG', 'NWOG'],
                "STDV":        ['OB', 'BB', 'FVG', 'IFVG', 'RB',
                                 'PDH', 'PDL',
                                 'Asia_H', 'Asia_L', 'London_H', 'London_L',
                                 'NYPre_H', 'NYPre_L', 'NYAM_H', 'NYAM_L',
                                 'NYLunch_H', 'NYLunch_L', 'NYPM_H', 'NYPM_L',
                                 'Daily_H', 'Daily_L', 'NDOG', 'NWOG'],
                "SESSION_OTE": ['OB', 'BB', 'FVG', 'IFVG', 'RB',
                                 'PDH', 'PDL',
                                 'Asia_H', 'Asia_L', 'London_H', 'London_L',
                                 'NYPre_H', 'NYPre_L', 'NYAM_H', 'NYAM_L',
                                 'NYLunch_H', 'NYLunch_L', 'NYPM_H', 'NYPM_L',
                                 'Daily_H', 'Daily_L', 'NDOG', 'NWOG'],
            },
            "session_ote_anchors": [
                'PDH', 'PDL',
                'Asia_H', 'Asia_L', 'London_H', 'London_L',
                'NYPre_H', 'NYPre_L', 'NYAM_H', 'NYAM_L',
            ],
            "cancel_pct_to_tp":       1.0,
            "min_ote_size_atr_mult":  0.0,
            "allowed_setup_types":    ['OTE', 'STDV', 'SESSION_OTE'],
            "stdv_reverse":           False,
            "ml_model":               ml_model,
        },
    )


def run_one(label, date_from, date_to, out, full_data, loader, ml_model):
    print(f"\n{'='*60}")
    print(f"  Run: {label}  ({date_from} -> {date_to})")
    print(f"{'='*60}\n")

    data = filter_data(full_data, loader, date_from, date_to)
    print(f"  {len(data.df_1m):,} 1m bars | {len(data.trading_dates):,} trading days\n")

    config = make_config(ml_model)

    with _step("Running backtest"):
        result = run_backtest(ICTSMCStrategy, config, data)
    result.print_summary()

    save_trade_log(result, data, f"trade_logs/ICTSMCStrategy_{label}.csv")

    if result.uses_trailing_stop:
        result_rev = None
    else:
        result_rev = reverse_trades(result)

    print("Computing performance metrics...")
    perf     = PerformanceEngine().compute(result, data)
    perf_rev = PerformanceEngine().compute(result_rev, data) if result_rev is not None else None

    # Prop firm
    prop_firm_results = prop_firm_results_rev = None
    _trades_with_sl = sum(1 for t in result.trades if t.sl_price is not None)
    if _trades_with_sl / max(1, len(result.trades)) >= 0.80:
        import time as _t
        prop_firm_results = {}
        for acc_name, account in LUCIDFLEX_ACCOUNTS.items():
            print(f"  Prop firm [{acc_name}]...", end=" ", flush=True)
            t0 = _t.perf_counter()
            prop_firm_results[acc_name] = run_propfirm_grid(
                trades=result.trades, account=account, n_sims=2_000, sizing_mode="micros")
            print(f"{_t.perf_counter()-t0:.1f}s")
            if result_rev is not None:
                if prop_firm_results_rev is None:
                    prop_firm_results_rev = {}
                prop_firm_results_rev[acc_name] = run_propfirm_grid(
                    trades=result_rev.trades, account=account, n_sims=2_000, sizing_mode="micros")

    # Regime
    regime_analysis = None
    if len(result.trades) >= 20:
        import time as _t
        print("Computing regime analysis (HMM)...")
        t0 = _t.perf_counter()
        rth_mask = (data.df_1m.index.time >= dtime(9,30)) & (data.df_1m.index.time <= dtime(16,0))
        df_rth      = data.df_1m[rth_mask]
        daily_close = df_rth["close"].resample("D").last().dropna()
        daily_log_ret = np.log(daily_close / daily_close.shift(1)).dropna()
        daily_dates = [d.date() for d in daily_log_ret.index]
        daily_rets  = daily_log_ret.values.astype(np.float64)
        if len(daily_rets) >= 20:
            regime_result = fit_regimes(
                daily_returns=daily_rets, daily_dates=daily_dates,
                n_states=2, train_ratio=0.5, mode="rolling",
            )
            regime_analysis = run_regime_analysis(
                trades=result.trades, regime_result=regime_result,
                data=data, allowed_regimes=None, n_permutations=5_000,
            )
            print(f"  done  ({_t.perf_counter()-t0:.1f}s)\n")

    with _step(f"Rendering tearsheet -> {out}"):
        TearsheetRenderer().render(
            perf,
            output_path=out,
            auto_open=True,
            reversed_results=perf_rev,
            prop_firm_results=prop_firm_results,
            prop_firm_results_rev=prop_firm_results_rev,
            regime_analysis=regime_analysis,
            run_result=result,
            run_result_rev=result_rev,
            market_data=data,
        )


if __name__ == "__main__":
    full_data, loader = load_full_data()
    ml_model = MLModel.load("models/ict_smc.pkl")
    # Use the walk-forward threshold (unbiased — from OOS folds on train split only).
    # The saved threshold (0.2771) was tuned on validation which causes leakage.
    ml_model.threshold = -0.0772
    print(f"ML model loaded  |  threshold (WF): {ml_model.threshold:.4f}\n")

    for run in RUNS:
        run_one(
            label=run["label"],
            date_from=run["date_from"],
            date_to=run["date_to"],
            out=run["out"],
            full_data=full_data,
            loader=loader,
            ml_model=ml_model,
        )

    print("\nDone. Tearsheets:")
    for run in RUNS:
        print(f"  {run['out']}")
