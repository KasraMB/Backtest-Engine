"""
Generate tearsheets for ML-filtered backtest runs.

Loads the pre-collected dataset (data/ml_dataset.parquet), scores every trade
with the ensemble model, then simulates a chronological equity curve that:
  - processes signals in entry_bar order
  - allows only one position at a time (enforced via exit_bar)
  - at each bar picks the highest-scoring signal above threshold (if any)
  - never peeks at future bars to decide whether to take a current signal

This matches how the strategy would run live with all LHS configs active
simultaneously, with the ML model acting as the real-time filter/arbiter.

Outputs:
  tearsheet_val.html           — validation period only (2023)
  tearsheet_train_val.html     — train + validation period (2019–2023)
"""
import json
import os
import time
from datetime import time as dtime
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.data.loader import DataLoader
from backtest.data.market_data import MarketData
from backtest.ml.features import ALL_FEATURE_NAMES
from backtest.ml.model import EnsembleMLModel
from backtest.ml.splits import TRAIN_START, TRAIN_END, VALIDATION_START, VALIDATION_END, TZ
from backtest.performance.engine import PerformanceEngine
from backtest.performance.tearsheet import TearsheetRenderer
from backtest.performance.trade_log import save_trade_log
from backtest.propfirm.lucidflex import LUCIDFLEX_ACCOUNTS, run_propfirm_grid
from backtest.regime.hmm import fit_regimes
from backtest.regime.analysis import run_regime_analysis
from backtest.runner.config import RunConfig
from backtest.runner.runner import RunResult, reverse_trades
from backtest.strategy.enums import ExitReason
from backtest.strategy.update import Trade, round_to_tick


CACHE_1M      = "data/NQ_1m.parquet"
CACHE_5M      = "data/NQ_5m.parquet"
CACHE_BAR_MAP = "data/NQ_bar_map.npy"
DATASET_PATH  = Path("data/ml_dataset.parquet")

STARTING_CAPITAL         = 100_000.0
SLIPPAGE_POINTS          = 0.25
COMMISSION_PER_CONTRACT  = 4.50

RUNS = [
    {"label": "val",       "split": "validation",            "out": "tearsheet_val.html"},
    {"label": "train_val", "split": ["train", "validation"], "out": "tearsheet_train_val.html"},
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

    rth_mask_f      = (df_1m_f.index.time >= dtime(9,30)) & (df_1m_f.index.time <= dtime(16,0))
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


def _simulate_filtered_trades(
    df_split: pd.DataFrame,
    ml_model: EnsembleMLModel,
    data: MarketData,
) -> tuple[list[Trade], list[float]]:
    """
    Score, threshold-filter, and chronologically simulate all trades in df_split.

    Rules (matching live multi-config operation):
      - Only one open position at a time; skip any signal whose entry_bar falls
        before or at the current position's exit_bar.
      - At each bar, if multiple signals pass the threshold, take the one with
        the highest ML score. This is NOT lookahead — same-bar signals are
        simultaneous.
      - Never defer a current bar's signal because a higher-scoring future signal
        exists (that would be lookahead).

    Returns (trades_in_chronological_order, bar_resolution_equity_curve).
    """
    if df_split.empty:
        n = len(data.df_1m)
        return [], [STARTING_CAPITAL] * (n + 1)

    X      = df_split[ALL_FEATURE_NAMES]
    scores = ml_model.predict_r(X)

    df = df_split.copy()
    df['_score'] = scores

    df = df[df['_score'] > ml_model.threshold].copy()
    if df.empty:
        n = len(data.df_1m)
        return [], [STARTING_CAPITAL] * (n + 1)

    # Sort: entry_bar ascending, score descending (same-bar tie-break: highest wins)
    df = df.sort_values(['entry_bar', '_score'], ascending=[True, False]).reset_index(drop=True)

    entry_bars = df['entry_bar'].values
    exit_bars  = df['exit_bar'].values

    taken_indices: list[int] = []
    in_position_until = -1
    i = 0
    n = len(df)

    while i < n:
        eb = int(entry_bars[i])
        if eb <= in_position_until:
            i += 1
            continue

        # All same-bar entries follow; df is sorted score-desc within a bar,
        # so index i is the best one at this bar.
        j = i + 1
        while j < n and int(entry_bars[j]) == eb:
            j += 1

        taken_indices.append(i)
        in_position_until = int(exit_bars[i])
        i = j

    if not taken_indices:
        n_bars = len(data.df_1m)
        return [], [STARTING_CAPITAL] * (n_bars + 1)

    df_taken = df.iloc[taken_indices]

    # Build Trade objects from dataset rows
    trades: list[Trade] = []
    for _, row in df_taken.iterrows():
        direction   = int(row['direction'])
        entry_price = float(row['entry_price'])
        exit_price  = float(row['exit_price'])
        sl_pts      = float(row['sl_pts'])
        tp_pts      = float(row['tp_pts'])
        sl_price    = round_to_tick(entry_price - direction * sl_pts)
        tp_price    = round_to_tick(entry_price + direction * tp_pts)
        r_multiple  = float(row['r_multiple'])

        trade = Trade(
            entry_bar               = int(row['entry_bar']),
            exit_bar                = int(row['exit_bar']),
            entry_price             = entry_price,
            exit_price              = exit_price,
            direction               = direction,
            contracts               = 1.0,
            slippage_points         = SLIPPAGE_POINTS,
            commission_per_contract = COMMISSION_PER_CONTRACT,
            exit_reason             = ExitReason.TP if r_multiple > 0 else ExitReason.SL,
            initial_sl_price        = sl_price,
            sl_price                = sl_price,
            initial_tp_price        = tp_price,
            tp_price                = tp_price,
        )
        trades.append(trade)

    # Build bar-resolution equity curve (step function at each exit)
    n_bars = len(data.df_1m)
    delta  = np.zeros(n_bars + 1, dtype=np.float64)
    for trade in trades:
        eb = min(trade.exit_bar, n_bars)
        delta[eb] += trade.net_pnl_dollars
    equity_curve = (STARTING_CAPITAL + np.cumsum(delta)).tolist()

    return trades, equity_curve


def _make_run_result(trades: list[Trade], equity_curve: list[float], label: str) -> RunResult:
    config = RunConfig(
        starting_capital        = STARTING_CAPITAL,
        slippage_points         = SLIPPAGE_POINTS,
        commission_per_contract = COMMISSION_PER_CONTRACT,
        eod_exit_time           = dtime(17, 0),
        params                  = {},
    )
    return RunResult(
        trades        = trades,
        equity_curve  = equity_curve,
        config        = config,
        strategy_name = f"ICTSMCStrategy_ML_{label}",
    )


def run_one(label, split, out, df_dataset, full_data, loader, ml_model):
    print(f"\n{'='*60}")
    print(f"  Run: {label}  (split={split})")
    print(f"{'='*60}\n")

    is_combined = isinstance(split, list)

    if is_combined:
        # Combined train+val: 2019-01-01 → 2023-12-31
        date_from = TRAIN_START
        date_to   = VALIDATION_END
        df_split  = df_dataset[df_dataset['split'].isin(split)].copy()

        # Validation entry_bar / exit_bar are indexed from 2023-01-01 (bar 0 = first 2023 bar).
        # Offset them so they're relative to the combined data start (2019-01-01).
        train_start_ts = pd.Timestamp(TRAIN_START, tz=TZ)
        train_end_ts   = pd.Timestamp(TRAIN_END,   tz=TZ) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        n_train_bars   = int(((full_data.df_1m.index >= train_start_ts) &
                              (full_data.df_1m.index <= train_end_ts)).sum())
        val_mask = df_split['split'] == 'validation'
        df_split.loc[val_mask, 'entry_bar'] += n_train_bars
        df_split.loc[val_mask, 'exit_bar']  += n_train_bars
    else:
        df_split  = df_dataset[df_dataset['split'] == split].copy()
        date_from = VALIDATION_START if split == 'validation' else TRAIN_START
        date_to   = VALIDATION_END   if split == 'validation' else TRAIN_END

    data = filter_data(full_data, loader, date_from, date_to)
    print(f"  Dataset rows: {len(df_split):,}  |  {len(data.trading_dates):,} trading days\n")

    with _step("Scoring and simulating ML-filtered trades"):
        trades, equity_curve = _simulate_filtered_trades(df_split, ml_model, data)

    result = _make_run_result(trades, equity_curve, label)
    result.print_summary()

    save_trade_log(result, data, f"trade_logs/ICTSMCStrategy_{label}.csv")

    # No trailing stops in this path
    result_rev = reverse_trades(result)

    print("Computing performance metrics...")
    perf     = PerformanceEngine().compute(result, data)
    perf_rev = PerformanceEngine().compute(result_rev, data) if result_rev is not None else None

    # Prop firm
    prop_firm_results = prop_firm_results_rev = None
    _trades_with_sl = sum(1 for t in result.trades if t.sl_price is not None)
    if result.trades and _trades_with_sl / max(1, len(result.trades)) >= 0.80:
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
        rth_mask    = (data.df_1m.index.time >= dtime(9,30)) & (data.df_1m.index.time <= dtime(16,0))
        df_rth      = data.df_1m[rth_mask]
        daily_close = df_rth["close"].resample("D").last().dropna()
        daily_log_ret = np.log(daily_close / daily_close.shift(1)).dropna()
        daily_dates   = [d.date() for d in daily_log_ret.index]
        daily_rets    = daily_log_ret.values.astype(np.float64)
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
    _t0 = time.perf_counter()

    if not DATASET_PATH.exists():
        print(f"ERROR: {DATASET_PATH} not found. Run run_ml_collect.py first.")
        raise SystemExit(1)

    df_dataset = pd.read_parquet(DATASET_PATH)
    if 'sl_pts' not in df_dataset.columns or 'tp_pts' not in df_dataset.columns:
        print("ERROR: dataset missing sl_pts/tp_pts. Re-run run_ml_collect.py first.")
        raise SystemExit(1)
    df_dataset['date'] = pd.to_datetime(df_dataset['date'])

    full_data, loader = load_full_data()

    ml_model = EnsembleMLModel.load("models/ict_smc_ensemble.pkl")
    ml_model.threshold = 0.0
    print(f"Ensemble model loaded  |  threshold: {ml_model.threshold:.4f}\n")

    for run in RUNS:
        run_one(
            label      = run["label"],
            split      = run["split"],
            out        = run["out"],
            df_dataset = df_dataset,
            full_data  = full_data,
            loader     = loader,
            ml_model   = ml_model,
        )

    print("\nDone. Tearsheets:")
    for run in RUNS:
        print(f"  {run['out']}")
    print(f"\nTotal time: {time.perf_counter() - _t0:.1f}s")
