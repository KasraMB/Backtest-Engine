"""
Quick sanity-check for the ML-collect speedup.

Runs N_CONFIGS configs through the same worker pool as run_ml_collect.py
and prints per-config wall times.

Expected behaviour (4 workers, after all Tier 1+2 optimisations):
  - First config per worker: ~30-40s (cold — Phase 1A + manip legs + Phase 2)
  - Configs 2+ in each worker:
      * same swing_n group: ~15-20s (swing cache hit + Phase 1A cache hit)
      * different swing_n:  ~20-30s (swing cache miss for this swing_n value)
  - Validation is skipped (validate=False) — saves ~8s per config vs standalone

Does NOT write any files — no dataset, no cache, no parquet.
"""
from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import time as dtime
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
N_CONFIGS   = 16      # 4 per worker — enough to see cold + multiple warm runs
N_WORKERS   = 4       # matches production
LHS_SEED    = 42

BASE_EXEC = dict(
    starting_capital         = 100_000,
    slippage_points          = 0.25,
    commission_per_contract  = 4.50,
    eod_exit_time            = dtime(17, 0),
    order_cancel_time        = dtime(11, 0),
)

CACHE_1M      = ROOT / "data" / "NQ_1m.parquet"
CACHE_5M      = ROOT / "data" / "NQ_5m.parquet"
CACHE_BAR_MAP = ROOT / "data" / "NQ_bar_map.npy"

# ---------------------------------------------------------------------------
# Worker globals
# ---------------------------------------------------------------------------
_g_data        = None
_g_exec_kwargs = None
_g_worker_id   = None


def _init_worker(
    cache_1m: str, cache_5m: str, cache_bar_map: str,
    exec_kwargs: dict,
) -> None:
    global _g_data, _g_exec_kwargs, _g_worker_id
    # Mirror run_ml_collect.py: 2 Phase 1 threads per worker (4 workers × 2 = 8 total)
    os.environ["BACKTEST_PHASE1_THREADS"] = "2"
    _g_worker_id = os.getpid()

    from backtest.data.loader import DataLoader
    from backtest.data.market_data import MarketData
    from backtest.ml.splits import filter_market_data

    df_1m   = pd.read_parquet(cache_1m)
    df_5m   = pd.read_parquet(cache_5m)
    bar_map = np.load(cache_bar_map)

    rth = (df_1m.index.time >= dtime(9, 30)) & (df_1m.index.time <= dtime(16, 0))
    a1  = {c: df_1m[c].to_numpy(dtype="float64") for c in ["open", "high", "low", "close", "volume"]}
    a5  = {c: df_5m[c].to_numpy(dtype="float64") for c in ["open", "high", "low", "close", "volume"]}

    loader    = DataLoader()
    full_data = MarketData(
        df_1m=df_1m, df_5m=df_5m,
        open_1m=a1["open"],  high_1m=a1["high"],  low_1m=a1["low"],
        close_1m=a1["close"], volume_1m=a1["volume"],
        open_5m=a5["open"],  high_5m=a5["high"],  low_5m=a5["low"],
        close_5m=a5["close"], volume_5m=a5["volume"],
        bar_map=bar_map,
        trading_dates=sorted(set(df_1m[rth].index.date)),
    )
    _g_data        = filter_market_data(full_data, "train", loader)
    _g_exec_kwargs = exec_kwargs


def _task_run_config(args: tuple) -> dict:
    """Run one config, return timing + trade count."""
    from backtest.runner.runner import run_backtest
    from backtest.runner.config import RunConfig
    from strategies.ict_smc import ICTSMCStrategy

    config_idx, params = args

    config = RunConfig(
        starting_capital        = _g_exec_kwargs["starting_capital"],
        slippage_points         = _g_exec_kwargs["slippage_points"],
        commission_per_contract = _g_exec_kwargs["commission_per_contract"],
        eod_exit_time           = _g_exec_kwargs["eod_exit_time"],
        order_cancel_time       = _g_exec_kwargs["order_cancel_time"],
        params                  = {**params, "ml_model": None},
    )

    t0     = time.perf_counter()
    # validate=False mirrors run_ml_collect.py — saves ~8s per config
    result = run_backtest(ICTSMCStrategy, config, _g_data, validate=False)
    elapsed = time.perf_counter() - t0

    return {
        "config_idx": config_idx,
        "worker_id":  _g_worker_id,
        "elapsed":    elapsed,
        "n_trades":   len(result.trades),
        "swing_n":    params.get("swing_n", "?"),
    }


def main() -> None:
    if not CACHE_1M.exists():
        print(f"ERROR: {CACHE_1M} not found. Run run.py once to build the cache.")
        return

    from backtest.ml.configs import sample_configs, ROUND_RANGES
    ranges      = ROUND_RANGES[1]
    all_configs = sample_configs(N_CONFIGS, ranges, seed=LHS_SEED)

    print("=== ML Collect Speedup Test ===")
    print(f"Configs: {N_CONFIGS}   Workers: {N_WORKERS}  (validate=False)")
    print("Expected (Tier 1+2 optimisations active):")
    print("  Cold (first config per worker):  ~30-40s")
    print("  Warm (same swing_n group):        ~15-20s")
    print("  Warm (new swing_n value):         ~20-30s")
    print(f"  Total wall ≈ ceil({N_CONFIGS}/{N_WORKERS}) × ~20s avg ≈ "
          f"{-((-N_CONFIGS) // N_WORKERS) * 20:.0f}s\n")

    init_args = (
        str(CACHE_1M), str(CACHE_5M), str(CACHE_BAR_MAP),
        {k: v for k, v in BASE_EXEC.items()},
    )

    tasks = [(i, params) for i, params in enumerate(all_configs)]

    t_total = time.perf_counter()
    results = []

    with ProcessPoolExecutor(
        max_workers = N_WORKERS,
        initializer = _init_worker,
        initargs    = init_args,
    ) as pool:
        future_map = {pool.submit(_task_run_config, task): task for task in tasks}
        done = 0
        for fut in as_completed(future_map):
            try:
                r = fut.result()
            except Exception as exc:
                task = future_map[fut]
                print(f"  [error] config {task[0]}: {exc}")
                continue
            results.append(r)
            done += 1
            print(f"  config {r['config_idx']:>2}  "
                  f"worker={r['worker_id']}  "
                  f"elapsed={r['elapsed']:>5.1f}s  "
                  f"trades={r['n_trades']:>4}  "
                  f"swing_n={r['swing_n']}")

    wall = time.perf_counter() - t_total

    elapsed_vals = sorted(r["elapsed"] for r in results)
    print(f"\n--- Summary ---")
    print(f"Wall clock ({N_CONFIGS} configs, {N_WORKERS} workers): {wall:.1f}s")
    print(f"Per-config: min={elapsed_vals[0]:.1f}s  "
          f"median={elapsed_vals[len(elapsed_vals)//2]:.1f}s  "
          f"max={elapsed_vals[-1]:.1f}s")

    # Extrapolate to full 150-config ML collect
    avg = sum(r["elapsed"] for r in results) / len(results)
    per_worker = -((-150) // N_WORKERS)   # ceil(150 / N_WORKERS)
    extrapolated = per_worker * avg
    print(f"\nExtrapolated to 150 configs / {N_WORKERS} workers:")
    print(f"  avg per config = {avg:.1f}s")
    print(f"  configs per worker = {per_worker}")
    print(f"  estimated wall ≈ {extrapolated:.0f}s = {extrapolated/60:.1f} min")


if __name__ == "__main__":
    main()
