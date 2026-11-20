# Futures Backtest Engine

Python software for intraday futures research, with bar-based execution, parameter sweeps, an optional ML pipeline, and account/reinvestment simulations. The main scripts use NQ one-minute and five-minute bars; some strategies accept instrument-specific settings for other contracts.

The project separates strategy signals from execution, risk, performance reporting, and account simulation. Results describe the configured model and input data. They do not establish a profitable live strategy.

## Capabilities

- Bar-based order handling with configurable commissions, slippage, tick size, stop/target exits, and session-end liquidation.
- A `BaseStrategy` interface with signal, fill, and position-management callbacks. Strategies can supply a `signal_bar_mask()` to skip irrelevant bars while flat.
- Parameter sweeps that reuse cached signals or trade arrays where the strategy and filter permit it, with checkpoints in selected sweep scripts.
- Walk-forward ML training, validation-based threshold selection, and optional HMM-based regime analysis.
- Monte Carlo evaluation/funded-account models and a reinvestment simulator in which accounts within a simulated path share market draws.

## Getting started

Run commands from the repository root in a virtual environment. For the core backtester and tests:

```bash
python -m venv .venv
# Activate .venv before continuing.
python -m pip install numpy pandas scipy numba pyarrow pytest
python -m pytest tests/test_core_objects.py tests/test_engine.py tests/test_runner.py -q
```

The selected tests exercise core objects, execution, and the runner with synthetic inputs. The broader suite includes strategy and research tests with additional dependencies or data requirements:

```bash
python -m pip install hmmlearn xgboost scikit-learn
python -m pytest tests/ -q
```

Dependency versions are not pinned in this repository. Record the Python and package versions used for any reported experiment. No fixed passing-test count is claimed here.

### Run a backtest with local data

Supply your own data. The loader accepts text exports with date/time and OHLCV fields and maps supported headers to the engine's column names. See [the loader](backtest/data/loader.py) and [data cleaning](backtest/data/cleaner.py) for parsing, timezone, and filtering behavior.

This example expects `data/NQ_1m.txt` and `data/NQ_5m.txt`. Adjust the filenames and date range to your data. The strategy is an existing example, not a recommendation or a claim of positive expected returns.

```python
from backtest.data.loader import DataLoader
from backtest.runner.config import RunConfig
from backtest.runner.runner import run_backtest
from strategies.no_edge.asia_breakout_strategy import AsiaBreakoutStrategy

data = DataLoader(data_dir="data").load(
    path_1m="NQ_1m.txt",
    path_5m="NQ_5m.txt",
    start_date="2022-01-01",
    end_date="2022-01-31",
)
config = RunConfig(
    starting_capital=100_000,
    slippage_points=0.25,
    commission_per_contract=1.75,
    fill_mode="next_open",
)
result = run_backtest(AsiaBreakoutStrategy, config, data=data)
print(f"Trades: {len(result.trades)}")
```

The example costs are illustrative. `slippage_points` is in price points per fill, and commission is charged per contract per side. Review strategy parameters and session settings before interpreting an output.

Historical driver scripts also use `data/NQ_1m.parquet`, `data/NQ_5m.parquet`, and `data/NQ_bar_map.npy`. Other strategies can require additional daily or auxiliary data. Data coverage depends on the files supplied; the repository does not guarantee a complete history through the present.

Some drivers retain imports from earlier strategy locations. For example, `run.py` imports `strategies.asia_breakout_strategy`, while the current file is under `strategies/no_edge/`. Update those imports and configure the inputs before using a historical driver. The direct API example above uses the current path.

## Execution assumptions

Market-order timing is configured in [RunConfig](backtest/runner/config.py):

- `signal_close` is the legacy default. Filling at the same bar's close can introduce lookahead when the signal requires that close to be known.
- `next_open` defers a market-order fill until the next bar's open. It addresses that timing assumption; it does not validate every feature, signal, or exit rule.

OHLC bars do not identify the complete intrabar price path, queue position, or available liquidity. Stop/target ordering and fills are model assumptions, particularly when both levels fall inside one bar. Inspect [the runner](backtest/runner/runner.py) and [execution engine](backtest/engine/) and test sensitivity to costs and fill settings.

## Research status and data separation

[The ML split definitions](backtest/ml/splits.py) currently label 2019-2022 as training, 2023 as validation, 2024 as `test1`, and 2025 onward as `test2`. These labels describe code partitions, not proof that those periods were untouched throughout development. Individual experiments can use different boundaries.

Some historical parameters were tuned using later dates. `run.py` explicitly notes tuning on 2022-2025 for one parameter set. Those dates cannot be treated as an independent holdout for that selection process. Historical result files should be read with their experiment history, costs, and configuration.

For a new evaluation, record the data version, information available at each decision, selection procedure, and permitted training/validation dates. Freeze the selected procedure before examining a final evaluation period, and document any later reuse. Broad parameter searches also require accounting for selection effects and dependence between observations.

The ML entry points are `run_ml_collect.py` and `run_ml_train.py`; collection produces `data/ml_dataset.parquet`, and training can save models under `models/`. They are research drivers with their own configuration and input requirements, rather than a single end-to-end validation guarantee.

## Account and reinvestment simulation

[`backtest/propfirm/lucidflex.py`](backtest/propfirm/lucidflex.py) contains four configured account sizes, evaluation/funded risk grids, drawdown and consistency logic, and payout accounting. These are coded assumptions and historical configurations, not a verified reproduction of every current firm rule or agreement. Check the implementation and configuration used by a particular run.

[`backtest/propfirm/correlated_mc.py`](backtest/propfirm/correlated_mc.py) compares account-opening schedules, concurrency limits, and cash reserves. Accounts within a simulated investor path share daily market draws; separate investor simulations use separate draws. HMM-conditioned resampling and these shared draws impose a particular dependence model, rather than establishing the true joint distribution of future returns.

Simulation pass rates, payouts, and cash outcomes depend on the supplied trade pools, costs, account rules, and sampling assumptions. They do not establish that a strategy's historical behavior will persist.

## Sweeps, performance, and checkpoints

The AMR sweep workflow separates structural backtests, post-processing of cached trades, account-risk selection, and reinvestment simulations. Filter reuse is appropriate only when the change does not require resimulating fills or position management.

The saved [AMR v2 sweep log](sweeps/logs/amr_v2_sweep.log) records 4,951 base configurations and 153 filter combinations, or 757,503 combinations. It reports Sweep B completing in **402.7 seconds**, with the best 10,000 results retained in a heap. Sweep A had already completed and was resumed from a checkpoint. The log does not supply a complete machine/environment record, so this is a historical workload example, not a portable or end-to-end benchmark.

Selected Monte Carlo kernels use Numba and `prange`. First-call compilation, input size, thread count, and output collection affect runtime. No universal speedup or near-linear core scaling is claimed.

Some sweep scripts write a temporary pickle, flush it, attempt `fsync`, then replace the checkpoint with `os.replace`. This reduces exposure to partial replacement; it does not guarantee recovery from every interruption, filesystem failure, or concurrent writer. See [`tmp_amr_v2_sweep.py`](tmp_amr_v2_sweep.py) for one implementation.

`python make_baseline.py --check` compares selected trade records with saved hashes when the required data and strategy imports are available. Matching a baseline detects output changes for those cases; it does not prove the strategy, execution model, or statistical evaluation is correct.

## Code map

| Path | Purpose |
| --- | --- |
| `backtest/data/` | Data loading, cleaning, and `MarketData` |
| `backtest/strategy/` | Strategy callbacks, orders, and position updates |
| `backtest/runner/` | Bar loop, configuration, and validation |
| `backtest/engine/` | Execution, accounts, and risk handling |
| `backtest/performance/` | Metrics, trade logs, and tearsheets |
| `backtest/ml/` | Features, split definitions, training, and evaluation |
| `backtest/regime/` | HMM and volatility-regime analysis |
| `backtest/propfirm/` | Account and reinvestment simulations |
| `strategies/` | Strategy implementations and historical experiments |
| `tests/` | Unit, synthetic execution, and research tests |
| `sweeps/logs/` | Saved experiment logs |

Strategy folder names such as `profitable` or `no_edge` are organizational labels from past research. They do not certify a strategy's present profitability or performance on independent data.
