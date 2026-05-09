# NQ Futures Backtest Engine

Intraday NQ futures backtesting framework with an ML signal pipeline and propfirm Monte Carlo simulation. Built for speed — parameter sweeps over tens of thousands of configs run on a single machine in hours, not days.

---

## Features

**Engine**
- 1-minute and 5-minute bar execution with realistic fill logic (slippage, commission, SL/TP, EOD exits)
- ATR-tiered position sizing — MNQ micros → NQ minis based on account risk target
- Numba-JIT signal kernels: pre-compute all signals in one forward pass, then `generate_signals()` is O(1) per bar
- `signal_bar_mask()` hook: strategy declares which bars are active, runner skips the rest — eliminates most of the per-bar Python overhead
- Module-level signal cache keyed by `(data_id, params)` — sweep combos sharing a dataset pay setup cost once

**ML Pipeline**
- Walk-forward cross-validation (2019–2022 train, 2023 validation, held-out test sets never touched)
- HMM regime detection feeds regime-conditional trade pools into the propfirm sim
- Ensemble model with sample weighting and threshold optimisation

**Propfirm Monte Carlo**
- Full LucidFlex eval + funded lifecycle simulation: MLL locking, consistency rule (52% cushion), payout triggering (5 profitable days), balance deduction after each payout
- Numba kernels with `prange` parallelize across N simulation paths
- Vectorised reinvestment MC — draws from pre-simulated lifecycle pools, no Python loop over sims
- Sweeps the full ERP × FRP risk grid (0.1–1.0 × 0.1–1.0, 100 combos) per config so optimal risk is found, not assumed

**Parameter Sweeps**
- Decomposed two-phase sweep: structural params first, filter params on survivors
- `ThreadPoolExecutor(max_workers=2)` for parallel backtests — Numba releases GIL so true parallel execution
- Checkpoint every 100 configs → resume after kill without losing progress
- Results auto-saved to `sweeps/logs/` (git-tracked) so collaborators can pull results without re-running

---

## Architecture

```
DataLoader → MarketData → run_backtest() → RunResult → PerformanceEngine → TearsheetRenderer
                                    └→ PropfirmMC (Numba) → EV/day, P($0), P(>$10K)
                    ↑
               ML Pipeline (walk-forward, HMM regime, ensemble filter)
```

| Module | Purpose |
|---|---|
| `backtest/data/` | Data loading, cleaning, MarketData container |
| `backtest/runner/` | Per-bar execution loop, fills, SL/TP logic |
| `backtest/engine/` | Position and order state machines |
| `backtest/performance/` | Metrics, tearsheet, trade log |
| `backtest/propfirm/` | LucidFlex Monte Carlo — eval and funded kernels |
| `backtest/ml/` | Dataset collection, features, walk-forward training |
| `backtest/regime/` | HMM regime detection |
| `strategies/` | Strategy implementations |
| `sweeps/` | Sweep scripts, logs, results |

---

## Quick Start

```bash
pip install pandas numpy pytest plotly hmmlearn pyarrow fastparquet scipy numba

python run.py                    # full backtest → tearsheet.html
python run_ml_collect.py         # collect ML dataset → data/ml_dataset.parquet
python run_ml_train.py           # train model → models/ict_smc.pkl
python run_ml_validate.py        # evaluate on validation split
python make_baseline.py --check  # verify trade hashes unchanged after any change
python -m pytest tests/ -v       # run tests

python build_md_cache.py         # build market data cache before running sweeps
```

---

## Writing a Strategy

Subclass `BaseStrategy` and implement three methods:

```python
class MyStrategy(BaseStrategy):
    def generate_signals(self, data: MarketData, i: int) -> Optional[Order]:
        # called each bar when flat — return Order or None
        ...

    def on_fill(self, position: OpenPosition, data: MarketData, i: int) -> None:
        # set SL/TP immediately after fill
        position.set_initial_sl_tp(sl_price=..., tp_price=...)

    def manage_position(self, data: MarketData, i: int,
                        position: OpenPosition) -> Optional[PositionUpdate]:
        # trail stops, partial exits, etc. — return PositionUpdate or None
        ...
```

Optionally override `signal_bar_mask(data)` to return a boolean array marking candidate bars — the runner skips all other bars entirely, giving a significant speedup on selective strategies.

---

## Data

- Instrument: NQ (E-mini Nasdaq 100 futures) — `POINT_VALUE = 20.0`, `TICK_SIZE = 0.25`
- Bars: 1-minute and 5-minute, 2019–present
- ML splits: train 2019–2022, validation 2023, test sets held out

---

## Propfirm Simulation

The `backtest/propfirm/` module simulates the full **LucidFlex** account lifecycle — eval phase (consistency rule, trailing MLL → lock), funded phase (payout triggers, balance deduction, up to 6 payouts), and a reinvestment loop that cascades payouts into new eval accounts over a configurable horizon.

ERP (eval risk %) and FRP (funded risk %) are swept across a full grid per strategy config rather than fixed, so optimal risk sizing is found rather than assumed. Configurable for any account size, budget, horizon, and profit goal.
