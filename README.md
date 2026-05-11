# NQ Futures Backtest Engine

Intraday NQ futures backtesting framework with an ML signal pipeline, prop firm Monte Carlo, and a correlated multi-account reinvestment simulator. Built for sweep-scale workloads — hundreds of thousands of configurations run on a single laptop in hours, not days.

---

## Highlights

**Engine**
- 1m + 5m bar execution with realistic fill logic (slippage, commission, SL/TP, EOD exits)
- Numba-JIT signal kernels: signals pre-computed in one forward pass, `generate_signals()` is O(1) per bar
- `signal_bar_mask()` hook lets strategies declare candidate bars — runner skips the rest
- Module-level signal cache keyed by `(data_id, params)` — sweep configs sharing a dataset pay setup once
- ATR-tiered position sizing — MNQ micros ↔ NQ minis based on risk target

**ML Pipeline**
- Walk-forward CV (2019–2022 train, 2023 validation, held-out test sets never touched)
- HMM regime detection feeds regime-conditional trade pools into the prop firm sim
- Ensemble model with sample weighting and threshold optimisation

**Prop Firm Monte Carlo**
- Full LucidFlex eval + funded lifecycle: trailing EOD MLL with Initial-Trail lock, consistency rule, 5-day payout cycle, payout cap, balance deduction
- Sweeps **all 4 account sizes** (25K / 50K / 100K / 150K) and the full 10×10 ERP × FRP grid per config
- **Decomposed optimization**: ERP picked to maximise pass rate; FRP picked to maximise median EV/day given that ERP
- 10× kernel call reduction by separating eval and funded sweeps (eval depends only on ERP, funded only on FRP)

**Correlated Reinvestment Simulator** (`backtest/propfirm/correlated_mc.py`)
- Replaces the independence assumption: accounts within one sim share daily market draws (correlated blowups), accounts across sims are independent
- Regime-aware PnL draws via fitted HMM transition matrix — every sim runs its own market path
- Account management strategy space: `greedy`, `on_fail`, `on_pass`, `on_close`, `on_payout`, `staggered` × `max_concurrent` × reserve cushion × stagger schedule
- Multi-config slots with `1/N` risk scaling and intraday conflict resolution by `entry_time_min`
- **Numba JIT kernel with `prange`** — ~15 µs per sim → 80 ms per combo at 5,000 sims; ~70× speedup over the vectorised NumPy baseline

**Parameter Sweeps**
- Decomposed: structural params (Sweep A) → filter params (Sweep B, post-processing) → risk × account (Phase 2) → reinvestment MC (Phase 3) → account management (final sweep)
- Sweep B applies skip / TP-beyond-fair filters as NumPy boolean masks on cached A trade arrays — **757k filter combos in seconds**, no new backtests
- Atomic checkpoints (tmp + `os.replace`) survive mid-write kills; resume picks up where it stopped
- Lazy `_parent` + `_indices` records: B keeps top-N by Sortino in a heap, never instantiates the full PnL arrays

---

## Architecture

```
DataLoader → MarketData → run_backtest() → RunResult → PerformanceEngine → TearsheetRenderer
                                                  │
                                                  ├→ Prop firm MC (Numba)         → EV/day, pass rate, payout
                                                  └→ Correlated reinvestment MC   → P($0), P(>budget), P(goal)
                              ↑
                         ML pipeline (walk-forward, HMM regime, ensemble filter)
```

| Module | Purpose |
|---|---|
| `backtest/data/` | Data loading, MarketData container |
| `backtest/runner/` | Per-bar execution loop, fills, SL/TP |
| `backtest/engine/` | Position and order state machines |
| `backtest/performance/` | Metrics, tearsheet, trade log |
| `backtest/propfirm/lucidflex.py` | Per-account Monte Carlo (eval + funded Numba kernels) |
| `backtest/propfirm/correlated_mc.py` | Multi-account correlated reinvestment sim, JIT'd |
| `backtest/ml/` | Walk-forward training, features, ensemble |
| `backtest/regime/` | HMM regime detection |
| `strategies/` | Strategy implementations |
| `sweeps/logs/` | Sweep results (git-tracked, shareable) |

---

## Quick Start

```bash
pip install pandas numpy pytest plotly hmmlearn pyarrow fastparquet scipy numba

python run.py                    # full backtest → tearsheet.html
python run_ml_collect.py         # ML dataset → data/ml_dataset.parquet
python run_ml_train.py           # train model → models/<name>.pkl
python make_baseline.py --check  # verify trade hashes unchanged
python -m pytest tests/ -v       # run tests (414 passing)
```

---

## Writing a Strategy

```python
class MyStrategy(BaseStrategy):
    def generate_signals(self, data: MarketData, i: int) -> Optional[Order]:
        # called each bar when flat — return Order or None
        ...

    def on_fill(self, position: OpenPosition, data: MarketData, i: int) -> None:
        position.set_initial_sl_tp(sl_price=..., tp_price=...)

    def manage_position(self, data: MarketData, i: int,
                        position: OpenPosition) -> Optional[PositionUpdate]:
        ...
```

Override `signal_bar_mask(data)` to declare candidate bars — the runner skips everything else.

---

## Data

- **Instrument**: NQ (E-mini Nasdaq 100 futures) — `POINT_VALUE = 20.0`, `TICK_SIZE = 0.25`
- **Bars**: 1m + 5m, 2019–present
- **ML splits**: train 2019–2022, validation 2023, test sets held out

---

## Prop Firm Pipeline

The full pipeline answers four questions in order:

1. **What strategy parameters work?** — Sweep A/B searches the config space, ranked by daily Sortino
2. **At what risk level?** — Phase 2 sweeps ERP × FRP × account size for the top survivors
3. **What survival rate?** — Phase 3 reinvestment MC on each survivor
4. **How should you manage multiple accounts?** — Correlated MC sweeps account management strategies × all pairs of top configs

LucidFlex MLL rules implemented faithfully: initial MLL = `start − mll_amount`, trails the EOD peak, caps and locks at `start + $100` once the Initial Trail Balance (`start + mll_amount + $100`) is crossed, and locks immediately on a payout request.

---

## Performance Notes

- All Monte Carlo kernels use `@njit(parallel=True, cache=True, fastmath=True)` — released GIL, vectorised across paths
- Correlated MC: per-sim state lives as `(n_sims, max_c)` NumPy arrays; `prange` across sims gives near-linear scaling on additional cores
- Sweep checkpoints are atomic (tmp + `os.replace` + `fsync`) — never corrupted on interrupt
- ML feature extraction uses a module-level cache so repeated calls during walk-forward training are instant
