# CLAUDE.md

## Commands

```bash
# Run all tests
python -m pytest tests/ -v

# Run the full backtest pipeline (outputs tearsheet.html + trade_logs/ICTSMCStrategy.csv)
python run.py

# ML pipeline (run in order)
python run_ml_collect.py   # multi-config data collection → data/ml_dataset.parquet
python run_ml_train.py     # walk-forward training → models/ict_smc.pkl
python run_ml_validate.py  # evaluate on validation split
```

No build step, no setup.py. Dependencies: `pandas numpy pytest plotly hmmlearn pyarrow fastparquet scipy`

## Architecture

This is an **intraday NQ futures backtesting framework**. The instrument is hard-coded throughout: `POINT_VALUE = 20.0`, `TICK_SIZE = 0.25` in `backtest/strategy/update.py`.

```
DataLoader → MarketData → run_backtest() → RunResult → PerformanceEngine → Results → TearsheetRenderer → tearsheet.html
                                                      └→ save_trade_log() → trade_logs/<strategy>.csv
```

`run.py` is the only entry point. Set `STRATEGY = <StrategyClass>` on line 33 to switch strategies.

### Data layer (`backtest/data/`)

`MarketData` holds both the Pandas DataFrames (`df_1m`, `df_5m`) and pre-extracted NumPy arrays (`open_1m`, `close_1m`, …). The hot loop uses the arrays exclusively for speed. `bar_map[i]` maps a 1m bar index to its parent 5m bar index.

First run reads raw NinjaTrader CSVs and writes parquet + npy caches; subsequent runs load from cache.

### Strategy contract (`backtest/strategy/`)

Strategies subclass `BaseStrategy` and implement three methods:

| Method | Called when | Returns |
|---|---|---|
| `generate_signals(data, i)` | Position is flat and `i >= min_lookback` | `Optional[Order]` |
| `on_fill(position, data, i)` | Order just filled | `None` (call `position.set_initial_sl_tp(sl, tp)` here) |
| `manage_position(data, i, position)` | Every bar while in position | `Optional[PositionUpdate]` |

`Order` supports `MARKET`, `LIMIT`, `STOP`, `STOP_LIMIT` types and `CONTRACTS`, `DOLLARS`, `PCT_RISK` sizing. Trailing stops via `trail_points` and `trail_activation_points`. `cancel_above`/`cancel_below` cancel a pending limit if price crosses those levels before fill (§9.3).

### Execution engine (`backtest/engine/`)

**Critical invariant:** SL always has priority over TP within a bar.

**Trail SL timing:** `tick_trail()` is called *after* `check_exits()` returns no exit. The trail SL used for a bar's exits is always the state from the end of the previous bar. Changing this ordering reintroduces a bug where the trail SL moves on the exit bar and triggers a false gap-open exit.

`apply_position_update()` enforces favorable-only SL moves and prevents SL/TP from crossing.

### Runner (`backtest/runner/runner.py`)

Per-bar order:
1. Tick pending order expiry
2. **§9.3 cancel check**: `cancel_above`/`cancel_below` threshold → cancel order (zone stays consumed)
3. Attempt fills for pending orders
4. On fill: open position, call `on_fill()`
5. Check same-bar exit (SL/TP on fill bar)
6. If position open: call `manage_position()`, apply `PositionUpdate`
7. Call `check_exits()` for SL/TP/EOD
8. If no exit: call `tick_trail()`

MAE/MFE tracked on `OpenPosition`, copied to `Trade` on close. `reverse_trades(result)` is a post-hoc flip — not a re-simulation. Skipped when `result.uses_trailing_stop` is True.

### Performance & tearsheet (`backtest/performance/`)

`PerformanceEngine.compute(result, data)` → `Results`. `TearsheetRenderer.render()` produces a self-contained HTML with Plotly. The Trade Inspector serializes OHLC windows per trade as `TC_WINDOWS`/`TC_WINDOWS_R` in the HTML; `trade_reason` appears as grey monospace text in the inspector.

`save_trade_log()` in `backtest/performance/trade_log.py` writes a CSV per trade.

### Prop-firm analysis (`backtest/propfirm/lucidflex.py`)

Vectorized Monte Carlo simulation of LucidFlex eval and funded phases. `run_propfirm_grid()` searches eval risk % × funded risk % to maximise `ev_per_day` at optimal payout count K.

### Regime analysis (`backtest/regime/`)

`fit_regimes()` trains a Gaussian HMM on daily log-returns. States sorted by mean return. `mode="rolling"` is forward-safe. `run_regime_analysis()` computes per-regime trade stats and a permutation p-value.

---

## ML Pipeline (`backtest/ml/`, `run_ml_*.py`)

### Data splits
train 2019–2022 (walk-forward) · validation 2023 (threshold tuning only) · test1/test2 **never touch**

### Collection (`run_ml_collect.py`)
LHS samples `N_CONFIGS` configs per round; appends to `ml_dataset.parquet`.

- Collects from **all configs**, not just sensitivity-passing ones — survivorship bias. `cfg_is_valid` / `cfg_base_metric` encode filter result as features.
- Sensitivity filter (`min_base_metric=0.05`, `max_degradation_pct=30%`) gates `validated_configs` for deployment only, not dataset inclusion.
- Perturbation runs (±15% on non-LHS params) are **currently inert** — all params are LHS axes in Round 1/2. Activate when a future round holds some params fixed outside LHS.

### Caches (`cache/`)
`sens_runs_cache.json` / `sensitivity_cache.json` — survive schema changes (backtests don't re-run, only dataset rebuild)
`trades_cache.json` / `val_trades_cache.json` — delete on schema change or full re-run

### Deferred — implement before perturbations activate
- **Round 2+ sample weights**: `weight = volume(round_N_space) / volume(round_1_space)`
- **Perturbation stability features**: `cfg_perturb_mean/worst/std_metric` on parent config (from `sens_runs_cache`)
- **Perturbation sample downweight**: `1 / (2 × n_non_lhs_params)` per perturb row

---

## ICT/SMC Strategy (`strategies/ict_smc.py`)

### Overview

Power of Three (PO3) framework on NQ 1m futures:
- **Phase 1** (once per day at first 09:30 bar): detects overnight accumulation zones on 5m, finds manipulation legs (swing → CISD), draws OTE/STDV fibs from the leg and SESSION_OTE fibs from session level anchors, validates each level against POI/swing confluence across timeframes
- **Phase 2** (09:30–11:00 ET, every bar while flat): monitors validated levels, detects AFZ entry pattern, places limit orders

### Key concepts

**OTE levels** (50/61.8/70.5/79%): retracement entries anchored to the manipulation leg's swing high/low.

**STDV extension levels**: where the distribution leg ends → reversal entries in the opposite direction to the manipulation leg.

**SESSION_OTE levels**: independent of manipulation legs. For each session level anchor (PDH, PDL, Asia H/L, London H/L, NYPre H/L, NYAM H/L): 100% = anchor price, 0% = overnight extreme (min for high anchors, max for low anchors). Formula: `level = extreme + f × (anchor - extreme)`. Pre-session touches (after the extreme forms, before 09:30) invalidate individual levels. During the session: price returning to the anchor invalidates the whole group; a new extreme recomputes all level prices and resets touched state.

**AFZ (Algorithmic Fibonacci Zone)**: 1m candle entry pattern. Zone identity = `furthest_bar` index. A zone is consumed once a limit order is placed from it.

**§9.3 TP-before-fill cancellation**: if price reaches TP before the limit fills, the order is cancelled. Zone stays consumed.

**ATR**: `_wilder_atr_scalar` uses a short rolling window (`start = max(0, i - period*4)`). This intentionally differs from a full-history recursive computation. **Do not replace with a full-history cached array** — it changes results.

**Resampling alignment**: 15m/30m bars resampled from 5m with a trim offset to align to clock boundaries: `trim = (N - (start_min % N)) % N // 5` bars trimmed from the front.

### Session level kind names

Session POIs use H/L split `session_kind` values: `PDH`/`PDL`, `Asia_H`/`Asia_L`, `London_H`/`London_L`, `NYPre_H`/`NYPre_L`, `NYAM_H`/`NYAM_L`, `NYLunch_H`/`NYLunch_L`, `NYPM_H`/`NYPM_L`, `Daily_H`/`Daily_L`, `NDOG`, `NWOG`. These are the strings to use in `validation_poi_types` and `session_ote_anchors`.

### Known performance bottlenecks

These are the measured/identified slow spots — do not introduce additional instances of these patterns:

1. **`_detect_fvg_vectorized` called 4×/day (1m, 5m, 15m, 30m)** — two Python loops over a growing `active` list per bar. The 1m call processes up to 2,500 bars; active list can reach 30–50 entries. Dominant Phase 1 cost.

2. **`_detect_ob_vectorized` called 4×/day** — same pattern, one Python loop over active OBs per bar.

3. **`_detect_accum_zones`** — tight Python loop where each iteration makes ~8 numpy calls on 6-element slices (`lb=6`). Python function call overhead dominates the actual arithmetic.

4. **`_fvg_step` called every bar while in a position** — Python loop over `fvg_active` (invalidation pass + RB detection pass). O(active_fvgs) per bar, 10–100 bars per trade.

5. **`data.df_1m.index[i].time()` in `runner.py` bar constructor** — pandas `Timestamp` allocation on every processed bar. The strategy already precomputes `_bar_times_min` as a numpy array; the runner does not use it.

6. **`_cisd_scan` in `_detect_manip_legs`** — all-Python O(n) scan per eligible swing, called up to `po3_max_accum_gap_bars` times per zone per scenario.
