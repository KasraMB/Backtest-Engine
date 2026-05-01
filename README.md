# NQ Futures Backtest Engine

Intraday and swing backtesting framework for NQ futures. Propfirm-aware: all sweep results are evaluated against LucidFlex eval/funded account structures via Monte Carlo simulation.

---

## Architecture

```
DataLoader → MarketData → run_backtest() → RunResult → PerformanceEngine → TearsheetRenderer
                                                   └→ PropfirmMC → EV/day
```

| Folder | Purpose |
|--------|---------|
| `backtest/` | Core engine — data loading, runner, order/position state, performance |
| `backtest/propfirm/` | LucidFlex Monte Carlo sim (micros sizing, Numba kernels) |
| `strategies/profitable/` | Tested strategies with confirmed positive propfirm EV |
| `strategies/no_edge/` | Tested strategies with zero positive EV across all param combos |
| `strategies/untested/` | Implemented but never swept through propfirm screen |
| `sweeps/` | All sweep scripts, logs, and results CSVs |
| `data/` | NQ 1m, 5m, daily parquet files |
| `models/` | Trained ML models (ICT/SMC ensemble) |
| `baseline/` | Trade hash snapshots for regression testing |
| `tearsheets/` | HTML tearsheets from past backtests |
| `archive/` | Old diagnostic and verification scripts |

---

## Entry Points

```bash
python run.py                    # full backtest → tearsheet
python run_ml_collect.py         # collect ML dataset
python run_ml_train.py           # train model → models/ict_smc.pkl
python run_ml_validate.py        # evaluate on validation split
python make_baseline.py --check  # verify trade hashes unchanged
python -m pytest tests/ -v       # run tests
```

---

## Strategy Results Summary

### Data splits
- **IS (in-sample):** 2019–2024 — used for sweeps and parameter selection
- **Soft OOS:** 2025 — used for stability checks in this project
- **Hard OOS:** 2026 — never touched, reserved for final evaluation

---

### Profitable Strategies (`strategies/profitable/`)

#### 1. TrendFollowing — `trend_following_strategy.py`
**Best IS EV: $31.35/day | Best 2025 OOS EV: $19.87/day**

D1 SMA crossover + N-bar range breakout entry. Enters long at RTH open (9:30) when SMA(fast) > SMA(slow) and prior close broke above the lookback-bar high. Wide ATR trailing stop (rarely triggers — trades exit EOD at 23:59). Edge is directional continuation in confirmed uptrends.

| Param | Best value |
|-------|-----------|
| sma_fast | 20 |
| sma_slow | 100 |
| lookback | 50 bars (D1) |
| atr_mult | 3.0 |

**IS year-by-year (6/6 profitable):**

| Year | Trades | WR | PnL |
|------|--------|----|-----|
| 2019 | 30 | 90% | +$18,855 |
| 2020 | 46 | 83% | +$48,676 |
| 2021 | 33 | 94% | +$54,083 |
| 2022 | 11 | 100% | +$44,431 |
| 2023 | 26 | 96% | +$67,111 |
| 2024 | 27 | 78% | +$58,517 |

**2025 OOS:** n=25, WR=92%, avgR=+0.109, PnL=+$57,650

**Best propfirm setup:** 150K account, floor_aware scheme, eval_risk=70%, funded_risk=100%
- IS EV: $31.35/day → $2,633/account per 84-day cycle
- OOS EV: $19.87/day → $1,669/account per 84-day cycle
- Pass rate: 99% IS / 91% OOS

---

#### 2. ORB ATR Breakout — `orb_atr_breakout_strategy.py`
**Best robust IS EV: $11.14/day | 2025 OOS: +$67,212 (+$262/day raw)**

Opening Range Breakout on M30. Enters when price breaks above/below the ORB high/low by ATR × multiplier. Optional D1 SMA(200) filter.

**IS best combo** (overfit — fails 2025): `atr_space=0.4, sl=0.7%, rr=1.0, SMA filter`
**Robust combo** (6/6 IS years, passes 2025): `atr_space=0.15, sl=0.3%, rr=2.0, no filter`

| Metric | IS best | Robust combo |
|--------|---------|-------------|
| IS EV/day | $13.34 | $11.14 |
| IS min year | -$2,537 (2019) | +$2,994 |
| 2025 PnL | -$13,405 | +$67,212 |
| Pass rate | 48% | 41% |

**Best propfirm setup (robust):** 25K fixed_dollar, eval=20%, funded=100%
- EV: $9.67/day → $812/account per 84-day cycle, 54% pass rate

---

#### 3. SessionMeanRev — `session_mean_rev.py`
**Best IS EV: ~$14.84/day (sweep3)**

Intraday mean reversion within NY session. Tested across sweeps 2–4 with breakeven stops, vol filters, and daily momentum filters.

Best config: rr=0.75, sl=1.5, disp=2.0, atr=10, NY-only.
**Best propfirm setup:** 25K account → EV ~$8.50/day

---

#### 4. ICT/SMC — `ict_smc.py`
**Best IS EV: positive with ML filter**

ICT/SMC entry model with optional ML ensemble filter. Pure strategy has low EV; ML-filtered version improves signal quality. Trained on 2019–2022, validated on 2023.

- Model: `models/ict_smc.pkl` (base), `models/ict_smc_ensemble.pkl`
- Threshold optimisation: `models/threshold_opt.json`

---

#### 5. MACD — `macd_strategy.py`
**Best IS EV: $5.55/day (marginal, sweep5)**

Positive but weak. Not yet validated OOS.

#### 6. Parabolic SAR — `parabolic_sar_strategy.py`
**Best IS EV: $1.04/day (marginal, sweep5)**

Marginally positive. Not yet validated OOS.

---

### No Edge (`strategies/no_edge/`)

| Strategy | Sweep | Best IS EV | Notes |
|----------|-------|-----------|-------|
| VWAPMeanRev | sweep6 | $-0.46/day | 0/2160 combos positive EV |
| IBS | sweep7 | $-14.21/day | 0/512 combos positive EV |
| TurtleSoup | sweep7 | $-1.41/day | 0/36 combos positive EV |
| AsiaBreakout | sweep_asia | negative | Year-inconsistent, fails propfirm screen |
| RSI | sweep5 | $-1.33/day | 0/363 combos positive EV |

---

### Untested (`strategies/untested/`)

Implemented but never swept through a full propfirm grid:
- `enhanced_orb.py` — ORB variant with additional filters
- `orb_stop_strategy.py` — earlier ORB implementation
- `london_breakout_strategy.py` — London session breakout
- `dual_thrust_strategy.py` — dual-thrust momentum
- `vwap_band_reversion.py` — VWAP band variant
- `overnight_momentum_rosa.py` / `overnight_reversal_cooc.py` — academic paper strategies
- `intraday_momentum_baltussen.py` / `intraday_momentum_jin.py` / `intraday_interval_momentum_huang.py` — academic intraday momentum
- `noise_boundary_momentum.py`, `shooting_star_strategy.py`, `heikin_ashi_strategy.py`, `bollinger_bands_strategy.py`, `awesome_oscillator_strategy.py`

---

## Propfirm Context

**Target:** $10,000 net profit in ≤84 calendar days, ≤$300 budget (LucidFlex)

**LucidFlex accounts:** 25K / 50K / 100K / 150K. All MC sims use MNQ micros sizing.

**EV/84-day by strategy (best OOS estimate):**

| Strategy | EV/day | EV/84d | Accounts for $10K |
|----------|--------|--------|-------------------|
| TrendFollowing | $19.87 | $1,669 | 6.0 |
| ORB (robust) | $9.67 | $812 | 12.3 |
| SessionMeanRev | ~$8.50 | ~$714 | ~14 |

**Within $300 budget (~2 accounts):** best realistic outcome is ~$3,300 with TrendFollowing + ORB running concurrently on separate accounts.

---

## Sweep History

| Sweep | Script | Strategies | Combos | Best EV |
|-------|--------|-----------|--------|---------|
| 1 | `sweep_propfirm.py` | ICT/SMC | ~500 | negative |
| 2 | `sweep_propfirm2.py` | SessionMeanRev | 180 | $16.30 |
| 3 | `sweep3_structural.py` | SessionMeanRev | 40 | $23.87 |
| 4 | `sweep4_aggressive.py` | SessionMeanRev | 192 | $19.92 |
| 5 | `sweep5_technical.py` | MACD / ParabolicSAR / RSI | 1009 | $5.55 |
| 6 | `sweep6_orb_vwap.py` | ORB + VWAP | 2544 | $13.34 |
| 7 | `sweep7_new_strategies.py` | IBS + TrendFollowing + TurtleSoup | 596 | $31.35 |

All results in `sweeps/sweep_results{1-7}/results.csv`.

---

## Code Notes

- `POINT_VALUE = 20.0`, `TICK_SIZE = 0.25` — NQ, hardcoded throughout
- Strategy contract: subclass `BaseStrategy`, implement `generate_signals`, `on_fill`, `manage_position`
- `signal_bar_mask()` — optional bool array hook; runner uses it as a pre-screen to skip flat bars (major sweep speedup)
- Module-level array caches in strategy files — keyed by `(id(data.open_1m), params)` so all sweep combos share one setup pass over the full dataset
- ML data splits: train=2019–2022, val=2023, test1/test2=never touch
