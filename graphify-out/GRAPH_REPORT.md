# Graph Report - .  (2026-04-19)

## Corpus Check
- 80 files · ~198,322 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1515 nodes · 5808 edges · 43 communities detected
- Extraction: 40% EXTRACTED · 60% INFERRED · 0% AMBIGUOUS · INFERRED: 3506 edges (avg confidence: 0.56)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Strategy Execution Engine|Strategy Execution Engine]]
- [[_COMMUNITY_ML Training & Sensitivity|ML Training & Sensitivity]]
- [[_COMMUNITY_Performance & Tearsheet|Performance & Tearsheet]]
- [[_COMMUNITY_Trade Lifecycle & Position Mgmt|Trade Lifecycle & Position Mgmt]]
- [[_COMMUNITY_ML Ensemble & Evaluation|ML Ensemble & Evaluation]]
- [[_COMMUNITY_Data Loading & Cleaning|Data Loading & Cleaning]]
- [[_COMMUNITY_Parameter Config & LHS Sampling|Parameter Config & LHS Sampling]]
- [[_COMMUNITY_DoubleSweep Strategy|DoubleSweep Strategy]]
- [[_COMMUNITY_Core Runner & Strategy Contracts|Core Runner & Strategy Contracts]]
- [[_COMMUNITY_Project Architecture Docs|Project Architecture Docs]]
- [[_COMMUNITY_Propfirm Monte Carlo Sim|Propfirm Monte Carlo Sim]]
- [[_COMMUNITY_Core Object Tests|Core Object Tests]]
- [[_COMMUNITY_Strategy Validator Tests|Strategy Validator Tests]]
- [[_COMMUNITY_ML Feature Engineering|ML Feature Engineering]]
- [[_COMMUNITY_Trade Log & CSV Tests|Trade Log & CSV Tests]]
- [[_COMMUNITY_Reverse Trades & Runner Tests|Reverse Trades & Runner Tests]]
- [[_COMMUNITY_HMM Regime Detection|HMM Regime Detection]]
- [[_COMMUNITY_TrainValTest Splits|Train/Val/Test Splits]]
- [[_COMMUNITY_Core Order & Position Types|Core Order & Position Types]]
- [[_COMMUNITY_Risk Manager & Sizing|Risk Manager & Sizing]]
- [[_COMMUNITY_Statistical Analysis|Statistical Analysis]]
- [[_COMMUNITY_Trade Chart Visualization|Trade Chart Visualization]]
- [[_COMMUNITY_Order Validation Logic|Order Validation Logic]]
- [[_COMMUNITY_Architecture Module Docs|Architecture Module Docs]]
- [[_COMMUNITY_Huang Momentum Strategy|Huang Momentum Strategy]]
- [[_COMMUNITY_MarketData Container|MarketData Container]]
- [[_COMMUNITY_ML Collect Cleanup|ML Collect Cleanup]]
- [[_COMMUNITY_Order Block & Breaker Block Spec|Order Block & Breaker Block Spec]]
- [[_COMMUNITY_backtest __init__|backtest __init__]]
- [[_COMMUNITY_backtestdata __init__|backtest/data __init__]]
- [[_COMMUNITY_backtestengine __init__|backtest/engine __init__]]
- [[_COMMUNITY_backtestml __init__|backtest/ml __init__]]
- [[_COMMUNITY_backtestperformance __init__|backtest/performance __init__]]
- [[_COMMUNITY_backtestpropfirm __init__|backtest/propfirm __init__]]
- [[_COMMUNITY_backtestregime __init__|backtest/regime __init__]]
- [[_COMMUNITY_backtestrunner __init__|backtest/runner __init__]]
- [[_COMMUNITY_backteststrategy __init__|backtest/strategy __init__]]
- [[_COMMUNITY_backtestvisualization __init__|backtest/visualization __init__]]
- [[_COMMUNITY_engine Module Doc|engine Module Doc]]
- [[_COMMUNITY_engine Module Doc (alt)|engine Module Doc (alt)]]
- [[_COMMUNITY_propfirm Module Doc|propfirm Module Doc]]
- [[_COMMUNITY_ML Train Split Doc|ML Train Split Doc]]
- [[_COMMUNITY_make_baseline Script|make_baseline Script]]

## God Nodes (most connected - your core abstractions)
1. `Order` - 321 edges
2. `MarketData` - 312 edges
3. `OpenPosition` - 249 edges
4. `PositionUpdate` - 240 edges
5. `OrderType` - 234 edges
6. `BaseStrategy` - 192 edges
7. `SizeType` - 191 edges
8. `RunConfig` - 176 edges
9. `ExitReason` - 168 edges
10. `Trade` - 144 edges

## Surprising Connections (you probably didn't know these)
- `One-shot model evaluation on held-out test data.  WARNING ------- Running this s` --uses--> `MLModel`  [INFERRED]
  run_ml_test.py → backtest\ml\model.py
- `Threshold optimisation for EnsembleMLModel.  Evaluates all threshold candidates` --uses--> `EnsembleMLModel`  [INFERRED]
  run_ml_threshold_opt.py → backtest\ml\model.py
- `For each threshold candidate build (pnl_pts, sl_dists, tpd, regime_labels).` --uses--> `EnsembleMLModel`  [INFERRED]
  run_ml_threshold_opt.py → backtest\ml\model.py
- `Scan all (threshold × scheme × erp × frp) cells and return the combo with     th` --uses--> `EnsembleMLModel`  [INFERRED]
  run_ml_threshold_opt.py → backtest\ml\model.py
- `Single source of truth for train / validation / test split boundaries.  Edit the` --uses--> `MarketData`  [INFERRED]
  backtest\ml\splits.py → backtest\data\market_data.py

## Hyperedges (group relationships)
- **Core Backtest Pipeline** — claudemd_dataloader, claudemd_marketdata, claudemd_run_backtest, claudemd_runresult, claudemd_performanceengine, claudemd_tearsheetrenderer [EXTRACTED 1.00]
- **ML Training & Evaluation Pipeline** — run_ml_collect, run_ml_train, run_ml_validate, run_ml_tearsheets, ml_dataset_parquet, model_ict_smc_pkl, model_ict_smc_ensemble_pkl [EXTRACTED 1.00]
- **PO3 Three-Phase Structure** — spec_po3_accumulation, spec_po3_manipulation, spec_po3_distribution [EXTRACTED 1.00]
- **POI Types (OB, BB, FVG, IFVG, RB, Session Levels)** — spec_ob, spec_fvg, spec_ifvg, spec_rb, spec_bb, spec_session_levels [EXTRACTED 1.00]
- **Strategy Two-Phase Trading Workflow** — spec_phase1_premarket, spec_phase2_session, spec_validated_levels, spec_afz [EXTRACTED 1.00]
- **Ensemble ML System** — propfirm_model_a, propfirm_model_b, propfirm_ensemble_model, propfirm_ensemble_walkforward [EXTRACTED 1.00]
- **Propfirm EV Optimization Loop** — propfirm_threshold_opt, propfirm_lucidflex, propfirm_regime_mc, propfirm_threshold_opt_json [EXTRACTED 1.00]

## Communities

### Community 0 - "Strategy Execution Engine"
Cohesion: 0.04
Nodes (211): BaseStrategy, Flip direction and mirror SL/TP through anchor (stop or limit price)., Return an Order to submit, or None. Called when flat and in trading_hours., Optional. Called immediately after any order fills.         Use to set initial, Abstract base class for all strategies.      Subclass this and implement gener, Flip direction and mirror SL/TP through entry price, in-place., Called every bar while in a position. Return PositionUpdate or None.         Th, Parse a trading_hours value from params into a list of (start, end) time tuples. (+203 more)

### Community 1 - "ML Training & Sensitivity"
Cohesion: 0.04
Nodes (88): normalize_config(), Convert a params dict into normalised config features (0–1 range).     Returns a, ExitReason, evaluate_filter(), Compare metrics for taken trades vs. all trades.     Useful for threshold analys, Scan skip thresholds and return the one that maximises `metric` on the     provi, search_threshold(), Stop triggers first, then limit fills using limit rules. (+80 more)

### Community 2 - "Performance & Tearsheet"
Cohesion: 0.04
Nodes (68): _cagr(), compute_benchmark(), _daily_sharpe(), _max_dd(), Benchmark — two buy-and-hold variants for fair strategy comparison.  B&H Fixed, Sharpe from daily dollar P&L — vectorised., Compute both B&H benchmarks and return the richer BenchmarkResult.      The pr, _build_day_index() (+60 more)

### Community 3 - "Trade Lifecycle & Position Mgmt"
Cohesion: 0.06
Nodes (17): ExitResult, make_bar(), make_engine(), make_long_position(), make_short_position(), TestCheckExits, TestDeltaResolution, TestFills (+9 more)

### Community 4 - "ML Ensemble & Evaluation"
Cohesion: 0.06
Nodes (49): _empty_result(), evaluate_ensemble(), majority_vote(), per_config_metrics(), Ensemble evaluation for the ICT/SMC ML pipeline.  An ensemble groups multiple Ph, Simulate an ensemble of Phase 1 configs on a pre-collected dataset.      For eac, True if strictly more than half of votes are True., True only if every vote is True. (+41 more)

### Community 5 - "Data Loading & Cleaning"
Cohesion: 0.06
Nodes (39): CleaningReport, DataCleaner, NQ daily maintenance window is roughly 16:00–18:00 ET., Flag bars that look suspicious. Does not remove them — adds an         'anomalo, Trim bars at the very start and end of the dataset that may be partial., Cleans a raw OHLCV DataFrame before it enters the engine.      Operations perf, Clean a DataFrame and return the cleaned version plus a report.          Args:, Summary of what the cleaner found and fixed. (+31 more)

### Community 6 - "Parameter Config & LHS Sampling"
Cohesion: 0.06
Nodes (51): get_phase2_candidates(), Parameter space definitions and LHS sampling for the ICT/SMC ML pipeline.  Round, Sample n parameter configs using Latin Hypercube Sampling.      Continuous/integ, Return a small grid of Phase 2 param combinations for inference-time     config, sample_configs(), _cisd_check_at_bar(), _cisd_scan(), _cisd_scan_nb() (+43 more)

### Community 7 - "DoubleSweep Strategy"
Cohesion: 0.05
Nodes (23): _atr(), DoubleSessionSweep, _in_session(), _atr(), _calc_sl(), _calc_tp(), _atr(), _ema() (+15 more)

### Community 8 - "Core Runner & Strategy Contracts"
Cohesion: 0.07
Nodes (22): ABC, generate_signals(), manage_position(), _mirror_order(), _mirror_position(), _parse_trading_hours(), _apply_delta(), build_active_bar_set() (+14 more)

### Community 9 - "Project Architecture Docs"
Cohesion: 0.05
Nodes (58): backtest/ml/ (ML pipeline), backtest/regime/ (HMM regime detection), BaseStrategy, ICT/SMC Strategy (ict_smc.py), ML Test Splits (test1/test2 â€” OOS), ML Validation Split (2023), data/ml_dataset.parquet, models/ict_smc_ensemble.pkl (EnsembleMLModel) (+50 more)

### Community 10 - "Propfirm Monte Carlo Sim"
Cohesion: 0.06
Nodes (49): _add_stability_scores_3d(), _build_multi_counts(), _build_regime_arrays(), _build_regime_seq(), _build_transition_matrix_from_labels(), compute_target_risk_vec(), _estimate_trading_days(), _eval_loop_nb() (+41 more)

### Community 11 - "Core Object Tests"
Cohesion: 0.05
Nodes (6): TestEnums, TestOpenPositionEffectiveSl, TestOrderInvalidConstruction, TestOrderValidConstruction, TestTrade, Returns the most protective SL currently active.         If both a fixed SL and

### Community 12 - "Strategy Validator Tests"
Cohesion: 0.07
Nodes (6): make_config(), make_data(), run_check(), TestOrderSanityCheck, _make_synthetic_data(), _make_synthetic_position()

### Community 13 - "ML Feature Engineering"
Cohesion: 0.12
Nodes (8): build_dataset(), Build a training DataFrame from a completed backtest RunResult.  Each row = one, Convert a RunResult into a feature DataFrame suitable for ML training.      Trad, encode_signal_features(), Feature definitions for the ICT/SMC ML pipeline.  Signal features are captured i, Build the signal_features dict from raw strategy values.     All values are plai, TestBuildDataset, TestSignalFeatureEncoding

### Community 14 - "Trade Log & CSV Tests"
Cohesion: 0.3
Nodes (7): _FakeResult, load_csv(), make_market_data(), make_trade(), TestSaveTradeLog, Save a CSV trade log for the given RunResult.      Columns per trade:         st, save_trade_log()

### Community 15 - "Reverse Trades & Runner Tests"
Cohesion: 0.3
Nodes (4): reverse_trades(), _make_trade(), TestReverseTrades, _wrap()

### Community 16 - "HMM Regime Detection"
Cohesion: 0.14
Nodes (17): _avg_durations_from_trans(), _build_transition_matrix(), _fit_hmm(), fit_regimes(), backtest/regime/hmm.py ────────────────────── Gaussian HMM regime detection fo, Empirical transition matrix from label sequence., Fit HMM and return regime labels for every day.      Parameters     ─────────, Return a permutation array that maps model states to sorted order     (ascendin (+9 more)

### Community 17 - "Train/Val/Test Splits"
Cohesion: 0.19
Nodes (6): filter_df(), Single source of truth for train / validation / test split boundaries.  Edit the, Return (start_ts, end_ts) as tz-aware Timestamps for the named split., Filter a dataset DataFrame (produced by build_dataset) to the given split.     E, split_bounds(), TestSplits

### Community 18 - "Core Order & Position Types"
Cohesion: 0.13
Nodes (1): Enum

### Community 19 - "Risk Manager & Sizing"
Cohesion: 0.22
Nodes (3): Returns the number of whole contracts to trade.          Args:             or, Distance in points between fill price and the effective stop.         Always re, TestRiskManager

### Community 20 - "Statistical Analysis"
Cohesion: 0.26
Nodes (12): _daily_sharpe(), _permutation_test(), backtest/regime/analysis.py ─────────────────────────── Regime breakdown stati, Permutation test: randomly shuffle regime labels across OOS days and     recomp, Full regime analysis:       1. Tag each trade with its regime       2. Compute, Get the calendar date of a trade's entry bar., Annualised Sharpe from a list of per-trade dollar PnLs., _regime_stats() (+4 more)

### Community 21 - "Trade Chart Visualization"
Cohesion: 0.24
Nodes (10): TradingView-style candlestick chart showing one winning trade and one         l, _add_overlay(), _get_slice(), _pick_trades(), plot_trade_examples(), trade_chart.py — TradingView-style position visualization for individual trades., Generate a TradingView-style candlestick chart showing one winning trade     and, Return (winner, loser) — one representative trade of each outcome.      Prefers (+2 more)

### Community 22 - "Order Validation Logic"
Cohesion: 0.18
Nodes (1): For LIMIT and STOP_LIMIT orders the limit price must be on the correct

### Community 23 - "Architecture Module Docs"
Cohesion: 0.28
Nodes (9): backtest/data/ (DataLoader + MarketData), backtest/performance/ (metrics, tearsheet, trade log), backtest/runner/ (per-bar execution loop), DataLoader, MarketData, PerformanceEngine, run_backtest(), RunResult (+1 more)

### Community 24 - "Huang Momentum Strategy"
Cohesion: 0.36
Nodes (3): _atr(), _build_intervals(), IntradayIntervalMomentumHuang

### Community 25 - "MarketData Container"
Cohesion: 0.67
Nodes (0): 

### Community 26 - "ML Collect Cleanup"
Cohesion: 1.0
Nodes (1): Delete all run_ml_collect outputs for a clean re-run.

### Community 27 - "Order Block & Breaker Block Spec"
Cohesion: 1.0
Nodes (2): Breaker Block (BB) â€” POI (inverse of invalidated OB), Order Block (OB) â€” POI

### Community 28 - "backtest __init__"
Cohesion: 1.0
Nodes (0): 

### Community 29 - "backtest/data __init__"
Cohesion: 1.0
Nodes (0): 

### Community 30 - "backtest/engine __init__"
Cohesion: 1.0
Nodes (0): 

### Community 31 - "backtest/ml __init__"
Cohesion: 1.0
Nodes (0): 

### Community 32 - "backtest/performance __init__"
Cohesion: 1.0
Nodes (0): 

### Community 33 - "backtest/propfirm __init__"
Cohesion: 1.0
Nodes (0): 

### Community 34 - "backtest/regime __init__"
Cohesion: 1.0
Nodes (0): 

### Community 35 - "backtest/runner __init__"
Cohesion: 1.0
Nodes (0): 

### Community 36 - "backtest/strategy __init__"
Cohesion: 1.0
Nodes (0): 

### Community 37 - "backtest/visualization __init__"
Cohesion: 1.0
Nodes (0): 

### Community 38 - "engine Module Doc"
Cohesion: 1.0
Nodes (1): Backtest Engine Project (CLAUDE.md)

### Community 39 - "engine Module Doc (alt)"
Cohesion: 1.0
Nodes (1): backtest/engine/ (position & order state machines)

### Community 40 - "propfirm Module Doc"
Cohesion: 1.0
Nodes (1): backtest/propfirm/ (prop firm Monte Carlo)

### Community 41 - "ML Train Split Doc"
Cohesion: 1.0
Nodes (1): ML Train Split (2019-2022)

### Community 42 - "make_baseline Script"
Cohesion: 1.0
Nodes (1): make_baseline.py (verify trade hashes)

## Knowledge Gaps
- **94 isolated node(s):** `Delete all run_ml_collect outputs for a clean re-run.`, `Summary of what the cleaner found and fixed.`, `Cleans a raw OHLCV DataFrame before it enters the engine.      Operations perf`, `Clean a DataFrame and return the cleaned version plus a report.          Args:`, `Detect gaps larger than the expected bar interval.         Gaps during the dail` (+89 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `ML Collect Cleanup`** (2 nodes): `clean_ml_collect.py`, `Delete all run_ml_collect outputs for a clean re-run.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Order Block & Breaker Block Spec`** (2 nodes): `Breaker Block (BB) â€” POI (inverse of invalidated OB)`, `Order Block (OB) â€” POI`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `backtest __init__`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `backtest/data __init__`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `backtest/engine __init__`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `backtest/ml __init__`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `backtest/performance __init__`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `backtest/propfirm __init__`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `backtest/regime __init__`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `backtest/runner __init__`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `backtest/strategy __init__`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `backtest/visualization __init__`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `engine Module Doc`** (1 nodes): `Backtest Engine Project (CLAUDE.md)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `engine Module Doc (alt)`** (1 nodes): `backtest/engine/ (position & order state machines)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `propfirm Module Doc`** (1 nodes): `backtest/propfirm/ (prop firm Monte Carlo)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `ML Train Split Doc`** (1 nodes): `ML Train Split (2019-2022)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `make_baseline Script`** (1 nodes): `make_baseline.py (verify trade hashes)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `MarketData` connect `Strategy Execution Engine` to `ML Training & Sensitivity`, `Performance & Tearsheet`, `ML Ensemble & Evaluation`, `Data Loading & Cleaning`, `Parameter Config & LHS Sampling`, `DoubleSweep Strategy`, `Core Runner & Strategy Contracts`, `Strategy Validator Tests`, `ML Feature Engineering`, `Trade Log & CSV Tests`, `Reverse Trades & Runner Tests`, `Train/Val/Test Splits`, `Huang Momentum Strategy`, `MarketData Container`?**
  _High betweenness centrality (0.245) - this node is a cross-community bridge._
- **Why does `Order` connect `Strategy Execution Engine` to `ML Training & Sensitivity`, `Trade Lifecycle & Position Mgmt`, `Parameter Config & LHS Sampling`, `DoubleSweep Strategy`, `Core Runner & Strategy Contracts`, `Core Object Tests`, `Strategy Validator Tests`, `Reverse Trades & Runner Tests`, `Core Order & Position Types`, `Risk Manager & Sizing`, `Order Validation Logic`, `Huang Momentum Strategy`?**
  _High betweenness centrality (0.125) - this node is a cross-community bridge._
- **Why does `ExitReason` connect `ML Training & Sensitivity` to `Strategy Execution Engine`, `Performance & Tearsheet`, `Trade Lifecycle & Position Mgmt`, `ML Ensemble & Evaluation`, `DoubleSweep Strategy`, `Core Runner & Strategy Contracts`, `Propfirm Monte Carlo Sim`, `Core Object Tests`, `ML Feature Engineering`, `Trade Log & CSV Tests`, `Reverse Trades & Runner Tests`, `Train/Val/Test Splits`, `Core Order & Position Types`, `Risk Manager & Sizing`?**
  _High betweenness centrality (0.102) - this node is a cross-community bridge._
- **Are the 308 inferred relationships involving `Order` (e.g. with `BaseStrategy` and `Parse a trading_hours value from params into a list of (start, end) time tuples.`) actually correct?**
  _`Order` has 308 INFERRED edges - model-reasoned connections that need verification._
- **Are the 308 inferred relationships involving `MarketData` (e.g. with `Generate baseline trade records for regression testing.  Usage:     python make_` and `Context manager that prints step name and elapsed time.`) actually correct?**
  _`MarketData` has 308 INFERRED edges - model-reasoned connections that need verification._
- **Are the 244 inferred relationships involving `OpenPosition` (e.g. with `BaseStrategy` and `Parse a trading_hours value from params into a list of (start, end) time tuples.`) actually correct?**
  _`OpenPosition` has 244 INFERRED edges - model-reasoned connections that need verification._
- **Are the 237 inferred relationships involving `PositionUpdate` (e.g. with `BaseStrategy` and `Parse a trading_hours value from params into a list of (start, end) time tuples.`) actually correct?**
  _`PositionUpdate` has 237 INFERRED edges - model-reasoned connections that need verification._