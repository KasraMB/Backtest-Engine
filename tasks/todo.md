# AnchoredMeanReversion v2 — Full Spec Parity + Parameter Sweep

## Approved plan (user approved 2026-05-07)

### Phase 1: Save current strategy (v1 backup)
- [x] Copy amr_strategy.py → amr_strategy_v1.py

### Phase 2: Rewrite AnchoredMeanReversionStrategy — full spec parity
Bugs to fix:
- [x] BOS direction: long needs brokeHigh, short needs brokeLow (not OR of both)
- [x] Direction filter: after against_fair_mins, only mean-rev trades allowed
- [x] Window off-by-one: < window_mins (was <=)

New features:
- [x] displacement_style: "Upper Wick" (default) | "Marubozu"
- [x] threshold_mw: float (Marubozu wick ratio, default 0.15)
- [x] threshold_uw: float (Upper Wick close position, replaces old `threshold`, default 0.8)
- [x] against_fair_mins: int (direction filter window, default 15)
- [x] skip_first_mins: int (skip first N mins of session window, default 0)
- [x] filter_tp_beyond_fair: bool (TP must cross fair price by pct%, default False)
- [x] tp_beyond_fair_pct: float (default 50.0)
- [x] move_sl_to_entry: bool (move SL to entry when price hits fair, default False)
- [x] link1400to930: bool (1400 session fair = 930 fair, default False)
- [x] window_mins_per_session: dict (per-session window override, e.g. {"930": 60, "1400": 90})

### Phase 3: Parameter sweep (tmp_amr_v2_sweep.py)
Grid:
- sessions × window_per_session: 13 combos
- displacement_style: Upper Wick / Marubozu (2)
- against_fair_mins: [0, 15, 30] (3)
- tp_multiple: [1.0, 1.5, 2.0] (3)
- swing_periods: [1, 2] (2)
- filter_tp_beyond_fair: [False, True@50%] (2)
Total: ~936 configs

Metric: P($0) asc in reinvestment MC (budget=$300, horizon=84d, goal=$10K)
Fixed: account=25K LucidFlex, ERP=0.2, FRP=1.0
- [x] Write sweep script
- [x] Run sweep
- [x] Report top configs in table
- [x] Explain best config + account strategy

## Notes
- Numba kernel recompiles on first run after signature change (expected ~30-60s)
- sweep uses ThreadPoolExecutor(4) for parallel runs
- simulate_lifecycles N=5000, n_mc=1000 for sweep speed
- v1 strategy preserved for continuity
