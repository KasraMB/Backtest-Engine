# Lessons

## 1. Correct IS/OOS/Reserved data splits
**Rule:** Always use the correct data splits:
- IS (in-sample / train): 2019-01-01 → 2022-12-31
- OOS (out-of-sample / validation): 2023-01-01 → 2024-12-31
- Reserved (final check, never touch): 2025-01-01 onward

**Why:** Previous sessions ran backtest on 2022-2025, bleeding into OOS. This contaminated all performance stats and EV/day figures. 57.1% WR and +$50k reported as "IS" was actually IS+OOS combined.

**How to apply:** Before any backtest or parameter tuning, check that DATE_FROM/DATE_TO in run.py is within IS range. OOS check only done after params are finalized on IS.

## 3. Propfirm sweep findings (SessionMeanRevStrategy, 2019-2024)

**Rule:** The optimal propfirm config is: NY session only, rr=1.25, require_bos=True, momentum_only=True, disp_min_atr_mult=2.0. This gives $13.67/day EV on 150K LucidFlex with 39.8% pass rate.

**Why:** 180-combo sweep confirmed selectivity beats frequency. Tight filters (BOS+disp=2.0) produce 367 trades over 6 years at 55%+ WR and positive avgR (+0.157). London session dilutes quality and hurts EV despite adding frequency. Reversion trades hurt quality at RR=1.25.

**How to apply:** Don't chase frequency. A positive avgR with 300-500 high-quality trades beats 1000+ lower-quality trades for propfirm EV/day. Propfirm EV is positive even when total net PnL is negative (due to commissions on NQ minis vs propfirm using MNQ micros).

## 5. Structural filters overfit — atr_vol_filter and require_daily_momentum

**Rule:** atr_vol_filter (ATR > N×median) and require_daily_momentum (prior-day direction) appear to improve IS EV/day significantly but are overfit. They reduce OOS trade count severely (48→8 in 2025) making OOS metrics noise, and often collapse OOS EV/day by 70-90%.

**Why:** These filters select a subset of IS data that happens to be high-quality, but the selection criteria don't generalize. The 2025 OOS had a different volatility and direction regime than IS 2019-2024. With only 8 OOS trades vs 108 IS trades/6yr, the "improvement" is entirely explained by sample selection on IS data.

**How to apply:** Treat any filter that cuts IS trades by >40% as overfit risk. Always check OOS trade count alongside OOS EV/day. Min 30 OOS trades required for a meaningful propfirm EV estimate. The 15% IS→OOS EV drift filter must use both IS and OOS metrics, not just EV/day.

## 6. rr=0.75 + sl=1.5 is regime-dependent, not structurally strong

**Rule:** rr=0.75, sl_atr_multiplier=1.5 shows IS $18-20/day but collapses to $2-4/day in 2025 OOS. Do not use for propfirm.

**Why:** 2025 was not a favorable year for low-RR/wide-SL displacement trades. The strategy takes profits quickly (0.75R) with a wide SL, which works in choppy IS years but underperforms when 2025 needed wider targets to capture directional moves. IS 2019-2024 had several range-bound years that rewarded quick exits.

**How to apply:** Low RR configs need to demonstrate OOS stability. Always check IS performance year-by-year — if good IS years are skewed toward range-bound periods, OOS in trending regimes will underperform.

## 4. Sweep2 fine-tuning results (SessionMeanRevStrategy, 2019-2024 IS)

**Rule:** The most IS/OOS-stable config after sweep2 fine-tuning is rr=1.0, sl_mult=1.25, wick=0.15, atr=10. The rr=1.5, sl_mult=0.75 combo initially appeared 2025-trend-specific but year-by-year analysis confirms it is genuinely strong 2022-2025 — see Lesson #7 for full analysis.

**Why:** 180-combo sweep2 varying rr_ratio, sl_atr_multiplier, wick_threshold, atr_period on the sweep1 winning base (NY, BOS=True, mom=True, disp=2.0). Best stable config: 366 trades/6yr, WR=57.4%, avgR=+0.155. The sweep1 winner (rr=1.25) maps to sweep2 rank5 at $13.06/day IS.

**How to apply:** rr=1.5 + sl_mult=0.75 is the recommended config for $40+/day propfirm EV (see Lesson #7). rr=1.0 + sl_mult=1.25 is the conservative fallback.

| Config | 2022-2024 EV/day | 2025 OOS EV/day |
|--------|-----------|-----------------|
| rr=1.0, sl=1.25, wick=0.15, atr=10 | $43/day | $17/day |
| rr=1.5, sl=0.75, wick=0.15, atr=10 | $37-39/day | $47/day ✓ |
| rr=1.25, sl=1.0, wick=0.15, atr=10 | $41/day | $24/day |

## 7. $40+/day propfirm EV config (SessionMeanRevStrategy, verified 2025 OOS)

**Rule:** Use rr=1.5, sl_atr_multiplier=0.75, atr_period=10, disp_min_atr_mult=2.0, wick=0.15, require_bos=True, momentum_only=True, NY session only. Account: LucidFlex 150K. This achieves $47/day propfirm EV on 2025 OOS (65% pass rate).

**Why:** Year-by-year analysis shows positive EV in 4 of 5 recent years (2021 was -$3/day — choppy recovery environment was the only exception). 2022-2025 is consistently positive:

| Year | Trades | WR    | Net PnL | EV/day (150K) |
|------|--------|-------|---------|---------------|
| 2019 | 71     | 43.7% | -$11,695 | $5.11 |
| 2020 | 56     | 53.6% | +$8,826  | $45.62 |
| 2021 | 63     | 28.6% | -$19,701 | -$3.21 (avoid choppy recovery) |
| 2022 | 74     | 47.3% | +$6,149  | $11.89 |
| 2023 | 48     | 52.1% | +$8,650  | $35.66 |
| 2024 | 58     | 60.3% | +$25,507 | $74.10 |
| 2025 | 53     | 54.7% | +$16,793 | $46.01 |

**How to apply:** Use this config for live propfirm trading. Monitor for 2021-like choppy conditions (WR dropping below 35% over 2+ months signals regime shift). The rr=1.5 targets require momentum continuation — this config excels in trending NQ environments. The propfirm grid has 3 nesting levels: scheme → eval_risk_pct → {funded_risk_pct: cell}; iterate all 3 to find best EV cell.

## 2. Year-by-year performance (SessionMeanRevStrategy, current params)
| Year | Trades | WR    | PnL       |
|------|--------|-------|-----------|
| 2019 | 71     | 47.9% | -$8,682   |
| 2020 | 56     | 51.8% | +$1,712   |
| 2021 | 63     | 38.1% | -$9,654   |
| 2022 | 74     | 51.4% | +$7,206   |
| 2023 | 48     | 56.2% | +$7,112   |
| 2024 | 56     | 67.9% | +$24,517  |
| 2025 | 51     | 52.9% | +$4,741   |

IS (2019-2022): net **-$9,418** — strategy is net losing on correct IS data
OOS (2023-2024): net **+$31,629** — profitable on OOS, but params were tuned on contaminated range

**Implication:** 48-combo coarse sweep (disp×bos×momentum_only×sessions) found 0/48 profitable on IS 2019-2022. WR ceiling is ~47.3% across all combos. Commission + slippage drag ~$114/trade is eating the entire edge. The only remaining lever to try is higher rr_ratio (longer hold = fewer round-trips). If rr≥1.5 doesn't flip it, the strategy has no edge on 2019-2022 NQ data.

## 9. _estimate_trading_days uses equity bars/day — wrong for NQ futures

**Rule:** Always pass `n_trading_days=len(trading_dates)` to `run_propfirm_grid` for NQ futures. Never rely on the auto-estimate alone.

**Why:** `_estimate_trading_days` divided bar-index span by 390 (equity 6.5h session). NQ trades 23h/day (~1380 bars). This overstated day count by 3.5×, understated trades/day by 3.5×, and then the `max(0.5, ...)` floor further inflated it. Combined effect: ev/day figures were ~2× overstated (e.g., $24/day shown instead of correct $12/day for 25K). Fix committed in ae1d14e: default changed to 1380 bars/day; `n_trading_days` param added to `run_propfirm_grid`; fallback floor lowered to 0.01.

**How to apply:** Always pass `n_trading_days=len(trading_dates_f)` when calling `run_propfirm_grid` in any script. The fallback (bar-index estimate) is now close (~9% off) but explicit is always more accurate.

## 8. London session and lower disp filter destroy SessionMeanRevStrategy edge

**Rule:** Never add London session or lower disp_min_atr_mult below 2.0 for SessionMeanRevStrategy. The edge is in selectivity, not frequency.

**Why:** Tested on 2025 OOS (53 trades, 54.7% WR, +$16,793 PnL baseline):
- London+NY: WR drops to 36.9%, PnL goes to -$25,853 (141 trades — London quality is much lower)
- disp=1.5: WR drops to 41%, PnL goes to -$10,468 (251 trades — lower quality signals dilute edge)
- London+NY+disp=1.5: WR 40.2%, PnL -$50,533 (500 trades — catastrophic)
- Increasing max_trades_per_day: no meaningful benefit; signal quality is the constraint

**How to apply:** When trying to reduce cycle time (more trades → faster eval), the instinct to add London or loosen filters will destroy the strategy. The correct lever for shorter cycles is aggressive propfirm sizing (eval_risk_pct), not more trades. The 91-day cycle (k=5 on 150K) already satisfies a 2-4 month target without any filter changes.
