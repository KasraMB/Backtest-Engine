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

**Rule:** The most IS/OOS-stable config after sweep2 fine-tuning is rr=1.0, sl_mult=1.25, wick=0.15, atr=10 (IS: $16.30/day, OOS 2025: $16.65/day). The rr=1.5, sl_mult=0.75 combo gives huge OOS ($45.33/day) but is probably 2025-trend-specific.

**Why:** 180-combo sweep2 varying rr_ratio, sl_atr_multiplier, wick_threshold, atr_period on the sweep1 winning base (NY, BOS=True, mom=True, disp=2.0). Best stable config: 366 trades/6yr, WR=57.4%, avgR=+0.155. The sweep1 winner (rr=1.25) maps to sweep2 rank5 at $13.06/day IS.

**How to apply:** rr=1.0 + sl_mult=1.25 is the recommended default. For aggressive propfirm attempts, rr=1.5 + sl_mult=0.75 showed 78.2% pass rate on 2025 OOS but treat with caution — high RR strategies are sensitive to regime changes.

| Config | IS EV/day | OOS 2025 EV/day |
|--------|-----------|-----------------|
| rr=1.0, sl=1.25, wick=0.15, atr=10 | $16.30 | $16.65 (stable) |
| rr=1.5, sl=0.75, wick=0.15, atr=10 | $12.53 | $45.33 (OOS boom) |
| rr=1.25, sl=1.0, wick=0.15, atr=10 | $13.07 | $20.75 |
| rr=1.0, sl=1.50, wick=0.15, atr=10 | $14.92 | $26.76 |

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
