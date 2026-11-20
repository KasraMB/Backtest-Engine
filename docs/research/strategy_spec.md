# NQ 1m ICT/SMC Strategy — Full Specification
*Version 1.2 — Ready for Claude Code implementation*

---

## 1. INSTRUMENT & TIMEFRAMES

- **Instrument:** NQ (Nasdaq 100 Futures) only
- **Execution timeframe:** 1-minute
- **Analysis timeframes:** 5m (primary), 15m, 30m
- **Session:** New York only — 09:30 to 11:00 ET
- No trades are taken outside this window. All entry logic must be evaluated against this constraint.

---

## 2. CORE CONCEPTS & DEFINITIONS

> **GLOBAL NOTE — ATR:** Every ATR calculation in this strategy uses **Wilder's Smoothing Method** (also known as RMA/SMMA). This is the standard ATR as defined by J. Welles Wilder. No simple or exponential moving average variants are used for ATR anywhere in this spec. The period is 14 unless stated otherwise.
> Formula: ATR[0] = ((ATR[1] × (period − 1)) + TR[0]) / period

### 2.1 Candle / Leg Terminology

A **leg** is a directional price move from one swing point to another:
- Bullish leg: swing low → swing high
- Bearish leg: swing high → swing low

### 2.2 Swing High / Swing Low

Swings are identified using a standard pivot detection with a single lookback parameter `n` (default 1), applied on the **5-minute** timeframe.

**Swing High:**
A candle at position `bar[n]` is a swing high if its high is **strictly greater** than the high of every candle within `n` bars on both sides:
```
high[n] > high[n-i]  AND  high[n] > high[n+i]   for all i in 1..n
```
The swing high candle is the one whose high formed the pivot. It is confirmed `n` bars after it forms (right-side confirmation required).

**Swing Low:**
A candle at position `bar[n]` is a swing low if its low is **strictly lower** than the low of every candle within `n` bars on both sides:
```
low[n] < low[n-i]  AND  low[n] < low[n+i]   for all i in 1..n
```

**Parameters:**
- `swing_n` (int, default 1): number of candles on each side required to confirm a swing point. Higher values produce fewer, more significant swings.

**Notes:**
- Strict inequality — ties (equal highs/lows) do not qualify
- Applied on 5m only for PO3 identification and CISD triggering
- Because right-side confirmation is required, a swing is only known `n` bars after it occurred — this is inherently forward-safe as long as `n` bars have elapsed

### 2.3 CISD (Change In State of Delivery)

CISD is a market structure break defined by a candle closing through the open of the first candle of an opposing series. It signals that the manipulation phase has ended and distribution is beginning.

**Bullish CISD:**
1. Identify the most recent series of consecutive **bearish** candles (close < open), with each candle's body size ≥ `cisd_min_body_ratio` × the previous candle's body size
2. The **CISD level** is set to the **open of the first candle** in that bearish series
3. A bullish CISD fires when a subsequent candle's **body** (i.e. `max(open, close)`) closes **above** that level
4. Full body close required — a wick above the level does not constitute a CISD

**Bearish CISD:**
1. Identify the most recent series of consecutive **bullish** candles (close > open), with each candle's body size ≥ `cisd_min_body_ratio` × the previous candle's body size
2. The **CISD level** is set to the **open of the first candle** in that bullish series
3. A bearish CISD fires when a subsequent candle's **body** (i.e. `min(open, close)`) closes **below** that level
4. Full body close required

**Parameters:**

| Parameter | Default | Description |
|---|---|---|
| `cisd_min_series_candles` | 2 | Minimum number of consecutive candles required in the series before a CISD level is valid. E.g. with default 2, a single green candle followed immediately by a red candle would not qualify — there must be at least 2 consecutive green candles before the level is drawn. |
| `cisd_min_body_ratio` | 0.5 | Each candle in the series must have a body size ≥ this fraction of the previous candle's body size. Prevents tiny doji-like candles from qualifying as part of a series. Set to 0.0 to disable — any candle regardless of size will count. E.g. with default 0.5, a candle with a body of 2 pts only qualifies if the previous candle's body was ≤ 4 pts. |

**Notes:**
- CISD operates on the **5m** timeframe for PO3 identification
- The same concept is used on the **1m** within the AFZ entry pattern (Section 5), but these are distinct applications — the AFZ has its own series detection logic
- Only the most recent series is used

### 2.4 PO3 — Power of Three (Accumulation, Manipulation, Distribution)

PO3 is the macro structure within which all trades are identified. It consists of three phases:

**Accumulation:** A consolidation range detected mechanically on the 5m using all four of the following conditions over a rolling `lookback` window:

1. **Low volatility:** `ATR% < atrMult × avg(ATR%, lookback)` — current volatility below a fraction of recent average
2. **Tight price band:** `range% < adaptiveBand` where `adaptiveBand = bandPct × (currentATR/avgATR)^volSens` — the price range as a % of close is narrow, scaled by current volatility
3. **No directional trend:** R² of closes over the window < `maxR2` — price is not trending in a consistent direction
4. **Sufficient choppiness:** Number of candle direction flips (bull→bear or bear→bull) within the window ≥ `minDirChanges`

A zone is **confirmed** (`zoneVisible = true`) once it has spanned at least `minCandles` bars. Zones can merge with a recent prior zone if the gap between them is ≤ `lookback` bars and their price ranges overlap.

**Accumulation parameters:**

| Parameter | Default | Description |
|---|---|---|
| `po3_lookback` | 6 | Rolling window size (candles) for all accumulation conditions |
| `po3_atr_mult` | 0.95 | ATR% must be below this fraction of its average |
| `po3_atr_len` | 14 | ATR calculation period |
| `po3_band_pct` | 0.3 | Base price band % threshold at average volatility |
| `po3_vol_sens` | 1.0 | Volatility sensitivity exponent (0=flat, 1=linear, 2=aggressive) |
| `po3_max_r2` | 0.4 | Maximum R² — rejects zones with directional trending |
| `po3_min_dir_changes` | 2 | Minimum candle direction flips within the window |
| `po3_min_candles` | 3 | Minimum bars zone must span before it is confirmed |

**Manipulation:** A directional leg that moves away from the accumulation range. There are two scenarios:

- *Scenario A (Bearish setup):* Price moves upward from accumulation (manipulation up), then a **bearish CISD** forms on the 5m (a candle closes below the open of the lowest candle of the most recent 5m swing high). The manipulation leg is the bullish leg from the last swing low to the swing high that preceded the CISD.
- *Scenario B (Bullish setup):* Price moves downward from accumulation (manipulation down), then a **bullish CISD** forms on the 5m (a candle closes above the open of the highest candle of the most recent 5m swing low). The manipulation leg is the bearish leg from the last swing high to the swing low that preceded the CISD.

**Temporal constraint — accumulation to manipulation gap:**

The manipulation leg must begin within `po3_max_accum_gap_bars` bars of the accumulation zone ending. This is a parameter to allow experimentation:

| Parameter | Default | Description |
|---|---|---|
| `po3_max_accum_gap_bars` | 10 | Max 5m bars allowed between accumulation zone end and manipulation leg start. Set to a large number (e.g. 999) to disable the constraint. |

**Temporal constraint — manipulation leg origin:**
- The manipulation leg may originate from **overnight** (prior to the NY session)
- It must **not** originate from the previous calendar day or earlier
- In practice: the swing point anchoring the manipulation leg must be from the current trading day's overnight session or later

**Parameter:**
- `po3_min_manipulation_size_pts` (float, default 0.0): minimum size of the manipulation leg in NQ points. Set to 0 to disable. Exists because there may be an unconscious minimum size filter being applied in manual trading.

**Distribution:** Price moves in the true direction (opposite to the manipulation) after the CISD confirms. The distribution phase is where trades are entered.

---

## 3. FIB TOOL — ANCHORING RULES

The standard TradingView fib retracement tool is used throughout. All levels are defined as fractions of the anchored range.

### 3.1 STDV Entry Fib (Manipulation Leg)

Used to identify the entry zone for the trade.

- **Bearish setup:** Anchor from the **swing low** of the manipulation leg (0%) to the **swing high** of the manipulation leg (100%). Price retracing back down into the fib zone after CISD is the setup.
- **Bullish setup:** Anchor from the **swing high** of the manipulation leg (0%) to the **swing low** of the manipulation leg (100%). Price retracing back up into the fib zone after CISD is the setup.
- Anchored on the **5m** timeframe.

**Entry fib levels used:**
- OTE zone: 61.8%, 70.5%, 79%
- Equilibrium: 50%
- STDV extension levels (for entry, not TP): -2, -2.25, -2.5, -3, -3.25, -3.5, -4, -4.25, -4.5
  - These are extensions beyond the 0% anchor in the direction of the trade
  - Example: for a bearish setup where 0% is the swing low, -2 is two full ranges below the swing low

### 3.2 OTE Entry Fib (Session Level)

Used as an alternative or additional entry anchor. Different from the manipulation leg fib.

- **Anchor start:** A session level (e.g. session open, midnight open, or equivalent significant price anchor)
- **Anchor end:** The highest or lowest point price has reached from that initial anchor
- **Bullish:** Anchor from session level (0%) to the highest point (100%)
- **Bearish:** Anchor from session level (0%) to the lowest point (100%)
- Uses the same OTE levels: 61.8%, 70.5%, 79%, and 50% (equilibrium)

> **Note:** OTE and STDV are the same fib tool applied with different anchors. They represent two separate entry frameworks that can be used independently or in confluence.

---

## 4. POINTS OF INTEREST (POI)

POIs are price zones used for entry confluence matching. Four types are recognised.

### 4.1 Order Block (OB)

**Definition:** The last candle in the opposing direction immediately before an impulsive move in the opposite direction.
- Bullish OB: last bearish candle (close < open) before a bullish impulse
- Bearish OB: last bullish candle (close > open) before a bearish impulse

**Levels (body only — wicks excluded):**
- 0% = near edge of body (top of bearish OB body, bottom of bullish OB body)
- 50% = midpoint of body
- 100% = far edge of body (bottom of bearish OB body, top of bullish OB body)

**Invalidation:** A candle closes beyond the 100% level (through the far edge of the body).

### 4.2 Fair Value Gap (FVG)

**Definition:** A three-candle pattern where there is a gap between candle 1 and candle 3:
- Bullish FVG: candle 1 high < candle 3 low (gap above candle 1, below candle 3)
- Bearish FVG: candle 1 low > candle 3 high (gap below candle 1, above candle 3)

**Levels:**
- 0% = near edge (candle 1 high for bullish, candle 1 low for bearish)
- 50% = midpoint of the gap
- 100% = far edge (candle 3 low for bullish, candle 3 high for bearish)

**Invalidation:** A candle closes beyond the 0% level (closes into the gap from the near side). At this point the FVG becomes an IFVG.

### 4.3 Inverse Fair Value Gap (IFVG)

**Definition:** A FVG that has been closed through — a candle has closed beyond the FVG's 0% level. The zone retains its original levels (same 0%, 50%, 100%) but now acts as resistance/support in the opposite direction.
- A bullish FVG that becomes an IFVG now acts as a bearish POI
- A bearish FVG that becomes an IFVG now acts as a bullish POI

**Levels:** Same as the original FVG (0%, 50%, 100% unchanged).

**Invalidation:** A candle closes beyond the IFVG's 100% level (back through the far edge). At this point the structure is fully consumed and discarded — an inverse of an IFVG is not used.

### 4.4 Rejection Block (RB)

**Definition:** A candle whose wick penetrates into an active FVG, where that wick is greater than 30% of the candle's body size.

**Formation rules:**
- There must be an active (not yet invalidated) FVG on the chart
- **Bullish FVG:** a candle's **lower wick** must reach into the FVG zone (wick low is inside the FVG range)
- **Bearish FVG:** a candle's **upper wick** must reach into the FVG zone (wick high is inside the FVG range)
- The wick size must be > `rb_min_wick_ratio` × candle body size (default 30%)
- The resulting rejection block is the **wick** of that candle (not the body)

**Levels (wick only):**
- 0% = tip of the wick (furthest point into the FVG)
- 50% = midpoint of the wick (CE)
- 100% = base of the wick (where wick meets the body)

**Additional matching levels — OTE of the rejection block:**
For rejection blocks only, confluence can also be matched against the OTE fib levels derived from the wick range:
- 50% (equilibrium/CE of the wick)
- 61.8% of the wick range
- 70.5% of the wick range
- 79% of the wick range

These OTE levels are measured as a retracement of the wick from tip (0%) to base (100%).

**Invalidation:** Price interacts with the 50% level (any wick touch counts — not required to be a close). Once the 50% has been touched, the rejection block is no longer valid as a POI.

**Parameter:**
- `rb_min_wick_ratio` (float, default 0.3): minimum wick size as a fraction of the candle body size. A wick must be > 30% of the body to qualify. Set to 0.0 to disable.

### 4.5 Breaker Block (BB)

**Definition:** The inverse of an order block — an order block that has been invalidated (price closed beyond its 100% level). At that point the former OB zone flips and acts as a POI in the opposite direction. Exactly analogous to how an IFVG is the inverse of an FVG.

- A bullish OB that gets closed through its 100% (bottom of body) becomes a **bearish breaker block**
- A bearish OB that gets closed through its 100% (top of body) becomes a **bullish breaker block**

**Levels:** Same as the original order block body (0%, 50%, 100% unchanged — body only, no wicks).

**Invalidation:** A candle closes beyond the breaker block's 100% level (back through the far edge of the body). At this point the structure is discarded — an inverse of a breaker block is not used.

### 4.6 POI Timeframes

POIs are identified on **any timeframe** — no restriction. The following timeframes have appeared in the trade journal and are all valid: 1m, 3m, 5m, 15m, 30m, 1h, 4h. Higher timeframe POIs are not weighted differently from lower timeframe ones — any match on any timeframe counts.

**Note on BPR:** Balanced Price Range (BPR) is treated identically to an IFVG throughout this strategy. Any reference to BPR in the trade journal should be interpreted as IFVG.

### 4.7 POI Level Matching — Confluence with Fib

For a POI to serve as confluence with a fib level (OTE or STDV), the POI level must fall within `confluence_tolerance_pts` (default 2.5 pts) of the fib level.

**Valid matching levels per POI type:**

| POI Type | Valid Levels for Matching |
|---|---|
| Order Block (OB) | 0%, 50% (CE), 100% of body |
| Breaker Block (BB) | 0%, 50% (CE), 100% of body |
| FVG | 0%, 50% (CE), 100% of gap |
| IFVG / BPR | 0%, 50% (CE), 100% of gap |
| Rejection Block (RB) | 0%, 50% (CE), 100% of wick **AND** OTE levels (50%, 61.8%, 70.5%, 79%) of the wick range |
| Session Level | The level itself (single price — see Section 4.8) |

**Note:** CE =Centrepoint/Equilibrium = 50% level. These terms are used interchangeably throughout the trade journal.

**Parameter:**
- `confluence_tolerance_pts` (float, default 2.5): maximum distance in NQ points between a fib level and a POI level for them to be considered confluent

### 4.8 Session Levels

Session levels are the high and low of each completed session period. They act as single-price POIs — no 0%/50%/100% range. Confluence is matched when a fib level falls within `confluence_tolerance_pts` of the session level price.

**Session definitions (all times in America/New_York timezone):**

| Level Name | Session Window | Produces |
|---|---|---|
| Asia High / Asia Low | 20:00 – 00:00 ET | High and low of the Asia session |
| London High / London Low | 02:00 – 05:00 ET | High and low of the London session |
| NY Pre High / NY Pre Low | 08:00 – 09:30 ET | High and low of the NY pre-market session |
| NY AM High / NY AM Low | 09:30 – 11:00 ET | High and low of the NY AM session |
| NY Lunch High / NY Lunch Low | 12:00 – 13:00 ET | High and low of the NY lunch session |
| NY PM High / NY PM Low | 13:30 – 16:00 ET | High and low of the NY PM session |
| PDH / PDL | Previous calendar day | Previous day's high and low |
| Daily High / Daily Low | Current calendar day | Running high and low of the current day |
| NDOG | New Day Opening Gap | The open price at 00:00 ET (midnight open) |
| NWOG | New Week Opening Gap | The open price at the start of the trading week |

**Validity:**
- Session levels remain valid for **2 calendar days** after the session that produced them closes
- A session level is not invalidated by price trading through it — it simply ages out after 2 days
- All valid session levels (within the 2-day window) are checked for confluence on every potential trade

**ML hook:** All session level types are treated equally in the current implementation. A future ML module will learn which session level types are most predictive of successful trades, and weight or filter them accordingly. The implementation must accept a pluggable `session_level_weight` function that can assign a weight to each level type per trade context.

**Parameter:**
- `session_level_validity_days` (int, default 2): number of calendar days a session level remains valid after its session closes

---

## 5. ENTRY MODEL — AFZ (Algorithmic Fibonacci Zone)

The AFZ is the 1m candle pattern that defines the exact entry price and stop loss. It only fires after the full pre-market preparation workflow has completed and price has touched an active validated level during the session.

---

### 5.0 Full Trading Workflow

The strategy operates in two distinct phases: pre-market preparation and session execution.

---

#### Phase 1 — Pre-Market Preparation (before 09:30 ET)

This phase runs once per day, before the NY session opens. All steps must complete before 09:30 ET.

**Step 1 — Identify manipulation legs:**
- Scan the 5m chart for all valid PO3 structures from the current trading day's overnight session
- A manipulation leg is valid if its originating swing point is from the current day's overnight session (i.e. after the previous day's 16:00 ET close — not from any earlier date)
- Multiple manipulation legs may be identified on a single day

**Step 2 — Draw fibs on each manipulation leg:**
- For each manipulation leg, draw the STDV entry fib (Section 3.1) producing levels: 50%, 61.8%, 70.5%, 79%, -2, -2.25, -2.5, -3, -3.25, -3.5, -4, -4.25, -4.5
- Draw the OTE fib (Section 3.2) where applicable, producing levels: 50%, 61.8%, 70.5%, 79%
- This generates a list of candidate price levels for the day

**Step 3 — Validate each level:**
- For each candidate level, check whether any POI (on any timeframe — see Section 4) has a 0%, 50%, or 100% level within `confluence_tolerance_pts` (default 2.5 pts) of the candidate level
- For Rejection Blocks only, also check OTE levels (50%, 61.8%, 70.5%, 79%) of the wick range
- If a match exists → mark the level as **validated**
- If no match exists → **discard** the level
- Once validated, the POI used for matching is no longer tracked. Only the price of the validated level is retained.

**Output of Phase 1:** A list of validated price levels for the trading day.

---

#### Phase 2 — Session Execution (09:30–11:00 ET)

**Step 4 — Monitor for level touch:**
- For each validated level, monitor whether price **touches** it during the session
- **Touch definition:** the candle's range (high to low, including wicks) contains the validated level price. Specifically: `candle_low ≤ level_price ≤ candle_high`
- Being close does not count — price must have literally traded at the level

**Step 5 — Evaluate penetration on touch:**
When a candle touches a validated level, evaluate the candle's body penetration:

*Body penetration check (hard rule):*
- Compute `penetration_threshold = level_penetration_atr_mult × ATR(14, Wilder)`
- If the candle's **body** (i.e. `min(open, close)` for bearish, `max(open, close)` for bullish) closes more than `penetration_threshold` beyond the level → level is **invalidated** (discard it)
- If the body closes within the threshold → level remains **active**

*Wick-only penetration (current rule — ML will refine):*
- If only the **wick** exceeds the threshold but the body does not:
  - If the wick does **not** reach another validated level → assume the level is still active (no invalidation)
  - If the wick **does** reach another validated level → shift attention to that new level and apply Steps 4–5 from that level's perspective

**Step 6 — Watch for AFZ:**
- Once a level is active (touched, body within threshold), watch for the AFZ pattern to form on the 1m
- The AFZ must form at or near the active level — price must have reached the level before the AFZ formed
- If price moves away and touches a different validated level instead, attention shifts to the new level

**Step 7 — Enter:**
- When a valid AFZ forms at an active level → place limit order at the AFZ middle line, SL at the AFZ external line (see Sections 5.2–5.3)
- The first valid AFZ that fires at any active level triggers the trade
- No preference between levels — all validated, active levels are treated equally

**ML hook:** The selection between multiple simultaneously active levels (Step 7), and the wick penetration decision (Step 5), are both candidates for ML optimisation. The implementation must expose these decision points as pluggable functions:
- `level_selection_policy(active_levels) → chosen_level`: currently returns the first active level with an AFZ
- `wick_penetration_policy(level, candle, atr) → bool`: currently always returns False (not invalidated)

**Parameters:**
| Parameter | Default | Description |
|---|---|---|
| `level_penetration_atr_mult` | 0.5 | Multiplier on ATR(14 Wilder) for body penetration threshold. Calibrate against trade history. |
| `confluence_tolerance_pts` | 2.5 pts | Max distance (pts) for POI/fib level matching in Phase 1 |

---

### 5.1 AFZ Pre-conditions (all must be true before evaluating the pattern)

1. Current time is within the NY session killzone: 09:30–11:00 ET
2. Phase 1 preparation has completed — at least one validated level exists for the day
3. Price has touched an active validated level (Step 4 above)
4. The level has not been invalidated by body penetration (Step 5 above)

### 5.2 AFZ Pattern — Bullish (Long Entry)

Triggered when the current 1m candle is bullish (close > open).

**Look-back logic:**
1. Skip any bullish candles immediately preceding bar 0 (bars[1], bars[2]... until a bearish candle is found)
2. From the first non-bullish candle onward, collect a consecutive series of **bearish** candles
3. The series ends when a non-bearish candle is encountered

**Validity check:**
- Current candle (bar 0) must close **at or above the open of the furthest bearish candle** in the series (`close[0] >= open[furthestBar]`)
- This is the structural close-through that confirms the AFZ

**Zone construction (body levels only — wicks excluded for zone boundaries):**
- **Zone top:** `min(high[furthestBar], open[0])` if `close[0] < high[furthestBar]`, else `high[furthestBar]`
  - More precisely: if `close[0] >= high[furthestBar]`, top = `high[furthestBar]`; else top = `open[furthestBar]`
- **Zone bottom:** the lowest `min(open[i], close[i])` across all candles from bar 0 to furthestBar inclusive (bodies only)
- **lowestLow:** the lowest wick low across all candles from bar 0 to furthestBar inclusive

**Entry (middle line):**
```
entry_price = round_up_to_tick((zone_top + zone_bottom) / 2)
```

**Stop loss (external line):**
```
sl_price = round_down_to_tick((zone_bottom - (zone_bottom - lowestLow) / 2)) - tick_offset
```
Where `tick_offset` (default 2 ticks = 0.5 NQ pts) is a buffer below the external level.

**Order type:** Limit order at `entry_price`

### 5.3 AFZ Pattern — Bearish (Short Entry)

Exact mirror of the bullish pattern.

**Look-back logic:**
1. Skip any bearish candles immediately preceding bar 0
2. Collect a consecutive series of **bullish** candles
3. Series ends at first non-bullish candle

**Validity check:**
- Current candle must close **at or below the open of the furthest bullish candle** (`close[0] <= open[furthestBar]`)

**Zone construction:**
- **Zone bottom:** `max(low[furthestBar], open[0])` — more precisely: if `close[0] <= low[furthestBar]`, bottom = `low[furthestBar]`; else bottom = `open[furthestBar]`
- **Zone top:** the highest `max(open[i], close[i])` across all candles from bar 0 to furthestBar inclusive (bodies only)
- **highestHigh:** the highest wick high across all candles from bar 0 to furthestBar inclusive

**Entry (middle line):**
```
entry_price = round_down_to_tick((zone_top + zone_bottom) / 2)
```

**Stop loss (external line):**
```
sl_price = round_up_to_tick((zone_top + (highestHigh - zone_top) / 2)) + tick_offset
```

**Order type:** Limit order at `entry_price`

### 5.4 One Entry Per AFZ

The `alreadyUsed` check in the code prevents the same AFZ zone from generating a second entry. Once an AFZ fires, that specific zone is consumed regardless of whether the limit order was filled. A new AFZ must form for a new entry to be possible.

### 5.5 AFZ at the Validated Level

The AFZ zone is defined purely by the 1m candle pattern — it does not need to be anchored exactly at the validated level. The requirement is that:
1. Price touched the validated level (Step 4 of Phase 2)
2. The level was not invalidated by body penetration (Step 5)
3. The AFZ pattern subsequently forms on the 1m in the vicinity of that level
4. The limit order is placed at the AFZ middle line, SL at the AFZ external line

### 5.6 Parameters

| Parameter | Default | Description |
|---|---|---|
| `tick_offset` | 2 ticks (0.5 pts) | Buffer added to SL beyond external level |
| `order_expiry_bars` | 10 | 1m bars before limit order is cancelled if not filled |


---

## 6. STOP LOSS

The stop loss is generated directly by the AFZ pattern (Section 5.2 / 5.3). It is the **external line** of the AFZ zone:

**Bullish (long):**
```
SL = round_down_to_tick((zone_bottom - (zone_bottom - lowestLow) / 2)) - tick_offset
```
This places the SL halfway between the zone bottom (lowest body in the range) and the lowestLow (lowest wick), minus a 2-tick buffer. It sits below all wicks in the pattern with margin.

**Bearish (short):**
```
SL = round_up_to_tick((zone_top + (highestHigh - zone_top) / 2)) + tick_offset
```
This places the SL halfway between the zone top (highest body in the range) and the highestHigh (highest wick), plus a 2-tick buffer.

**ML hook:** The `tick_offset` parameter (default 2 ticks) is a candidate for ML optimisation on a per-trade basis. The implementation must accept it as a variable input.

---

## 7. TAKE PROFIT

### 7.1 STDV TP Fib — Anchoring

The TP fib is anchored from the **AFZ zone itself**, not from the manipulation leg. This is different from the entry STDV fib.

**Bullish setup TP fib:**
- Anchor 100% = `topOfZone` (top of AFZ zone body)
- Anchor 0% = `lowestLow` (lowest wick across all candles in the AFZ pattern)
- Extensions go **below** the 0% anchor (below lowestLow)
- Extension levels: -2, -2.25, -2.5, -3, -3.25, -3.5, -4, -4.25, -4.5

**Bearish setup TP fib:**
- Anchor 100% = `bottomOfZone` (bottom of AFZ zone body)
- Anchor 0% = `highestHigh` (highest wick across all candles in the AFZ pattern)
- Extensions go **above** the 0% anchor (above highestHigh)
- Same extension levels

### 7.2 TP Level Selection

The TP is the **closest** extension level (starting from -2 and moving outward) that satisfies **both** of the following conditions simultaneously:

**Condition 1 — POI or Swing confluence:**
The extension level is within `confluence_tolerance_pts` (default 2.5 pts) of either:
- A swing high (for bearish) or swing low (for bullish) on the 1m, 5m, 15m, or 30m
- OR a POI level (0%, 50%, or 100% of an OB, BB, FVG, IFVG, or RB) on the 5m, 15m, or 30m

**Condition 2 — Minimum R:R:**
The distance from entry to TP divided by the distance from entry to SL must be ≥ `min_rr` (default 5.0):
```
(|entry - tp|) / (|entry - sl|) >= min_rr
```

**If no extension level meets both conditions:** The trade is not taken. The AFZ fired but there is no valid TP, so the setup is skipped entirely.

### 7.3 TP is Fixed at Entry

The TP level is calculated and set at the time the AFZ fires. It is not adjusted as the trade develops.

### 7.4 Parameters

| Parameter | Default | Description |
|---|---|---|
| `min_rr` | 5.0 | Minimum R:R required for a valid TP |
| `confluence_tolerance_pts` | 2.5 | Max distance (pts) for TP confluence match |

---

## 8. TRADE MANAGEMENT

### 8.1 Breakeven Rule

Move SL to breakeven (entry price) when **either** of the following occurs first — whichever comes first triggers the move:

**Condition A — First Swing:**
A swing high (for shorts) or swing low (for longs) forms on the 1m after entry and is in profit relative to entry.

**Condition B — New IFVG Formation:**
A new IFVG forms on the 1m in the direction of the trade after entry, and it is in profit relative to entry.

**ML Hook:**
This is a deliberate simplification. The breakeven decision is context-dependent in live trading. A future ML module will replace the "first to occur" rule with a learned policy. The implementation must:
- Accept a pluggable `breakeven_policy` function that takes the current trade state and returns True/False
- Default to the "first to occur" rule described above
- Be designed so the ML module can be swapped in without changing the surrounding trade management logic

### 8.2 Trailing Stop — Protected Swings

After breakeven is triggered, the SL trails to protected swings.

**Protected swing definition:**
A swing point (high or low on the 1m) that is **preceded by a CISD** in the direction of the trade. Specifically:
- For a long trade: a swing low that has a bullish CISD forming before it (meaning price made a CISD, then the swing low formed — this swing low is "protected" because the CISD signals structure in our favour)
- For a short trade: a swing high that has a bearish CISD forming before it

**Trailing rule:**
- When a new protected swing forms on the 1m that is **in profit** relative to current SL, move SL to that swing point
- SL moves to the swing high (for shorts) or swing low (for longs) — with the same buffer used at entry (from the entry model code)
- SL only moves in the direction of the trade (never widen it)

### 8.3 No Partial Exits

The strategy is all-in, all-out. No partial profit taking. Full position exits at TP or SL.

---

## 9. LEVEL AND TRADE INVALIDATION

### 9.1 Level Invalidation — During Pre-Market (Phase 1)

A candidate level is discarded during Phase 1 if no POI match is found within `confluence_tolerance_pts`. No further conditions apply during pre-market.

### 9.2 Level Invalidation — During Session (Phase 2)

A validated level becomes invalidated during the session when:

1. **Body penetration:** The candle that touches the level has its body close more than `level_penetration_atr_mult × ATR(14 Wilder)` beyond the level (see Section 5.0 Step 5)
2. **Session expiry:** The NY killzone closes (11:00 ET) — all remaining active levels are discarded, no new entries

### 9.3 Limit Order Cancellation — After AFZ But Before Fill

Once an AFZ has fired and a limit order is placed, the order is cancelled under any of the following conditions (whichever occurs first):

1. **TP hit before fill:** Price reaches the TP level before the limit order is filled → cancel the order immediately. The trade opportunity has passed.
2. **Bar expiry:** The limit order has been active for more than `order_expiry_bars` 1m bars without being filled → cancel the order. Default: 10 bars.
3. **Session expiry:** 11:00 ET is reached before the order is filled → cancel the order. No new entries after session end.

### 9.4 Notes

- The confluent POI used during Phase 1 validation is **not** tracked after validation — its subsequent invalidation has no effect on the already-validated level
- There is no explicit CISD negation invalidation rule — if the manipulation context changes, the pre-market levels simply may not be touched or may be body-penetrated during the session
- Multiple levels can be active simultaneously — if one is invalidated, the others remain active

---

## 10. PARAMETER SUMMARY

### Core Strategy Parameters

| Parameter | Default | Description |
|---|---|---|
| `confluence_tolerance_pts` | 2.5 pts | Max distance for POI/fib confluence matching |
| `level_penetration_atr_mult` | 0.5 | ATR(14 Wilder) multiplier for body penetration threshold |
| `min_rr` | 5.0 | Minimum R:R required to take a trade |
| `session_start` | 09:30 ET | Start of NY killzone |
| `session_end` | 11:00 ET | End of NY killzone |
| `tick_offset` | 0.5 pts (2 ticks) | SL buffer beyond AFZ external level |
| `swing_n` | 1 | Pivot candles on each side for swing detection (5m) |
| `cisd_min_series_candles` | 2 | Min consecutive candles in series for a valid CISD level |
| `cisd_min_body_ratio` | 0.5 | Min body size ratio vs previous candle to qualify for CISD series |
| `rb_min_wick_ratio` | 0.3 | Min wick size as fraction of candle body to qualify as rejection block |
| `session_level_validity_days` | 2 | Calendar days a session level remains valid after its session closes |
| `order_expiry_bars` | 10 | 1m bars before unfilledlimit order is cancelled |

### PO3 Accumulation Parameters

| Parameter | Default | Description |
|---|---|---|
| `po3_lookback` | 6 | Rolling window size for accumulation conditions |
| `po3_atr_mult` | 0.95 | ATR% threshold fraction |
| `po3_atr_len` | 14 | ATR calculation period |
| `po3_band_pct` | 0.3 | Base price band % |
| `po3_vol_sens` | 1.0 | Volatility sensitivity exponent |
| `po3_max_r2` | 0.4 | Maximum R² for directional filter |
| `po3_min_dir_changes` | 2 | Minimum candle direction flips |
| `po3_min_candles` | 3 | Minimum bars for confirmed zone |
| `po3_max_accum_gap_bars` | 10 | Max 5m bars between accumulation end and manipulation start |
| `po3_min_manipulation_size_pts` | 0.0 | Min manipulation leg size in pts (0 = disabled) |

### ML Candidate Parameters (fixed for now, future per-trade optimisation)

| Parameter | Notes |
|---|---|
| `tick_offset` | SL buffer — per-trade ML |
| `confluence_tolerance_pts` | Matching strictness — per-trade ML |
| `min_rr` | Minimum R:R threshold — per-trade ML |
| Breakeven trigger policy | Section 8.1 — pluggable function |
| POI type/timeframe weighting | Section 4 — which POI types predict best outcomes |

---

## 11. CALIBRATION NOTES (post-backtesting)

All parameters are now defined. The following items are flagged for calibration once initial backtesting is complete:

1. **`level_penetration_atr_mult`** (default 0.5) — calibrate against trade journal to find the value that best matches manual trading decisions
2. **`po3_min_manipulation_size_pts`** (default 0.0) — may need a non-zero value if an implicit minimum is discovered during backtesting

---

## 12. ML HOOKS (future implementation)

The following components are flagged for future ML replacement. Each must be implemented with a pluggable interface so the ML module can be swapped in without changing surrounding logic:

| Component | Current Rule | ML Goal |
|---|---|---|
| Breakeven trigger | First of: swing or IFVG on 1m | Learn optimal breakeven timing from trade outcomes |
| Confluence weighting | Binary match within tolerance | Learn which POI type + timeframe combination predicts best entries |
| Manipulation leg minimum size | Parameter (default 0) | Learn if there is an implicit minimum |
| SL tick offset | Fixed 2 ticks | Learn optimal per-trade buffer |
| Confluence tolerance | Fixed 2.5 pts | Learn optimal matching strictness per setup |
| Min R:R | Fixed 5.0 | Learn optimal per-trade R:R threshold |
| TP level selection | Closest extension meeting 5R + confluence | Learn which extension level performs best given context |
| Session level weighting | All session levels treated equally | Learn which session level types (Asia H/L, London H/L, PDH/PDL etc.) are most predictive per trade context |

---

*End of spec v1.3 — All parameters defined. Ready for Claude Code implementation.*
