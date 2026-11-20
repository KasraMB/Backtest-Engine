# Strategy Specification: Session Open Mean Reversion & Momentum (NQ Futures)

**Version:** 1.1  
**Instrument:** NQ (Nasdaq-100 E-mini Futures)  
**Data Required:** 1-minute OHLC bars  
**Scope:** Backtesting only  
**Timezone:** All times in Eastern Time (ET)

---

## 1. Core Hypothesis

The closing price of the candle immediately before each major session open represents the "true price" for that session. Price may displace away from this level due to momentum and hedging activity, but will tend to mean-revert back to it within the session window. Momentum moves away from the true price are also tradeable when confirmed by a displacement candle.

---

## 2. Session Definitions

Three sessions are traded. Each session has a defined window during which trades may be entered and must be exited.

| Session  | Window Start | Window End | Reference Candle   |
|----------|-------------|------------|--------------------|
| Asia     | 20:00 ET    | 22:00 ET   | 19:59 ET 1m candle |
| London   | 03:00 ET    | 05:00 ET   | 02:59 ET 1m candle |
| New York | 09:30 ET    | 11:00 ET   | 09:29 ET 1m candle |

> Note: Sessions do not overlap. Each session is treated independently. Swing state resets each session. The daily trade counter persists across all three sessions (see Section 10.5).

---

## 3. Key Price Levels

All levels are derived from the **Reference Candle (RC)** — the 1-minute candle that closes immediately before the session window opens. All levels are fixed at session start and do not change.

### 3.1 Reference Candle Direction

- **Bullish RC:** `RC.close > RC.open`
- **Bearish RC:** `RC.close < RC.open`
- **Doji RC:** `RC.close == RC.open` → skip session entirely, no trades taken

### 3.2 Level Definitions

| Level | Definition |
|-------|-----------|
| **True Price** | `RC.close` — the close of the reference candle |
| **Body High** | `max(RC.open, RC.close)` |
| **Body Low** | `min(RC.open, RC.close)` |
| **Max Reversion (Bullish RC)** | `Body Low` — bottom of the RC body |
| **Max Reversion (Bearish RC)** | `Body High` — top of the RC body |

> Wicks of the reference candle are ignored. Only the body (open-to-close range) defines the reversion boundaries.

### 3.3 Reversion Targets

| Target | Level |
|--------|-------|
| **True Reversion** | `True Price` (`RC.close`) |
| **Max Reversion** | `Body Low` (bullish RC) or `Body High` (bearish RC) |

---

## 4. Indicators

### 4.1 ATR (Average True Range)

- **Period:** 14
- **Timeframe:** 1-minute bars
- **Calculation:** Standard Wilder ATR (14-period)
- **Timing:** Rolling. The ATR value used for any candle is computed at the **close of that candle**.

---

## 5. Position Sizing

Risk per trade is fixed at **1% of account equity**.

### 5.1 Equity Modes

| Mode | Behavior |
|------|---------|
| `dynamic` (default) | Equity updated after each closed trade. Each new trade risks 1% of current equity at time of entry. |
| `fixed` | A starting balance is set at backtest initialization. Every trade risks 1% of that fixed starting balance. |

### 5.2 Contract Sizing Calculation

```
risk_amount  = account_equity × 0.01
stop_distance = 1 × ATR  (in points)
point_value  = $20 per point (NQ full contract)
contracts    = floor(risk_amount / (stop_distance × point_value))
```

- Always round **down** to the nearest whole contract.
- Minimum position: 1 contract. If calculated size < 1, the trade is **skipped**.
- For MNQ (micro): substitute `point_value = $2`.

---

## 6. Trade Types

Two trade types exist per session. Both may trigger in the same session if separate valid signals occur.

### 6.1 Reversion Trade

Price has displaced away from the True Price and is expected to return.

- **Long reversion:** price is below True Price + bullish displacement candle fires (moving back toward True Price)
- **Short reversion:** price is above True Price + bearish displacement candle fires (moving back toward True Price)

### 6.2 Momentum Trade

Price is displacing away from the True Price with strength and is expected to continue.

- **Long momentum:** price is above True Price + bullish displacement candle fires (moving further above True Price)
- **Short momentum:** price is below True Price + bearish displacement candle fires (moving further below True Price)

---

## 7. Entry Signal: Displacement Candle

The displacement candle is the sole required entry trigger. BOS is an optional confluence filter (Section 8).

### 7.1 Bullish Displacement Candle

All of the following must be true on candidate candle `C`:

1. `C.close > C_prev.high` — closes **above** the previous candle's high
2. `C.close > C.open` — candle body is bullish
3. Top wick ratio ≤ 15%:
   - `top_wick = C.high - C.close`
   - `candle_range = C.high - C.low`
   - `top_wick / candle_range ≤ 0.15`
   - If `candle_range == 0`: skip candle (invalid)

### 7.2 Bearish Displacement Candle

All of the following must be true on candidate candle `C`:

1. `C.close < C_prev.low` — closes **below** the previous candle's low
2. `C.close < C.open` — candle body is bearish
3. Bottom wick ratio ≤ 15%:
   - `bottom_wick = C.close - C.low`
   - `candle_range = C.high - C.low`
   - `bottom_wick / candle_range ≤ 0.15`
   - If `candle_range == 0`: skip candle (invalid)

### 7.3 Entry Price

Entry is taken at the **close of the displacement candle**. No pullback or retest required.

> In live trading this corresponds to a market order at the open of the next 1m candle. For backtesting, record entry as `C.close`.

---

## 8. Optional Confluence: Break of Structure (BOS)

Togglable via `require_bos: bool` (default `False`). When enabled, a valid stored swing must exist and be broken by the displacement candle's close before an entry is taken.

### 8.1 Swing Point Definitions

**Swing High:** Candle `C` is a confirmed swing high if:
```
C.high > C_prev.high  AND  C.high > C_next.high
```

**Swing Low:** Candle `C` is a confirmed swing low if:
```
C.low < C_prev.low  AND  C.low < C_next.low
```

> Confirmation requires a right neighbor (one-candle lag). The current candle is never a confirmed swing.

### 8.2 Swing State Tracking

Two slots are maintained per session:
- `stored_swing_high` — most recently confirmed swing high this session
- `stored_swing_low` — most recently confirmed swing low this session

**Rules:**

1. **Initialization:** Both slots empty at session open.
2. **Replacement:** When a new swing high is confirmed, it replaces `stored_swing_high` unconditionally — regardless of BOS state. Same for swing lows. Always store the most recent.
3. **BOS consumption:** When a candle's close breaks a stored swing (close > `stored_swing_high` or close < `stored_swing_low`), that slot is **immediately cleared**. This happens whether or not a trade is taken. The slot stays empty until a new swing of that direction is confirmed.
4. **BOS candle as new candidate:** The candle that triggered the BOS is itself eligible to be evaluated as a new swing point candidate in the same pass.
5. **Session reset:** Both slots clear at session end. Swing state does not carry between sessions.

### 8.3 BOS Condition

**Bullish BOS:** `C.close > stored_swing_high.high` AND `stored_swing_high` is not empty.

**Bearish BOS:** `C.close < stored_swing_low.low` AND `stored_swing_low` is not empty.

If the relevant slot is empty → BOS cannot be satisfied → no trade (when `require_bos = True`).

---

## 9. Stop Loss and Take Profit

### 9.1 ATR-Based Levels

ATR value is taken from the displacement candle's close.

| Level | Long | Short |
|-------|------|-------|
| **Stop Loss (SL)** | `entry − (1 × ATR)` | `entry + (1 × ATR)` |
| **Take Profit (TP)** | `entry + (1.5 × ATR)` | `entry − (1.5 × ATR)` |

Fixed **1.5R** reward-to-risk on every trade.

### 9.2 Reversion Trade Validity Filter

For **reversion trades only**: if the TP overshoots the True Price, the setup has insufficient edge and is **skipped**.

| Direction | Validity Condition |
|-----------|-------------------|
| Long reversion | `TP ≤ True Price` |
| Short reversion | `TP ≥ True Price` |

> Momentum trades have no TP boundary restriction.

---

## 10. Trade Management

### 10.1 Exit Conditions (priority order)

1. **Stop Loss hit:** price touches or crosses SL → exit at SL price.
2. **Take Profit hit:** price touches or crosses TP → exit at TP price.
3. **Session window closes:** neither SL nor TP hit by session end → close at market.

### 10.2 Backtesting Exit Execution

- Check each 1m candle's `high` and `low` against SL and TP levels each bar.
- If a single candle spans both SL and TP: assume **SL hit first** (conservative).
- Session end exit: use the `close` of the last candle within the session window.

### 10.3 One Trade Per Signal

Each valid displacement candle generates at most one trade. No pyramiding.

### 10.4 Concurrent Session Trades

Trades from different active sessions may be open simultaneously. Within a single session, once a trade is open, all further signals in that session are **ignored** until the trade closes.

### 10.5 Maximum Trades Per Day

A maximum of **3 trades** may be taken per trading day.

- A **trading day** = the 24-hour cycle from 20:00 ET (Asia open) through 11:00 ET the following morning (NY close).
- Trade count increments at the moment of **entry**.
- Once 3 trades are entered, all further signals across all remaining sessions that day are ignored.
- Counter **resets at 20:00 ET** each evening.

---

## 11. Signal Logic: Decision Tree Per Candle

Execute the following in order for each 1-minute candle `C` within a session window. The reference candle (19:59 / 02:59 / 09:29) is excluded from entry evaluation.

```
PRE-CHECKS:
1. Is RC a Doji (RC.close == RC.open)?       → YES: skip entire session
2. Has the daily trade limit (3) been reached? → YES: skip candle
3. Is a trade already open for this session?   → YES: skip candle (go to swing update)

SWING STATE UPDATE (runs every candle, regardless of trade state):
4. Is the candle two bars ago a confirmed swing high?
     → YES: replace stored_swing_high
5. Is the candle two bars ago a confirmed swing low?
     → YES: replace stored_swing_low

BOS DETECTION (runs every candle, clears slots on break):
6. Is stored_swing_high not empty AND C.close > stored_swing_high.high?
     → YES: bullish_bos = True, clear stored_swing_high
7. Is stored_swing_low not empty AND C.close < stored_swing_low.low?
     → YES: bearish_bos = True, clear stored_swing_low

ENTRY EVALUATION (only if pre-checks passed):
8. Is C a bullish displacement candle? (Section 7.1)
     → NO: go to step 12
     → YES:
         a. Determine type:
              C.close > True Price → Momentum Long
              C.close < True Price → Reversion Long
              C.close == True Price → skip
         b. If require_bos=True: was bullish_bos=True this candle? → NO: skip
         c. SL = C.close − ATR,  TP = C.close + (1.5 × ATR)
         d. If Reversion Long: TP ≤ True Price? → NO: skip
         e. contracts = floor(equity × 0.01 / (ATR × point_value))
              contracts < 1 → skip
         f. Enter Long at C.close

12. Is C a bearish displacement candle? (Section 7.2)
     → NO: skip candle
     → YES:
         a. Determine type:
              C.close < True Price → Momentum Short
              C.close > True Price → Reversion Short
              C.close == True Price → skip
         b. If require_bos=True: was bearish_bos=True this candle? → NO: skip
         c. SL = C.close + ATR,  TP = C.close − (1.5 × ATR)
         d. If Reversion Short: TP ≥ True Price? → NO: skip
         e. contracts = floor(equity × 0.01 / (ATR × point_value))
              contracts < 1 → skip
         f. Enter Short at C.close
```

---

## 12. Parameters Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `atr_period` | 14 | ATR lookback period |
| `atr_timeframe` | 1m | Timeframe for ATR calculation |
| `wick_threshold` | 0.15 | Max wick ratio in displacement direction (15% of candle range) |
| `rr_ratio` | 1.5 | Take profit as multiple of stop distance |
| `sl_atr_multiplier` | 1.0 | Stop loss = this × ATR |
| `risk_per_trade` | 0.01 | Fraction of equity risked per trade (1%) |
| `equity_mode` | `dynamic` | `dynamic` or `fixed` |
| `starting_equity` | user-defined | Required when `equity_mode = fixed` |
| `point_value` | 20 | USD per point per contract (NQ = $20, MNQ = $2) |
| `require_bos` | `False` | Toggle BOS as required confluence |
| `max_trades_per_day` | 3 | Max entries per trading day across all sessions |
| `sessions` | Asia, London, NY | Active sessions |
| `instrument` | NQ | Futures instrument |

---

## 13. Definitions Glossary

| Term | Definition |
|------|-----------|
| **RC** | Reference Candle. The 1m candle immediately before the session window opens (19:59, 02:59, 09:29 ET). |
| **True Price** | `RC.close`. The hypothesized fair value for that session. |
| **Body High** | `max(RC.open, RC.close)`. Top of the RC body, wicks excluded. |
| **Body Low** | `min(RC.open, RC.close)`. Bottom of the RC body, wicks excluded. |
| **True Reversion** | Price returns to touch the True Price. |
| **Max Reversion** | Price touches the opposite body boundary of the RC from the True Price. |
| **Displacement Candle** | A candle closing beyond the previous candle's range with ≤15% wick in the direction of the move. |
| **BOS** | Break of Structure. A candle closes beyond the most recently stored swing high (bullish) or swing low (bearish). Consumes that swing slot. |
| **Swing High** | A candle whose high is strictly greater than both its immediate neighbors' highs. Confirmed one candle after formation. |
| **Swing Low** | A candle whose low is strictly less than both its immediate neighbors' lows. Confirmed one candle after formation. |
| **Stored Swing** | The most recently confirmed swing high or low for the current session, held in state for BOS evaluation. Replaced by newer swings; cleared on BOS or session end. |
| **Reversion Trade** | Entry toward the True Price — price has moved away and is returning. |
| **Momentum Trade** | Entry away from the True Price — price is displacing further from it. |
| **ATR** | Average True Range, 14-period Wilder smoothing, on 1m bars. |
| **1R** | Risk unit. Entry to stop loss = 1× ATR. |
| **1.5R** | Take profit. Entry to TP = 1.5× ATR. |
| **Trading Day** | 20:00 ET through 11:00 ET the following morning (Asia open through NY close). |

