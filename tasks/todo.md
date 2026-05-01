# New Strategies Implementation Plan

## Strategies to implement (from Profitable Strategies zip)
1. IBS (Internal Bar Strength) — daily mean reversion
2. TrendFollowing (TradingWithRayner) — swing/multi-day trend
3. TurtleSoup (Linda Raschke) — 4H contrarian breakout fade

## Data available
- `data/NQ_1day_full_data.parquet` — D1 OHLCV (used by ORB already)
- `data/NQ_1m.parquet` — 1m bars (engine backbone)

---

## Step 1: IBS Strategy [ ]
File: `strategies/ibs_strategy.py`

Logic:
- IBS = (D1_close - D1_low) / (D1_high - D1_low) from prior day
- At RTH open (9:30 ET): if IBS < low_thresh → long; if IBS > high_thresh → short
- SL = ATR(14) × sl_atr_mult
- TP = SL × rr_ratio
- Force exit at eod_exit_time (e.g. 15:00)
- signal_bar_mask: only 9:30 bars → very fast sweep

Grid params: ibs_low_thresh, ibs_high_thresh, sl_atr_mult, rr_ratio, use_200sma_filter

## Step 2: TrendFollowing Strategy [ ]
File: `strategies/trend_following_strategy.py`

Logic:
- D1 indicators: SMA(50), SMA(100), 200-bar highest-high, 200-bar lowest-low
- At each RTH open: check if D1 close broke out of 200-bar range + SMA trend aligned
- Enter market at 9:30 open, hold position (no EOD exit — eod_exit_time=23:59)
- SL: ATR(14) × atr_mult trailing (updated each bar in manage_position)
- No TP (pure trend following exits on stop only)
- One position at a time, can hold multiple days

Grid params: sma_fast, sma_slow, lookback, atr_mult (sl multiplier)

## Step 3: TurtleSoup Strategy [ ]
File: `strategies/turtle_soup_strategy.py`

Logic:
- Derive 4H bars from 1m data (bars at 00:00, 04:00, 08:00, 12:00, 16:00, 20:00)
- Track 20-bar 4H rolling high/low
- Breakout above 20H high: watch for reversal → enter SHORT when 1m closes back below high
- Breakout below 20H low: watch for reversal → enter LONG when 1m closes back above low
- SL at the breakout extreme (recent 4H bar high/low)
- TP = SL × rr_ratio (2:1 or 3:1)

Grid params: lookback_4h, rr_ratio, sl_buffer_atr, session filter

## Step 4: Sweep scripts [ ]
- `sweep7_ibs_trend.py` — sweep IBS + TrendFollowing on IS data (2019-2024)
- `sweep8_turtle_soup.py` — sweep TurtleSoup

## Step 5: Review results and pick best combos [ ]

## Rules
- Implement one at a time, test after each
- signal_bar_mask on every strategy for sweep speed
- IS data only (2019-2024) — same as sweep6
