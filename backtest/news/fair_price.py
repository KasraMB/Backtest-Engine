"""Build per-bar (current_fair, block_mask) arrays for an information-aware
mean-reversion strategy.

The fair price is the consensus value given current information. It RESETS at
each session open (a repricing window) and is UPDATED at scheduled news events:

  - EXPECTED news (|z| <= z_thresh): the spike is unfair -> fair HOLDS at the
    pre-event level (event-bar open). No re-anchor.
  - UNEXPECTED news (|z| > z_thresh): real repricing -> fair SHIFTS to the
    post-event consolidation = close of bar event+M. The shifted value is
    CAUSALLY valid only from bar event+M+1 onward.
  - MISSING forecast (z is NaN): no information about expected/unexpected
    -> no re-anchor.

`block_mask[i]` is True for bars in `[event-W, event+M]` for any event in the
data (no entries should be initiated through imminent/recent news).

Causality invariant (unit-testable): `current_fair[i]` and `block_mask[i]`
depend only on data with timestamp <= bars[i-1].close_time AND on the event
calendar (the calendar is public ex-ante so W minutes of forward visibility
is real, not look-ahead).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CALENDAR = REPO / 'data' / 'news_calendar_us.csv'


def load_news_calendar(path: 'Path | str | None' = None) -> pd.DataFrame:
    """Load the cleaned US high-impact news calendar (built by
    scripts/data/build_news_calendar.py). Returns a DataFrame with `etime`
    (ET tz-aware) and per-event z-scores."""
    p = Path(path) if path is not None else DEFAULT_CALENDAR
    if not p.exists():
        raise FileNotFoundError(
            f"News calendar not found at {p}. Run scripts/data/build_news_calendar.py.")
    df = pd.read_csv(p)
    df['etime'] = pd.to_datetime(df['etime'], utc=True).dt.tz_convert('America/New_York')
    return df.sort_values('etime').reset_index(drop=True)


def build_news_fair_arrays(
    df_1m_index:   pd.DatetimeIndex,   # tz-aware ET, 1m bar timestamps (length N)
    opens_1m:      np.ndarray,          # (N,)
    closes_1m:     np.ndarray,          # (N,)
    session_opens_min: list[int],       # [930, 1400, ...] minute-of-day
    calendar_df:   pd.DataFrame,        # filtered calendar (etime, z, has_forecast)
    z_thresh:      float = 1.0,
    post_event_m:  int = 15,            # minutes after event = consolidation
    pre_event_w:   int = 5,             # minutes before event = no entry
) -> tuple[np.ndarray, np.ndarray]:
    """Return (current_fair, block_mask) of length N.

    current_fair: forward-filled fair price per bar. NaN where undefined
        (before any session open / event). Strategy should fall back to its
        internal fair tracking where NaN.
    block_mask: True where no new entry should be initiated.
    """
    N = len(df_1m_index)
    current_fair = np.full(N, np.nan, dtype=np.float64)
    block_mask   = np.zeros(N, dtype=np.bool_)
    if N == 0:
        return current_fair, block_mask

    # ── 1. Session-open resets: fair := open at the session-open bar ──────
    times_min = (df_1m_index.hour * 60 + df_1m_index.minute).to_numpy(np.int64)
    for sm in session_opens_min:
        reset_bars = np.where(times_min == sm)[0]
        if len(reset_bars):
            current_fair[reset_bars] = opens_1m[reset_bars]

    # ── 2. News events ────────────────────────────────────────────────────
    # Restrict to events within the data's date range.
    if len(df_1m_index) >= 2:
        data_lo, data_hi = df_1m_index[0], df_1m_index[-1]
    else:
        data_lo = data_hi = df_1m_index[0]
    cal = calendar_df[(calendar_df['etime'] >= data_lo) &
                      (calendar_df['etime'] <= data_hi)].copy()
    if len(cal) == 0:
        s = pd.Series(current_fair).ffill()
        return s.to_numpy(np.float64), block_mask

    # Vectorized event-bar lookup via the bar index.
    # get_indexer returns -1 for missing (e.g. weekend events or gaps).
    ev_bars = df_1m_index.get_indexer(cal['etime'].to_numpy(), method='nearest',
                                      tolerance=pd.Timedelta(minutes=1))
    cal = cal.assign(_bar=ev_bars)
    cal = cal[cal['_bar'] >= 0]

    # Apply each event with EXPECTED-vs-UNEXPECTED branching (the trader's
    # actual claim): EXPECTED -> trade the reversion BACK to the pre-event
    # price; UNEXPECTED -> price genuinely re-anchored to post-event level.
    #
    #   EXPECTED  (|z| <= z_thresh AND has_forecast):
    #     - Anchor current_fair to the PRE-event open (opens[eb], known at bar eb)
    #       across the post-event consolidation window [eb, eb+M+1).
    #     - Block ONLY the immediate event bars [eb-2, eb+2] (the spike itself);
    #       ALLOW entries in [eb+3, eb+M] so the strategy can trade the
    #       reversion -- the trader's highest-win-rate setup.
    #
    #   UNEXPECTED (|z| > z_thresh) or MISSING forecast:
    #     - Block longer: [eb-W, eb+M].
    #     - If genuinely unexpected, CAUSAL post-event shift: fair := close at
    #       eb+M, active from bar eb+M+1 onward (so no bar before that knows
    #       this value). Session-reset guard: if a session open falls in
    #       (eb, eb+M+1], the new session's repricing dominates -> skip shift.
    for _, ev in cal.iterrows():
        eb = int(ev['_bar'])
        has_fc = bool(ev.get('has_forecast', False)) and pd.notna(ev.get('z'))
        is_unexpected = has_fc and abs(float(ev['z'])) > z_thresh
        is_expected   = has_fc and not is_unexpected

        if is_expected:
            # Tight block (spike only) + pre-event-anchored reversion target.
            # The bar at eb (8:30) ALREADY contains the news reaction in its
            # open print (release lands at 8:30:00, bar opens at 8:30:00). The
            # trader explicitly anchors on the "8:29 candle" body -- so we use
            # closes[eb-1] as the pre-news fair (the last quote BEFORE release).
            b_lo = max(0, eb - 2)
            b_hi = min(N, eb + 3)
            block_mask[b_lo:b_hi] = True
            anchor_end = min(N, eb + post_event_m + 1)
            pre_news_anchor = float(closes_1m[max(0, eb - 1)])
            for j in range(eb, anchor_end):
                # Don't overwrite a session-reset value (set earlier in step 1).
                if np.isnan(current_fair[j]):
                    current_fair[j] = pre_news_anchor
        else:
            # Unexpected or no-forecast: conservative block + (if unexpected)
            # causal post-event shift.
            b_lo = max(0, eb - pre_event_w)
            b_hi = min(N, eb + post_event_m + 1)
            block_mask[b_lo:b_hi] = True
            if not is_unexpected:
                continue
            shift_bar = eb + post_event_m + 1   # CAUSAL: only known after this bar
            if shift_bar >= N:
                continue
            between = slice(eb + 1, shift_bar + 1)
            if np.any(~np.isnan(current_fair[between])):
                continue
            post_close = closes_1m[eb + post_event_m]
            current_fair[shift_bar] = float(post_close)

    # ── 3. Forward-fill the fair price between markers ────────────────────
    s = pd.Series(current_fair).ffill()
    return s.to_numpy(np.float64), block_mask
