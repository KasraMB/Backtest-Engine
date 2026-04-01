"""
ICT / SMC Strategy for NQ Futures — v1.2
==========================================
Implements the full Power-of-Three (PO3) / ICT framework:
  Phase 1 (pre-market, once per day at first bar >= 09:30 ET):
    - Detect overnight 5m accumulation zones (PO3 conditions)
    - Detect 5m swing highs/lows (swing_n bars each side)
    - Find PO3 manipulation legs (accumulation -> swing -> CISD)
    - Draw STDV entry fib (manipulation leg) + OTE session fib (NDOG to extreme)
    - Detect POIs (OB, FVG, IFVG, BB, RB) on 1m / 5m / 15m / 30m
    - Compute session levels (PDH/PDL, Asia, London, NYPre, NDOG, NWOG, ...)
    - Validate each candidate fib level by POI/session-level confluence

  Phase 2 (09:30-11:00 ET, per 1m bar while flat):
    - Monitor validated levels for touch / body-penetration invalidation
    - Detect AFZ pattern on 1m at touched level
    - Compute TP (closest STDV extension with swing/POI confluence and min_rr)
    - Place limit order

  Trade management:
    - Breakeven: SL -> entry on first 1m swing in profit OR new 1m IFVG in direction
    - After breakeven: trail to protected swings (1m CISD then subsequent 1m swing)

All ATR uses Wilder's Smoothing (RMA / SMMA).
Performance: bar metadata pre-computed once; 15m/30m resampled with numpy only;
             POI detection vectorised with array shifts; swing cache used for TP.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date as _date_cls
from typing import Callable, List, Optional, Tuple

import numpy as np

try:
    from numba import njit as _numba_njit
    _njit = lambda fn: _numba_njit(fn, cache=True)   # always cache compiled objects
except ImportError:
    _njit = lambda fn: fn                             # plain Python fallback

from backtest.data.market_data import MarketData
from backtest.strategy.base import BaseStrategy
from backtest.strategy.enums import OrderType, SizeType
from backtest.strategy.order import Order
from backtest.strategy.update import OpenPosition, PositionUpdate, TICK_SIZE

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SESSION_START_MIN = 9 * 60 + 30    # 09:30 ET in minutes since midnight
SESSION_END_MIN   = 11 * 60         # 11:00 ET
ASIA_START_MIN    = 20 * 60         # 20:00 ET (prior calendar day)
LONDON_START_MIN  = 2 * 60          # 02:00 ET
LONDON_END_MIN    = 5 * 60          # 05:00 ET
NYPRE_START_MIN   = 8 * 60          # 08:00 ET
NYPRE_END_MIN     = SESSION_START_MIN
NYAM_START_MIN    = SESSION_START_MIN
NYAM_END_MIN      = SESSION_END_MIN
NYLUNCH_START_MIN = 12 * 60
NYLUNCH_END_MIN   = 13 * 60
NYPM_START_MIN    = 13 * 60 + 30
NYPM_END_MIN      = 16 * 60

OTE_FIBS   = [0.50, 0.618, 0.705, 0.79]
STDV_MULTS = [2.0, 2.25, 2.5, 3.0, 3.25, 3.5, 4.0, 4.25, 4.5]

# Session level kinds that represent highs (→ short SESSION_OTE setup)
# and lows (→ long SESSION_OTE setup).
_SESSION_KINDS_HIGH = frozenset({
    'PDH', 'Asia_H', 'London_H', 'NYPre_H', 'NYAM_H',
    'NYLunch_H', 'NYPM_H', 'Daily_H',
})
_SESSION_KINDS_LOW = frozenset({
    'PDL', 'Asia_L', 'London_L', 'NYPre_L', 'NYAM_L',
    'NYLunch_L', 'NYPM_L', 'Daily_L',
})

# ---------------------------------------------------------------------------
# Internal data structures
# ---------------------------------------------------------------------------

@dataclass
class AccumZone:
    start: int   # 5m bar index (absolute, inclusive)
    end:   int   # 5m bar index (absolute, inclusive)
    high:  float
    low:   float


@dataclass
class ManipLeg:
    """One confirmed PO3 manipulation leg."""
    direction:       int    # +1 = bullish manip (short trade), -1 = bearish manip (long trade)
    swing_lo_idx:    int    # 5m bar index of swing low
    swing_hi_idx:    int    # 5m bar index of swing high
    swing_lo_price:  float
    swing_hi_price:  float
    cisd_bar_idx:    int    # 5m bar index where CISD fired
    ndog:            float  # midnight open price for session fib anchor


@dataclass
class ValidLevel:
    price:            float
    direction:        int    # trade direction: +1 = long, -1 = short
    touched:          bool  = False
    invalidated:      bool  = False
    afz_zone_bar:     int   = -1   # furthest bar of the last consumed AFZ; -1 = none yet
    manip_leg:        Optional[ManipLeg] = None
    fib_type:         str   = ''   # 'OTE', 'STDV', 'SESSION_OTE'
    fib_value:        float = 0.0  # e.g. 0.618 for OTE, 2.0 for STDV
    confluence_kind:  str   = ''   # POI kind that validated this level (e.g. 'FVG', 'PDH')
    confluence_price: float = 0.0  # midpoint of the matching POI zone
    confluence_tf:    str   = ''   # timeframe of the matching POI ('1m','5m','15m','30m','session')
    ote_group:        Optional['SessionOTEGroup'] = None   # set for SESSION_OTE levels only


@dataclass
class SessionOTEGroup:
    """
    Tracks one SESSION_OTE anchor+extreme pair.
    0% = extreme (second anchor), 100% = first anchor.
    Levels: extreme + f * (anchor - extreme) for f in OTE_FIBS.
    """
    anchor_kind:  str    # session kind of the first anchor ('PDH', 'Asia_H', …)
    anchor_price: float  # first anchor (100% level) — invalidated when price returns here
    direction:    int    # -1 = short (high anchor), +1 = long (low anchor)
    extreme:      float  # current second anchor (0% level) — updated live during session
    extreme_kind: str   = ''   # named session level of the extreme if applicable (e.g. 'NYAM_L'), else ''
    invalidated:  bool  = False
    levels:       List['ValidLevel'] = field(default_factory=list)  # linked ValidLevels


@dataclass
class POI:
    """A Point-of-Interest zone (OB, BB, FVG, IFVG, RB, SESSION)."""
    kind:      str   # 'OB', 'BB', 'FVG', 'IFVG', 'RB', 'SESSION'
    direction: int   # +1 bullish, -1 bearish, 0 neutral (session)
    near:      float # 0% level
    mid:       float # 50% level
    far:       float # 100% level
    wick_tip:  float = 0.0
    wick_base: float = 0.0
    invalidated:  bool = False
    created_bar:  int  = -1   # absolute bar index (timeframe-specific) when POI formed; -1 = unknown
    session_kind: str  = ''   # non-empty only for SESSION POIs: 'PDH','PDL','Asia','London', etc.
    levels: List[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.levels = [self.near, self.mid, self.far]
        if self.kind == 'RB':
            wr = abs(self.wick_base - self.wick_tip)
            self.levels = [
                self.near, self.mid, self.far,
                self.wick_tip + 0.50  * wr,
                self.wick_tip + 0.618 * wr,
                self.wick_tip + 0.705 * wr,
                self.wick_tip + 0.79  * wr,
            ]


@dataclass
class PosCtx:
    """Per-trade management state."""
    direction:       int
    entry_price:     float
    sl_price:        float
    tp_price:        float
    at_breakeven:    bool  = False
    last_cisd_bar:   int   = -1
    last_cisd_price: float = 0.0
    # Incremental pivot tracking — avoids O(n²) rescans of growing windows.
    # be_next_pivot: next absolute bar index whose pivot confirmation to check
    #   for breakeven Condition A.  -1 = not yet initialised.
    be_next_pivot:      int   = -1
    # trail_best_sl: running best swing value for the trailing section.
    #   long:  running max of (swing_low - tick_offset) candidates; init -inf
    #   short: running min of (swing_high + tick_offset) candidates; init +inf
    trail_best_sl:      float = float('-inf')
    trail_next_pivot:   int   = -1   # next absolute pivot index to check
    trail_window_start: int   = -1   # sw_s value when trail state was last reset
    # Incremental FVG/IFVG/RB state for Condition B breakeven.
    # Each entry: (bar3_abs, bar1_abs, poi)
    #   bar3_abs — formation bar of the originating FVG (guards k<=b3 skip)
    #   bar1_abs — bar1 of the originating FVG (used for scan_s window filter)
    fvg_active: List[Tuple[int, int, 'POI']] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Standalone helpers (pure functions, no state)
# ---------------------------------------------------------------------------

def _round_up(price: float) -> float:
    return math.ceil(price / TICK_SIZE) * TICK_SIZE


def _round_down(price: float) -> float:
    return math.floor(price / TICK_SIZE) * TICK_SIZE


@_njit
def _wilder_atr(high: np.ndarray, low: np.ndarray,
                close: np.ndarray, period: int = 14) -> np.ndarray:
    """Wilder's ATR (RMA/SMMA). Returns array of same length as inputs."""
    n = len(high)
    if n == 0:
        return np.empty(0, dtype=np.float64)
    tr = np.empty(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for k in range(1, n):
        tr[k] = max(high[k] - low[k],
                    abs(high[k] - close[k - 1]),
                    abs(low[k]  - close[k - 1]))
    atr = np.empty(n, dtype=np.float64)
    atr[0] = tr[0]
    alpha = 1.0 / period
    for k in range(1, n):
        atr[k] = atr[k - 1] * (1.0 - alpha) + tr[k] * alpha
    return atr


def _wilder_atr_scalar(high: np.ndarray, low: np.ndarray,
                        close: np.ndarray, period: int, end_idx: int) -> float:
    """Compute a single ATR value at end_idx using Wilder's smoothing."""
    start = max(0, end_idx - period * 4)
    arr = _wilder_atr(high[start:end_idx + 1],
                      low[start:end_idx + 1],
                      close[start:end_idx + 1], period)
    return float(arr[-1]) if len(arr) > 0 else 0.0


@_njit
def _detect_swings(high: np.ndarray, low: np.ndarray,
                    n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Standard n-bar pivot detection (strict inequality).
    Returns (sh, sl) arrays — value at the pivot bar index, NaN elsewhere.
    """
    length = len(high)
    sh = np.full(length, np.nan)
    sl = np.full(length, np.nan)
    for k in range(n, length - n):
        # swing high
        if (np.all(high[k] > high[k - n:k]) and
                np.all(high[k] > high[k + 1:k + n + 1])):
            sh[k] = high[k]
        # swing low
        if (np.all(low[k] < low[k - n:k]) and
                np.all(low[k] < low[k + 1:k + n + 1])):
            sl[k] = low[k]
    return sh, sl


@_njit
def _detect_swings_confirmed_at(high: np.ndarray, low: np.ndarray,
                                  n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Same as _detect_swings but value placed at confirmation bar (pivot + n).
    Useful when you want to know when the pivot became known.
    """
    length = len(high)
    sh = np.full(length, np.nan)
    sl = np.full(length, np.nan)
    for k in range(n, length - n):
        conf = k + n
        if (np.all(high[k] > high[k - n:k]) and
                np.all(high[k] > high[k + 1:k + n + 1])):
            sh[conf] = high[k]
        if (np.all(low[k] < low[k - n:k]) and
                np.all(low[k] < low[k + 1:k + n + 1])):
            sl[conf] = low[k]
    return sh, sl


def _resample_5m_to_Nm(o5: np.ndarray, h5: np.ndarray, l5: np.ndarray,
                         c5: np.ndarray, group_size: int
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Resample 5m bars to N-minute bars by grouping `group_size` bars.
    Only complete groups are returned.  Pure numpy — no pandas.
    """
    n = len(o5)
    n_complete = (n // group_size) * group_size
    if n_complete == 0:
        empty = np.empty(0)
        return empty, empty, empty, empty
    o5c = o5[:n_complete].reshape(-1, group_size)
    h5c = h5[:n_complete].reshape(-1, group_size)
    l5c = l5[:n_complete].reshape(-1, group_size)
    c5c = c5[:n_complete].reshape(-1, group_size)
    return o5c[:, 0], h5c.max(axis=1), l5c.min(axis=1), c5c[:, -1]


def _detect_ob_vectorized(o: np.ndarray, h: np.ndarray,
                            l: np.ndarray, c: np.ndarray,
                            bar_offset: int = 0) -> List[POI]:
    """
    Detect Order Blocks and promote to Breaker Blocks when invalidated.
    Bullish OB: last bearish bar before bullish impulse closing above its high.
    Bearish OB: last bullish bar before bearish impulse closing below its low.
    Processes invalidation inline as bars advance.
    """
    n = len(o)
    pois: List[POI] = []
    if n < 2:
        return pois

    body_top = np.maximum(o, c)
    body_bot = np.minimum(o, c)
    bull = c > o
    bear = c < o

    # Track active OBs/BBs for inline invalidation
    active: List[POI] = []

    for k in range(1, n):
        # Update invalidation for active structures
        still_active: List[POI] = []
        for ob in active:
            if ob.invalidated:
                continue
            if ob.kind == 'OB':
                if ob.direction == 1 and c[k] < ob.far:
                    # Bullish OB invalidated → becomes bearish BB
                    ob.invalidated = True
                    bb = POI(kind='BB', direction=-1,
                             near=ob.near, mid=ob.mid, far=ob.far,
                             created_bar=k + bar_offset)
                    pois.append(bb)
                    still_active.append(bb)
                elif ob.direction == -1 and c[k] > ob.far:
                    # Bearish OB invalidated → becomes bullish BB
                    ob.invalidated = True
                    bb = POI(kind='BB', direction=1,
                             near=ob.near, mid=ob.mid, far=ob.far,
                             created_bar=k + bar_offset)
                    pois.append(bb)
                    still_active.append(bb)
                else:
                    still_active.append(ob)
            elif ob.kind == 'BB':
                if ob.direction == 1 and c[k] < ob.near:
                    ob.invalidated = True   # BB consumed
                elif ob.direction == -1 and c[k] > ob.near:
                    ob.invalidated = True
                else:
                    still_active.append(ob)
        active = still_active

        prev = k - 1
        # Bullish OB
        if bear[prev] and bull[k] and c[k] > h[prev]:
            ob = POI(kind='OB', direction=1,
                     near=body_top[prev],
                     mid=(body_top[prev] + body_bot[prev]) / 2.0,
                     far=body_bot[prev],
                     created_bar=k + bar_offset)
            pois.append(ob)
            active.append(ob)
        # Bearish OB
        if bull[prev] and bear[k] and c[k] < l[prev]:
            ob = POI(kind='OB', direction=-1,
                     near=body_bot[prev],
                     mid=(body_top[prev] + body_bot[prev]) / 2.0,
                     far=body_top[prev],
                     created_bar=k + bar_offset)
            pois.append(ob)
            active.append(ob)

    return pois


def _detect_fvg_vectorized(o: np.ndarray, h: np.ndarray,
                             l: np.ndarray, c: np.ndarray,
                             rb_min_wick_ratio: float = 0.3,
                             bar_offset: int = 0) -> List[POI]:
    """
    Detect FVGs, promote to IFVGs when invalidated, and detect Rejection Blocks.
    Processes bars sequentially; inline invalidation ensures no look-ahead.
    """
    n = len(o)
    pois: List[POI] = []
    if n < 3:
        return pois

    body_top = np.maximum(o, c)
    body_bot = np.minimum(o, c)
    body_sz  = body_top - body_bot

    # (bar3_abs_idx, poi) tuples for active structures
    active: List[Tuple[int, POI]] = []

    for k in range(2, n):
        bar1, bar3 = k - 2, k

        # Update invalidation for active FVGs/IFVGs (only check bars after bar3 formed)
        still_active: List[Tuple[int, POI]] = []
        for (b3, fvg) in active:
            if fvg.invalidated:
                continue
            if k <= b3:
                still_active.append((b3, fvg))
                continue
            if fvg.kind == 'FVG':
                if fvg.direction == 1 and c[k] < fvg.near:
                    fvg.invalidated = True
                    ifvg = POI(kind='IFVG', direction=-1,
                               near=fvg.near, mid=fvg.mid, far=fvg.far,
                               created_bar=k + bar_offset)
                    pois.append(ifvg)
                    still_active.append((b3, ifvg))
                    continue
                elif fvg.direction == -1 and c[k] > fvg.near:
                    fvg.invalidated = True
                    ifvg = POI(kind='IFVG', direction=1,
                               near=fvg.near, mid=fvg.mid, far=fvg.far,
                               created_bar=k + bar_offset)
                    pois.append(ifvg)
                    still_active.append((b3, ifvg))
                    continue
                else:
                    still_active.append((b3, fvg))
            elif fvg.kind == 'IFVG':
                if fvg.direction == 1 and c[k] < fvg.far:
                    fvg.invalidated = True
                    continue
                elif fvg.direction == -1 and c[k] > fvg.far:
                    fvg.invalidated = True
                    continue
                else:
                    still_active.append((b3, fvg))
            elif fvg.kind == 'RB':
                # Invalidated when any wick touches the 50% (mid) level
                if fvg.direction == 1 and l[k] <= fvg.mid:
                    fvg.invalidated = True
                    continue
                elif fvg.direction == -1 and h[k] >= fvg.mid:
                    fvg.invalidated = True
                    continue
                else:
                    still_active.append((b3, fvg))
            # Check Rejection Blocks against active (non-invalidated) FVGs
        # (done inside the above loop separately below)
        active = still_active

        # Check RB against all currently active bullish/bearish FVGs
        for (b3, fvg) in active:
            if fvg.invalidated or fvg.kind != 'FVG':
                continue
            if fvg.direction == 1:
                # Bullish FVG: lower wick must reach INTO the FVG zone
                wick_tip = l[k]
                wick_base = body_bot[k]
                if fvg.near <= wick_tip <= fvg.far:   # tip inside FVG gap
                    wick_sz = wick_base - wick_tip
                    if body_sz[k] > 1e-9 and wick_sz > rb_min_wick_ratio * body_sz[k]:
                        rb = POI(kind='RB', direction=1,
                                 near=wick_tip,
                                 mid=(wick_tip + wick_base) / 2.0,
                                 far=wick_base,
                                 wick_tip=wick_tip, wick_base=wick_base,
                                 created_bar=k + bar_offset)
                        pois.append(rb)
                        active.append((k, rb))
            elif fvg.direction == -1:
                # Bearish FVG: upper wick must reach INTO the FVG zone
                wick_tip = h[k]
                wick_base = body_top[k]
                if fvg.near >= wick_tip >= fvg.far:
                    wick_sz = wick_tip - wick_base
                    if body_sz[k] > 1e-9 and wick_sz > rb_min_wick_ratio * body_sz[k]:
                        rb = POI(kind='RB', direction=-1,
                                 near=wick_tip,
                                 mid=(wick_tip + wick_base) / 2.0,
                                 far=wick_base,
                                 wick_tip=wick_tip, wick_base=wick_base,
                                 created_bar=k + bar_offset)
                        pois.append(rb)
                        active.append((k, rb))

        # Detect new FVGs at bar k (bar1=k-2, bar3=k)
        if h[bar1] < l[bar3]:
            fvg = POI(kind='FVG', direction=1,
                      near=h[bar1],
                      mid=(h[bar1] + l[bar3]) / 2.0,
                      far=l[bar3],
                      created_bar=bar3 + bar_offset)
            pois.append(fvg)
            active.append((bar3, fvg))
        if l[bar1] > h[bar3]:
            fvg = POI(kind='FVG', direction=-1,
                      near=l[bar1],
                      mid=(l[bar1] + h[bar3]) / 2.0,
                      far=h[bar3],
                      created_bar=bar3 + bar_offset)
            pois.append(fvg)
            active.append((bar3, fvg))

    return pois


def _fvg_step(
    o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray,
    i: int, rb_min_wick_ratio: float,
    fvg_active: List[Tuple[int, int, 'POI']],
) -> None:
    """
    Advance the incremental FVG/IFVG/RB state machine by one bar.

    Replicates exactly one iteration of the _detect_fvg_vectorized inner loop
    at absolute bar index i, mutating fvg_active in-place.  No lookahead —
    only reads o/h/l/c at indices <= i.

    fvg_active entries: (bar3_abs, bar1_abs, poi)
      bar3_abs — absolute bar3 of the originating FVG (k<=b3 guard)
      bar1_abs — absolute bar1 of the originating FVG (scan_s window filter)
    """
    body_top_i = max(o[i], c[i])
    body_bot_i = min(o[i], c[i])
    body_sz_i  = body_top_i - body_bot_i

    # Step 1: invalidation pass
    still: List[Tuple[int, int, POI]] = []
    for b3, b1, poi in fvg_active:
        if poi.invalidated:
            continue
        if i <= b3:                         # not yet active — skip this bar
            still.append((b3, b1, poi))
            continue
        if poi.kind == 'FVG':
            if poi.direction == 1 and c[i] < poi.near:
                poi.invalidated = True
                still.append((b3, b1, POI(kind='IFVG', direction=-1,
                                          near=poi.near, mid=poi.mid, far=poi.far)))
            elif poi.direction == -1 and c[i] > poi.near:
                poi.invalidated = True
                still.append((b3, b1, POI(kind='IFVG', direction=1,
                                          near=poi.near, mid=poi.mid, far=poi.far)))
            else:
                still.append((b3, b1, poi))
        elif poi.kind == 'IFVG':
            if poi.direction == 1 and c[i] < poi.far:
                poi.invalidated = True
            elif poi.direction == -1 and c[i] > poi.far:
                poi.invalidated = True
            else:
                still.append((b3, b1, poi))
        elif poi.kind == 'RB':
            if poi.direction == 1 and l[i] <= poi.mid:
                poi.invalidated = True
            elif poi.direction == -1 and h[i] >= poi.mid:
                poi.invalidated = True
            else:
                still.append((b3, b1, poi))
    fvg_active.clear()
    fvg_active.extend(still)

    # Step 2: check RBs against active FVGs
    new_rbs: List[Tuple[int, int, POI]] = []
    for b3, b1, poi in fvg_active:
        if poi.kind != 'FVG':
            continue
        if poi.direction == 1:
            wick_tip  = l[i]
            wick_base = body_bot_i
            if poi.near <= wick_tip <= poi.far:
                wick_sz = wick_base - wick_tip
                if body_sz_i > 1e-9 and wick_sz > rb_min_wick_ratio * body_sz_i:
                    new_rbs.append((i, i, POI(kind='RB', direction=1,
                                              near=wick_tip,
                                              mid=(wick_tip + wick_base) / 2.0,
                                              far=wick_base,
                                              wick_tip=wick_tip, wick_base=wick_base)))
        elif poi.direction == -1:
            wick_tip  = h[i]
            wick_base = body_top_i
            if poi.far <= wick_tip <= poi.near:
                wick_sz = wick_tip - wick_base
                if body_sz_i > 1e-9 and wick_sz > rb_min_wick_ratio * body_sz_i:
                    new_rbs.append((i, i, POI(kind='RB', direction=-1,
                                              near=wick_tip,
                                              mid=(wick_tip + wick_base) / 2.0,
                                              far=wick_base,
                                              wick_tip=wick_tip, wick_base=wick_base)))
    fvg_active.extend(new_rbs)

    # Step 3: detect new FVG at bar i (bar1 = i-2, bar3 = i)
    if i >= 2:
        if h[i - 2] < l[i]:
            fvg_active.append((i, i - 2,
                               POI(kind='FVG', direction=1,
                                   near=h[i - 2],
                                   mid=(h[i - 2] + l[i]) / 2.0,
                                   far=l[i])))
        if l[i - 2] > h[i]:
            fvg_active.append((i, i - 2,
                               POI(kind='FVG', direction=-1,
                                   near=l[i - 2],
                                   mid=(l[i - 2] + h[i]) / 2.0,
                                   far=h[i])))


def _detect_all_pois(o: np.ndarray, h: np.ndarray, l: np.ndarray,
                      c: np.ndarray, rb_min_wick_ratio: float = 0.3,
                      bar_offset: int = 0) -> List[POI]:
    """Combine OB/BB and FVG/IFVG/RB detection."""
    return (_detect_ob_vectorized(o, h, l, c, bar_offset) +
            _detect_fvg_vectorized(o, h, l, c, rb_min_wick_ratio, bar_offset))


def _poi_matches_price(poi: POI, price: float, tol: float) -> bool:
    """True if any of the POI's valid levels is within tol of price."""
    if poi.invalidated:
        return False
    for lv in poi.levels:
        if abs(lv - price) <= tol:
            return True
    return False


def _cisd_check_at_bar(
    o: np.ndarray, c: np.ndarray,
    i: int, scan_start: int, direction: int,
    min_series: int, body_ratio: float,
) -> bool:
    """
    Check whether bar i completes a CISD pattern.

    Looks backward from i-1 for a consecutive series of bars in the opposite
    direction, then tests whether bar i closes through the series origin.
    O(series_length) — call once per bar instead of re-scanning the window.
    """
    j           = i - 1
    series_len  = 0
    series_last = -1
    prev_body   = 0.0

    if direction == 1:
        while j >= scan_start:
            body = o[j] - c[j] if c[j] < o[j] else 0.0
            if c[j] >= o[j]:
                break
            if series_len > 0 and body < body_ratio * prev_body:
                break
            series_len += 1
            series_last = j
            prev_body   = body
            j -= 1
        if series_len >= min_series and series_last >= 0:
            hi = o[i] if o[i] > c[i] else c[i]
            return hi > o[series_last]
    else:
        while j >= scan_start:
            body = c[j] - o[j] if c[j] > o[j] else 0.0
            if c[j] <= o[j]:
                break
            if series_len > 0 and body < body_ratio * prev_body:
                break
            series_len += 1
            series_last = j
            prev_body   = body
            j -= 1
        if series_len >= min_series and series_last >= 0:
            lo = o[i] if o[i] < c[i] else c[i]
            return lo < o[series_last]

    return False


def _cisd_scan(o: np.ndarray, c: np.ndarray, direction: int,
               min_series: int, min_body_ratio: float,
               start: int, end: int) -> Optional[int]:
    """
    Scan bars [start, end] for a CISD in `direction`.
    direction=+1  → bullish CISD (look for bearish series, then close above its open)
    direction=-1  → bearish CISD (look for bullish series, then close below its open)
    Returns absolute bar index of CISD bar, or None.
    """
    if end - start < min_series:
        return None
    for i in range(start + min_series, end + 1):
        j = i - 1
        series: List[int] = []
        if direction == 1:
            # Gather consecutive bearish bars going left from j
            while j >= start:
                body = o[j] - c[j] if c[j] < o[j] else 0.0
                if c[j] >= o[j]:  # not bearish
                    break
                if series and body < min_body_ratio * (o[series[-1]] - c[series[-1]]):
                    break
                series.append(j)
                j -= 1
            if len(series) < min_series:
                continue
            first_bar = series[-1]
            cisd_level = o[first_bar]
            if max(o[i], c[i]) > cisd_level:
                return i
        else:
            # Gather consecutive bullish bars going left from j
            while j >= start:
                body = c[j] - o[j] if c[j] > o[j] else 0.0
                if c[j] <= o[j]:  # not bullish
                    break
                if series and body < min_body_ratio * (c[series[-1]] - o[series[-1]]):
                    break
                series.append(j)
                j -= 1
            if len(series) < min_series:
                continue
            first_bar = series[-1]
            cisd_level = o[first_bar]
            if min(o[i], c[i]) < cisd_level:
                return i
    return None


# ---------------------------------------------------------------------------
# Main strategy class
# ---------------------------------------------------------------------------

class ICTSMCStrategy(BaseStrategy):
    """
    ICT / SMC NQ 1-minute futures strategy implementing strategy_spec.md v1.2.
    """

    trading_hours = None   # All hours; 09:30-11:00 ET gated inside generate_signals
    min_lookback  = 300

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def __init__(self, params: dict = None) -> None:
        super().__init__(params)
        p = params or {}

        # Core strategy parameters
        self.swing_n:                     int   = p.get('swing_n', 1)
        self.cisd_min_series_candles:     int   = p.get('cisd_min_series_candles', 2)
        self.cisd_min_body_ratio:         float = p.get('cisd_min_body_ratio', 0.5)
        self.rb_min_wick_ratio:           float = p.get('rb_min_wick_ratio', 0.3)
        self.confluence_tolerance_atr_mult:    float = p.get('confluence_tolerance_atr_mult', 0.18)
        self.tp_confluence_tolerance_atr_mult: float = p.get('tp_confluence_tolerance_atr_mult', 0.18)
        self.level_penetration_atr_mult:  float = p.get('level_penetration_atr_mult', 0.5)
        self.min_rr:                      float = p.get('min_rr', 5.0)
        self.tick_offset:                 float = p.get('tick_offset', 0.5)
        self.order_expiry_bars:           int   = p.get('order_expiry_bars', 10)
        self.session_level_validity_days: int   = p.get('session_level_validity_days', 2)
        self.contracts:                   int   = p.get('contracts', 1)
        # Fraction of the entry→TP distance that price must NOT reach before fill.
        # 1.0 (default) = cancel only if price hits TP exactly (current behaviour).
        # 0.5 = cancel if price travels 50 % of the way from entry to TP.
        self.cancel_pct_to_tp:            float = p.get('cancel_pct_to_tp', 1.0)
        # Minimum OTE (0%→100%) size expressed as a multiple of ATR.
        # Applies to OTE and SESSION_OTE levels; 0.0 (default) = no filter.
        self.min_ote_size_atr_mult:       float = p.get('min_ote_size_atr_mult', 0.0)
        # Max validated levels kept per session for each fib type.
        # Priority: biggest manipulation leg range; tiebreak by proximity to 09:30 open.
        # 0 = keep all (no limit).
        self.max_ote_per_session:         int   = p.get('max_ote_per_session', 1)
        self.max_stdv_per_session:        int   = p.get('max_stdv_per_session', 1)
        self.max_session_ote_per_session: int   = p.get('max_session_ote_per_session', 1)

        # PO3 accumulation parameters
        self.po3_lookback:               int   = p.get('po3_lookback', 6)
        self.po3_atr_mult:               float = p.get('po3_atr_mult', 0.95)
        self.po3_atr_len:                int   = p.get('po3_atr_len', 14)
        self.po3_band_pct:               float = p.get('po3_band_pct', 0.3)
        self.po3_vol_sens:               float = p.get('po3_vol_sens', 1.0)
        self.po3_max_r2:                 float = p.get('po3_max_r2', 0.4)
        self.po3_min_dir_changes:        int   = p.get('po3_min_dir_changes', 2)
        self.po3_min_candles:            int   = p.get('po3_min_candles', 3)
        self.po3_max_accum_gap_bars:     int   = p.get('po3_max_accum_gap_bars', 10)
        self.po3_min_manipulation_size_atr_mult: float = p.get('po3_min_manipulation_size_atr_mult', 0.0)

        # POI detection lookback limit (5m bars).  500 ≈ 2 weeks; keeps daily
        # POI recomputation O(constant) instead of O(growing history).
        self.poi_lookback_5m_bars: int = p.get('poi_lookback_5m_bars', 500)

        # allowed_setup_types: which fib_type values are eligible for entry.
        # Any subset of {'OTE', 'STDV', 'SESSION_OTE'}.  Default = all three.
        self.allowed_setup_types: frozenset = frozenset(
            p.get('allowed_setup_types', ['OTE', 'STDV', 'SESSION_OTE'])
        )

        # stdv_reverse: if True, STDV entries trade opposite to the distribution leg
        # (reversal at the extension — the original behaviour).
        # If False (default), STDV entries trade with the distribution leg (continuation).
        self.stdv_reverse: bool = p.get('stdv_reverse', False)

        # ---- Configurable validation sources ----
        # validation_poi_types: per fib_type, which POI kinds and session level names
        # are eligible to validate a level.  Candle-based kinds: OB BB FVG IFVG RB.
        # Session kinds: PDH PDL Asia London NYPre NYAM NYLunch NYPM Daily NDOG NWOG.
        _all_candle  = ['OB', 'BB', 'FVG', 'IFVG', 'RB']
        _all_session = [
            'PDH', 'PDL',
            'Asia_H', 'Asia_L', 'London_H', 'London_L',
            'NYPre_H', 'NYPre_L', 'NYAM_H', 'NYAM_L',
            'NYLunch_H', 'NYLunch_L', 'NYPM_H', 'NYPM_L',
            'Daily_H', 'Daily_L', 'NDOG', 'NWOG',
        ]
        _all_sources = _all_candle + _all_session
        self.validation_poi_types: dict = p.get('validation_poi_types', {
            'OTE':         list(_all_sources),
            'STDV':        list(_all_sources),
            'SESSION_OTE': list(_all_sources),
        })

        # validation_timeframes: per fib_type, which candle-based POI timeframes
        # are searched.  Does not affect session POIs (they have no timeframe).
        # Default: STDV uses 5m and above; OTE and SESSION_OTE use all timeframes.
        self.validation_timeframes: dict = p.get('validation_timeframes', {
            'OTE':         ['1m', '5m', '15m', '30m'],
            'STDV':        ['5m', '15m', '30m'],
            'SESSION_OTE': ['1m', '5m', '15m', '30m'],
        })

        # manip_leg_timeframe: timeframe used for swing detection and CISD when
        # identifying the manipulation leg.  '5m' or '1m'.
        self.manip_leg_timeframe: str = p.get('manip_leg_timeframe', '5m')

        # manip_leg_swing_depth: how many prior swings to skip when anchoring the
        # opposite end of the manipulation leg.
        # 1 = use the nearest prior swing (current default behaviour).
        # 2 = skip the nearest and use the second prior swing; etc.
        self.manip_leg_swing_depth: int = p.get('manip_leg_swing_depth', 1)

        # session_ote_anchors: which session level kinds act as SESSION_OTE first anchors.
        # Each anchor generates an independent fib: 100% = anchor, 0% = overnight extreme.
        # Options: 'PDH', 'PDL', 'Asia_H', 'Asia_L', 'London_H', 'London_L',
        #          'NYPre_H', 'NYPre_L', 'NYAM_H', 'NYAM_L', 'NYLunch_H', 'NYLunch_L',
        #          'NYPM_H', 'NYPM_L', 'Daily_H', 'Daily_L'
        self.session_ote_anchors: set = set(p.get('session_ote_anchors', [
            'PDH', 'PDL',
            'Asia_H', 'Asia_L', 'London_H', 'London_L',
            'NYPre_H', 'NYPre_L', 'NYAM_H', 'NYAM_L',
        ]))

        # Daily trade limit.  None = unlimited.
        # If set to N: take at most N trades per day.
        # Additionally, if a completed trade reached breakeven (proxy for a win),
        # no further trades are taken that day regardless of the count.
        self.max_trades_per_day: Optional[int] = p.get('max_trades_per_day', None)

        # ML hooks (pluggable callables)
        self.level_selection_policy: Callable = p.get(
            'level_selection_policy', self._default_level_policy)
        self.wick_penetration_policy: Callable = p.get(
            'wick_penetration_policy', lambda lv, ohlc, atr: False)
        self.breakeven_policy: Optional[Callable] = p.get('breakeven_policy', None)
        self.session_level_weight: Callable = p.get(
            'session_level_weight', lambda name, ctx: 1.0)

        # Per-day state
        self._phase1_done:        bool = False
        self._phase1_date_ord:    int  = -1
        self._validated_levels:   List[ValidLevel] = []
        self._session_ote_groups: List[SessionOTEGroup] = []
        self._session_atr:        float = 14.0   # fallback; overwritten by Phase 1

        # Daily trade limit tracking
        self._daily_trade_count: int  = 0
        self._daily_won:         bool = False
        self._consumed_afz_bars: set  = set()   # furthest_bar indices used for any order today
        self._pending_fib_levels: dict = {}     # bar_index → fib level list, for on_fill

        # Position management context
        self._pos_ctx: Optional[PosCtx] = None

        # Pre-computed bar metadata arrays (1m and 5m, filled on first call)
        self._bar_dates_ord:    Optional[np.ndarray] = None
        self._bar_times_min:    Optional[np.ndarray] = None
        self._bar_dates_5m_ord: Optional[np.ndarray] = None
        self._bar_times_5m_min: Optional[np.ndarray] = None

        # Cached POIs (computed in phase1)
        self._poi_1m:  List[POI] = []
        self._poi_5m:  List[POI] = []
        self._poi_15m: List[POI] = []
        self._poi_30m: List[POI] = []

        # Cached swing prices for TP confluence (computed in phase1)
        self._sw_1m_hi:  np.ndarray = np.empty(0)
        self._sw_1m_lo:  np.ndarray = np.empty(0)
        self._sw_5m_hi:  np.ndarray = np.empty(0)
        self._sw_5m_lo:  np.ndarray = np.empty(0)
        self._sw_15m_hi: np.ndarray = np.empty(0)
        self._sw_15m_lo: np.ndarray = np.empty(0)
        self._sw_30m_hi: np.ndarray = np.empty(0)
        self._sw_30m_lo: np.ndarray = np.empty(0)
        # Pre-concatenated swing arrays for _compute_tp (rebuilt each day in phase1)
        self._sw_lo_all: np.ndarray = np.empty(0)
        self._sw_hi_all: np.ndarray = np.empty(0)

    # ------------------------------------------------------------------
    # Pre-computation of bar metadata (called once)
    # ------------------------------------------------------------------

    @staticmethod
    def _to_ordinals(idx) -> np.ndarray:
        """
        Vectorised Gregorian ordinal computation for a pandas DatetimIndex.
        Equivalent to calling ts.toordinal() per element but ~100x faster.
        Formula: ordinal = (y-1)*365 + (y-1)//4 - (y-1)//100 + (y-1)//400 + doy
        Works correctly for timezone-aware indexes (uses local-timezone date).
        """
        y1  = idx.year.to_numpy(np.int64) - 1
        doy = idx.day_of_year.to_numpy(np.int64)
        return (y1 * 365 + y1 // 4 - y1 // 100 + y1 // 400 + doy).astype(np.int32)

    def _ensure_bar_metadata(self, data: MarketData) -> None:
        if self._bar_dates_ord is not None:
            return
        idx = data.df_1m.index
        self._bar_dates_ord = self._to_ordinals(idx)
        self._bar_times_min = (idx.hour * 60 + idx.minute).to_numpy(dtype=np.int32)
        idx5 = data.df_5m.index
        self._bar_dates_5m_ord = self._to_ordinals(idx5)
        self._bar_times_5m_min = (idx5.hour * 60 + idx5.minute).to_numpy(dtype=np.int32)

    # ------------------------------------------------------------------
    # Session levels (fully vectorised)
    # ------------------------------------------------------------------

    def _compute_session_levels(
            self, data: MarketData, bar_i: int
    ) -> Tuple[List[POI], Optional[float], Optional[float]]:
        """
        Build a list of SESSION POIs using precomputed date/time arrays.
        Also returns (ndog_price, nwog_price).
        All operations are numpy vectorised — no Python loops over bar indices.
        """
        h1 = data.high_1m
        l1 = data.low_1m
        o1 = data.open_1m
        dates = self._bar_dates_ord
        times = self._bar_times_min
        n = bar_i + 1       # bars up to and including bar_i
        tod = dates[bar_i]
        validity = self.session_level_validity_days
        results: List[POI] = []

        # Helper: bar indices up to bar_i
        arange_n = np.arange(n)

        def _make_sess(price: float, sk: str) -> POI:
            return POI(kind='SESSION', direction=0,
                       near=price, mid=price, far=price,
                       session_kind=sk)

        def _hl(mask: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
            """Return (max_high, min_low) for bars where mask is True, else None."""
            if not np.any(mask):
                return None, None
            return float(h1[:n][mask].max()), float(l1[:n][mask].min())

        # ---- PDH / PDL: prev calendar day 09:30 to 16:00 ----
        prev_d = tod - 1
        m = (dates[:n] == prev_d)
        hi, lo = _hl(m)
        if hi is not None:
            results.append(_make_sess(hi, 'PDH'))
            results.append(_make_sess(lo, 'PDL'))

        # ---- Asia H/L: 20:00 prev_d to 00:00 tod ----
        # 20:00-24:00 on prev_d (times >= 1200) plus 00:00 on tod (times == 0)
        for delta in range(0, validity + 1):
            pd = tod - 1 - delta
            am = (((dates[:n] == pd)    & (times[:n] >= ASIA_START_MIN)) |
                  ((dates[:n] == pd + 1) & (times[:n] == 0)))
            hi, lo = _hl(am)
            if hi is not None:
                results.append(_make_sess(hi, 'Asia_H'))
                results.append(_make_sess(lo, 'Asia_L'))

        # ---- London H/L: 02:00-05:00 ----
        for delta in range(0, validity + 1):
            pd = tod - delta
            m = (dates[:n] == pd) & (times[:n] >= LONDON_START_MIN) & (times[:n] < LONDON_END_MIN)
            hi, lo = _hl(m)
            if hi is not None:
                results.append(_make_sess(hi, 'London_H'))
                results.append(_make_sess(lo, 'London_L'))

        # ---- NY Pre H/L: 08:00-09:30 ----
        for delta in range(0, validity + 1):
            pd = tod - delta
            m = (dates[:n] == pd) & (times[:n] >= NYPRE_START_MIN) & (times[:n] < NYPRE_END_MIN)
            hi, lo = _hl(m)
            if hi is not None:
                results.append(_make_sess(hi, 'NYPre_H'))
                results.append(_make_sess(lo, 'NYPre_L'))

        # ---- NY AM H/L: 09:30-11:00 from prev day(s) ----
        for delta in range(1, validity + 1):
            pd = tod - delta
            m = (dates[:n] == pd) & (times[:n] >= NYAM_START_MIN) & (times[:n] < NYAM_END_MIN)
            hi, lo = _hl(m)
            if hi is not None:
                results.append(_make_sess(hi, 'NYAM_H'))
                results.append(_make_sess(lo, 'NYAM_L'))

        # ---- NY Lunch H/L: 12:00-13:00 from prev day(s) ----
        for delta in range(1, validity + 1):
            pd = tod - delta
            m = (dates[:n] == pd) & (times[:n] >= NYLUNCH_START_MIN) & (times[:n] < NYLUNCH_END_MIN)
            hi, lo = _hl(m)
            if hi is not None:
                results.append(_make_sess(hi, 'NYLunch_H'))
                results.append(_make_sess(lo, 'NYLunch_L'))

        # ---- NY PM H/L: 13:30-16:00 from prev day(s) ----
        for delta in range(1, validity + 1):
            pd = tod - delta
            m = (dates[:n] == pd) & (times[:n] >= NYPM_START_MIN) & (times[:n] < NYPM_END_MIN)
            hi, lo = _hl(m)
            if hi is not None:
                results.append(_make_sess(hi, 'NYPM_H'))
                results.append(_make_sess(lo, 'NYPM_L'))

        # ---- Daily H/L: current day up to bar_i ----
        m = dates[:n] == tod
        hi, lo = _hl(m)
        if hi is not None:
            results.append(_make_sess(hi, 'Daily_H'))
            results.append(_make_sess(lo, 'Daily_L'))

        # ---- NDOG: midnight open (00:00 ET today) ----
        ndog_mask = (dates[:n] == tod) & (times[:n] == 0)
        ndog_idx = np.where(ndog_mask)[0]
        ndog: Optional[float] = None
        if len(ndog_idx) > 0:
            ndog = float(o1[ndog_idx[0]])
            results.append(_make_sess(ndog, 'NDOG'))

        # ---- NWOG: open at most recent Monday 00:00 ET ----
        # Monday of current week; if today is Monday → previous Monday
        d_obj = _date_cls.fromordinal(tod)
        dow = d_obj.weekday()  # 0=Mon … 6=Sun
        monday_ord = tod - dow  # 0 on Monday (today), correct for all days
        nwog_mask = (dates[:n] == monday_ord) & (times[:n] == 0)
        nwog_idx = np.where(nwog_mask)[0]
        nwog: Optional[float] = None
        if len(nwog_idx) > 0:
            nwog = float(o1[nwog_idx[0]])
            results.append(_make_sess(nwog, 'NWOG'))

        return results, ndog, nwog

    # ------------------------------------------------------------------
    # SESSION_OTE level computation
    # ------------------------------------------------------------------

    def _compute_session_ote_levels(
            self,
            session_pois: List[POI],
            h1: np.ndarray,
            l1: np.ndarray,
            overnight_start_1m: int,
            n_1m: int,
            poi_by_fib: dict,
            phase1_atr: float = 0.0,
    ) -> Tuple[List[ValidLevel], List[SessionOTEGroup]]:
        """
        Build SESSION_OTE ValidLevels from session level POIs (independent of manip legs).

        For each anchor in session_ote_anchors:
          - High anchor (PDH, Asia_H, …): direction=-1, extreme=overnight min
          - Low anchor (PDL, Asia_L, …):  direction=+1, extreme=overnight max
          - level = extreme + f * (anchor - extreme)  for f in OTE_FIBS
          - Pre-session touch: skip levels already touched AFTER the extreme forms

        Returns (validated_levels, groups).
        """
        if n_1m <= overnight_start_1m:
            return [], []

        tol     = self.confluence_tolerance_atr_mult * phase1_atr
        groups:    List[SessionOTEGroup] = []
        validated: List[ValidLevel]      = []
        seen_prices: set                 = set()
        poi_list = poi_by_fib.get('SESSION_OTE', [])

        # Pre-compute NYAM prev-session bar order once so the loop can use it.
        _nyam_h_price: Optional[float] = None
        _nyam_l_price: Optional[float] = None
        _nyam_hi_abs: int = -1   # bar index of prev-NYAM session high
        _nyam_lo_abs: int = -1   # bar index of prev-NYAM session low
        if 'NYAM_H' in self.session_ote_anchors or 'NYAM_L' in self.session_ote_anchors:
            for _sp in session_pois:
                if _sp.session_kind == 'NYAM_H' and _nyam_h_price is None:
                    _nyam_h_price = _sp.near
                elif _sp.session_kind == 'NYAM_L' and _nyam_l_price is None:
                    _nyam_l_price = _sp.near
            if _nyam_h_price is not None and _nyam_l_price is not None:
                _tod  = self._bar_dates_ord[n_1m - 1]
                _nm   = ((self._bar_dates_ord[:n_1m] == _tod - 1)
                         & (self._bar_times_min[:n_1m] >= NYAM_START_MIN)
                         & (self._bar_times_min[:n_1m] < NYAM_END_MIN))
                _ni   = np.where(_nm)[0]
                if len(_ni) > 0:
                    _nyam_hi_abs = int(_ni[int(np.argmax(h1[_ni]))])
                    _nyam_lo_abs = int(_ni[int(np.argmin(l1[_ni]))])

        for poi in session_pois:
            sk = poi.session_kind
            if sk not in self.session_ote_anchors:
                continue

            anchor_price = poi.near
            ov_h = h1[overnight_start_1m:n_1m]
            ov_l = l1[overnight_start_1m:n_1m]

            if sk == 'NYAM_H':
                # Anchor = NYAM_H (100%).  Extreme must be NYAM_L and must have
                # formed *before* NYAM_H in the previous NYAM session.
                if (_nyam_l_price is None or _nyam_lo_abs < 0
                        or _nyam_lo_abs >= _nyam_hi_abs):
                    continue
                direction   = -1
                extreme     = _nyam_l_price
                # Touch check after the anchor (NYAM_H) formed — price passed through
                # the OTE levels on the way UP to NYAM_H, so only retouches after the
                # high matter.
                extreme_idx = _nyam_hi_abs
            elif sk == 'NYAM_L':
                # Anchor = NYAM_L (100%).  Extreme must be NYAM_H and must have
                # formed *before* NYAM_L.
                if (_nyam_h_price is None or _nyam_hi_abs < 0
                        or _nyam_hi_abs >= _nyam_lo_abs):
                    continue
                direction   = 1
                extreme     = _nyam_h_price
                extreme_idx = _nyam_lo_abs  # touch check after NYAM_L (anchor) formed
            elif sk in _SESSION_KINDS_HIGH:
                direction   = -1
                extreme     = float(ov_l.min())
                extreme_idx = int(np.argmin(ov_l)) + overnight_start_1m
            elif sk in _SESSION_KINDS_LOW:
                direction   = 1
                extreme     = float(ov_h.max())
                extreme_idx = int(np.argmax(ov_h)) + overnight_start_1m
            else:
                continue

            ote_size = abs(anchor_price - extreme)
            if ote_size < 1e-9:
                continue
            if (self.min_ote_size_atr_mult > 0.0
                    and ote_size < self.min_ote_size_atr_mult * phase1_atr):
                continue

            if sk == 'NYAM_H':
                _extreme_kind = 'NYAM_L'
            elif sk == 'NYAM_L':
                _extreme_kind = 'NYAM_H'
            else:
                # Check if the extreme price coincides with a named session level
                _extreme_kind = ''
                for _spoi in session_pois:
                    if abs(_spoi.near - extreme) <= 0.25 and _spoi.session_kind:
                        _extreme_kind = _spoi.session_kind
                        break
            group = SessionOTEGroup(
                anchor_kind=sk,
                anchor_price=anchor_price,
                direction=direction,
                extreme=extreme,
                extreme_kind=_extreme_kind,
            )

            any_valid = False
            for f in OTE_FIBS:
                level_price = extreme + f * (anchor_price - extreme)
                if level_price < 100:
                    continue

                # Pre-session touch: was this level hit AFTER the extreme?
                # (price passing through on the way TO the extreme doesn't count)
                after_start = extreme_idx + 1
                if after_start < n_1m:
                    if direction == -1:
                        touched_pre = bool(np.any(h1[after_start:n_1m] >= level_price))
                    else:
                        touched_pre = bool(np.any(l1[after_start:n_1m] <= level_price))
                    if touched_pre:
                        continue

                # POI confluence check
                confirmed_poi: Optional[POI] = None
                confirmed_tf: str = ''
                for (ctf, cpoi) in poi_list:
                    if cpoi.invalidated:
                        continue
                    # Direction filter: POI must match trade direction or be neutral (SESSION)
                    if cpoi.direction != 0 and cpoi.direction != direction:
                        continue
                    if _poi_matches_price(cpoi, level_price, tol):
                        confirmed_poi = cpoi
                        confirmed_tf = ctf
                        break

                if confirmed_poi is None:
                    continue

                key = (round(level_price, 2), direction)
                if key in seen_prices:
                    continue
                seen_prices.add(key)

                lv = ValidLevel(
                    price=level_price,
                    direction=direction,
                    fib_type='SESSION_OTE',
                    fib_value=f,
                    confluence_kind=(confirmed_poi.kind if confirmed_poi.kind != 'SESSION'
                                     else confirmed_poi.session_kind),
                    confluence_price=round((confirmed_poi.near + confirmed_poi.far) / 2, 2),
                    confluence_tf=confirmed_tf,
                    ote_group=group,
                )
                group.levels.append(lv)
                validated.append(lv)
                any_valid = True

            if any_valid:
                groups.append(group)

        return validated, groups

    # ------------------------------------------------------------------
    # Accumulation zone detection on 5m
    # ------------------------------------------------------------------

    def _detect_accum_zones(self, o5: np.ndarray, h5: np.ndarray,
                              l5: np.ndarray, c5: np.ndarray,
                              atr5: np.ndarray) -> List[AccumZone]:
        """
        Detect PO3 accumulation zones on 5m using four conditions:
        1. Low ATR%    2. Tight price band    3. Low R²    4. Sufficient direction flips
        """
        n = len(c5)
        lb = self.po3_lookback
        if n <= lb:
            return []

        atr_pct = atr5 / np.maximum(c5, 1e-9)
        zones: List[AccumZone] = []
        active_start: Optional[int] = None
        active_hi = -np.inf
        active_lo =  np.inf

        xs = np.arange(lb, dtype=float)
        xs_mean = xs.mean()
        xs_dev  = xs - xs_mean
        xs_sq   = float((xs_dev ** 2).sum())

        for k in range(lb, n):
            win_sl = slice(k - lb, k)
            c_win   = c5[win_sl]
            h_win   = h5[win_sl]
            l_win   = l5[win_sl]

            # Condition 1: low volatility
            cur_atr_pct = float(atr_pct[k])
            avg_atr_pct = float(atr_pct[win_sl].mean())
            cond1 = cur_atr_pct < self.po3_atr_mult * avg_atr_pct

            # Condition 2: tight price band
            rng_pct = (float(h_win.max()) - float(l_win.min())) / max(float(c5[k]), 1e-9)
            cur_atr = float(atr5[k])
            avg_atr = float(atr5[win_sl].mean())
            if avg_atr > 1e-9:
                adaptive_band = self.po3_band_pct * ((cur_atr / avg_atr) ** self.po3_vol_sens)
            else:
                adaptive_band = self.po3_band_pct
            cond2 = rng_pct < adaptive_band

            # Condition 3: no trend (R²)
            y_dev  = c_win - float(c_win.mean())
            ss_tot = float((y_dev ** 2).sum())
            if ss_tot < 1e-10 or xs_sq < 1e-10:
                r2 = 0.0
            else:
                cov = float((xs_dev * y_dev).sum())
                r2  = (cov ** 2) / (xs_sq * ss_tot)
            cond3 = r2 < self.po3_max_r2

            # Condition 4: direction flips
            dirs  = np.sign(np.diff(c_win))
            flips = int(np.sum(dirs[:-1] != dirs[1:]))
            cond4 = flips >= self.po3_min_dir_changes

            in_accum = cond1 and cond2 and cond3 and cond4

            if in_accum:
                if active_start is None:
                    active_start = k - lb
                active_hi = max(active_hi, float(h_win.max()))
                active_lo = min(active_lo, float(l_win.min()))
            else:
                if active_start is not None:
                    zone_len = k - active_start
                    if zone_len >= self.po3_min_candles:
                        zones.append(AccumZone(
                            start=active_start, end=k - 1,
                            high=active_hi, low=active_lo))
                    active_start = None
                    active_hi = -np.inf
                    active_lo =  np.inf

        if active_start is not None:
            zone_len = n - 1 - active_start
            if zone_len >= self.po3_min_candles:
                zones.append(AccumZone(
                    start=active_start, end=n - 1,
                    high=active_hi, low=active_lo))

        return self._merge_accum_zones(zones)

    def _merge_accum_zones(self, zones: List[AccumZone]) -> List[AccumZone]:
        if len(zones) < 2:
            return zones
        merged = [zones[0]]
        for z in zones[1:]:
            prev = merged[-1]
            gap     = z.start - prev.end - 1
            overlap = prev.high >= z.low and z.high >= prev.low
            if gap <= self.po3_lookback and overlap:
                merged[-1] = AccumZone(
                    start=prev.start, end=z.end,
                    high=max(prev.high, z.high),
                    low=min(prev.low,  z.low))
            else:
                merged.append(z)
        return merged

    # ------------------------------------------------------------------
    # PO3 manipulation leg detection
    # ------------------------------------------------------------------

    def _detect_manip_legs(self,
                            of_: np.ndarray, hf_: np.ndarray,
                            lf_: np.ndarray, cf_: np.ndarray,
                            abs_zones: List[AccumZone],
                            overnight_start: int,
                            ndog: float,
                            bar_map: Optional[np.ndarray] = None,
                            min_size: float = 0.0,
                            ) -> List[ManipLeg]:
        """
        For each accumulation zone, search for a swing followed by CISD.

        abs_zones and overnight_start must be in the same bar-index space as the
        passed OHLCV arrays (either 5m or 1m depending on manip_leg_timeframe).

        bar_map: if provided (1m mode), used to (a) compute gap in 5m bars for the
        po3_max_accum_gap_bars constraint, and (b) convert stored indices to 5m
        space so they match POI created_bar values in ManipLeg.
        """
        nf = len(cf_)
        sh_, sl_ = _detect_swings_confirmed_at(hf_, lf_, self.swing_n)
        legs: List[ManipLeg] = []
        depth = self.manip_leg_swing_depth

        for zone in abs_zones:
            search_start = zone.end + 1
            if search_start >= nf:
                continue

            # Best leg per direction for this zone.  A later leg that extends the
            # swing further in the same direction (higher high for bullish, lower low
            # for bearish) supersedes the earlier one — it is the true manipulation.
            best_a: Optional[ManipLeg] = None
            best_b: Optional[ManipLeg] = None

            # --- Scenario A: bullish manipulation (manip up → bearish setup / short trade)
            for sh_idx in range(search_start, nf):
                if np.isnan(sh_[sh_idx]):
                    continue
                # Gap constraint: always measured in 5m bars regardless of timeframe
                gap_5m = (int(bar_map[sh_idx]) - zone.end) if bar_map is not None else (sh_idx - zone.end)
                if gap_5m > self.po3_max_accum_gap_bars:
                    break

                # Find the Nth prior swing low (depth=1 → nearest, depth=2 → second, …)
                sl_idx_a: Optional[int] = None
                sl_price_a: Optional[float] = None
                found = 0
                for j in range(sh_idx - 1, max(overnight_start, 0) - 1, -1):
                    if not np.isnan(sl_[j]):
                        found += 1
                        if found == depth:
                            sl_idx_a = j
                            sl_price_a = float(sl_[j])
                            break
                if sl_idx_a is None:
                    continue

                leg_size = float(sh_[sh_idx]) - sl_price_a
                if leg_size < min_size:
                    continue

                cisd_bar = _cisd_scan(
                    of_, cf_, direction=-1,
                    min_series=self.cisd_min_series_candles,
                    min_body_ratio=self.cisd_min_body_ratio,
                    start=sh_idx, end=nf - 1)
                if cisd_bar is None:
                    continue

                # Convert indices to 5m space if operating in 1m mode
                sl_5m   = int(bar_map[sl_idx_a]) if bar_map is not None else sl_idx_a
                sh_5m   = int(bar_map[sh_idx])   if bar_map is not None else sh_idx
                cisd_5m = int(bar_map[cisd_bar])  if bar_map is not None else cisd_bar

                candidate = ManipLeg(
                    direction=1,
                    swing_lo_idx=sl_5m,
                    swing_hi_idx=sh_5m,
                    swing_lo_price=sl_price_a,
                    swing_hi_price=float(sh_[sh_idx]),
                    cisd_bar_idx=cisd_5m,
                    ndog=ndog,
                )
                # Replace if this swing is higher (further in the manip direction)
                if best_a is None or candidate.swing_hi_price > best_a.swing_hi_price:
                    best_a = candidate

            # --- Scenario B: bearish manipulation (manip down → bullish setup / long trade)
            for sl_idx_b in range(search_start, nf):
                if np.isnan(sl_[sl_idx_b]):
                    continue
                gap_5m = (int(bar_map[sl_idx_b]) - zone.end) if bar_map is not None else (sl_idx_b - zone.end)
                if gap_5m > self.po3_max_accum_gap_bars:
                    break

                # Find the Nth prior swing high
                sh_idx_b: Optional[int] = None
                sh_price_b: Optional[float] = None
                found = 0
                for j in range(sl_idx_b - 1, max(overnight_start, 0) - 1, -1):
                    if not np.isnan(sh_[j]):
                        found += 1
                        if found == depth:
                            sh_idx_b = j
                            sh_price_b = float(sh_[j])
                            break
                if sh_idx_b is None:
                    continue

                leg_size = sh_price_b - float(sl_[sl_idx_b])
                if leg_size < min_size:
                    continue

                cisd_bar = _cisd_scan(
                    of_, cf_, direction=1,
                    min_series=self.cisd_min_series_candles,
                    min_body_ratio=self.cisd_min_body_ratio,
                    start=sl_idx_b, end=nf - 1)
                if cisd_bar is None:
                    continue

                sl_5m   = int(bar_map[sl_idx_b]) if bar_map is not None else sl_idx_b
                sh_5m   = int(bar_map[sh_idx_b]) if bar_map is not None else sh_idx_b
                cisd_5m = int(bar_map[cisd_bar])  if bar_map is not None else cisd_bar

                candidate = ManipLeg(
                    direction=-1,
                    swing_lo_idx=sl_5m,
                    swing_hi_idx=sh_5m,
                    swing_lo_price=float(sl_[sl_idx_b]),
                    swing_hi_price=sh_price_b,
                    cisd_bar_idx=cisd_5m,
                    ndog=ndog,
                )
                # Replace if this swing is lower (further in the manip direction)
                if best_b is None or candidate.swing_lo_price < best_b.swing_lo_price:
                    best_b = candidate

            if best_a is not None:
                legs.append(best_a)
            if best_b is not None:
                legs.append(best_b)

        return legs

    # ------------------------------------------------------------------
    # Fib level generation from a manipulation leg
    # ------------------------------------------------------------------

    def _get_fib_levels(self, leg: ManipLeg) -> List[Tuple[float, int, str, float]]:
        """
        Return list of (candidate_price, trade_direction, fib_type, fib_value)
        from STDV and OTE fibs anchored to the manipulation leg.
        trade_direction: +1 = long, -1 = short.
        fib_type: 'OTE' | 'STDV'
        fib_value: fib ratio (OTE) or multiplier (STDV)
        Note: SESSION_OTE levels are computed independently in _compute_session_ote_levels.
        """
        sh = leg.swing_hi_price
        sl = leg.swing_lo_price
        rng = sh - sl
        if rng <= 0:
            return []

        levels: List[Tuple[float, int, str, float]] = []

        stdv_long  =  1 if self.stdv_reverse else -1
        stdv_short = -1 if self.stdv_reverse else  1

        if leg.direction == 1:
            # Bullish manipulation → bearish setup (short trade)
            # OTE measured from swing high downward (sh - f*rng)
            for f in OTE_FIBS:
                levels.append((sh - f * rng, -1, 'OTE', f))
            for m in STDV_MULTS:
                levels.append((sl - m * rng, stdv_long, 'STDV', m))
        else:
            # Bearish manipulation → bullish setup (long trade)
            # OTE measured from swing low upward (sl + f*rng)
            for f in OTE_FIBS:
                levels.append((sl + f * rng, 1, 'OTE', f))
            for m in STDV_MULTS:
                levels.append((sh + m * rng, stdv_short, 'STDV', m))

        return levels

    # ------------------------------------------------------------------
    # Phase 1 — pre-market preparation
    # ------------------------------------------------------------------

    def _run_phase1(self, data: MarketData, bar_i: int, tod_ord: int) -> None:
        tod = tod_ord
        dates5 = self._bar_dates_5m_ord   # precomputed once — no per-day reallocation
        times5 = self._bar_times_5m_min

        completed_5m = int(data.bar_map[bar_i])
        if completed_5m < 0:
            return

        # ATR at phase-1 bar — used for OTE size filter and tolerance scaling
        _phase1_atr = _wilder_atr_scalar(data.high_1m, data.low_1m, data.close_1m, 14, bar_i)
        self._session_atr = _phase1_atr

        # Identify overnight 5m bar index range (prev-day 16:00 to today 09:30)
        prev_d = tod - 1
        overnight_mask = (
            ((dates5 == prev_d) & (times5 >= NYPM_END_MIN)) |
            ((dates5 == tod)    & (times5 < SESSION_START_MIN))
        )
        # Restrict to completed bars
        overnight_mask[completed_5m + 1:] = False
        ov_indices = np.where(overnight_mask)[0]
        if len(ov_indices) < self.po3_min_candles:
            return

        overnight_start_5m = int(ov_indices[0])
        overnight_end_5m   = int(ov_indices[-1])

        # Slice full 5m arrays up to completed bar (for POI detection and swing detection)
        o5f = data.open_5m[:completed_5m + 1]
        h5f = data.high_5m[:completed_5m + 1]
        l5f = data.low_5m[:completed_5m + 1]
        c5f = data.close_5m[:completed_5m + 1]

        # Overnight-only slices for accumulation zone detection
        o5_ov = data.open_5m[overnight_start_5m:overnight_end_5m + 1]
        h5_ov = data.high_5m[overnight_start_5m:overnight_end_5m + 1]
        l5_ov = data.low_5m[overnight_start_5m:overnight_end_5m + 1]
        c5_ov = data.close_5m[overnight_start_5m:overnight_end_5m + 1]

        # ATR on extended window (for proper Wilder initialisation)
        atr5_start = max(0, overnight_start_5m - self.po3_atr_len * 4)
        atr5_ext = _wilder_atr(
            data.high_5m[atr5_start:overnight_end_5m + 1],
            data.low_5m[atr5_start:overnight_end_5m + 1],
            data.close_5m[atr5_start:overnight_end_5m + 1],
            self.po3_atr_len)
        atr5_ov = atr5_ext[overnight_start_5m - atr5_start:]

        # Detect accumulation zones (indices relative to overnight_start_5m)
        zones_rel = self._detect_accum_zones(o5_ov, h5_ov, l5_ov, c5_ov, atr5_ov)
        if not zones_rel:
            return

        # Convert relative indices to absolute 5m bar indices
        abs_zones = [AccumZone(
            start=z.start + overnight_start_5m,
            end=z.end   + overnight_start_5m,
            high=z.high, low=z.low)
            for z in zones_rel]

        # Session levels + NDOG
        session_pois, ndog, nwog = self._compute_session_levels(data, bar_i)
        if ndog is None:
            ndog = 0.0

        # Overnight 1m window — computed always (needed for SESSION_OTE even without manip legs)
        n_1m = bar_i + 1
        dates1 = self._bar_dates_ord
        times1 = self._bar_times_min
        ov_mask_1m = (
            ((dates1[:n_1m] == prev_d) & (times1[:n_1m] >= NYPM_END_MIN)) |
            ((dates1[:n_1m] == tod)    & (times1[:n_1m] < SESSION_START_MIN))
        )
        ov_idx_1m          = np.where(ov_mask_1m)[0]
        overnight_start_1m = int(ov_idx_1m[0]) if len(ov_idx_1m) > 0 else 0

        # Detect manipulation legs — timeframe determined by manip_leg_timeframe
        if self.manip_leg_timeframe == '1m':
            # Convert 5m zone boundaries to 1m space.
            # bar_map[i] = last completed 5m bar at 1m bar i.
            # The first 1m bar strictly after 5m zone.end completes is the first i
            # where bar_map[i] > zone.end.
            bm = data.bar_map[:n_1m]
            abs_zones_1m = [
                AccumZone(
                    start=int(np.searchsorted(bm, z.start, side='left')),
                    end=int(np.searchsorted(bm, z.end,   side='right')) - 1,
                    high=z.high, low=z.low)
                for z in abs_zones
            ]

            legs = self._detect_manip_legs(
                data.open_1m[:n_1m], data.high_1m[:n_1m],
                data.low_1m[:n_1m],  data.close_1m[:n_1m],
                abs_zones_1m, overnight_start_1m, ndog,
                bar_map=bm,
                min_size=self.po3_min_manipulation_size_atr_mult * _phase1_atr)
        else:
            legs = self._detect_manip_legs(
                o5f, h5f, l5f, c5f,
                abs_zones, overnight_start_5m, ndog,
                min_size=self.po3_min_manipulation_size_atr_mult * _phase1_atr)

        # Detect POIs on all timeframes — use a bounded lookback window so
        # cost stays O(poi_lookback_5m_bars) per day rather than O(total_history).
        n_1m = bar_i + 1
        poi5_start = max(0, completed_5m + 1 - self.poi_lookback_5m_bars)
        poi1_start = max(0, n_1m - self.poi_lookback_5m_bars * 5)

        o1r = data.open_1m[poi1_start:n_1m]
        h1r = data.high_1m[poi1_start:n_1m]
        l1r = data.low_1m[poi1_start:n_1m]
        c1r = data.close_1m[poi1_start:n_1m]
        self._poi_1m = _detect_all_pois(o1r, h1r, l1r, c1r, self.rb_min_wick_ratio,
                                         bar_offset=poi1_start)
        # Convert 1m created_bar indices to 5m indices so temporal comparisons
        # use the same scale as leg.swing_lo_idx / swing_hi_idx (both in 5m).
        for _p in self._poi_1m:
            if _p.created_bar >= 0:
                _cb = min(_p.created_bar, len(data.bar_map) - 1)
                _p.created_bar = data.bar_map[_cb]

        o5r = data.open_5m[poi5_start:completed_5m + 1]
        h5r = data.high_5m[poi5_start:completed_5m + 1]
        l5r = data.low_5m[poi5_start:completed_5m + 1]
        c5r = data.close_5m[poi5_start:completed_5m + 1]
        self._poi_5m = _detect_all_pois(o5r, h5r, l5r, c5r, self.rb_min_wick_ratio,
                                         bar_offset=poi5_start)

        # 15m and 30m via numpy resample — trim leading bars so groups align to
        # real clock boundaries (:00/:15/:30/:45 for 15m; :00/:30 for 30m).
        start_min_5m = int(self._bar_times_5m_min[poi5_start])
        trim15 = (15 - (start_min_5m % 15)) % 15 // 5
        trim30 = (30 - (start_min_5m % 30)) % 30 // 5
        o15, h15, l15, c15 = _resample_5m_to_Nm(o5r[trim15:], h5r[trim15:], l5r[trim15:], c5r[trim15:], 3)
        o30, h30, l30, c30 = _resample_5m_to_Nm(o5r[trim30:], h5r[trim30:], l5r[trim30:], c5r[trim30:], 6)

        self._poi_15m = _detect_all_pois(o15, h15, l15, c15, self.rb_min_wick_ratio) if len(c15) else []
        self._poi_30m = _detect_all_pois(o30, h30, l30, c30, self.rb_min_wick_ratio) if len(c30) else []

        # Build swing caches
        def _sw_prices(sh: np.ndarray, sl: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            return sh[~np.isnan(sh)], sl[~np.isnan(sl)]

        sw_start_1m = max(0, n_1m - 200)
        sh1, sl1 = _detect_swings(data.high_1m[sw_start_1m:n_1m],
                                   data.low_1m[sw_start_1m:n_1m], self.swing_n)
        self._sw_1m_hi, self._sw_1m_lo = _sw_prices(sh1, sl1)

        sw_start_5m = max(0, completed_5m + 1 - 78)
        sh5s, sl5s = _detect_swings(h5f[sw_start_5m:], l5f[sw_start_5m:], self.swing_n)
        self._sw_5m_hi, self._sw_5m_lo = _sw_prices(sh5s, sl5s)

        if len(h15):
            sw15 = max(0, len(h15) - 26)
            sh15, sl15 = _detect_swings(h15[sw15:], l15[sw15:], self.swing_n)
            self._sw_15m_hi, self._sw_15m_lo = _sw_prices(sh15, sl15)
        else:
            self._sw_15m_hi = self._sw_15m_lo = np.empty(0)

        if len(h30):
            sw30 = max(0, len(h30) - 13)
            sh30, sl30 = _detect_swings(h30[sw30:], l30[sw30:], self.swing_n)
            self._sw_30m_hi, self._sw_30m_lo = _sw_prices(sh30, sl30)
        else:
            self._sw_30m_hi = self._sw_30m_lo = np.empty(0)

        # Cache concatenated swing arrays used every bar in _compute_tp
        self._sw_lo_all = np.concatenate([self._sw_1m_lo, self._sw_5m_lo,
                                           self._sw_15m_lo, self._sw_30m_lo])
        self._sw_hi_all = np.concatenate([self._sw_1m_hi, self._sw_5m_hi,
                                           self._sw_15m_hi, self._sw_30m_hi])

        # Pre-build per-fib-type POI lists once per day.
        # candle-based POIs are filtered by validation_timeframes;
        # session POIs are filtered by session_kind via validation_poi_types.
        # Both are always filtered by validation_poi_types (kind membership).
        _tf_to_pois = {
            '1m':  self._poi_1m,
            '5m':  self._poi_5m,
            '15m': self._poi_15m,
            '30m': self._poi_30m,
        }
        _default_tfs     = ['1m', '5m', '15m', '30m']
        _default_sources = (set(self.validation_poi_types.get('OTE', [])) |
                            set(self.validation_poi_types.get('STDV', [])) |
                            set(self.validation_poi_types.get('SESSION_OTE', [])))

        def _build_poi_list(fib_type: str) -> List[Tuple[str, POI]]:
            allowed: set = set(self.validation_poi_types.get(fib_type, list(_default_sources)))
            tfs = self.validation_timeframes.get(fib_type, _default_tfs)
            result: List[Tuple[str, POI]] = []
            for tf in tfs:
                for poi in _tf_to_pois.get(tf, []):
                    if poi.kind in allowed:
                        result.append((tf, poi))
            for poi in session_pois:
                if poi.session_kind in allowed:
                    result.append(('session', poi))
            return result

        poi_by_fib: dict = {ft: _build_poi_list(ft) for ft in ('OTE', 'STDV', 'SESSION_OTE')}

        tol = self.confluence_tolerance_atr_mult * _phase1_atr
        seen: set = set()
        validated: List[ValidLevel] = []

        for leg in legs:
            # OTE size filter — skip this leg's OTE levels if the leg span is too small
            leg_span = leg.swing_hi_price - leg.swing_lo_price
            if (self.min_ote_size_atr_mult > 0.0
                    and leg_span < self.min_ote_size_atr_mult * _phase1_atr):
                # Only exclude OTE fib_types; STDV levels on this leg are still allowed
                _skip_ote_for_leg = True
            else:
                _skip_ote_for_leg = False

            # Only POIs that fully formed BEFORE the manipulation leg started
            # can validate levels anchored to that leg.
            leg_start = min(leg.swing_lo_idx, leg.swing_hi_idx)
            for (price, fdir, fib_type, fib_value) in self._get_fib_levels(leg):
                if _skip_ote_for_leg and fib_type == 'OTE':
                    continue
                if price < 100:           # implausible NQ price
                    continue
                key = (round(price, 2), fdir)
                if key in seen:
                    continue
                # Pre-session touch check: if price reached this level during overnight/pre-market,
                # the level is already consumed and should not be traded at session open.
                _h1 = data.high_1m[overnight_start_1m:n_1m]
                _l1 = data.low_1m[overnight_start_1m:n_1m]
                if fdir == 1 and bool(np.any(_l1 <= price)):
                    continue
                if fdir == -1 and bool(np.any(_h1 >= price)):
                    continue
                poi_list = poi_by_fib.get(fib_type, [])
                for (tf, poi) in poi_list:
                    if poi.invalidated:
                        continue
                    # Direction filter: POI must match trade direction or be neutral (SESSION)
                    if poi.direction != 0 and poi.direction != fdir:
                        continue
                    # Skip POIs that formed at or after the leg started.
                    # created_bar == -1 means unknown (15m/30m/session POIs) — allow them.
                    if poi.created_bar >= 0 and poi.created_bar >= leg_start:
                        continue
                    if _poi_matches_price(poi, price, tol):
                        seen.add(key)
                        validated.append(ValidLevel(
                            price=price,
                            direction=fdir,
                            manip_leg=leg,
                            fib_type=fib_type,
                            fib_value=fib_value,
                            confluence_kind=poi.kind if poi.kind != 'SESSION' else poi.session_kind,
                            confluence_price=round((poi.near + poi.far) / 2, 2),
                            confluence_tf=tf,
                        ))
                        break

        self._validated_levels = validated

        # SESSION_OTE levels — independent of manipulation legs;
        # computed from session level anchors and the overnight extreme.
        sote_levels, self._session_ote_groups = self._compute_session_ote_levels(
            session_pois, data.high_1m, data.low_1m,
            overnight_start_1m, n_1m, poi_by_fib, _phase1_atr,
        )
        self._validated_levels.extend(sote_levels)

        # Reduce to 1 OTE, 1 STDV, 1 SESSION_OTE for the session.
        # Priority: biggest manipulation leg range first; tiebreak by proximity to 09:30 open.
        session_open = float(data.open_1m[bar_i + 1]) if bar_i + 1 < len(data.open_1m) else None

        def _leg_range(lv: ValidLevel) -> float:
            if lv.manip_leg is not None:
                return lv.manip_leg.swing_hi_price - lv.manip_leg.swing_lo_price
            return 0.0

        def _dist_to_open(lv: ValidLevel) -> float:
            return abs(lv.price - session_open) if session_open is not None else 0.0

        _limits = {
            'OTE':         self.max_ote_per_session,
            'STDV':        self.max_stdv_per_session,
            'SESSION_OTE': self.max_session_ote_per_session,
        }
        filtered: List[ValidLevel] = []
        for fib_type in ('OTE', 'STDV', 'SESSION_OTE'):
            group = [lv for lv in self._validated_levels if lv.fib_type == fib_type]
            if not group:
                continue
            limit = _limits[fib_type]
            if limit == 0 or limit >= len(group):
                filtered.extend(group)
            else:
                ranked = sorted(group, key=lambda lv: (-_leg_range(lv), _dist_to_open(lv)))
                filtered.extend(ranked[:limit])
        self._validated_levels = filtered

    # ------------------------------------------------------------------
    # AFZ pattern detection
    # ------------------------------------------------------------------

    def _find_afz(self, data: MarketData, bar_i: int, trade_dir: int
                   ) -> Optional[Tuple[float, float, float, float, float, int]]:
        """
        Detect AFZ pattern at bar_i.
        Returns (entry_price, sl_price, zone_top, zone_bot, extreme) or None.
        extreme = lowestLow (long) or highestHigh (short).
        """
        o = data.open_1m
        h = data.high_1m
        l = data.low_1m
        c = data.close_1m

        if trade_dir == 1:
            # Bullish AFZ: current bar must be bullish
            if c[bar_i] <= o[bar_i]:
                return None
            # Skip immediately preceding bullish bars
            j = bar_i - 1
            while j >= 0 and c[j] > o[j]:
                j -= 1
            # Collect consecutive bearish bars
            bear_bars: List[int] = []
            while j >= 0 and c[j] < o[j]:
                bear_bars.append(j)
                j -= 1
            if not bear_bars:
                return None
            # Iterate all bearish bars: pick the one giving the highest zone top
            # where close[0] >= open[barIdx] (mirrors Pine Script logic exactly)
            zone_top = None
            furthest = -1
            for idx in bear_bars:
                if c[bar_i] >= o[idx]:
                    candidate_top = h[idx] if c[bar_i] >= h[idx] else o[idx]
                    if zone_top is None or candidate_top > zone_top:
                        zone_top = candidate_top
                        furthest = idx
            if furthest < 0 or zone_top is None:
                return None
            # Zone bottom (min open/close) and lowest wick across full range
            zone_bot  = float(np.minimum(o[furthest:bar_i + 1], c[furthest:bar_i + 1]).min())
            lowest_lo = float(l[furthest:bar_i + 1].min())
            entry = _round_up((zone_top + zone_bot) / 2.0)
            sl    = _round_down(zone_bot - (zone_bot - lowest_lo) / 2.0) - self.tick_offset
            return (entry, sl, zone_top, zone_bot, lowest_lo, furthest)

        else:
            # Bearish AFZ: current bar must be bearish
            if c[bar_i] >= o[bar_i]:
                return None
            j = bar_i - 1
            while j >= 0 and c[j] < o[j]:
                j -= 1
            bull_bars: List[int] = []
            while j >= 0 and c[j] > o[j]:
                bull_bars.append(j)
                j -= 1
            if not bull_bars:
                return None
            # Iterate all bullish bars: pick the one giving the lowest zone bottom
            # where close[0] <= open[barIdx]
            zone_bot = None
            furthest = -1
            for idx in bull_bars:
                if c[bar_i] <= o[idx]:
                    candidate_bot = l[idx] if c[bar_i] <= l[idx] else o[idx]
                    if zone_bot is None or candidate_bot < zone_bot:
                        zone_bot = candidate_bot
                        furthest = idx
            if furthest < 0 or zone_bot is None:
                return None
            zone_top   = float(np.maximum(o[furthest:bar_i + 1], c[furthest:bar_i + 1]).max())
            highest_hi = float(h[furthest:bar_i + 1].max())
            entry = _round_down((zone_top + zone_bot) / 2.0)
            sl    = _round_up(zone_top + (highest_hi - zone_top) / 2.0) + self.tick_offset
            return (entry, sl, zone_top, zone_bot, highest_hi, furthest)

    # ------------------------------------------------------------------
    # TP computation
    # ------------------------------------------------------------------

    def _compute_tp(self, trade_dir: int, entry: float, sl: float,
                     zone_top: float, zone_bot: float,
                     extreme: float) -> Optional[float]:
        """
        Compute TP as the closest STDV extension from the AFZ zone that:
        1) Has swing or POI confluence on 1m/5m/15m/30m
        2) Gives RR >= min_rr
        extreme = lowestLow (long) or highestHigh (short).
        """
        risk = abs(entry - sl)
        if risk < 1e-9:
            return None

        tol = self.tp_confluence_tolerance_atr_mult * self._session_atr

        if trade_dir == 1:
            # Bullish: fib_range = zone_top - lowestLow; extensions above zone_top
            fib_range = zone_top - extreme
            if fib_range <= 0:
                return None
            tp_candidates = [zone_top + m * fib_range for m in STDV_MULTS]
            sw_targets = self._sw_lo_all
        else:
            # Bearish: fib_range = highestHigh - zone_bot; extensions below zone_bot
            fib_range = extreme - zone_bot
            if fib_range <= 0:
                return None
            tp_candidates = [zone_bot - m * fib_range for m in STDV_MULTS]
            sw_targets = self._sw_hi_all

        # TP POIs: 5m + 15m + 30m only (per spec 7.2)
        tp_pois = self._poi_5m + self._poi_15m + self._poi_30m

        fallback_tp: Optional[float] = None
        for tp_price in tp_candidates:
            # RR check
            if abs(tp_price - entry) / risk < self.min_rr:
                continue
            # Track first extension clearing min_rr as fallback (no confluence required)
            if fallback_tp is None:
                fallback_tp = tp_price
            # Swing confluence (vectorised distance check)
            has_conf = (len(sw_targets) > 0 and
                        float(np.abs(sw_targets - tp_price).min()) <= tol)
            # POI confluence
            if not has_conf:
                for poi in tp_pois:
                    if not poi.invalidated and _poi_matches_price(poi, tp_price, tol):
                        has_conf = True
                        break
            if has_conf:
                return tp_price

        # No confluent level found — fall back to closest extension meeting min_rr
        return fallback_tp

    # ------------------------------------------------------------------
    # ML hook defaults
    # ------------------------------------------------------------------

    def _default_level_policy(self, active_levels: List[ValidLevel]) -> Optional[ValidLevel]:
        """Default: return the first active level."""
        return active_levels[0] if active_levels else None

    # ------------------------------------------------------------------
    # generate_signals — Phase 2 execution
    # ------------------------------------------------------------------

    def generate_signals(self, data: MarketData, i: int) -> Optional[Order]:
        self._ensure_bar_metadata(data)

        tod   = int(self._bar_dates_ord[i])
        t_min = int(self._bar_times_min[i])

        # Day reset on date change
        if tod != self._phase1_date_ord:
            self._phase1_done        = False
            self._phase1_date_ord    = tod
            self._validated_levels   = []
            self._session_ote_groups = []
            self._pos_ctx            = None   # any prior trade should be closed by EOD
            self._daily_trade_count  = 0
            self._daily_won          = False
            self._consumed_afz_bars  = set()
            self._pending_fib_levels = {}

        # Detect a completed trade from earlier today: generate_signals is only
        # called when flat, so _pos_ctx still set means the trade just closed.
        if self._pos_ctx is not None:
            if self._pos_ctx.at_breakeven:
                self._daily_won = True
            self._pos_ctx = None

        # Phase 1: run once at first bar with time >= 09:30 ET
        if not self._phase1_done and t_min >= SESSION_START_MIN:
            # Use bar_i-1 as data cutoff: phase 1 must complete before 09:30 (spec §5.0)
            self._run_phase1(data, max(0, i - 1), tod)
            self._phase1_done = True

        # Only enter trades 09:30-11:00 ET
        if t_min < SESSION_START_MIN or t_min >= SESSION_END_MIN:
            return None

        if not self._validated_levels:
            return None

        o_i = data.open_1m[i]
        h_i = data.high_1m[i]
        l_i = data.low_1m[i]
        c_i = data.close_1m[i]
        atr = _wilder_atr_scalar(data.high_1m, data.low_1m, data.close_1m, 14, i)
        pen  = self.level_penetration_atr_mult * atr

        # SESSION_OTE group management (runs before level touch checks)
        for grp in self._session_ote_groups:
            if grp.invalidated:
                continue
            # 100% invalidation: price returns to the first anchor
            if grp.direction == -1 and h_i >= grp.anchor_price:
                grp.invalidated = True
                for lv in grp.levels:
                    lv.invalidated = True
                continue
            if grp.direction == 1 and l_i <= grp.anchor_price:
                grp.invalidated = True
                for lv in grp.levels:
                    lv.invalidated = True
                continue
            # Dynamic extreme: if price makes a new extreme, recompute level prices
            # and reset touched state (only while no group level has been touched yet)
            any_touched = any(lv.touched for lv in grp.levels)
            if not any_touched:
                if grp.direction == -1 and l_i < grp.extreme:
                    grp.extreme = l_i
                    for lv in grp.levels:
                        lv.price  = grp.extreme + lv.fib_value * (grp.anchor_price - grp.extreme)
                        lv.touched = False
                elif grp.direction == 1 and h_i > grp.extreme:
                    grp.extreme = h_i
                    for lv in grp.levels:
                        lv.price  = grp.extreme + lv.fib_value * (grp.anchor_price - grp.extreme)
                        lv.touched = False

        # Update touch and invalidation state for all levels
        for lv in self._validated_levels:
            if lv.invalidated:
                continue
            lp = lv.price
            if l_i <= lp <= h_i:
                lv.touched = True
                body_lo = min(o_i, c_i)
                body_hi = max(o_i, c_i)
                if lv.direction == 1:
                    penetration = lp - body_lo   # how far body is below level
                    if penetration > pen:
                        lv.invalidated = True
                        continue
                else:
                    penetration = body_hi - lp   # how far body is above level
                    if penetration > pen:
                        lv.invalidated = True
                        continue
                # ML hook: wick-only penetration
                if self.wick_penetration_policy(lv, (o_i, h_i, l_i, c_i), atr):
                    lv.invalidated = True

        # Find candidates (touched, not invalidated)
        candidates = [lv for lv in self._validated_levels
                      if lv.touched and not lv.invalidated
                      and lv.fib_type in self.allowed_setup_types]
        if not candidates:
            return None

        chosen = self.level_selection_policy(candidates)
        if chosen is None:
            return None

        # Try AFZ at chosen level
        afz = self._find_afz(data, i, chosen.direction)
        if afz is None:
            return None

        entry, sl, zone_top, zone_bot, extreme, furthest_bar = afz

        # AFZ zone already consumed by any prior order today (globally, not per-level)
        if furthest_bar in self._consumed_afz_bars:
            return None

        # Compute TP
        tp = self._compute_tp(chosen.direction, entry, sl, zone_top, zone_bot, extreme)
        if tp is None:
            return None

        # Daily trade limit gate
        if self.max_trades_per_day is not None:
            if self._daily_trade_count >= self.max_trades_per_day:
                return None
            if self._daily_won:
                return None

        chosen.afz_zone_bar = furthest_bar
        self._consumed_afz_bars.add(furthest_bar)
        chosen.invalidated = True   # level fully consumed — no further AFZ entries

        # Snapshot all validated levels for trade inspector rendering
        self._pending_fib_levels[i] = [
            {"p": lv.price, "t": lv.fib_type, "d": lv.direction, "v": lv.fib_value}
            for lv in self._validated_levels
        ]

        if chosen.fib_type == 'OTE':
            fib_str = f"OTE {chosen.fib_value * 100:.1f}%"
        elif chosen.fib_type == 'STDV':
            fib_str = f"STDV {chosen.fib_value:.2f}x reversal"
        else:
            grp = chosen.ote_group
            if grp is not None:
                extreme_label = grp.extreme_kind if grp.extreme_kind else f"{grp.extreme:.2f}"
                anchor_label = f"{grp.anchor_kind} → {extreme_label}"
            else:
                anchor_label = ""
            fib_str = f"Session OTE {chosen.fib_value * 100:.1f}% ({anchor_label})" if anchor_label else f"Session OTE {chosen.fib_value * 100:.1f}%"
        if chosen.confluence_kind:
            tf_str = f"[{chosen.confluence_tf}] " if chosen.confluence_tf else ""
            conf_str = f"{tf_str}{chosen.confluence_kind} @ {chosen.confluence_price:.2f}"
        else:
            conf_str = ""
        trade_reason = f"{fib_str} | {conf_str}" if conf_str else fib_str

        return Order(
            direction=chosen.direction,
            order_type=OrderType.LIMIT,
            size_type=SizeType.CONTRACTS,
            size_value=float(self.contracts),
            limit_price=entry,
            sl_price=sl,
            tp_price=tp,
            expiry_bars=self.order_expiry_bars,
            cancel_above=(entry + self.cancel_pct_to_tp * (tp - entry)
                          if chosen.direction == 1 else None),
            cancel_below=(entry + self.cancel_pct_to_tp * (tp - entry)
                          if chosen.direction == -1 else None),
            trade_reason=trade_reason,
        )

    # ------------------------------------------------------------------
    # on_fill
    # ------------------------------------------------------------------

    def on_fill(self, position: OpenPosition, data: MarketData, bar_index: int) -> None:
        """Called immediately after fill. SL/TP already on position from Order."""
        position.set_initial_sl_tp(position.sl_price, position.tp_price)
        position.fib_levels = self._pending_fib_levels.pop(position.order_placed_bar, [])
        self._daily_trade_count += 1
        self._pos_ctx = PosCtx(
            direction=position.direction,
            entry_price=position.entry_price,
            sl_price=position.sl_price or 0.0,
            tp_price=position.tp_price or 0.0,
        )

    # ------------------------------------------------------------------
    # manage_position — breakeven + protected-swing trailing
    # ------------------------------------------------------------------

    def manage_position(self, data: MarketData, i: int,
                         position: OpenPosition) -> Optional[PositionUpdate]:
        if self._pos_ctx is None:
            return None

        ctx       = self._pos_ctx
        direction = ctx.direction
        entry     = ctx.entry_price
        current_sl = position.sl_price

        o = data.open_1m
        h = data.high_1m
        l = data.low_1m
        c = data.close_1m
        entry_bar = position.entry_bar

        # ----------------------------------------------------------------
        # Breakeven
        # ----------------------------------------------------------------
        if not ctx.at_breakeven:
            be_triggered = False

            if self.breakeven_policy is not None:
                be_triggered = bool(self.breakeven_policy(position, data, i))
            else:
                # Advance incremental FVG state machine one bar (feeds Condition B).
                if i >= entry_bar + 2:
                    _fvg_step(o, h, l, c, i, self.rb_min_wick_ratio, ctx.fvg_active)

                # Condition A: first 1m swing in profit after entry.
                # Incremental: at bar i the newly-confirmable pivot is i-sn.
                # We scan forward from the last unchecked pivot rather than
                # re-running _detect_swings_confirmed_at on the full window.
                n_since_entry = i - entry_bar + 1
                if n_since_entry > 2 * self.swing_n + 1:
                    sn = self.swing_n
                    if ctx.be_next_pivot < 0:
                        ctx.be_next_pivot = entry_bar + sn
                    new_last = i - sn
                    p = ctx.be_next_pivot
                    while p <= new_last and not be_triggered:
                        if direction == 1:
                            if (np.all(l[p] < l[p - sn:p]) and
                                    np.all(l[p] < l[p + 1:p + sn + 1]) and
                                    l[p] > entry):
                                be_triggered = True
                        else:
                            if (np.all(h[p] > h[p - sn:p]) and
                                    np.all(h[p] > h[p + 1:p + sn + 1]) and
                                    h[p] < entry):
                                be_triggered = True
                        p += 1
                    if not be_triggered:
                        ctx.be_next_pivot = new_last + 1

                # Condition B: new 1m IFVG in trade direction that is in profit.
                # Incremental: scan ctx.fvg_active instead of re-running
                # _detect_fvg_vectorized from scratch each bar.
                if not be_triggered and n_since_entry > 3:
                    scan_s = max(entry_bar, i - 15)
                    if i - scan_s >= 2:
                        for _b3, _b1, _poi in ctx.fvg_active:
                            if _b1 < scan_s:
                                continue
                            if _poi.kind == 'IFVG' and _poi.direction == direction:
                                if direction == 1 and _poi.near > entry:
                                    be_triggered = True
                                    break
                                elif direction == -1 and _poi.near < entry:
                                    be_triggered = True
                                    break

            if be_triggered:
                ctx.at_breakeven = True
                if current_sl is None:
                    return PositionUpdate(new_sl_price=entry)
                if direction == 1 and entry > current_sl:
                    return PositionUpdate(new_sl_price=entry)
                elif direction == -1 and entry < current_sl:
                    return PositionUpdate(new_sl_price=entry)

        # ----------------------------------------------------------------
        # Protected swing trailing (only after breakeven)
        # ----------------------------------------------------------------
        if ctx.at_breakeven:
            # Detect new 1m CISD in trade direction.
            # Incremental: only check bar i as a new CISD completion bar.
            # All earlier bars in the window were already checked on their
            # respective calls, so we never re-scan the same bars twice.
            scan_start = max(entry_bar, i - 25)
            if i - scan_start >= self.cisd_min_series_candles:
                if _cisd_check_at_bar(
                    o, c, i, scan_start, direction,
                    self.cisd_min_series_candles, self.cisd_min_body_ratio,
                ) and i > ctx.last_cisd_bar:
                    ctx.last_cisd_bar   = i
                    ctx.last_cisd_price = float(c[i])

            # Trail to protected swing after CISD.
            # Incremental: maintain a running best swing value (trail_best_sl)
            # and only scan newly-confirmable pivots each bar.  When sw_s moves
            # (new CISD detected), reset the trail state and rescan the new
            # (small, bounded-by-25-bar CISD-scan) window from scratch.
            if ctx.last_cisd_bar >= entry_bar:
                sw_s = ctx.last_cisd_bar + 1
                sw_e = i
                sn   = self.swing_n
                if sw_e - sw_s > 2 * sn:
                    # Reset if window start changed (new CISD fired)
                    if sw_s != ctx.trail_window_start:
                        ctx.trail_window_start = sw_s
                        ctx.trail_next_pivot   = sw_s + sn
                        ctx.trail_best_sl      = (float('-inf') if direction == 1
                                                  else float('inf'))
                    new_last = sw_e - sn
                    p = ctx.trail_next_pivot
                    if direction == 1:
                        while p <= new_last:
                            if (np.all(l[p] < l[p - sn:p]) and
                                    np.all(l[p] < l[p + 1:p + sn + 1])):
                                cand = float(l[p]) - self.tick_offset
                                if cand > ctx.trail_best_sl:
                                    ctx.trail_best_sl = cand
                            p += 1
                        ctx.trail_next_pivot = new_last + 1
                        base = current_sl if current_sl is not None else entry
                        best = max(base, ctx.trail_best_sl)
                        if current_sl is None or best > current_sl:
                            return PositionUpdate(new_sl_price=best)
                    else:
                        while p <= new_last:
                            if (np.all(h[p] > h[p - sn:p]) and
                                    np.all(h[p] > h[p + 1:p + sn + 1])):
                                cand = float(h[p]) + self.tick_offset
                                if cand < ctx.trail_best_sl:
                                    ctx.trail_best_sl = cand
                            p += 1
                        ctx.trail_next_pivot = new_last + 1
                        base = current_sl if current_sl is not None else entry
                        best = min(base, ctx.trail_best_sl)
                        if current_sl is None or best < current_sl:
                            return PositionUpdate(new_sl_price=best)

        return None
