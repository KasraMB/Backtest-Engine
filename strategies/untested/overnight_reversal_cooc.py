"""
Overnight-Intraday Reversal (CO-OC) — Della Corte, Kosowski & Wang (2015/2021)
"Market Closure and Short-Term Reversal"
Confirmed across equity index futures including NQ. Sharpe ratio ~4 in pure
futures replications (QuantReturns, Oct 2025).

Mechanism
─────────
The overnight return (previous RTH close → today's RTH open) negatively
predicts the intraday return (RTH open → RTH close).

If NQ gaps UP overnight, it tends to REVERSE (drift down) during RTH.
If NQ gaps DOWN overnight, it tends to REVERSE (drift up) during RTH.

Signal:
    overnight_return = (today_open - prev_close) / prev_close

Trade rule:
    overnight_return > +threshold → SHORT at RTH open
    overnight_return < -threshold → LONG  at RTH open

The threshold filters noise and improves Sharpe (Rosa 2022 confirms this
for related strategies). Default threshold = 0.10% (10 bps).

The mechanism: overnight moves are driven by thin liquidity and order
imbalances from institutional hedgers. When RTH liquidity restores, the
imbalance unwinds — creating the reversal. Stronger in high-VIX regimes.

Entry:  MARKET at RTH open (09:30 bar open price)
SL:     ATR multiple from fill (protects against gap continuations)
TP:     risk_reward multiple or ATR multiple
Exit:   SL, TP, or EOD (16:00 ET)
One trade per day.

Params dict keys:
    threshold_pct       : float — min |overnight_return| % to trade. Default 0.10
    gap_filter_mult     : float — additional filter: |gap| must exceed
                                  gap_filter_mult × ATR14. Default 0.0 (off)
    sl_type             : str   — "atr_multiple"|"fixed_points"|"pct". Default "atr_multiple"
    sl_value            : float — multiplier for sl_type. Default 1.5
    tp_type             : str   — "risk_reward"|"atr_multiple"|"fixed_points". Default "risk_reward"
    tp_value            : float — multiplier. Default 2.0
    atr_length          : int   — ATR period. Default 14
    trail_points        : float — trailing stop distance. None = off. Default None
    trail_activation    : float — profit to activate trail. Default 0.0
    contracts           : int   — fixed contracts. Default 1
"""
from __future__ import annotations

from datetime import time, date
from typing import Optional

import numpy as np

from backtest.data.market_data import MarketData
from backtest.strategy.base import BaseStrategy
from backtest.strategy.enums import OrderType, SizeType
from backtest.strategy.order import Order
from backtest.strategy.update import OpenPosition, PositionUpdate


def _atr(highs, lows, closes, period, i) -> float:
    if i < 1:
        return 0.0
    start = max(1, i - period * 3)
    trs = np.maximum(
        highs[start:i+1] - lows[start:i+1],
        np.abs(highs[start:i+1] - closes[start-1:i]),
        np.abs(lows[start:i+1]  - closes[start-1:i]),
    )
    if len(trs) == 0:
        return 0.0
    atr = float(trs[0])
    for tr in trs[1:]:
        atr = (atr * (period - 1) + float(tr)) / period
    return atr


class OvernightReversalCOOC(BaseStrategy):
    """
    Overnight-intraday reversal: overnight gap predicts intraday direction
    in the OPPOSITE direction.  One trade per day, entered at RTH open.
    """

    trading_hours = [(time(9, 30), time(16, 0))]
    min_lookback  = 20

    def __init__(self, params: dict = None):
        super().__init__(params)
        p = params or {}

        self.threshold_pct:   float          = p.get("threshold_pct",   0.10)
        self.gap_filter_mult: float          = p.get("gap_filter_mult",  0.0)
        self.sl_type:         str            = p.get("sl_type",          "atr_multiple")
        self.sl_value:        float          = p.get("sl_value",         1.5)
        self.tp_type:         str            = p.get("tp_type",          "risk_reward")
        self.tp_value:        float          = p.get("tp_value",         2.0)
        self.atr_length:      int            = p.get("atr_length",       14)
        self.trail_points:    Optional[float] = p.get("trail_points",    None)
        self.trail_activation: float         = p.get("trail_activation", 0.0)
        self.contracts:       int            = p.get("contracts",        1)

        # Per-day state
        self._current_date:  Optional[date]  = None
        self._prev_close:    float           = 0.0
        self._today_open:    float           = 0.0
        self._traded_today:  bool            = False
        self._open_captured: bool            = False

    def _reset_day(self) -> None:
        self._today_open    = 0.0
        self._traded_today  = False
        self._open_captured = False

    def _find_prev_close(self, data: MarketData, i: int, bar_date: date) -> float:
        """Walk back to find the last close of the previous RTH session."""
        for j in range(i - 1, max(0, i - 1000), -1):
            jtime = data.df_1m.index[j].time()
            jdate = data.df_1m.index[j].date()
            if jdate != bar_date and time(9, 30) <= jtime <= time(16, 0):
                return float(data.close_1m[j])
        return 0.0

    def generate_signals(self, data: MarketData, i: int) -> Optional[Order]:
        ts       = data.df_1m.index[i]
        bar_time = ts.time()
        bar_date = ts.date()

        # ── Day rollover ──────────────────────────────────────────────────────
        if bar_date != self._current_date:
            self._reset_day()
            self._current_date = bar_date
            self._prev_close   = self._find_prev_close(data, i, bar_date)

        # ── Capture today's open ──────────────────────────────────────────────
        if bar_time == time(9, 30) and not self._open_captured:
            self._today_open    = float(data.open_1m[i])
            self._open_captured = True

        # ── Entry: only at the 09:30 open bar ────────────────────────────────
        if self._traded_today or bar_time != time(9, 30):
            return None
        if self._prev_close <= 0 or self._today_open <= 0:
            return None

        # ── Overnight return ──────────────────────────────────────────────────
        overnight_ret = (self._today_open - self._prev_close) / self._prev_close

        if abs(overnight_ret) < self.threshold_pct / 100.0:
            return None   # gap too small — noise

        # ── Optional ATR gap filter ───────────────────────────────────────────
        if self.gap_filter_mult > 0:
            atr = _atr(data.high_1m, data.low_1m, data.close_1m,
                       self.atr_length, i)
            gap_pts = abs(self._today_open - self._prev_close)
            if atr > 0 and gap_pts < self.gap_filter_mult * atr:
                return None

        # ── CO-OC reversal: trade OPPOSITE to overnight direction ─────────────
        direction = -1 if overnight_ret > 0 else 1

        self._traded_today = True
        return Order(
            direction  = direction,
            order_type = OrderType.MARKET,
            size_type  = SizeType.CONTRACTS,
            size_value = self.contracts,
        )

    def on_fill(self, position: OpenPosition, data: MarketData,
                bar_index: int) -> None:
        entry   = position.entry_price
        is_long = position.is_long()
        atr     = _atr(data.high_1m, data.low_1m, data.close_1m,
                       self.atr_length, bar_index)

        # ── Stop loss ─────────────────────────────────────────────────────────
        if self.sl_type == "atr_multiple":
            offset = atr * self.sl_value if atr > 0 else entry * 0.005
            sl = entry - offset if is_long else entry + offset
        elif self.sl_type == "fixed_points":
            sl = entry - self.sl_value if is_long else entry + self.sl_value
        elif self.sl_type == "pct":
            sl = (entry * (1 - self.sl_value / 100.0) if is_long
                  else entry * (1 + self.sl_value / 100.0))
        else:
            offset = atr * 1.5 if atr > 0 else entry * 0.005
            sl = entry - offset if is_long else entry + offset

        # ── Take profit ───────────────────────────────────────────────────────
        risk = abs(entry - sl)
        if self.tp_type == "risk_reward":
            tp = (entry + risk * self.tp_value if is_long
                  else entry - risk * self.tp_value)
        elif self.tp_type == "atr_multiple":
            offset = atr * self.tp_value
            tp = entry + offset if is_long else entry - offset
        elif self.tp_type == "fixed_points":
            tp = entry + self.tp_value if is_long else entry - self.tp_value
        else:
            tp = (entry + risk * self.tp_value if is_long
                  else entry - risk * self.tp_value)

        position.set_initial_sl_tp(sl, tp)

        # ── Trailing stop ─────────────────────────────────────────────────────
        if self.trail_points is not None:
            position.trail_points            = self.trail_points
            position.trail_activation_points = self.trail_activation

    def manage_position(self, data: MarketData, i: int,
                        position: OpenPosition) -> Optional[PositionUpdate]:
        return None