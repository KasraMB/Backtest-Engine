"""
VWAP Mean Reversion Strategy (NQ Futures)

Enter long when price drops X ATR below the daily VWAP (expect reversion).
Enter short when price rises X ATR above the daily VWAP (expect reversion).

VWAP resets at RTH open (09:30 ET) each day.
Entries allowed from entry_start_min to 15:00 ET.
At most max_trades_per_day positions per day (one per side by default).

Exit: SL / TP hit, or force-close at eod_exit_time via runner.

Optional D1 200 SMA filter: only long when prev_day_close > SMA200, short below.
Optional reversion confirmation: only enter if the current bar is moving TOWARD VWAP
  (i.e., for a long entry the close must be higher than the open of the same bar).
"""
from __future__ import annotations

from datetime import time as _time
from typing import Optional

import numpy as np
import pandas as pd

from backtest.data.market_data import MarketData
from backtest.strategy.base import BaseStrategy
from backtest.strategy.enums import OrderType, SizeType
from backtest.strategy.order import Order
from backtest.strategy.update import OpenPosition, PositionUpdate

POINT_VALUE = 20.0
TICK        = 0.25

_RTH_OPEN_MIN = 9  * 60 + 30   # 09:30 — VWAP resets here
_WIN_END_MIN  = 15 * 60         # 15:00 — no new entries after this


def _tick(price: float) -> float:
    return round(round(price / TICK) * TICK, 10)


class VWAPMeanRevStrategy(BaseStrategy):
    """
    VWAP deviation mean-reversion for NQ intraday.
    """

    trading_hours = [(_time(9, 30), _time(14, 59))]
    min_lookback  = 20

    # ── Construction ─────────────────────────────────────────────────────────

    def __init__(self, params: dict = None) -> None:
        super().__init__(params)
        p = params or {}

        self.vwap_dev_atr    = float(p.get('vwap_deviation_atr',  1.5))
        self.sl_atr_mult     = float(p.get('sl_atr_multiplier',   1.0))
        self.rr_ratio        = float(p.get('rr_ratio',            1.5))
        self.risk_per_trade  = float(p.get('risk_per_trade',      0.01))
        self.starting_equity = float(p.get('starting_equity',     100_000))
        self.equity_mode     = str(p.get('equity_mode',           'dynamic'))
        self.atr_period      = int(p.get('atr_period',            14))
        self.use_sma_filter  = bool(p.get('use_200sma_filter',    False))
        self.require_confirm = bool(p.get('require_confirmation', False))
        self.max_trades_day  = int(p.get('max_trades_per_day',    2))
        # Minimum bars into RTH before entering (e.g. 30 = wait until 10:00)
        self.entry_delay_min = int(p.get('entry_delay_min',       30))

        # Lazy arrays set in _setup
        self._times_min:  Optional[np.ndarray] = None
        self._atr_arr:    Optional[np.ndarray] = None
        self._vwap_arr:   Optional[np.ndarray] = None  # full series, NaN outside RTH
        self._sma200_arr: Optional[np.ndarray] = None  # daily SMA200 prev-day mapped to 1m bars
        self._prev_cls:   Optional[np.ndarray] = None

        # Daily state
        self._today:         object = None
        self._long_taken:    bool   = False
        self._short_taken:   bool   = False
        self._trades_today:  int    = 0

        # Equity cache
        self._eq_cache   = self.starting_equity
        self._eq_cache_n = 0

    # ── Setup ─────────────────────────────────────────────────────────────────

    def _setup(self, data: MarketData) -> None:
        if self._times_min is not None:
            return

        idx = data.df_1m.index
        t_min_arr = (idx.hour * 60 + idx.minute).to_numpy(np.int32)
        self._times_min = t_min_arr

        n = len(data.close_1m)

        # Precompute integer day index per bar (days since epoch) for fast daily resets
        self._bar_day = idx.normalize().view(np.int64) // (86_400 * 10**9)

        # ATR (Wilder) — vectorised True Range then EMA
        hi, lo, cl_prev = data.high_1m, data.low_1m, np.empty(n)
        cl_prev[0] = data.close_1m[0]
        cl_prev[1:] = data.close_1m[:-1]
        tr = np.maximum(hi - lo, np.maximum(np.abs(hi - cl_prev), np.abs(lo - cl_prev)))
        period = self.atr_period
        atr = np.zeros(n)
        if n >= period:
            atr[period - 1] = tr[:period].mean()
            alpha = 1.0 / period
            inv   = 1.0 - alpha
            for i in range(period, n):
                atr[i] = atr[i - 1] * inv + tr[i] * alpha
        self._atr_arr = atr

        # VWAP — per-day cumsum via pandas groupby (C-speed, resets at midnight)
        # Only accumulate RTH bars (t_min >= 09:30); zero out pre-RTH.
        tp       = (data.high_1m + data.low_1m + data.close_1m) / 3.0
        vol      = data.volume_1m
        dates    = idx.date
        rth_mask = t_min_arr >= _RTH_OPEN_MIN

        tpv_rth = np.where(rth_mask, tp * vol, 0.0)
        vol_rth = np.where(rth_mask, vol, 0.0)

        df_tmp    = pd.DataFrame({'tpv': tpv_rth, 'vol': vol_rth, 'date': dates}, index=idx)
        cum_tpv   = df_tmp.groupby('date')['tpv'].cumsum().values
        cum_v     = df_tmp.groupby('date')['vol'].cumsum().values

        safe_v = np.where(cum_v > 0, cum_v, 1.0)
        vwap = np.where((rth_mask) & (cum_v > 0), cum_tpv / safe_v, np.nan)
        self._vwap_arr = vwap

        # D1 SMA(200) and prev-day close mapped to 1m bars
        dates = idx.date
        close_s = pd.Series(data.close_1m, index=idx)
        daily_close = close_s.groupby(dates).last()
        sma200 = daily_close.rolling(200, min_periods=200).mean().shift(1)
        prev_c = daily_close.shift(1)
        sma_map  = sma200.to_dict()
        prev_map = prev_c.to_dict()
        self._sma200_arr = np.array([sma_map.get(d, np.nan)  for d in dates])
        self._prev_cls   = np.array([prev_map.get(d, np.nan) for d in dates])

    # ── Equity ────────────────────────────────────────────────────────────────

    def _current_equity(self) -> float:
        if self.equity_mode == 'fixed':
            return self.starting_equity
        n = len(self.closed_trades)
        while self._eq_cache_n < n:
            self._eq_cache += self.closed_trades[self._eq_cache_n].net_pnl_dollars
            self._eq_cache_n += 1
        return self._eq_cache

    # ── Signal generation ─────────────────────────────────────────────────────

    def generate_signals(self, data: MarketData, i: int) -> Optional[Order]:
        self._setup(data)

        t_min = int(self._times_min[i])

        # Only trade within window and after entry delay
        entry_start = _RTH_OPEN_MIN + self.entry_delay_min
        if t_min < entry_start or t_min >= _WIN_END_MIN:
            return None

        today = int(self._bar_day[i])
        if today != self._today:
            self._today        = today
            self._long_taken   = False
            self._short_taken  = False
            self._trades_today = 0

        if self._trades_today >= self.max_trades_day:
            return None

        vwap = float(self._vwap_arr[i])
        if np.isnan(vwap):
            return None

        atr = float(self._atr_arr[i])
        if atr <= 0.0:
            return None

        cl   = float(data.close_1m[i])
        op   = float(data.open_1m[i])
        dev  = self.vwap_dev_atr * atr

        sma200   = float(self._sma200_arr[i])
        prev_cls = float(self._prev_cls[i])

        # Long: price is dev below VWAP
        if not self._long_taken and cl < vwap - dev:
            if self.use_sma_filter and not np.isnan(sma200) and prev_cls < sma200:
                pass  # below SMA200 — skip long
            elif self.require_confirm and cl <= op:
                pass  # no reversion bar yet
            else:
                order = self._build_order(1, cl, atr, 'vwap_long')
                if order is not None:
                    self._long_taken   = True
                    self._trades_today += 1
                    return order

        # Short: price is dev above VWAP
        if not self._short_taken and cl > vwap + dev:
            if self.use_sma_filter and not np.isnan(sma200) and prev_cls > sma200:
                pass  # above SMA200 — skip short
            elif self.require_confirm and cl >= op:
                pass  # no reversion bar yet
            else:
                order = self._build_order(-1, cl, atr, 'vwap_short')
                if order is not None:
                    self._short_taken  = True
                    self._trades_today += 1
                    return order

        return None

    def _build_order(self, direction: int, entry: float, atr: float, reason: str) -> Optional[Order]:
        sl_dist = self.sl_atr_mult * atr
        if sl_dist <= 0.0:
            return None
        if direction == 1:
            sl_price = _tick(entry - sl_dist)
            tp_price = _tick(entry + self.rr_ratio * sl_dist)
        else:
            sl_price = _tick(entry + sl_dist)
            tp_price = _tick(entry - self.rr_ratio * sl_dist)

        equity    = self._current_equity()
        contracts = int(equity * self.risk_per_trade / (sl_dist * POINT_VALUE))
        if contracts < 1:
            return None

        return Order(
            direction    = direction,
            order_type   = OrderType.MARKET,
            size_type    = SizeType.CONTRACTS,
            size_value   = float(contracts),
            sl_price     = sl_price,
            tp_price     = tp_price,
            trade_reason = reason,
        )

    # ── Position management ───────────────────────────────────────────────────

    def on_fill(self, position: OpenPosition, data: MarketData, bar_index: int) -> None:
        position.set_initial_sl_tp(position.sl_price, position.tp_price)

    def manage_position(
        self, data: MarketData, i: int, position: OpenPosition,
    ) -> Optional[PositionUpdate]:
        return None  # runner handles EOD exit via eod_exit_time
