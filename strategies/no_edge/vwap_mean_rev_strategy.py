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

Performance notes:
  - _setup() arrays are cached at module level keyed by (data_id, atr_period) so all
    combos in a sweep share one computation pass over the 2M-bar dataset.
  - signal_bar_mask returns a pre-screened subset of RTH bars where the VWAP deviation
    exceeds a conservative floor (0.5 ATR), cutting the runner's flat-bar loop by ~5-10x.
"""
from __future__ import annotations

from datetime import time as _time
from typing import Optional

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from backtest.data.market_data import MarketData
from backtest.strategy.base import BaseStrategy
from backtest.strategy.enums import OrderType, SizeType
from backtest.strategy.order import Order
from backtest.strategy.update import OpenPosition, PositionUpdate

POINT_VALUE = 20.0
TICK        = 0.25

_RTH_OPEN_MIN = 9  * 60 + 30   # 570
_WIN_END_MIN  = 15 * 60         # 900

# Minimum deviation ratio (ATR multiples) for a bar to be included in
# signal_bar_mask.  Must be ≤ the smallest vwap_deviation_atr in any sweep grid.
_MASK_MIN_DEV = 0.5


def _tick(price: float) -> float:
    return round(round(price / TICK) * TICK, 10)


# ---------------------------------------------------------------------------
# Module-level cache: (data_id, atr_period) → precomputed arrays
# Shared across all combo instances so _setup runs only once per sweep.
# ---------------------------------------------------------------------------
_ARRAY_CACHE: dict = {}


def _compute_arrays(data: MarketData, atr_period: int) -> dict:
    """Compute all per-bar arrays needed by every VWAPMeanRevStrategy combo."""
    idx = data.df_1m.index
    n   = len(data.close_1m)

    # Times / day index — use cached values from MarketData when available
    if data.bar_times_1m_min is not None:
        t_min_arr = data.bar_times_1m_min
    else:
        t_min_arr = (idx.hour * 60 + idx.minute).to_numpy(np.int32)

    if data.bar_day_int_1m is not None:
        day_arr = data.bar_day_int_1m
    else:
        day_arr = idx.normalize().view(np.int64) // (86_400 * 10**9)

    # Wilder ATR via lfilter (exact EMA, vectorised — avoids 2M Python loop)
    hi, lo = data.high_1m, data.low_1m
    cl_prev = np.empty(n)
    cl_prev[0] = data.close_1m[0]
    cl_prev[1:] = data.close_1m[:-1]
    tr = np.maximum(hi - lo, np.maximum(np.abs(hi - cl_prev), np.abs(lo - cl_prev)))
    alpha = 1.0 / atr_period
    # lfilter(b=[alpha], a=[1, -(1-alpha)]) → Wilder EMA
    atr = lfilter([alpha], [1.0, -(1.0 - alpha)], tr).astype(np.float64)

    # VWAP — daily cumsum using vectorised numpy (no pandas groupby)
    rth_mask = t_min_arr >= _RTH_OPEN_MIN
    tp   = (hi + lo + data.close_1m) / 3.0
    tpv  = np.where(rth_mask, tp * data.volume_1m, 0.0)
    vol  = np.where(rth_mask, data.volume_1m, 0.0)

    day_start  = np.concatenate([[True], day_arr[1:] != day_arr[:-1]])
    ds_idxs    = np.where(day_start)[0]

    cs_tpv = np.cumsum(tpv)
    cs_vol = np.cumsum(vol)

    # For each day-start, record the global cumsum at end of PREVIOUS bar
    prev_cs_tpv = np.empty(len(ds_idxs))
    prev_cs_vol = np.empty(len(ds_idxs))
    prev_cs_tpv[0] = 0.0
    prev_cs_vol[0] = 0.0
    if len(ds_idxs) > 1:
        prev_cs_tpv[1:] = cs_tpv[ds_idxs[1:] - 1]
        prev_cs_vol[1:] = cs_vol[ds_idxs[1:] - 1]

    # Map each bar to its day index
    bar_day_idx  = np.searchsorted(ds_idxs, np.arange(n), side='right') - 1
    cum_tpv_day  = cs_tpv - prev_cs_tpv[bar_day_idx]
    cum_vol_day  = cs_vol - prev_cs_vol[bar_day_idx]

    safe_vol = np.where(cum_vol_day > 0, cum_vol_day, 1.0)
    vwap = np.where(rth_mask & (cum_vol_day > 0), cum_tpv_day / safe_vol, np.nan)

    # D1 SMA(200) and prev-day close mapped to 1m bars
    # Build daily close array via day_arr
    last_bar_of_day = np.where(
        np.concatenate([day_arr[1:] != day_arr[:-1], [True]])
    )[0]
    daily_close_vals = data.close_1m[last_bar_of_day]
    n_days = len(last_bar_of_day)

    sma200_daily = np.full(n_days, np.nan)
    if n_days >= 200:
        cs_d = np.cumsum(daily_close_vals)
        sma200_daily[199:] = (cs_d[199:] - np.concatenate([[0.0], cs_d[:-200][:(n_days-199)]]))  / 200.0

    # Shift by 1 day (prior day's close/SMA visible at each bar)
    prev_close_daily = np.full(n_days, np.nan)
    prev_close_daily[1:] = daily_close_vals[:-1]
    prev_sma200_daily = np.full(n_days, np.nan)
    prev_sma200_daily[1:] = sma200_daily[:-1]

    # Map day-level to 1m bar level
    sma200_arr = prev_sma200_daily[bar_day_idx]
    prev_cls   = prev_close_daily[bar_day_idx]

    # Signal bar mask: entry window AND deviation exceeds conservative floor
    entry_window = rth_mask & (t_min_arr < _WIN_END_MIN)
    dev_ratio = np.where(
        (atr > 0) & np.isfinite(vwap),
        np.abs(data.close_1m - vwap) / atr,
        0.0,
    )
    sig_mask = entry_window & (dev_ratio > _MASK_MIN_DEV)

    return {
        't_min_arr': t_min_arr,
        'day_arr':   day_arr,
        'atr_arr':   atr,
        'vwap_arr':  vwap,
        'sma200_arr': sma200_arr,
        'prev_cls':  prev_cls,
        'sig_mask':  sig_mask,
    }


def _get_arrays(data: MarketData, atr_period: int) -> dict:
    key = (id(data.open_1m), atr_period)
    if key not in _ARRAY_CACHE:
        _ARRAY_CACHE[key] = _compute_arrays(data, atr_period)
    return _ARRAY_CACHE[key]


class VWAPMeanRevStrategy(BaseStrategy):
    """
    VWAP deviation mean-reversion for NQ intraday.
    """

    trading_hours = [(_time(9, 30), _time(14, 59))]
    min_lookback  = 20

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
        self.entry_delay_min = int(p.get('entry_delay_min',       30))

        self._arrays: Optional[dict] = None

        # Daily state
        self._today:         object = None
        self._long_taken:    bool   = False
        self._short_taken:   bool   = False
        self._trades_today:  int    = 0

        self._eq_cache   = self.starting_equity
        self._eq_cache_n = 0

    def _setup(self, data: MarketData) -> None:
        if self._arrays is not None:
            return
        self._arrays = _get_arrays(data, self.atr_period)

    def signal_bar_mask(self, data: MarketData) -> np.ndarray:
        self._setup(data)
        return self._arrays['sig_mask']

    def _current_equity(self) -> float:
        if self.equity_mode == 'fixed':
            return self.starting_equity
        n = len(self.closed_trades)
        while self._eq_cache_n < n:
            self._eq_cache += self.closed_trades[self._eq_cache_n].net_pnl_dollars
            self._eq_cache_n += 1
        return self._eq_cache

    def generate_signals(self, data: MarketData, i: int) -> Optional[Order]:
        self._setup(data)
        arr   = self._arrays
        t_min = int(arr['t_min_arr'][i])

        entry_start = _RTH_OPEN_MIN + self.entry_delay_min
        if t_min < entry_start or t_min >= _WIN_END_MIN:
            return None

        today = int(arr['day_arr'][i])
        if today != self._today:
            self._today        = today
            self._long_taken   = False
            self._short_taken  = False
            self._trades_today = 0

        if self._trades_today >= self.max_trades_day:
            return None

        vwap = float(arr['vwap_arr'][i])
        if np.isnan(vwap):
            return None

        atr = float(arr['atr_arr'][i])
        if atr <= 0.0:
            return None

        cl  = float(data.close_1m[i])
        op  = float(data.open_1m[i])
        dev = self.vwap_dev_atr * atr

        sma200   = float(arr['sma200_arr'][i])
        prev_cls = float(arr['prev_cls'][i])

        if not self._long_taken and cl < vwap - dev:
            if self.use_sma_filter and not np.isnan(sma200) and prev_cls < sma200:
                pass
            elif self.require_confirm and cl <= op:
                pass
            else:
                order = self._build_order(1, cl, atr, 'vwap_long')
                if order is not None:
                    self._long_taken   = True
                    self._trades_today += 1
                    return order

        if not self._short_taken and cl > vwap + dev:
            if self.use_sma_filter and not np.isnan(sma200) and prev_cls > sma200:
                pass
            elif self.require_confirm and cl >= op:
                pass
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

    def on_fill(self, position: OpenPosition, data: MarketData, bar_index: int) -> None:
        position.set_initial_sl_tp(position.sl_price, position.tp_price)

    def manage_position(
        self, data: MarketData, i: int, position: OpenPosition,
    ) -> Optional[PositionUpdate]:
        return None
