from __future__ import annotations
import logging
from datetime import time
from typing import Optional, Type, Union

import numpy as np

from backtest.data.market_data import MarketData
from backtest.engine.execution import Bar, ExecutionEngine, PendingOrder
from backtest.engine.risk import Account, RiskManager
from backtest.runner.config import RunConfig
from backtest.strategy.base import BaseStrategy
from backtest.strategy.enums import ExitReason, OrderType
from backtest.strategy.order import Order
from backtest.strategy.update import OpenPosition, Trade, round_to_tick

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RunResult — raw output of a single backtest run
# ---------------------------------------------------------------------------

class RunResult:
    """
    Raw output of a single backtest run.
    Passed to the PerformanceEngine in Phase 6 to compute metrics.
    """

    def __init__(
        self,
        trades: list[Trade],
        equity_curve: list[float],
        config: RunConfig,
        strategy_name: str,
    ):
        self.trades = trades
        self.equity_curve = equity_curve
        self.config = config
        self.strategy_name = strategy_name

    @property
    def n_trades(self) -> int:
        return len(self.trades)

    @property
    def uses_trailing_stop(self) -> bool:
        """True if any trade had an active trailing stop during its lifetime."""
        return any(t.had_trailing for t in self.trades)

    @property
    def total_net_pnl(self) -> float:
        return sum(t.net_pnl_dollars for t in self.trades)

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        return sum(1 for t in self.trades if t.is_winner) / len(self.trades)

    def print_summary(self) -> None:
        print(f"\n{'='*50}")
        print(f"  Strategy : {self.strategy_name}")
        print(f"  Trades   : {self.n_trades}")
        print(f"  Win Rate : {self.win_rate:.1%}")
        print(f"  Net PnL  : ${self.total_net_pnl:,.2f}")
        print(f"  Final Eq : ${self.equity_curve[-1]:,.2f}" if self.equity_curve else "")
        print(f"{'='*50}")

    def print_trades(self, max_trades: int = 20) -> None:
        """Print the first N trades in a readable format."""
        print(f"\n{'─'*90}")
        print(f"{'#':>4}  {'Entry Bar':>9}  {'Exit Bar':>8}  {'Dir':>4}  "
              f"{'Qty':>4}  {'Entry':>8}  {'Exit':>8}  "
              f"{'PnL pts':>8}  {'Net $':>9}  {'Reason'}")
        print(f"{'─'*90}")
        for idx, t in enumerate(self.trades[:max_trades]):
            direction = "LONG" if t.direction == 1 else "SHORT"
            print(
                f"{idx+1:>4}  {t.entry_bar:>9}  {t.exit_bar:>8}  {direction:>4}  "
                f"{t.contracts:>4.0f}  {t.entry_price:>8.2f}  {t.exit_price:>8.2f}  "
                f"{t.pnl_points:>+8.2f}  {t.net_pnl_dollars:>+9.2f}  {t.exit_reason.name}"
            )
        if len(self.trades) > max_trades:
            print(f"  ... and {len(self.trades) - max_trades} more trades")
        print(f"{'─'*90}")


def reverse_trades(result: "RunResult") -> "RunResult":
    """
    Derive a reversed RunResult from an existing one — no re-run needed.

    Each trade keeps identical entry/exit prices, timestamps, and costs.
    Only direction and exit_reason flip:
      - direction:    1 → -1, -1 → 1
      - SL  ↔ TP  (what was a stop-loss hit is now a take-profit, and vice versa)
      - SAME_BAR_SL ↔ SAME_BAR_TP
      - EOD / SIGNAL / FORCED_EXIT — unchanged (direction-agnostic)

    Because entry/exit prices are the same but direction is flipped, every
    winning trade becomes a loser and every loser becomes a winner.
    PnL is recomputed automatically via Trade.net_pnl_dollars (direction-aware).
    """
    from backtest.strategy.enums import ExitReason
    from copy import copy

    flip_reason = {
        ExitReason.SL:           ExitReason.TP,
        ExitReason.TP:           ExitReason.SL,
        ExitReason.SAME_BAR_SL:  ExitReason.SAME_BAR_TP,
        ExitReason.SAME_BAR_TP:  ExitReason.SAME_BAR_SL,
    }

    rev_trades = []
    for t in result.trades:
        rt = copy(t)
        rt.direction        = -t.direction
        rt.exit_reason      = flip_reason.get(t.exit_reason, t.exit_reason)
        # Swap SL ↔ TP prices: what was the long's SL becomes the short's TP and vice versa
        rt.sl_price         = t.tp_price
        rt.tp_price         = t.sl_price
        rt.initial_sl_price = t.initial_tp_price
        rt.initial_tp_price = t.initial_sl_price
        rev_trades.append(rt)

    # Rebuild equity curve from reversed trade PnLs
    capital = result.config.starting_capital
    rev_equity = [capital]
    balance = capital
    for t in rev_trades:
        balance += t.net_pnl_dollars
        rev_equity.append(balance)
    # Pad to same length as original bar-resolution curve
    # (performance engine only uses the trade-resolution curve now)

    return RunResult(
        trades        = rev_trades,
        equity_curve  = rev_equity,
        config        = result.config,
        strategy_name = result.strategy_name,
    )


# ---------------------------------------------------------------------------
# Active bar set builder
# ---------------------------------------------------------------------------

def build_active_bar_set(
    data: MarketData,
    trading_hours: Optional[list[tuple[time, time]]],
) -> set[int]:
    """
    Pre-compute the set of 1m bar indices that fall within the strategy's
    declared trading_hours. Used to gate generate_signals calls.

    If trading_hours is None, returns all bar indices.
    """
    if trading_hours is None:
        return set(range(len(data.df_1m)))

    active = set()
    timestamps = data.df_1m.index

    for i, ts in enumerate(timestamps):
        t = ts.time()
        for start, end in trading_hours:
            if start <= end:
                if start <= t <= end:
                    active.add(i)
                    break
            else:
                # Overnight window: e.g. (18:00, 08:00)
                if t >= start or t <= end:
                    active.add(i)
                    break

    return active


# ---------------------------------------------------------------------------
# EOD bar detection
# ---------------------------------------------------------------------------

def build_required_bar_set(data: MarketData, strategy_class) -> set[int]:
    """
    Pre-compute the set of bars that must be processed when flat.
    For strategies with trading_hours=None but that only care about specific
    sessions (e.g. DoubleSessionSweep), this avoids iterating dead bars.

    A bar is required if it falls within ANY of these windows:
      - The strategy's declared trading_hours (if not None)
      - 20:00-00:00 ET (Asia)
      - 02:00-05:00 ET (London)
      - 09:30-16:00 ET (NY)

    If trading_hours is not None, we use that directly (already restricted).
    If trading_hours is None, we use the union of all three session windows
    as a safe superset — this covers any overnight strategy.
    """
    # Get the trading_hours from a temporary strategy instance
    try:
        tmp = strategy_class.__new__(strategy_class)
        trading_hours = getattr(tmp, 'trading_hours', None)
    except Exception:
        trading_hours = None

    if trading_hours is not None:
        # Strategy already declares its hours — use active_bars directly
        return None   # signal to caller: use active_bars as required_bars

    # trading_hours=None: strategy processes all bars itself but we can still
    # skip bars that fall entirely outside all known session windows.
    session_windows = [
        (time(20, 0),  time(0, 0)),    # Asia (overnight)
        (time(2, 0),   time(5, 0)),    # London
        (time(9, 30),  time(16, 0)),   # NY RTH
    ]

    required = set()
    timestamps = data.df_1m.index

    for i, ts in enumerate(timestamps):
        t = ts.time()
        for start, end in session_windows:
            if start < end:
                if start <= t < end:
                    required.add(i)
                    break
            else:
                # Overnight: e.g. 20:00 to 00:00
                if t >= start or t < end:
                    required.add(i)
                    break

    return required


# ---------------------------------------------------------------------------
# EOD bar detection
# ---------------------------------------------------------------------------

def build_eod_bar_set(data: MarketData, eod_exit_time: time) -> set[int]:
    """
    Pre-compute bar indices that trigger EOD exits.

    NQ futures trade nearly 24 hours, so calendar-date grouping would put the
    23:59 globex bar and the 17:00 ET close bar on the same date — and picking
    the *last* bar at or after eod_exit_time would yield 23:59 instead of 17:00.

    Correct approach: for each RTH session date, find the *first* bar whose
    time is at or after eod_exit_time.  That is the bar where we close out.

    RTH is defined as 09:30–17:00 ET (futures daily close is 16:00 CT = 17:00 ET).
    For early-close sessions (holiday) where no bar reaches eod_exit_time,
    fall back to the last RTH bar of that session.
    """
    eod_bars = set()
    timestamps = data.df_1m.index

    from collections import defaultdict
    # Group by calendar date
    date_bars: dict = defaultdict(list)
    for i, ts in enumerate(timestamps):
        date_bars[ts.date()].append((i, ts.time()))

    for bars in date_bars.values():
        # Only consider RTH bars (09:30–17:00 ET) so late-night globex bars are excluded
        rth_bars = [(i, t) for i, t in bars
                    if time(9, 30) <= t <= time(16, 0)]

        if not rth_bars:
            continue

        # First RTH bar at or after eod_exit_time
        eod_candidates = [(i, t) for i, t in rth_bars if t >= eod_exit_time]
        if eod_candidates:
            eod_bars.add(eod_candidates[0][0])   # FIRST, not last
        else:
            # Early-close: use the last RTH bar of the session
            eod_bars.add(rth_bars[-1][0])

    return eod_bars


# ---------------------------------------------------------------------------
# Parallel Phase 1 pre-computation
# ---------------------------------------------------------------------------

def _precompute_phase1_parallel(
    strategy_class: Type[BaseStrategy],
    params: dict,
    data: MarketData,
    n_workers: int = 8,
) -> dict:
    """
    Pre-compute Phase 1 (validated levels, session OTE groups, swing arrays, etc.)
    for every trading day in parallel using ThreadPoolExecutor.

    Returns dict mapping date_ordinal → phase1_result_dict.

    Requires strategy_class._supports_precomputed_phase1 == True.
    Numba functions inside _run_phase1 (FVG/OB/cisd_scan) use nogil=True so
    threads truly run in parallel for the hot inner loops.
    """
    from concurrent.futures import ThreadPoolExecutor

    SESSION_START_MIN = 570  # 09:30 ET in minutes since midnight

    # Build date+time arrays once (same logic as _ensure_bar_metadata)
    idx = data.df_1m.index
    y1  = idx.year.to_numpy(np.int64) - 1
    doy = idx.day_of_year.to_numpy(np.int64)
    bar_dates = (y1 * 365 + y1 // 4 - y1 // 100 + y1 // 400 + doy).astype(np.int32)
    bar_times = (idx.hour * 60 + idx.minute).to_numpy(np.int32)

    # Collect (date_ord, bar_i) for every Phase 1 trigger
    trigger_bars: list = []
    seen_days: set = set()
    for i in range(len(bar_times)):
        if int(bar_times[i]) >= SESSION_START_MIN:
            tod = int(bar_dates[i])
            if tod not in seen_days:
                seen_days.add(tod)
                trigger_bars.append((tod, max(0, i - 1)))

    def _run_one(args):
        tod_ord, bar_i = args
        inst = strategy_class(params)
        inst._ensure_bar_metadata(data)
        inst._run_phase1(data, bar_i, tod_ord)
        return tod_ord, {
            'validated_levels':    inst._validated_levels,
            'session_ote_groups':  inst._session_ote_groups,
            'session_atr':         inst._session_atr,
            'overnight_range_atr': inst._overnight_range_atr,
            'sw_1m_hi':  inst._sw_1m_hi,  'sw_1m_lo':  inst._sw_1m_lo,
            'sw_5m_hi':  inst._sw_5m_hi,  'sw_5m_lo':  inst._sw_5m_lo,
            'sw_15m_hi': inst._sw_15m_hi, 'sw_15m_lo': inst._sw_15m_lo,
            'sw_30m_hi': inst._sw_30m_hi, 'sw_30m_lo': inst._sw_30m_lo,
            'sw_lo_all': inst._sw_lo_all,  'sw_hi_all': inst._sw_hi_all,
        }

    results: dict = {}
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        for tod_ord, result in pool.map(_run_one, trigger_bars):
            results[tod_ord] = result

    return results


# ---------------------------------------------------------------------------
# Core bar loop
# ---------------------------------------------------------------------------

def _run_single(
    strategy_class: Type[BaseStrategy],
    data: MarketData,
    config: RunConfig,
) -> RunResult:
    """
    Execute the full bar loop for a single strategy + config combination.

    Bar processing order (per bar i):
      1. If pending order: check expiry -> check fill -> if filled:
           validate SL/TP at fill, resolve contracts, check same-bar exit
           if same-bar exit: record trade, clear position -> step 4
           else: create OpenPosition -> step 2
      2. If position open:
           manage_position() -> apply PositionUpdate
           check exits (SL -> TP -> EOD -> signal-based)
           if exit: record trade, clear position -> step 4
           else: tick_trail() (advances trail SL for next bar)
      3. If no position AND i in active_bars AND i >= min_lookback:
           generate_signals() -> if Order: validate, resolve size, register pending
      4. Advance
    """
    # --- Setup ---
    strategy = strategy_class(config.params)
    strategy._reverse_mode = config.reverse_signals
    engine = ExecutionEngine(
        slippage_points=config.slippage_points,
        commission_per_contract=config.commission_per_contract,
        eod_exit_time=config.eod_exit_time,
    )
    risk_manager = RiskManager()
    account = Account(balance=config.starting_capital)

    active_bars = build_active_bar_set(data, strategy.trading_hours)
    eod_bars = build_eod_bar_set(data, config.eod_exit_time)

    # For strategies with trading_hours=None we compute a required_bars mask
    # covering all session windows. Bars outside this are skipped when flat.
    _req_set = build_required_bar_set(data, strategy_class)
    n_bars   = len(data.open_1m)
    if _req_set is not None:
        # Convert to boolean numpy array for fast O(1) indexing
        required_mask = np.zeros(n_bars, dtype=bool)
        for idx in _req_set:
            required_mask[idx] = True
    else:
        required_mask = None   # all bars required

    # Pre-extract numpy arrays for the hot path
    opens  = data.open_1m
    highs  = data.high_1m
    lows   = data.low_1m
    closes = data.close_1m

    # Precompute cancel mask — avoids allocating a Timestamp object per bar
    # (data.df_1m.index[i].time() is called 830K times without this).
    _cancel_min = config.order_cancel_time.hour * 60 + config.order_cancel_time.minute
    _times_min  = (data.df_1m.index.hour * 60 + data.df_1m.index.minute).to_numpy()
    cancel_mask = _times_min >= _cancel_min

    # Parallel Phase 1 pre-computation — each trading day is independent;
    # Numba functions inside _run_phase1 use nogil=True for true parallelism.
    if getattr(strategy, '_supports_precomputed_phase1', False):
        strategy._phase1_precomputed = _precompute_phase1_parallel(
            strategy_class, config.params, data
        )

    trades: list[Trade] = []
    equity_curve: list[float] = [account.balance]

    position: Optional[OpenPosition] = None
    pending: Optional[PendingOrder] = None

    # --- Main loop ---
    for i in range(n_bars):
        # ── Fast skip: when flat and no pending order, skip bars outside
        # session windows. Positions need every bar for SL/TP checks.
        if (position is None and pending is None
                and required_mask is not None
                and not required_mask[i]):
            equity_curve.append(account.balance)
            continue
        bar = Bar(
            index=i,
            open=opens[i],
            high=highs[i],
            low=lows[i],
            close=closes[i],
        )
        is_eod = i in eod_bars

        # ── Step 1: Pending order ──────────────────────────────────────────
        if pending is not None:

            # Cancel any pending order at or after session end (11:00 ET)
            if cancel_mask[i]:
                logger.debug(f"Bar {i}: pending order cancelled — session end")
                pending = None

            # Check expiry first
            elif engine.tick_expiry(pending):
                logger.debug(f"Bar {i}: pending order expired, cancelled")
                pending = None

            else:
                fill = None  # initialise; set below if fill attempt succeeds
                # §9.3 TP-before-fill cancellation (limit orders only)
                _ord = pending.order
                if (
                    (_ord.cancel_above is not None and
                     bar.high >= _ord.cancel_above and
                     _ord.limit_price is not None and bar.low > _ord.limit_price) or
                    (_ord.cancel_below is not None and
                     bar.low <= _ord.cancel_below and
                     _ord.limit_price is not None and bar.high < _ord.limit_price)
                ):
                    logger.debug(f"Bar {i}: pending order cancelled — TP reached before fill (§9.3)")
                    pending = None
                else:
                    # Market orders are handled at signal time (step 3), not here.
                    # Non-market orders: attempt fill
                    if pending.order.order_type != OrderType.MARKET:
                        fill = engine.attempt_fill(pending, bar)
                    else:
                        fill = None  # shouldn't reach here for MARKET

                if fill is not None:
                    fill.contracts = risk_manager.resolve_contracts(
                        pending.order, account, fill.fill_price
                    )

                    # Validate SL/TP against actual fill price
                    _validate_sl_tp_at_fill(pending.order, fill.fill_price)

                    if position is not None:
                        # Delta resolution — existing position
                        trade_or_position = _apply_delta(
                            engine, pending.order, fill, position,
                            bar, i, trades, account
                        )
                        position = trade_or_position  # None if fully closed
                    else:
                        # New position from scratch
                        position = _open_position(pending.order, fill, i, order_placed_bar=pending.registered_bar)

                        # Strategy sets SL/TP; wrapper handles reversal if needed
                        strategy._on_fill(position, data, i)

                        # Capture fill-bar excursion before same-bar exit check
                        _update_mae_mfe(position, bar)

                        # Check same-bar exit
                        same_bar = engine.check_same_bar_exit(position, bar, fill.fill_price)
                        if same_bar is not None:
                            trade = engine.build_trade(position, i, same_bar.exit_price, same_bar.exit_reason)
                            _record_trade(trade, trades, account, equity_curve)
                            position = None

                    pending = None  # order consumed regardless

        # ── Expose closed trades to strategy (read-only view, no copy overhead) ──
        strategy.closed_trades = trades

        # ── Step 2: Position management ────────────────────────────────────
        if position is not None:

            # Update MAE/MFE with this bar's range (max() is idempotent on fill bar)
            _update_mae_mfe(position, bar)

            # manage_position hook
            update = strategy.manage_position(data, i, position)
            if update is not None:
                forced = engine.apply_position_update(position, update, bar.close)
                if forced is not None:
                    trade = engine.build_trade(position, i, forced.exit_price, forced.exit_reason)
                    _record_trade(trade, trades, account, equity_curve)
                    position = None

        if position is not None:
            exit_result = engine.check_exits(position, bar, is_last_bar_of_session=is_eod)
            if exit_result is not None:
                trade = engine.build_trade(position, i, exit_result.exit_price, exit_result.exit_reason)
                _record_trade(trade, trades, account, equity_curve)
                position = None
            else:
                # Advance trail SL after exit checks so this bar's own price
                # action doesn't influence the SL used for this bar's exits.
                engine.tick_trail(position, bar)

        # ── Step 3: Signal generation ──────────────────────────────────────
        if position is None and pending is None:
            if i >= strategy.min_lookback and i in active_bars:
                order = strategy._generate_signals(data, i)
                if order is not None:
                    if order.order_type == OrderType.MARKET:
                        # Market orders fill at bar close — no remaining bar
                        # range exists, so same-bar SL/TP cannot be hit.
                        fill_price = bar.close
                        contracts = risk_manager.resolve_contracts(order, account, fill_price)
                        _validate_sl_tp_at_fill(order, fill_price)

                        from backtest.engine.execution import FillResult
                        fill = FillResult(fill_price=fill_price, contracts=contracts)
                        position = _open_position(order, fill, i, order_placed_bar=i)

                        # Strategy sets SL/TP; wrapper handles reversal if needed
                        strategy._on_fill(position, data, i)
                    else:
                        # Limit/Stop/StopLimit: register as pending
                        bars_remaining = order.expiry_bars  # None = GTC
                        pending = PendingOrder(
                            order=order,
                            registered_bar=i,
                            bars_remaining=bars_remaining,
                        )

        # ── Step 4: Advance (equity curve snapshot) ─────────────────────────
        # Mark-to-market the open position at bar close
        if position is not None:
            unrealized = _unrealized_pnl(position, bar.close)
            equity_curve.append(account.balance + unrealized)
        else:
            equity_curve.append(account.balance)

    # Force-close any position still open at end of data.
    # Step 4 already appended the last bar's mark-to-market snapshot, so we
    # overwrite that final point with the settled realised value after closing.
    if position is not None:
        last_bar = Bar(
            index=n_bars - 1,
            open=opens[-1], high=highs[-1], low=lows[-1], close=closes[-1],
            bar_time=data.df_1m.index[-1].time(),
        )
        trade = engine.build_trade(position, n_bars - 1, closes[-1], ExitReason.FORCED_EXIT)
        _record_trade(trade, trades, account, equity_curve)
        equity_curve[-1] = account.balance   # overwrite last MTM with final realised
        position = None

    return RunResult(
        trades=trades,
        equity_curve=equity_curve,
        config=config,
        strategy_name=strategy_class.__name__,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_backtest(
    strategy_class: Type[BaseStrategy],
    config: Union[RunConfig, list[RunConfig]],
    data: MarketData = None,
    validate: bool = True,
) -> Union[RunResult, list[RunResult]]:
    """
    Run a backtest for the given strategy class and config(s).

    Args:
        strategy_class: The strategy class (not an instance — runner instantiates fresh).
        config:         A single RunConfig, or a list for grid search.
        data:           Pre-loaded MarketData. If None, raises ValueError.
        validate:       Run the validation layer before the backtest (default True).
                        Set to False to skip for speed during grid search after
                        initial validation passes.

    Returns:
        RunResult for a single config, or list[RunResult] for grid search.
    """
    if data is None:
        raise ValueError("data must be provided. Load it with DataLoader first.")

    # Run validation on the first config (or the only config)
    if validate:
        from backtest.runner.validator import Validator, ValidationError
        first_config = config[0] if isinstance(config, list) else config
        report = Validator().run(strategy_class, first_config, data)
        report.print(strategy_class.__name__)
        if not report.passed:
            raise ValidationError(
                f"Strategy {strategy_class.__name__} failed validation. "
                f"Fix the above issues before running a full backtest."
            )

    if isinstance(config, list):
        return [_run_single(strategy_class, data, c) for c in config]
    else:
        return _run_single(strategy_class, data, config)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _open_position(order: Order, fill, entry_bar: int, order_placed_bar: int = None) -> OpenPosition:
    """Create an OpenPosition from a filled order. All prices rounded to tick grid."""
    return OpenPosition(
        direction=order.direction,
        entry_price=round_to_tick(fill.fill_price),
        entry_bar=entry_bar,
        contracts=fill.contracts,
        sl_price=round_to_tick(order.sl_price) if order.sl_price is not None else None,
        tp_price=round_to_tick(order.tp_price) if order.tp_price is not None else None,
        trail_points=order.trail_points,
        trail_activation_points=order.trail_activation_points,
        trade_reason=order.trade_reason,
        order_placed_bar=order_placed_bar,
    )


def _record_trade(
    trade: Trade,
    trades: list[Trade],
    account: Account,
    equity_curve: list[float],
) -> None:
    """Record a completed trade and update account balance.

    Does NOT append to equity_curve — step 4 of the bar loop appends a
    mark-to-market snapshot every bar.  Appending here too inserts an extra
    settled-cash point mid-bar, causing duplicate indices and a transient
    dip to realised cash whenever a position closes intra-bar.
    """
    trades.append(trade)
    account.balance += trade.net_pnl_dollars


def _update_mae_mfe(position: OpenPosition, bar: Bar) -> None:
    """Update MAE/MFE on the position using the current bar's high/low."""
    if position.direction == 1:
        adverse   = position.entry_price - bar.low
        favorable = bar.high - position.entry_price
    else:
        adverse   = bar.high - position.entry_price
        favorable = position.entry_price - bar.low
    position.mae_points = max(position.mae_points, max(0.0, adverse))
    position.mfe_points = max(position.mfe_points, max(0.0, favorable))


def _unrealized_pnl(position: OpenPosition, current_price: float) -> float:
    """Mark-to-market PnL for an open position."""
    from backtest.strategy.update import POINT_VALUE
    return (current_price - position.entry_price) * position.direction * position.contracts * POINT_VALUE


def _validate_sl_tp_at_fill(order: Order, fill_price: float) -> None:
    """
    Validate SL/TP against actual fill price. Log warnings for violations
    but don't crash — the order has already filled.
    """
    if order.sl_price is not None:
        if order.direction == 1 and order.sl_price >= fill_price:
            logger.warning(
                f"SL {order.sl_price} is at or above fill price {fill_price} on long — "
                f"position will exit immediately"
            )
        elif order.direction == -1 and order.sl_price <= fill_price:
            logger.warning(
                f"SL {order.sl_price} is at or below fill price {fill_price} on short — "
                f"position will exit immediately"
            )

    if order.tp_price is not None:
        if order.direction == 1 and order.tp_price <= fill_price:
            logger.warning(
                f"TP {order.tp_price} is at or below fill price {fill_price} on long"
            )
        elif order.direction == -1 and order.tp_price >= fill_price:
            logger.warning(
                f"TP {order.tp_price} is at or above fill price {fill_price} on short"
            )


def _apply_delta(
    engine: ExecutionEngine,
    order: Order,
    fill,
    position: OpenPosition,
    bar: Bar,
    bar_idx: int,
    trades: list[Trade],
    account: Account,
) -> Optional[OpenPosition]:
    """
    Apply delta resolution when a signal fires while a position is open.
    Returns the new OpenPosition (or None if fully closed with no new open).
    """
    equity_curve_stub: list[float] = []  # not used here, caller manages equity

    contracts_to_close, contracts_to_open = engine.resolve_delta(
        order, fill.contracts, position
    )

    new_position = position

    if contracts_to_close > 0:
        # Partial or full close
        if contracts_to_close >= position.contracts:
            # Full close
            trade = engine.build_trade(position, bar_idx, fill.fill_price, ExitReason.SIGNAL)
            trades.append(trade)
            account.balance += trade.net_pnl_dollars
            new_position = None
        else:
            # Partial close — reduce contract count, record partial trade
            partial = OpenPosition(
                direction=position.direction,
                entry_price=position.entry_price,
                entry_bar=position.entry_bar,
                contracts=contracts_to_close,
                sl_price=position.sl_price,
                tp_price=position.tp_price,
                trail_points=position.trail_points,
                trail_activation_points=position.trail_activation_points,
                trail_sl_price=position.trail_sl_price,
                trail_watermark=position.trail_watermark,
                order_placed_bar=position.order_placed_bar,
            )
            trade = engine.build_trade(partial, bar_idx, fill.fill_price, ExitReason.SIGNAL)
            trades.append(trade)
            account.balance += trade.net_pnl_dollars
            position.contracts -= contracts_to_close
            new_position = position

    if contracts_to_open > 0:
        from backtest.engine.execution import FillResult
        new_fill = FillResult(fill_price=fill.fill_price, contracts=contracts_to_open)
        new_position = _open_position(order, new_fill, bar_idx)

    return new_position