"""
PerformanceEngine — Phase 6
────────────────────────────
Takes a RunResult and computes all performance metrics into a Results object.
"""
from __future__ import annotations
import numpy as np
from collections import defaultdict
from typing import TYPE_CHECKING

from backtest.performance.results import (
    Results, DrawdownStats, MonteCarloResults, ConfidenceIntervals,
    ExitBreakdown, HourlyBreakdown, BenchmarkResult, TradeRow,
)

if TYPE_CHECKING:
    from backtest.runner.runner import RunResult
    from backtest.data.market_data import MarketData

POINT_VALUE = 20.0
N_MC_SIMS   = 2_000
MC_PERCENTILES = [5, 25, 50, 75, 95]


class PerformanceEngine:

    def compute(
        self,
        result: "RunResult",
        data: "MarketData",
        n_mc_sims: int = N_MC_SIMS,
    ) -> Results:
        import time as _time
        _t0 = _time.perf_counter()
        def _tick(label, t_prev, threshold=0.3):
            """Print only if this step took longer than threshold seconds."""
            t = _time.perf_counter()
            if t - t_prev >= threshold:
                print(f"    [{label}]  {t - t_prev:.2f}s")
            return t

        trades     = result.trades
        equity     = np.array(result.equity_curve, dtype=np.float64)
        capital    = result.config.starting_capital
        n_days     = len(data.trading_dates)
        strat_name = result.strategy_name

        if not trades:
            return self._empty_results(strat_name, capital, n_days, equity, data)

        pnls      = np.array([t.net_pnl_dollars for t in trades], dtype=np.float64)
        durations = np.array([t.exit_bar - t.entry_bar for t in trades], dtype=np.float64)

        # Pre-build a lightweight date→pnl lookup used by _daily_equity_returns
        # for trade-resolution equity curves (reversed runs).  Built early so
        # _sharpe/_sortino can use it even before the full trade log is ready.
        # Guard: exit_bar may exceed test data length, clamp to valid range.
        _max_bar = len(data.df_1m) - 1
        self._trade_log_for_daily = [
            {"exit": data.df_1m.index[min(t.exit_bar, _max_bar)].strftime("%Y-%m-%d"),
             "net_pnl": t.net_pnl_dollars}
            for t in trades
        ]

        total_commission = float(sum(t.commission_paid  for t in trades))
        total_slippage   = float(sum(t.slippage_paid    for t in trades))

        # ── Core stats ───────────────────────────────────────────────────────
        n_trades       = len(trades)
        winners        = pnls[pnls > 0]
        losers         = pnls[pnls < 0]
        win_rate       = len(winners) / n_trades
        avg_win        = float(winners.mean()) if len(winners) else 0.0
        avg_loss       = float(losers.mean())  if len(losers)  else 0.0
        avg_trade      = float(pnls.mean())
        payoff         = abs(avg_win / avg_loss) if avg_loss != 0 else np.inf
        profit_factor  = (winners.sum() / abs(losers.sum())
                          if len(losers) and losers.sum() != 0 else np.inf)
        expectancy_r   = avg_trade / abs(avg_loss) if avg_loss != 0 else np.inf
        largest_win    = float(pnls.max())
        largest_loss   = float(pnls.min())
        win_streak, loss_streak = self._streaks(pnls)
        _t = _tick("core stats + streaks", _t0)

        # ── Equity curve & drawdown ──────────────────────────────────────────
        dd_stats        = self._drawdown_stats(equity, n_days)
        dd_curve_pct    = self._drawdown_curve(equity)
        years = n_days / 252.0
        if equity[-1] <= 0:
            cagr = -1.0
        else:
            cagr = (equity[-1] / capital) ** (1.0 / max(years, 1e-9)) - 1.0
        _t = _tick("drawdown + CAGR", _t)

        # ── Risk-adjusted ────────────────────────────────────────────────────
        sharpe  = self._sharpe(trades, capital, data, equity)
        sortino = self._sortino(trades, capital, data, equity)
        calmar  = (cagr / abs(dd_stats.max_dd_pct)) if dd_stats.max_dd_pct != 0 else np.inf
        _t = _tick("Sharpe / Sortino / Calmar", _t)

        # ── MAE / MFE ────────────────────────────────────────────────────────
        mae_arr, mfe_arr = self._mae_mfe(trades, data)
        avg_mae = float(mae_arr.mean()) if len(mae_arr) else 0.0
        avg_mfe = float(mfe_arr.mean()) if len(mfe_arr) else 0.0
        _t = _tick("MAE / MFE", _t)

        # ── Exit / hourly breakdown ───────────────────────────────────────────
        exit_breakdown = self._exit_breakdown(trades)
        hourly = self._hourly_breakdown(trades, data)
        _t = _tick("exit + hourly breakdown", _t)

        # ── Monte Carlo ──────────────────────────────────────────────────────
        mc = self._monte_carlo(pnls, capital, n_mc_sims)
        _t = _tick(f"Monte Carlo ({n_mc_sims:,} sims)", _t)

        # ── Bootstrap p-value & CIs ──────────────────────────────────────────
        pvalue = self._bootstrap_pvalue(pnls, n_mc_sims)
        _t = _tick("bootstrap p-value", _t)
        cis    = self._confidence_intervals(pnls, capital, n_mc_sims, data, equity)
        _t = _tick("confidence intervals", _t)

        # ── Benchmark ────────────────────────────────────────────────────────
        from backtest.performance.benchmark import compute_benchmark
        benchmark = compute_benchmark(
            data, capital,
            slippage_points         = result.config.slippage_points,
            commission_per_contract = result.config.commission_per_contract,
        )
        _t = _tick("benchmark", _t)

        # ── Trade log ────────────────────────────────────────────────────────
        trade_log = self._build_trade_log(trades, data)
        _t = _tick("trade log", _t)
        print(f"    [TOTAL compute()]  {_t - _t0:.2f}s")

        return Results(
            strategy_name       = strat_name,
            starting_capital    = capital,
            n_trading_days      = n_days,
            total_net_pnl       = float(pnls.sum()),
            cagr                = float(cagr),
            final_equity        = float(equity[-1]),
            total_commission    = total_commission,
            total_slippage      = total_slippage,
            n_trades            = n_trades,
            win_rate            = win_rate,
            avg_win_dollars     = avg_win,
            avg_loss_dollars    = avg_loss,
            avg_trade_pnl_dollars = avg_trade,
            payoff_ratio        = payoff,
            profit_factor       = profit_factor,
            expectancy_r        = expectancy_r,
            largest_win         = largest_win,
            largest_loss        = largest_loss,
            longest_win_streak  = win_streak,
            longest_loss_streak = loss_streak,
            avg_trade_duration_bars = float(durations.mean()),
            sharpe              = sharpe,
            sortino             = sortino,
            calmar              = calmar,
            drawdown            = dd_stats,
            avg_mae_pct         = avg_mae,
            avg_mfe_pct         = avg_mfe,
            mae_per_trade       = mae_arr,
            mfe_per_trade       = mfe_arr,
            equity_curve        = equity,
            drawdown_curve_pct  = dd_curve_pct,
            equity_timestamps   = data.df_1m.index,
            exit_breakdown      = exit_breakdown,
            hourly_breakdown    = hourly,
            trade_durations_bars = durations,
            trade_pnls          = pnls,
            monte_carlo         = mc,
            bootstrap_pvalue    = pvalue,
            confidence_intervals = cis,
            benchmark           = benchmark,
            trade_log           = trade_log,
        )

    # ── Private helpers ──────────────────────────────────────────────────────

    def _drawdown_curve(self, equity: np.ndarray) -> np.ndarray:
        peak = np.maximum.accumulate(equity)
        return (equity - peak) / np.where(peak == 0, 1, peak)

    def _drawdown_stats(self, equity: np.ndarray, n_days: int) -> DrawdownStats:
        peak    = np.maximum.accumulate(equity)
        dd_abs  = equity - peak
        dd_pct  = dd_abs / np.where(peak == 0, 1, peak)

        max_dd_pct = float(dd_pct.min())
        max_dd_usd = float(dd_abs.min())

        # Average drawdown (only periods actually in drawdown)
        in_dd = dd_pct[dd_pct < 0]
        avg_dd = float(in_dd.mean()) if len(in_dd) else 0.0

        # Max drawdown duration (bars)
        max_dur = 0
        cur_dur = 0
        for i in range(len(equity)):
            if dd_pct[i] < 0:
                cur_dur += 1
                max_dur = max(max_dur, cur_dur)
            else:
                cur_dur = 0

        # Convert bars to days (approx: 390 bars per RTH session)
        max_dur_days = max_dur / 390.0

        return DrawdownStats(
            max_dd_dollars      = max_dd_usd,
            max_dd_pct          = max_dd_pct,
            avg_dd_pct          = avg_dd,
            max_dd_duration_bars = max_dur,
            max_dd_duration_days = max_dur_days,
        )

    @staticmethod
    def _build_day_index(data: "MarketData"):
        """
        Vectorised: returns (unique_dates, last_bar_idx) arrays, cached on data.
        unique_dates : numpy array of date objects, shape (n_days,)
        last_bar_idx : index into df_1m of the last bar each day, shape (n_days,)
        """
        cached = getattr(data, '_day_index_cache', None)
        if cached is not None:
            return cached
        ts = data.df_1m.index
        date_codes = ts.normalize().asi8           # int64, one per bar
        change = np.concatenate([[True], date_codes[1:] != date_codes[:-1]])
        first_bar_idx = np.where(change)[0]
        last_bar_idx  = np.concatenate([first_bar_idx[1:] - 1, [len(ts) - 1]])
        unique_dates  = ts[first_bar_idx].date    # numpy array of datetime.date
        result = (unique_dates, last_bar_idx)
        try:
            data._day_index_cache = result
        except Exception:
            pass
        return result

    def _daily_equity_returns(self, equity: np.ndarray, data: "MarketData") -> np.ndarray:
        """
        Dollar equity change per trading day.

        Bar-resolution  (len = n_bars+1): index equity[last_bar+1] per day.
        Trade-resolution (len != n_bars+1): group trade PnLs by exit date via
          self._trade_log_for_daily, walk forward across all trading days.
        """
        unique_dates, last_bar_idx = self._build_day_index(data)

        if len(equity) == len(data.df_1m) + 1:
            # fast vectorised path
            daily_eq = equity[last_bar_idx + 1]
        else:
            trade_log = getattr(self, '_trade_log_for_daily', None)
            if not trade_log:
                return np.array([])
            date_pnl: dict = {}
            for t in trade_log:
                key = t["exit"][:10]
                date_pnl[key] = date_pnl.get(key, 0.0) + t["net_pnl"]
            running = float(equity[0])
            daily_eq_list = []
            for d in unique_dates:
                key = d.isoformat() if hasattr(d, 'isoformat') else str(d)
                running += date_pnl.get(key, 0.0)
                daily_eq_list.append(running)
            daily_eq = np.array(daily_eq_list, dtype=np.float64)

        eq_with_start = np.concatenate([[float(equity[0])], daily_eq])
        return np.diff(eq_with_start)

    def _sharpe(self, trades, capital: float, data: "MarketData", equity: np.ndarray = None) -> float:
        """Sharpe ratio from daily dollar P&L. mean/std * sqrt(252)."""
        if equity is None or len(equity) < 2:
            return 0.0
        daily = self._daily_equity_returns(equity, data)
        if len(daily) < 2 or daily.std() == 0:
            return 0.0
        return float((daily.mean() / daily.std()) * np.sqrt(252))

    def _sortino(self, trades, capital: float, data: "MarketData", equity: np.ndarray = None) -> float:
        """Sortino ratio from daily dollar P&L. mean / downside_std * sqrt(252)."""
        if equity is None or len(equity) < 2:
            return 0.0
        daily    = self._daily_equity_returns(equity, data)
        downside = daily[daily < 0]
        if len(downside) < 2 or downside.std() == 0:
            return 0.0
        return float((daily.mean() / downside.std()) * np.sqrt(252))

    def _streaks(self, pnls: np.ndarray) -> tuple[int, int]:
        max_win = max_loss = cur_win = cur_loss = 0
        for p in pnls:
            if p > 0:
                cur_win  += 1
                cur_loss  = 0
            else:
                cur_loss += 1
                cur_win   = 0
            max_win  = max(max_win,  cur_win)
            max_loss = max(max_loss, cur_loss)
        return max_win, max_loss

    def _mae_mfe(self, trades, data: "MarketData"):
        """
        Compute MAE and MFE for each trade as % of entry price.
        Uses the bar range during the trade duration.
        """
        opens  = data.open_1m
        highs  = data.high_1m
        lows   = data.low_1m
        n_bars = len(opens)

        mae_list, mfe_list = [], []

        for t in trades:
            entry = t.entry_price
            if entry == 0:
                continue
            start = t.entry_bar
            end   = min(t.exit_bar + 1, n_bars)

            if start >= end:
                mae_list.append(0.0)
                mfe_list.append(0.0)
                continue

            bar_highs = highs[start:end]
            bar_lows  = lows[start:end]

            if t.direction == 1:
                mae = (entry - bar_lows.min())  / entry  # worst low vs entry
                mfe = (bar_highs.max() - entry) / entry  # best high vs entry
            else:
                mae = (bar_highs.max() - entry) / entry  # worst high vs entry (short)
                mfe = (entry - bar_lows.min())  / entry  # best low vs entry (short)

            mae_list.append(max(mae, 0.0))
            mfe_list.append(max(mfe, 0.0))

        return np.array(mae_list), np.array(mfe_list)

    def _exit_breakdown(self, trades) -> list[ExitBreakdown]:
        groups: dict[str, list[float]] = defaultdict(list)
        for t in trades:
            groups[t.exit_reason.name].append(t.net_pnl_dollars)

        result = []
        for reason, pnls in sorted(groups.items()):
            arr = np.array(pnls)
            result.append(ExitBreakdown(
                reason             = reason,
                count              = len(arr),
                win_rate           = float((arr > 0).mean()),
                avg_pnl_dollars    = float(arr.mean()),
                total_pnl_dollars  = float(arr.sum()),
            ))
        return result

    def _hourly_breakdown(self, trades, data: "MarketData") -> list[HourlyBreakdown]:
        groups: dict[int, list[float]] = defaultdict(list)
        timestamps = data.df_1m.index

        for t in trades:
            if t.entry_bar < len(timestamps):
                hour = timestamps[t.entry_bar].hour
                groups[hour].append(t.net_pnl_dollars)

        result = []
        for hour in sorted(groups.keys()):
            arr = np.array(groups[hour])
            result.append(HourlyBreakdown(
                hour               = hour,
                count              = len(arr),
                avg_pnl_dollars    = float(arr.mean()),
                win_rate           = float((arr > 0).mean()),
                total_pnl_dollars  = float(arr.sum()),
            ))
        return result

    def _monte_carlo(
        self,
        pnls: np.ndarray,
        capital: float,
        n_sims: int,
    ) -> MonteCarloResults:
        """
        Vectorised Monte Carlo.  Uses float32 curves (halves memory vs float64;
        $0.01 precision is more than sufficient).  Percentile curves are computed
        via np.partition (O(n_sims) per time-step) on the transposed buffer,
        avoiding a full sort across all n_sims × (n+1) values.

        Memory: n_sims × (n+1) × 4 bytes.
          2,000 sims × 12,451 steps = 100 MB   ← typical Huang strategy
          2,000 sims ×  3,100 steps =  25 MB   ← typical ORB strategy
        """
        rng = np.random.default_rng(42)
        n   = len(pnls)
        pct_ranks = {p: max(0, int(round(p / 100 * (n_sims - 1)))) for p in MC_PERCENTILES}

        def _run_sims(sample_fn):
            # Build all curves at once in float32
            samp   = sample_fn(n_sims)                   # (n_sims, n) float64
            curves = np.empty((n_sims, n + 1), dtype=np.float32)
            curves[:, 0] = capital
            np.cumsum(samp, axis=1, out=curves[:, 1:])
            curves[:, 1:] += capital

            finals  = curves[:, -1].astype(np.float64)

            # Max drawdown per sim (in float64 for accuracy)
            c64  = curves.astype(np.float64)
            peak = np.maximum.accumulate(c64, axis=1)
            safe = np.where(peak == 0, 1.0, peak)
            max_dds = ((c64 - peak) / safe).min(axis=1)
            del c64, peak, safe                          # free float64 copy early

            # Percentile curves via partition on transposed buffer
            # buf_T shape: (n+1, n_sims) — each row = one time-step across all sims
            buf_T = curves.T
            pcts  = {}
            for p, rank in pct_ranks.items():
                pcts[p] = np.partition(buf_T, rank, axis=1)[:, rank].astype(np.float64)

            return finals, max_dds, pcts

        shuf_finals, shuf_dd, shuf_pcts = _run_sims(
            lambda ns: pnls[np.argsort(rng.random((ns, n)), axis=1)]
        )
        boot_finals, boot_dd, boot_pcts = _run_sims(
            lambda ns: pnls[rng.integers(0, n, size=(ns, n))]
        )

        return MonteCarloResults(
            shuffle_percentiles    = shuf_pcts,
            bootstrap_percentiles  = boot_pcts,
            shuffle_final_equity   = shuf_finals,
            bootstrap_final_equity = boot_finals,
            shuffle_max_dd_pct     = shuf_dd,
            bootstrap_max_dd_pct   = boot_dd,
            shuffle_p5             = float(np.percentile(shuf_finals, 5)),
            shuffle_p50            = float(np.percentile(shuf_finals, 50)),
            shuffle_p95            = float(np.percentile(shuf_finals, 95)),
            bootstrap_p5           = float(np.percentile(boot_finals, 5)),
            bootstrap_p50          = float(np.percentile(boot_finals, 50)),
            bootstrap_p95          = float(np.percentile(boot_finals, 95)),
        )

    def _bootstrap_pvalue(self, pnls: np.ndarray, n_sims: int) -> float:
        """Bootstrap p-value for H0: E[trade PnL] <= 0."""
        rng = np.random.default_rng(42)
        n   = len(pnls)
        idx = rng.integers(0, n, size=(n_sims, n))
        boot_means = pnls[idx].mean(axis=1)
        return float((boot_means <= 0).mean())

    def _confidence_intervals(
        self,
        pnls: np.ndarray,
        capital: float,
        n_sims: int,
        data: "MarketData" = None,
        equity: np.ndarray = None,
    ) -> ConfidenceIntervals:
        """95% bootstrap CIs for key metrics — fully vectorised."""
        rng = np.random.default_rng(42)
        n   = len(pnls)

        # Resample trades: (n_sims, n)
        idx      = rng.integers(0, n, size=(n_sims, n))
        samples  = pnls[idx]                                 # (n_sims, n)

        # Win rate
        win_rates = (samples > 0).mean(axis=1)

        # Expectancy
        expectancies = samples.mean(axis=1)

        # Profit factor
        wins_sum  = np.where(samples > 0, samples, 0).sum(axis=1)
        loss_sum  = np.abs(np.where(samples < 0, samples, 0).sum(axis=1))
        valid_pf  = loss_sum > 0
        pfs       = np.where(valid_pf, wins_sum / np.where(valid_pf, loss_sum, 1), np.nan)

        # Max DD — float32 to halve memory (200MB→100MB for 12k trades / 2k sims)
        curves_f32       = np.empty((n_sims, n + 1), dtype=np.float32)
        curves_f32[:, 0] = capital
        np.cumsum(samples.astype(np.float32), axis=1, out=curves_f32[:, 1:])
        curves_f32[:, 1:] += capital
        c64      = curves_f32.astype(np.float64)
        peak     = np.maximum.accumulate(c64, axis=1)
        safe     = np.where(peak == 0, 1.0, peak)
        dd_pcts  = ((c64 - peak) / safe).min(axis=1)
        del curves_f32, c64, peak, safe

        # Sharpe CI: bootstrap over daily returns
        daily_returns = (
            self._daily_equity_returns(equity, data)
            if equity is not None and len(equity) > 1
            else np.array([])
        )
        if len(daily_returns) >= 2:
            nd   = len(daily_returns)
            didx = rng.integers(0, nd, size=(n_sims, nd))
            d    = daily_returns[didx]                       # (n_sims, nd)
            std  = d.std(axis=1)
            valid = std > 0
            sharpes = np.where(valid, (d.mean(axis=1) / np.where(valid, std, 1)) * np.sqrt(252), np.nan)
        else:
            sharpes = np.array([np.nan])

        def ci(arr):
            a = arr[~np.isnan(arr)] if hasattr(arr, '__len__') else arr
            if len(a) < 2:
                v = float(a[0]) if len(a) == 1 else 0.0
                return (v, v)
            return (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))

        pf_vals = pfs[~np.isnan(pfs)]
        return ConfidenceIntervals(
            sharpe        = ci(sharpes),
            win_rate      = ci(win_rates),
            expectancy    = ci(expectancies),
            profit_factor = ci(pf_vals) if len(pf_vals) >= 2 else (0.0, 0.0),
            max_dd_pct    = ci(dd_pcts),
        )

    def _empty_results(self, name, capital, n_days, equity, data) -> Results:
        """Return a zeroed Results when there are no trades."""
        from backtest.performance.benchmark import compute_benchmark
        benchmark = compute_benchmark(data, capital)  # _empty_results has no config
        zero_mc = MonteCarloResults(
            shuffle_percentiles={p: equity for p in MC_PERCENTILES},
            bootstrap_percentiles={p: equity for p in MC_PERCENTILES},
            shuffle_final_equity=np.array([capital]),
            bootstrap_final_equity=np.array([capital]),
            shuffle_max_dd_pct=np.array([0.0]),
            bootstrap_max_dd_pct=np.array([0.0]),
            shuffle_p5=capital, shuffle_p50=capital, shuffle_p95=capital,
            bootstrap_p5=capital, bootstrap_p50=capital, bootstrap_p95=capital,
        )
        zero_ci = ConfidenceIntervals(
            sharpe=(0.0, 0.0), win_rate=(0.0, 0.0), expectancy=(0.0, 0.0),
            profit_factor=(0.0, 0.0), max_dd_pct=(0.0, 0.0),
        )
        zero_dd = DrawdownStats(0.0, 0.0, 0.0, 0, 0.0)
        return Results(
            strategy_name=name, starting_capital=capital, n_trading_days=n_days,
            total_net_pnl=0.0, cagr=0.0, final_equity=capital,
            total_commission=0.0, total_slippage=0.0,
            n_trades=0, win_rate=0.0, avg_win_dollars=0.0, avg_loss_dollars=0.0,
            avg_trade_pnl_dollars=0.0, payoff_ratio=0.0, profit_factor=0.0,
            expectancy_r=0.0, largest_win=0.0, largest_loss=0.0,
            longest_win_streak=0, longest_loss_streak=0, avg_trade_duration_bars=0.0,
            sharpe=0.0, sortino=0.0, calmar=0.0, drawdown=zero_dd,
            avg_mae_pct=0.0, avg_mfe_pct=0.0,
            mae_per_trade=np.array([]), mfe_per_trade=np.array([]),
            equity_curve=equity, drawdown_curve_pct=np.zeros(len(equity)),
            equity_timestamps=data.df_1m.index,
            exit_breakdown=[], hourly_breakdown=[],
            trade_durations_bars=np.array([]), trade_pnls=np.array([]),
            monte_carlo=zero_mc, bootstrap_pvalue=1.0,
            confidence_intervals=zero_ci, benchmark=benchmark,
            trade_log=[],
        )

    def _build_trade_log(self, trades: list, data: "MarketData") -> list:
        """Convert Trade objects to TradeRow dicts for the tearsheet table."""
        ts = data.df_1m.index
        n  = len(ts)
        rows = []
        for i, t in enumerate(trades):
            entry_dt = str(ts[min(t.entry_bar, n - 1)])[:16]
            exit_dt  = str(ts[min(t.exit_bar,  n - 1)])[:16]
            rows.append({
                "num":       i + 1,
                "entry":     entry_dt,
                "exit":      exit_dt,
                "dir":       "LONG" if t.direction == 1 else "SHORT",
                "qty":       t.contracts,
                "ep":        round(t.entry_price, 2),
                "xp":        round(t.exit_price, 2),
                "pts":       round(t.pnl_points, 2),
                "gross":     round(t.pnl_dollars, 2),
                "net":       round(t.net_pnl_dollars, 2),
                "reason":    t.exit_reason.name,
                "dur":       t.exit_bar - t.entry_bar,
            })
        return rows