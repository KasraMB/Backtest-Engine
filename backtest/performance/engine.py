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
N_MC_SIMS   = 10_000
MC_PERCENTILES = [5, 25, 50, 75, 95]


class PerformanceEngine:

    def compute(
        self,
        result: "RunResult",
        data: "MarketData",
        n_mc_sims: int = N_MC_SIMS,
    ) -> Results:
        trades     = result.trades
        equity     = np.array(result.equity_curve, dtype=np.float64)
        capital    = result.config.starting_capital
        n_days     = len(data.trading_dates)
        strat_name = result.strategy_name

        if not trades:
            return self._empty_results(strat_name, capital, n_days, equity, data)

        pnls      = np.array([t.net_pnl_dollars for t in trades], dtype=np.float64)
        durations = np.array([t.exit_bar - t.entry_bar for t in trades], dtype=np.float64)

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

        # Streaks
        win_streak, loss_streak = self._streaks(pnls)

        # ── Equity curve & drawdown ──────────────────────────────────────────
        dd_stats        = self._drawdown_stats(equity, n_days)
        dd_curve_pct    = self._drawdown_curve(equity)

        # ── CAGR ─────────────────────────────────────────────────────────────
        years = n_days / 252.0
        if equity[-1] <= 0:
            # Equity wiped out — CAGR is -100% or worse; geometric formula undefined
            cagr = -1.0
        else:
            cagr = (equity[-1] / capital) ** (1.0 / max(years, 1e-9)) - 1.0

        # ── Risk-adjusted ────────────────────────────────────────────────────
        sharpe  = self._sharpe(trades, capital, data, equity)
        sortino = self._sortino(trades, capital, data, equity)
        # Calmar = CAGR / abs(max_dd_pct). Preserves sign of CAGR so a losing
        # strategy shows negative Calmar rather than a misleadingly large positive.
        calmar  = (cagr / abs(dd_stats.max_dd_pct)) if dd_stats.max_dd_pct != 0 else np.inf

        # ── MAE / MFE ────────────────────────────────────────────────────────
        mae_arr, mfe_arr = self._mae_mfe(trades, data)
        avg_mae = float(mae_arr.mean()) if len(mae_arr) else 0.0
        avg_mfe = float(mfe_arr.mean()) if len(mfe_arr) else 0.0

        # ── Exit breakdown ───────────────────────────────────────────────────
        exit_breakdown = self._exit_breakdown(trades)

        # ── Hourly breakdown ─────────────────────────────────────────────────
        hourly = self._hourly_breakdown(trades, data)

        # ── Monte Carlo ──────────────────────────────────────────────────────
        mc = self._monte_carlo(pnls, capital, n_mc_sims)

        # ── Bootstrap p-value & CIs ──────────────────────────────────────────
        pvalue = self._bootstrap_pvalue(pnls, n_mc_sims)
        cis    = self._confidence_intervals(pnls, capital, n_mc_sims, data, equity)

        # ── Benchmark ────────────────────────────────────────────────────────
        from backtest.performance.benchmark import compute_benchmark
        benchmark = compute_benchmark(
            data, capital,
            slippage_points         = result.config.slippage_points,
            commission_per_contract = result.config.commission_per_contract,
        )

        # ── Trade log ────────────────────────────────────────────────────────
        trade_log = self._build_trade_log(trades, data)

        return Results(
            strategy_name       = strat_name,
            starting_capital    = capital,
            n_trading_days      = n_days,
            total_net_pnl       = float(pnls.sum()),
            cagr                = float(cagr),
            final_equity        = float(equity[-1]),
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

    def _daily_equity_returns(self, equity: np.ndarray, data: "MarketData") -> np.ndarray:
        """
        Resample the bar-by-bar equity curve to one dollar P&L per trading day.

        Returns DOLLAR changes (not percentage returns) because:
        1. A fixed-contract futures strategy earns dollars, not reinvested %.
        2. Percentage returns blow up as equity approaches or crosses zero,
           producing outliers that inflate std and destroy the Sharpe CI.

        e.g. equity $10,000 → -$5,000 gives a -150% return which dominates
        the distribution even though it represents a normal sized day in dollars.

        The Sharpe computed from dollar daily P&L is:
            (mean_daily_pnl / std_daily_pnl) * sqrt(252)
        which is the standard definition for a constant-size strategy.
        """
        timestamps = data.df_1m.index   # length == len(equity) - 1

        # Build a map: date -> last bar index on that date
        last_bar_of_day: dict = {}
        for i, ts in enumerate(timestamps):
            last_bar_of_day[ts.date()] = i   # later bars overwrite earlier ones

        sorted_dates = sorted(last_bar_of_day.keys())
        # equity[i+1] corresponds to bar i (equity[0] = starting capital)
        daily_eq = np.array(
            [equity[last_bar_of_day[d] + 1] for d in sorted_dates],
            dtype=np.float64,
        )

        # Dollar change per day (prepend starting capital for first day's return)
        eq_with_start = np.concatenate([[equity[0]], daily_eq])
        dollar_pnl = np.diff(eq_with_start)   # simple difference, not divided by prior equity
        return dollar_pnl

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
        rng = np.random.default_rng(42)
        n   = len(pnls)

        def _equity_from_pnls(sim_pnls: np.ndarray) -> np.ndarray:
            eq = np.empty(n + 1)
            eq[0] = capital
            np.cumsum(sim_pnls, out=eq[1:])
            eq[1:] += capital
            return eq

        def _max_dd_pct(eq: np.ndarray) -> float:
            peak = np.maximum.accumulate(eq)
            dd   = (eq - peak) / np.where(peak == 0, 1, peak)
            return float(dd.min())

        # ── Shuffle (without replacement) ────────────────────────────────────
        shuf_finals = np.empty(n_sims)
        shuf_dd     = np.empty(n_sims)
        shuf_curves = []

        for s in range(n_sims):
            shuffled = rng.permutation(pnls)
            eq       = _equity_from_pnls(shuffled)
            shuf_finals[s] = eq[-1]
            shuf_dd[s]     = _max_dd_pct(eq)
            shuf_curves.append(eq)

        shuf_curves_arr = np.array(shuf_curves)
        shuf_pcts = {}
        for p in MC_PERCENTILES:
            try:
                shuf_pcts[p] = np.percentile(shuf_curves_arr, p, axis=0)
            except Exception:
                shuf_pcts[p] = shuf_curves_arr[0]

        # ── Bootstrap (with replacement) ─────────────────────────────────────
        boot_finals = np.empty(n_sims)
        boot_dd     = np.empty(n_sims)
        boot_curves = []

        for s in range(n_sims):
            resampled = rng.choice(pnls, size=n, replace=True)
            eq        = _equity_from_pnls(resampled)
            boot_finals[s] = eq[-1]
            boot_dd[s]     = _max_dd_pct(eq)
            boot_curves.append(eq)

        boot_curves_arr = np.array(boot_curves)
        boot_pcts = {}
        for p in MC_PERCENTILES:
            try:
                boot_pcts[p] = np.percentile(boot_curves_arr, p, axis=0)
            except Exception:
                boot_pcts[p] = boot_curves_arr[0]

        return MonteCarloResults(
            shuffle_percentiles   = shuf_pcts,
            bootstrap_percentiles = boot_pcts,
            shuffle_final_equity  = shuf_finals,
            bootstrap_final_equity= boot_finals,
            shuffle_max_dd_pct    = shuf_dd,
            bootstrap_max_dd_pct  = boot_dd,
            shuffle_p5            = float(np.percentile(shuf_finals, 5)),
            shuffle_p50           = float(np.percentile(shuf_finals, 50)),
            shuffle_p95           = float(np.percentile(shuf_finals, 95)),
            bootstrap_p5          = float(np.percentile(boot_finals, 5)),
            bootstrap_p50         = float(np.percentile(boot_finals, 50)),
            bootstrap_p95         = float(np.percentile(boot_finals, 95)),
        )

    def _bootstrap_pvalue(self, pnls: np.ndarray, n_sims: int) -> float:
        """
        Bootstrap p-value for H0: E[trade PnL] <= 0.
        p-value = fraction of bootstrap means <= 0.
        """
        rng = np.random.default_rng(42)
        n   = len(pnls)
        boot_means = np.empty(n_sims)
        for s in range(n_sims):
            boot_means[s] = rng.choice(pnls, size=n, replace=True).mean()
        return float((boot_means <= 0).mean())

    def _confidence_intervals(
        self,
        pnls: np.ndarray,
        capital: float,
        n_sims: int,
        data: "MarketData" = None,
        equity: np.ndarray = None,
    ) -> ConfidenceIntervals:
        """95% bootstrap CIs for key metrics."""
        rng = np.random.default_rng(42)
        n   = len(pnls)

        sharpes, win_rates, expectancies, pfs, dd_pcts = [], [], [], [], []

        # Bootstrap Sharpe CI: resample daily returns (not individual trades).
        # This avoids within-day trade-correlation problems and matches _sharpe().
        daily_returns = self._daily_equity_returns(equity, data) if equity is not None and len(equity) > 1 else np.array([])
        n_daily = len(daily_returns)

        for _ in range(n_sims):
            s = rng.choice(pnls, size=n, replace=True)
            winners = s[s > 0]
            losers  = s[s < 0]

            # Sharpe: bootstrap over daily returns
            if n_daily >= 2:
                d = rng.choice(daily_returns, size=n_daily, replace=True)
                if d.std() > 0:
                    sharpes.append(float((d.mean() / d.std()) * np.sqrt(252)))

            # Win rate
            win_rates.append((s > 0).mean())

            # Expectancy
            expectancies.append(s.mean())

            # Profit factor
            if len(losers) and losers.sum() != 0:
                pfs.append(winners.sum() / abs(losers.sum()))

            # Max DD %
            eq   = np.empty(n + 1)
            eq[0] = capital
            np.cumsum(s, out=eq[1:])
            eq[1:] += capital
            peak = np.maximum.accumulate(eq)
            dd   = (eq - peak) / np.where(peak == 0, 1, peak)
            dd_pcts.append(dd.min())

        def ci(arr):
            a = np.array(arr)
            if len(a) < 2:
                v = float(a[0]) if len(a) == 1 else 0.0
                return (v, v)
            return (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))

        return ConfidenceIntervals(
            sharpe        = ci(sharpes),
            win_rate      = ci(win_rates),
            expectancy    = ci(expectancies),
            profit_factor = ci(pfs) if pfs else (0.0, 0.0),
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
            n_trades=0, win_rate=0.0, avg_win_dollars=0.0, avg_loss_dollars=0.0,
            avg_trade_pnl_dollars=0.0, payoff_ratio=0.0, profit_factor=0.0,
            expectancy_r=0.0, largest_win=0.0, largest_loss=0.0,
            longest_win_streak=0, longest_loss_streak=0, avg_trade_duration_bars=0.0,
            sharpe=0.0, sortino=0.0, calmar=0.0, drawdown=zero_dd,
            avg_mae_pct=0.0, avg_mfe_pct=0.0,
            mae_per_trade=np.array([]), mfe_per_trade=np.array([]),
            equity_curve=equity, drawdown_curve_pct=np.zeros(len(equity)),
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