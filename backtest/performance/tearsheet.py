"""
TearsheetRenderer — Phase 6
─────────────────────────────
Produces a self-contained HTML tearsheet with interactive Plotly charts.
No external dependencies — opens in any browser.

Size optimisations vs naive approach:
  - Equity/drawdown curves downsampled to MAX_CURVE_PTS points
  - MC percentile curves downsampled to MAX_MC_CURVE_PTS points
  - MC histograms pre-binned in numpy — raw 10k arrays never hit the HTML
  - MAE/MFE scatter capped at MAX_SCATTER_PTS points
  - All floats rounded to 2dp in JSON
  - Compact JSON (no whitespace)
"""
from __future__ import annotations
import json
import os
import webbrowser
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from backtest.performance.results import Results

# ── Downsample limits ────────────────────────────────────────────────────────
MAX_CURVE_PTS    = 2_000   # equity curve, drawdown curve, benchmark curve
MAX_MC_CURVE_PTS = 2_000   # MC percentile fan lines — needs higher res to show spread
MAX_SCATTER_PTS  = 3_000   # MAE/MFE scatter
N_HIST_BINS      =    80   # pre-binned histograms


def _fmt_pct(v: float, decimals: int = 2) -> str:
    if v != v or abs(v) == float("inf"):
        return "N/A"
    return f"{v * 100:.{decimals}f}%"

def _fmt_dollar(v: float) -> str:
    sign = "+" if v >= 0 else ""
    return f"{sign}${v:,.2f}"

def _fmt_num(v: float, decimals: int = 2) -> str:
    if v != v:           # NaN
        return "N/A"
    if abs(v) == float("inf"):
        return "N/A"
    return f"{v:.{decimals}f}"

def _color(v: float) -> str:
    return "#26a69a" if v >= 0 else "#ef5350"


def _downsample_xy(arr: np.ndarray, max_pts: int):
    """
    Uniform-stride downsample. Returns (x_list, y_list) as Python lists.
    Always includes the last point so the curve reaches its final value.
    """
    n = len(arr)
    if n <= max_pts:
        return list(range(n)), _r(arr.tolist())
    idx = np.linspace(0, n - 1, max_pts, dtype=int)
    idx[-1] = n - 1  # always include final value
    return idx.tolist(), _r(arr[idx].tolist())


def _prebinned_histogram(arr: np.ndarray, n_bins: int = N_HIST_BINS):
    """
    Compute histogram in numpy and return (bin_centers, counts).
    Passing pre-binned data as a bar chart is ~100x smaller than raw float arrays.
    """
    counts, edges = np.histogram(arr, bins=n_bins)
    centers = ((edges[:-1] + edges[1:]) / 2)
    return _r(centers.tolist()), counts.tolist()


def _r(lst: list, d: int = 2) -> list:
    """Round a list of floats to d decimal places to reduce JSON size."""
    return [round(v, d) for v in lst]


def _j(obj) -> str:
    """Compact JSON — no extra whitespace."""
    return json.dumps(obj, separators=(",", ":"))


class TearsheetRenderer:

    def render(
        self,
        results: "Results",
        output_path: str = "tearsheet.html",
        auto_open: bool = True,
    ) -> str:
        """
        Render a self-contained HTML tearsheet.

        Args:
            results:     Computed Results from PerformanceEngine.
            output_path: Path to save the HTML file.
            auto_open:   Open in default browser immediately (default True).
        """
        html = self._build_html(results)

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

        abs_path = os.path.abspath(output_path)
        size_mb  = os.path.getsize(abs_path) / 1_000_000
        print(f"\nTearsheet saved: {abs_path}  ({size_mb:.1f} MB)")

        if auto_open:
            webbrowser.open(f"file:///{abs_path.replace(os.sep, '/')}")

        return abs_path

    # ── HTML assembly ────────────────────────────────────────────────────────

    def _build_html(self, r: "Results") -> str:
        sections = "\n".join([
            self._section_header(r),
            self._section_core_metrics(r),
            self._section_equity_curve(r),
            self._section_drawdown(r),
            self._section_monte_carlo(r),
            self._section_bootstrap(r),
            self._section_mae_mfe(r),
            self._section_exit_breakdown(r),
            self._section_hourly(r),
            self._section_trade_distribution(r),
            self._section_benchmark(r),
            self._section_trade_log(r),
        ])

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Tearsheet — {r.strategy_name}</title>
<script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:#0d1117;color:#e6edf3;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;font-size:14px}}
  .container{{max-width:1400px;margin:0 auto;padding:24px}}
  h1{{font-size:24px;font-weight:600;margin-bottom:4px}}
  h2{{font-size:16px;font-weight:600;color:#8b949e;text-transform:uppercase;letter-spacing:.05em;margin:32px 0 12px}}
  .subtitle{{color:#8b949e;font-size:13px;margin-bottom:32px}}
  .metric-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:12px;margin-bottom:8px}}
  .metric-card{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px}}
  .metric-label{{font-size:11px;color:#8b949e;text-transform:uppercase;letter-spacing:.05em;margin-bottom:6px}}
  .metric-value{{font-size:22px;font-weight:600}}
  .metric-sub{{font-size:11px;color:#8b949e;margin-top:4px}}
  .positive{{color:#26a69a}}.negative{{color:#ef5350}}.neutral{{color:#e6edf3}}
  .chart-box{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px;margin-bottom:16px}}
  .two-col{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
  table{{width:100%;border-collapse:collapse}}
  th{{text-align:left;padding:8px 12px;font-size:11px;color:#8b949e;text-transform:uppercase;letter-spacing:.05em;border-bottom:1px solid #30363d}}
  td{{padding:8px 12px;border-bottom:1px solid #21262d;font-size:13px}}
  tr:last-child td{{border-bottom:none}}
  .section-note{{font-size:11px;color:#8b949e;margin-top:8px}}
  .pvalue-badge{{display:inline-block;padding:4px 12px;border-radius:12px;font-size:13px;font-weight:600}}
  .pvalue-good{{background:rgba(38,166,154,.15);color:#26a69a;border:1px solid #26a69a}}
  .pvalue-bad{{background:rgba(239,83,80,.15);color:#ef5350;border:1px solid #ef5350}}
  /* Trade log */
  .tl-controls{{display:flex;flex-wrap:wrap;gap:10px;margin-bottom:14px;align-items:center}}
  .tl-controls input,.tl-controls select{{background:#0d1117;border:1px solid #30363d;color:#e6edf3;
    border-radius:6px;padding:6px 10px;font-size:13px;outline:none}}
  .tl-controls input{{width:220px}}.tl-controls input:focus,.tl-controls select:focus{{border-color:#26a69a}}
  .tl-controls select{{cursor:pointer}}
  .tl-controls label{{font-size:12px;color:#8b949e;white-space:nowrap}}
  .tl-btn{{background:#161b22;border:1px solid #30363d;color:#e6edf3;border-radius:6px;
    padding:6px 14px;font-size:13px;cursor:pointer;transition:border-color .15s}}
  .tl-btn:hover{{border-color:#26a69a;color:#26a69a}}
  .tl-btn.active{{border-color:#26a69a;color:#26a69a;background:rgba(38,166,154,.08)}}
  .tl-stats{{font-size:12px;color:#8b949e;margin-left:auto;white-space:nowrap}}
  #tl-table th{{cursor:pointer;user-select:none;white-space:nowrap}}
  #tl-table th:hover{{color:#e6edf3}}
  #tl-table th .sort-icon{{margin-left:4px;opacity:.4;font-style:normal}}
  #tl-table th.sort-asc .sort-icon::after{{content:"▲";opacity:1}}
  #tl-table th.sort-desc .sort-icon::after{{content:"▼";opacity:1}}
  #tl-table th .sort-icon::after{{content:"⇅"}}
  .tl-pagination{{display:flex;gap:6px;align-items:center;margin-top:14px;flex-wrap:wrap}}
  .tl-page-btn{{background:#161b22;border:1px solid #30363d;color:#8b949e;border-radius:6px;
    padding:4px 10px;font-size:12px;cursor:pointer;min-width:32px;text-align:center}}
  .tl-page-btn:hover{{border-color:#26a69a;color:#26a69a}}
  .tl-page-btn.active{{border-color:#26a69a;color:#26a69a;background:rgba(38,166,154,.12);font-weight:600}}
  .tl-page-btn:disabled{{opacity:.35;cursor:default;pointer-events:none}}
  .tl-page-info{{font-size:12px;color:#8b949e;margin:0 6px}}
</style>
</head>
<body>
<div class="container">
{sections}
</div>
</body>
</html>"""

    # ── Base layout template ─────────────────────────────────────────────────

    def _base_layout(self, **overrides) -> dict:
        base = {
            "paper_bgcolor": "#161b22",
            "plot_bgcolor":  "#161b22",
            "font":   {"color": "#e6edf3", "size": 11},
            "margin": {"l": 60, "r": 20, "t": 30, "b": 50},
            "legend": {"bgcolor": "rgba(0,0,0,0)"},
            "xaxis":  {"gridcolor": "#21262d"},
            "yaxis":  {"gridcolor": "#21262d"},
        }
        base.update(overrides)
        return base

    def _chart(self, chart_id: str, traces: list, layout: dict) -> str:
        return (f"<div id='{chart_id}'></div>"
                f"<script>Plotly.newPlot('{chart_id}',{_j(traces)},{_j(layout)},{{responsive:true}});</script>")

    # ── Sections ─────────────────────────────────────────────────────────────

    def _section_header(self, r: "Results") -> str:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        return (f"<h1>{r.strategy_name}</h1>"
                f"<p class='subtitle'>Generated {ts} &nbsp;·&nbsp; {r.n_trades:,} trades"
                f" &nbsp;·&nbsp; {r.n_trading_days:,} trading days"
                f" &nbsp;·&nbsp; Capital ${r.starting_capital:,.0f}</p>")

    def _section_core_metrics(self, r: "Results") -> str:
        ci = r.confidence_intervals
        dd = r.drawdown

        def card(label, value, sub="", cls="neutral"):
            sub_html = f"<div class='metric-sub'>{sub}</div>" if sub else ""
            return (f"<div class='metric-card'>"
                    f"<div class='metric-label'>{label}</div>"
                    f"<div class='metric-value {cls}'>{value}</div>"
                    f"{sub_html}</div>")

        pnl_cls    = "positive" if r.total_net_pnl >= 0 else "negative"
        sharpe_cls = "positive" if r.sharpe >= 1 else ("neutral" if r.sharpe >= 0 else "negative")
        pf_cls     = "positive" if r.profit_factor > 1 else "negative"
        exp_cls    = "positive" if r.avg_trade_pnl_dollars > 0 else "negative"

        cards = [
            card("Net PnL",       _fmt_dollar(r.total_net_pnl),
                 f"CAGR {_fmt_pct(r.cagr)}", pnl_cls),
            card("Sharpe Ratio",  _fmt_num(r.sharpe),
                 f"95% CI [{_fmt_num(ci.sharpe[0])}, {_fmt_num(ci.sharpe[1])}]", sharpe_cls),
            card("Sortino",       _fmt_num(r.sortino), "downside deviation"),
            card("Calmar",        _fmt_num(r.calmar),  "CAGR / max DD"),
            card("Win Rate",      _fmt_pct(r.win_rate),
                 f"95% CI [{_fmt_pct(ci.win_rate[0])}, {_fmt_pct(ci.win_rate[1])}]"),
            card("Profit Factor", _fmt_num(r.profit_factor),
                 f"95% CI [{_fmt_num(ci.profit_factor[0])}, {_fmt_num(ci.profit_factor[1])}]", pf_cls),
            card("Expectancy",    _fmt_dollar(r.avg_trade_pnl_dollars),
                 f"95% CI [{_fmt_dollar(ci.expectancy[0])}, {_fmt_dollar(ci.expectancy[1])}]", exp_cls),
            card("Payoff Ratio",  _fmt_num(r.payoff_ratio), "avg win / avg loss"),
            card("Max DD",        _fmt_pct(dd.max_dd_pct), _fmt_dollar(dd.max_dd_dollars), "negative"),
            card("Avg Win",       _fmt_dollar(r.avg_win_dollars),
                 f"Largest: {_fmt_dollar(r.largest_win)}", "positive"),
            card("Avg Loss",      _fmt_dollar(r.avg_loss_dollars),
                 f"Largest: {_fmt_dollar(r.largest_loss)}", "negative"),
            card("Total Trades",  f"{r.n_trades:,}",
                 f"Avg duration {r.avg_trade_duration_bars:.0f} bars"),
        ]

        return f"<h2>Performance Summary</h2><div class='metric-grid'>{''.join(cards)}</div>"

    def _section_equity_curve(self, r: "Results") -> str:
        eq_x,  eq_y  = _downsample_xy(r.equity_curve, MAX_CURVE_PTS)
        bm_x,  bm_y  = _downsample_xy(r.benchmark.equity_curve, MAX_CURVE_PTS)
        bmc_x, bmc_y = _downsample_xy(r.benchmark.equity_curve_compounding, MAX_CURVE_PTS)

        traces = [
            {"x": eq_x,  "y": eq_y,  "type": "scatter", "mode": "lines",
             "name": r.strategy_name, "line": {"color": "#26a69a", "width": 2}},
            {"x": bmc_x, "y": bmc_y, "type": "scatter", "mode": "lines",
             "name": "B&H Compounding", "line": {"color": "#ff9800", "width": 1.5, "dash": "dash"}},
            {"x": bm_x,  "y": bm_y,  "type": "scatter", "mode": "lines",
             "name": "B&H Fixed (1 contract)", "line": {"color": "#8b949e", "width": 1, "dash": "dot"}},
        ]
        layout = self._base_layout(
            xaxis={"gridcolor": "#21262d", "title": "Bar"},
            yaxis={"gridcolor": "#21262d", "title": "Equity ($)", "tickformat": "$,.0f"},
            hovermode="x unified", margin={"l": 60, "r": 20, "t": 20, "b": 40},
        )
        return (f"<h2>Equity Curve</h2><div class='chart-box'>"
                f"{self._chart('equity_chart', traces, layout)}"
                f"<p class='section-note'>B&H Fixed: 1 contract held entire period. "
                f"B&H Compounding: scales contracts daily as equity grows — "
                f"the truly apples-to-apples comparison against a compounding strategy.</p>"
                f"</div>")

    def _section_drawdown(self, r: "Results") -> str:
        dd_x, dd_y = _downsample_xy(r.drawdown_curve_pct * 100, MAX_CURVE_PTS)
        dd         = r.drawdown
        ci         = r.confidence_intervals

        traces = [{
            "x": dd_x, "y": dd_y, "type": "scatter", "mode": "lines",
            "fill": "tozeroy", "name": "Drawdown %",
            "line": {"color": "#ef5350", "width": 1},
            "fillcolor": "rgba(239,83,80,0.15)",
        }]
        layout = self._base_layout(
            xaxis={"gridcolor": "#21262d", "title": "Bar"},
            yaxis={"gridcolor": "#21262d", "title": "Drawdown (%)", "ticksuffix": "%"},
            hovermode="x unified", margin={"l": 60, "r": 20, "t": 20, "b": 40},
        )

        def dcard(label, value, sub="", cls="negative"):
            sub_html = f"<div class='metric-sub'>{sub}</div>" if sub else ""
            return (f"<div class='metric-card'><div class='metric-label'>{label}</div>"
                    f"<div class='metric-value {cls}'>{value}</div>{sub_html}</div>")

        stats = (f"<div class='metric-grid' style='margin-top:12px'>"
                 + dcard("Max DD $", _fmt_dollar(dd.max_dd_dollars))
                 + dcard("Max DD %", _fmt_pct(dd.max_dd_pct),
                         f"95% CI [{_fmt_pct(ci.max_dd_pct[0])}, {_fmt_pct(ci.max_dd_pct[1])}]")
                 + dcard("Avg DD %", _fmt_pct(dd.avg_dd_pct))
                 + dcard("Max DD Duration",
                         f"{dd.max_dd_duration_bars:,} bars",
                         f"{dd.max_dd_duration_days:.1f} days", "neutral")
                 + "</div>")

        return (f"<h2>Drawdown</h2><div class='chart-box'>"
                f"{self._chart('dd_chart', traces, layout)}{stats}</div>")

    def _section_monte_carlo(self, r: "Results") -> str:
        mc = r.monte_carlo

        def fan_chart(chart_id, pcts, title, p5, p50, p95, stat_label="Final equity"):
            traces = []

            # ── Filled bands: P5–P95 (outer, dark) and P25–P75 (inner, brighter) ──
            # Plotly filled areas use "tonexty" — lower band drawn first, upper fills to it
            def filled_band(p_lo, p_hi, fill_color, name):
                x_lo, y_lo = _downsample_xy(pcts[p_lo], MAX_MC_CURVE_PTS)
                x_hi, y_hi = _downsample_xy(pcts[p_hi], MAX_MC_CURVE_PTS)
                # Lower boundary (no fill, transparent line)
                traces.append({"x": x_lo, "y": y_lo, "type": "scatter", "mode": "lines",
                                "name": f"P{p_lo}", "showlegend": False,
                                "line": {"color": "rgba(0,0,0,0)", "width": 0}})
                # Upper boundary fills down to lower
                traces.append({"x": x_hi, "y": y_hi, "type": "scatter", "mode": "lines",
                                "name": name, "fill": "tonexty",
                                "fillcolor": fill_color,
                                "line": {"color": "rgba(0,0,0,0)", "width": 0}})

            filled_band(5,  95, "rgba(239,83,80,0.15)",   "P5–P95")
            filled_band(25, 75, "rgba(255,152,0,0.20)",   "P25–P75")

            # P50 median line
            x50, y50 = _downsample_xy(pcts[50], MAX_MC_CURVE_PTS)
            traces.append({"x": x50, "y": y50, "type": "scatter", "mode": "lines",
                           "name": "P50 (median)",
                           "line": {"color": "#26a69a", "width": 1.5, "dash": "dash"}})

            # Actual path on top — trade-sampled so x-scale matches MC curves
            actual_eq = np.empty(len(r.trade_pnls) + 1)
            actual_eq[0] = r.starting_capital
            np.cumsum(r.trade_pnls, out=actual_eq[1:])
            actual_eq[1:] += r.starting_capital
            eq_x, eq_y = _downsample_xy(actual_eq, MAX_MC_CURVE_PTS)
            traces.append({"x": eq_x, "y": eq_y, "type": "scatter", "mode": "lines",
                           "name": "Actual", "line": {"color": "#ffffff", "width": 2}})

            layout = self._base_layout(
                xaxis={"gridcolor": "#21262d", "title": "Trade #"},
                yaxis={"gridcolor": "#21262d", "title": "Equity ($)", "tickformat": "$,.0f"},
                title={"text": title, "font": {"size": 13}, "x": 0.5},
                legend={"bgcolor": "rgba(0,0,0,0)", "font": {"size": 10}},
            )
            fmt_val = (lambda v: f"{v:.1f}%") if stat_label == "Max DD" else _fmt_dollar
            stats = (f"<div style='display:flex;gap:24px;margin-top:8px;font-size:12px;color:#8b949e'>"
                     f"<span>P5 {stat_label}: <strong style='color:#ef5350'>{fmt_val(p5)}</strong></span>"
                     f"<span>P50 {stat_label}: <strong style='color:#26a69a'>{fmt_val(p50)}</strong></span>"
                     f"<span>P95 {stat_label}: <strong style='color:#ef5350'>{fmt_val(p95)}</strong></span>"
                     f"</div>")
            return (f"<div class='chart-box'>{self._chart(chart_id, traces, layout)}{stats}</div>")

        def hist_chart(chart_id, arr1, lbl1, col1, arr2, lbl2, col2, title, x_title, fmt=""):
            c1, v1 = _prebinned_histogram(arr1)
            c2, v2 = _prebinned_histogram(arr2)
            x_ax = {"gridcolor": "#21262d", "title": x_title}
            if fmt == "$":   x_ax["tickformat"] = "$,.0f"
            elif fmt == "%": x_ax["ticksuffix"] = "%"
            traces = [
                {"x": c1, "y": v1, "type": "bar", "name": lbl1,
                 "marker": {"color": col1}, "opacity": 0.7},
                {"x": c2, "y": v2, "type": "bar", "name": lbl2,
                 "marker": {"color": col2}, "opacity": 0.7},
            ]
            layout = self._base_layout(
                barmode="overlay", xaxis=x_ax,
                yaxis={"gridcolor": "#21262d", "title": "Count"},
                title={"text": title, "font": {"size": 13}, "x": 0.5},
            )
            return f"<div class='chart-box'>{self._chart(chart_id, traces, layout)}</div>"

        # Shuffle: same trades reordered → final equity always identical, but
        # drawdown path varies. Show max DD percentiles as the meaningful stat.
        shuf_dd_p5  = float(np.percentile(mc.shuffle_max_dd_pct * 100, 5))
        shuf_dd_p50 = float(np.percentile(mc.shuffle_max_dd_pct * 100, 50))
        shuf_dd_p95 = float(np.percentile(mc.shuffle_max_dd_pct * 100, 95))

        def fmt_pct_val(v): return f"{v:.1f}%"

        shuf = fan_chart("mc_shuffle",   mc.shuffle_percentiles,
                         "Sequence Shuffle (10,000 simulations)",
                         shuf_dd_p5, shuf_dd_p50, shuf_dd_p95,
                         stat_label="Max DD")
        boot = fan_chart("mc_bootstrap", mc.bootstrap_percentiles,
                         "Bootstrap Resample (10,000 simulations)",
                         mc.bootstrap_p5, mc.bootstrap_p50, mc.bootstrap_p95,
                         stat_label="Final equity")
        feq  = hist_chart("mc_feq",
                          mc.shuffle_final_equity,    "Shuffle",    "rgba(38,166,154,0.7)",
                          mc.bootstrap_final_equity,  "Bootstrap",  "rgba(239,83,80,0.7)",
                          "Final Equity Distribution", "Final Equity ($)", "$")
        fdd  = hist_chart("mc_fdd",
                          mc.shuffle_max_dd_pct * 100,   "Shuffle",   "rgba(38,166,154,0.7)",
                          mc.bootstrap_max_dd_pct * 100, "Bootstrap", "rgba(239,83,80,0.7)",
                          "Max Drawdown Distribution", "Max Drawdown (%)", "%")

        return (f"<h2>Monte Carlo Analysis</h2>"
                f"<p class='section-note'>Shuffle: reorders actual trades. "
                f"Bootstrap: resamples with replacement. White line = actual path.</p>"
                f"<div class='two-col'>{shuf}{boot}</div>"
                f"<div class='two-col'>{feq}{fdd}</div>")

    def _section_bootstrap(self, r: "Results") -> str:
        pv   = r.bootstrap_pvalue
        ci   = r.confidence_intervals
        good = pv < 0.05
        interp = ("Reject H₀ — statistically significant positive expectancy (p < 0.05)"
                  if good else
                  "Fail to reject H₀ — insufficient evidence of positive expectancy (p ≥ 0.05)")
        rows = [
            ("Sharpe Ratio",  _fmt_num(r.sharpe),                   f"[{_fmt_num(ci.sharpe[0])}, {_fmt_num(ci.sharpe[1])}]"),
            ("Win Rate",      _fmt_pct(r.win_rate),                 f"[{_fmt_pct(ci.win_rate[0])}, {_fmt_pct(ci.win_rate[1])}]"),
            ("Expectancy",    _fmt_dollar(r.avg_trade_pnl_dollars), f"[{_fmt_dollar(ci.expectancy[0])}, {_fmt_dollar(ci.expectancy[1])}]"),
            ("Profit Factor", _fmt_num(r.profit_factor),            f"[{_fmt_num(ci.profit_factor[0])}, {_fmt_num(ci.profit_factor[1])}]"),
            ("Max DD %",      _fmt_pct(r.drawdown.max_dd_pct),      f"[{_fmt_pct(ci.max_dd_pct[0])}, {_fmt_pct(ci.max_dd_pct[1])}]"),
        ]
        trows = "".join(
            f"<tr><td>{m}</td><td>{v}</td><td style='color:#8b949e'>{c}</td></tr>"
            for m, v, c in rows)
        badge = f"<span class='pvalue-badge {'pvalue-good' if good else 'pvalue-bad'}'>p = {pv:.4f}</span>"

        return (f"<h2>Bootstrap Statistics</h2><div class='chart-box'>"
                f"<div style='margin-bottom:16px'>"
                f"<span style='font-size:13px;color:#8b949e;margin-right:12px'>"
                f"H₀: E[trade PnL] ≤ 0 &nbsp;·&nbsp; 10,000 bootstrap resamples</span>"
                f"{badge}</div>"
                f"<p style='font-size:12px;color:#8b949e;margin-bottom:16px'>{interp}</p>"
                f"<table><thead><tr><th>Metric</th><th>Observed</th>"
                f"<th>95% Confidence Interval</th></tr></thead>"
                f"<tbody>{trows}</tbody></table>"
                f"<p class='section-note' style='margin-top:12px'>"
                f"CIs via bootstrap resampling (10,000 samples with replacement).</p></div>")

    def _section_mae_mfe(self, r: "Results") -> str:
        if len(r.mae_per_trade) == 0:
            return ""

        mae  = r.mae_per_trade * 100
        mfe  = r.mfe_per_trade * 100
        pnls = r.trade_pnls
        n    = len(mae)

        # Subsample scatter
        if n > MAX_SCATTER_PTS:
            rng = np.random.default_rng(0)
            idx = np.sort(rng.choice(n, MAX_SCATTER_PTS, replace=False))
            mae_s, mfe_s, pnl_s = mae[idx], mfe[idx], pnls[idx]
        else:
            mae_s, mfe_s, pnl_s = mae, mfe, pnls

        scatter_traces = [{
            "x": _r(mae_s.tolist()), "y": _r(mfe_s.tolist()),
            "mode": "markers", "type": "scatter", "name": "Trade",
            "marker": {"color": ["#26a69a" if p >= 0 else "#ef5350" for p in pnl_s],
                       "size": 4, "opacity": 0.5},
            "text": [_fmt_dollar(p) for p in pnl_s],
            "hovertemplate": "MAE:%{x:.2f}%<br>MFE:%{y:.2f}%<br>%{text}<extra></extra>",
        }]
        scatter_layout = self._base_layout(
            xaxis={"gridcolor": "#21262d", "title": "MAE % (adverse excursion)", "ticksuffix": "%"},
            yaxis={"gridcolor": "#21262d", "title": "MFE % (favorable excursion)", "ticksuffix": "%"},
            title={"text": f"MAE vs MFE (n={min(n, MAX_SCATTER_PTS):,}, green=winner, red=loser)",
                   "font": {"size": 13}, "x": 0.5},
        )

        def hist_chart(chart_id, arr, title, color):
            c, v = _prebinned_histogram(arr)
            traces = [{"x": c, "y": v, "type": "bar",
                       "marker": {"color": color}}]
            layout = self._base_layout(
                xaxis={"gridcolor": "#21262d", "ticksuffix": "%"},
                yaxis={"gridcolor": "#21262d", "title": "Count"},
                showlegend=False,
                title={"text": title, "font": {"size": 13}, "x": 0.5},
            )
            return (f"<div class='chart-box'>{self._chart(chart_id, traces, layout)}</div>")

        stats = (f"<div style='display:flex;gap:32px;margin-top:12px;font-size:13px'>"
                 f"<span>Avg MAE: <strong style='color:#ef5350'>{_fmt_pct(r.avg_mae_pct)}</strong></span>"
                 f"<span>Avg MFE: <strong style='color:#26a69a'>{_fmt_pct(r.avg_mfe_pct)}</strong></span>"
                 f"</div>")

        return (f"<h2>MAE / MFE Analysis</h2>"
                f"<div class='chart-box'>"
                f"{self._chart('mae_mfe_scatter', scatter_traces, scatter_layout)}"
                f"{stats}</div>"
                f"<div class='two-col'>"
                f"{hist_chart('mae_hist', mae, 'MAE Distribution', 'rgba(239,83,80,0.7)')}"
                f"{hist_chart('mfe_hist', mfe, 'MFE Distribution', 'rgba(38,166,154,0.7)')}"
                f"</div>")

    def _section_exit_breakdown(self, r: "Results") -> str:
        if not r.exit_breakdown:
            return ""

        trows = "".join(
            f"<tr><td>{eb.reason}</td><td>{eb.count:,}</td>"
            f"<td style='color:{_color(eb.win_rate-.5)}'>{_fmt_pct(eb.win_rate)}</td>"
            f"<td style='color:{_color(eb.avg_pnl_dollars)}'>{_fmt_dollar(eb.avg_pnl_dollars)}</td>"
            f"<td style='color:{_color(eb.total_pnl_dollars)}'>{_fmt_dollar(eb.total_pnl_dollars)}</td></tr>"
            for eb in r.exit_breakdown)

        reasons = [eb.reason for eb in r.exit_breakdown]
        totals  = [round(eb.total_pnl_dollars, 2) for eb in r.exit_breakdown]
        traces  = [{"x": reasons, "y": totals, "type": "bar",
                    "marker": {"color": [_color(t) for t in totals]}}]
        layout  = self._base_layout(
            xaxis={"gridcolor": "#21262d"},
            yaxis={"gridcolor": "#21262d", "title": "Total PnL ($)", "tickformat": "$,.0f"},
            showlegend=False, margin={"l": 60, "r": 20, "t": 20, "b": 40},
        )

        return (f"<h2>Exit Reason Breakdown</h2><div class='two-col'>"
                f"<div class='chart-box'><table>"
                f"<thead><tr><th>Reason</th><th>Count</th><th>Win Rate</th>"
                f"<th>Avg PnL</th><th>Total PnL</th></tr></thead>"
                f"<tbody>{trows}</tbody></table></div>"
                f"<div class='chart-box'>{self._chart('exit_bar', traces, layout)}</div>"
                f"</div>")

    def _section_hourly(self, r: "Results") -> str:
        if not r.hourly_breakdown:
            return ""

        labels = [f"{h.hour:02d}:00" for h in r.hourly_breakdown]
        avgs   = [round(h.avg_pnl_dollars, 2) for h in r.hourly_breakdown]
        counts = [h.count for h in r.hourly_breakdown]

        traces = [
            {"x": labels, "y": avgs, "type": "bar", "name": "Avg PnL",
             "marker": {"color": [_color(a) for a in avgs]}, "yaxis": "y"},
            {"x": labels, "y": counts, "type": "scatter", "mode": "lines+markers",
             "name": "Trade Count", "line": {"color": "#8b949e", "width": 1},
             "marker": {"size": 4}, "yaxis": "y2"},
        ]
        layout = self._base_layout(
            xaxis={"gridcolor": "#21262d", "title": "Entry Hour (ET)"},
            yaxis={"gridcolor": "#21262d", "title": "Avg PnL ($)", "tickformat": "$,.0f"},
            yaxis2={"title": "Trade Count", "overlaying": "y", "side": "right",
                    "showgrid": False, "color": "#8b949e"},
            margin={"l": 60, "r": 60, "t": 20, "b": 50},
        )

        return (f"<h2>Performance by Hour of Day</h2><div class='chart-box'>"
                f"{self._chart('hourly_chart', traces, layout)}"
                f"<p class='section-note'>Entry hour in Eastern Time.</p></div>")

    def _section_trade_distribution(self, r: "Results") -> str:
        pnl_c, pnl_v = _prebinned_histogram(r.trade_pnls)
        dur_c, dur_v = _prebinned_histogram(r.trade_durations_bars)

        pnl_traces = [{"x": pnl_c, "y": pnl_v, "type": "bar",
                       "marker": {"color": [_color(c) for c in pnl_c]}}]
        pnl_layout = self._base_layout(
            xaxis={"gridcolor": "#21262d", "title": "Net PnL ($)", "tickformat": "$,.0f"},
            yaxis={"gridcolor": "#21262d", "title": "Count"},
            showlegend=False,
            title={"text": "Trade PnL Distribution", "font": {"size": 13}, "x": 0.5},
            shapes=[{"type": "line", "x0": 0, "x1": 0, "y0": 0, "y1": 1,
                     "xref": "x", "yref": "paper",
                     "line": {"color": "#8b949e", "width": 1, "dash": "dot"}}],
        )

        dur_traces = [{"x": dur_c, "y": dur_v, "type": "bar",
                       "marker": {"color": "rgba(139,148,158,0.7)"}}]
        dur_layout = self._base_layout(
            xaxis={"gridcolor": "#21262d", "title": "Duration (bars)"},
            yaxis={"gridcolor": "#21262d", "title": "Count"},
            showlegend=False,
            title={"text": "Trade Duration Distribution", "font": {"size": 13}, "x": 0.5},
        )

        exp_cls = "positive" if r.expectancy_r >= 0 else "negative"
        extras = (
            f"<div class='metric-grid' style='margin-top:0'>"
            f"<div class='metric-card'><div class='metric-label'>Longest Win Streak</div>"
            f"<div class='metric-value positive'>{r.longest_win_streak}</div></div>"
            f"<div class='metric-card'><div class='metric-label'>Longest Loss Streak</div>"
            f"<div class='metric-value negative'>{r.longest_loss_streak}</div></div>"
            f"<div class='metric-card'><div class='metric-label'>Expectancy (R)</div>"
            f"<div class='metric-value {exp_cls}'>{_fmt_num(r.expectancy_r)}R</div></div>"
            f"<div class='metric-card'><div class='metric-label'>Avg Duration</div>"
            f"<div class='metric-value neutral'>{r.avg_trade_duration_bars:.0f} bars</div></div>"
            f"</div>"
        )

        return (f"<h2>Trade Distributions</h2><div class='two-col'>"
                f"<div class='chart-box'>{self._chart('pnl_hist', pnl_traces, pnl_layout)}</div>"
                f"<div class='chart-box'>{self._chart('dur_hist', dur_traces, dur_layout)}</div>"
                f"</div><div class='chart-box'>{extras}</div>")

    def _section_benchmark(self, r: "Results") -> str:
        bm  = r.benchmark
        bm_cal  = (bm.cagr / abs(bm.max_dd_pct)) if bm.max_dd_pct != 0 else 0.0
        bmc_cal = (bm.cagr_compounding / abs(bm.max_dd_pct_compounding)) \
                  if bm.max_dd_pct_compounding != 0 else 0.0
        alpha_fixed = r.cagr - bm.cagr
        alpha_comp  = r.cagr - bm.cagr_compounding

        def row3(label, sval, bval, bcval, better_higher=True, fmt_fn=_fmt_num):
            """3-column row: strategy | fixed B&H | compounding B&H."""
            try:
                sn  = float(sval);  bn = float(bval);  bcn = float(bcval)
                best_bh = max(bn, bcn) if better_higher else min(bn, bcn)
                worst_bh = min(bn, bcn) if better_higher else max(bn, bcn)
                if (sn > best_bh if better_higher else sn < best_bh):
                    cls = "positive"
                elif (sn > worst_bh if better_higher else sn < worst_bh):
                    cls = "neutral"
                else:
                    cls = "negative"
            except Exception:
                cls = "neutral"
            return (f"<tr><td>{label}</td>"
                    f"<td class='{cls}'>{fmt_fn(sval)}</td>"
                    f"<td>{fmt_fn(bval)}</td>"
                    f"<td>{fmt_fn(bcval)}</td></tr>")

        def alpha_row(label, alpha_val):
            cls = "positive" if alpha_val > 0 else "negative"
            return (f"<tr><td>{label}</td>"
                    f"<td class='{cls}'>{_fmt_pct(alpha_val)}</td>"
                    f"<td colspan='2' style='color:#8b949e;font-size:12px'>"
                    f"strategy CAGR − B&amp;H CAGR</td></tr>")

        rows = "\n".join([
            row3("Net PnL",  r.total_net_pnl, bm.total_pnl_dollars,
                 bm.total_pnl_dollars_compounding, fmt_fn=_fmt_dollar),
            row3("CAGR",     r.cagr, bm.cagr,
                 bm.cagr_compounding, fmt_fn=_fmt_pct),
            alpha_row("Alpha vs Fixed",      alpha_fixed),
            alpha_row("Alpha vs Compounding", alpha_comp),
            row3("Sharpe",   r.sharpe, bm.sharpe, bm.sharpe_compounding),
            row3("Max DD %", r.drawdown.max_dd_pct, bm.max_dd_pct,
                 bm.max_dd_pct_compounding, better_higher=False, fmt_fn=_fmt_pct),
            row3("Calmar",   r.calmar, bm_cal, bmc_cal),
        ])

        def bcard(label, val, cls="neutral"):
            return (f"<div class='metric-card'><div class='metric-label'>{label}</div>"
                    f"<div class='metric-value {cls}'>{val}</div></div>")

        a_cls  = "positive" if alpha_fixed > 0 else "negative"
        ac_cls = "positive" if alpha_comp  > 0 else "negative"
        cards = (
            f"<div class='metric-grid' style='margin-bottom:16px'>"
            + bcard("B&H Fixed PnL",       _fmt_dollar(bm.total_pnl_dollars),
                    "positive" if bm.total_pnl_dollars > 0 else "negative")
            + bcard("B&H Fixed CAGR",      _fmt_pct(bm.cagr))
            + bcard("B&H Fixed Max DD",    _fmt_pct(bm.max_dd_pct), "negative")
            + bcard("B&H Compound PnL",    _fmt_dollar(bm.total_pnl_dollars_compounding),
                    "positive" if bm.total_pnl_dollars_compounding > 0 else "negative")
            + bcard("B&H Compound CAGR",   _fmt_pct(bm.cagr_compounding))
            + bcard("B&H Compound Max DD", _fmt_pct(bm.max_dd_pct_compounding), "negative")
            + bcard("Alpha vs Fixed",      _fmt_pct(alpha_fixed),  a_cls)
            + bcard("Alpha vs Compound",   _fmt_pct(alpha_comp),   ac_cls)
            + "</div>"
        )

        return (
            f"<h2>Benchmark Comparison</h2><div class='chart-box'>"
            f"<p class='section-note' style='margin-bottom:16px'>"
            f"<strong>B&amp;H Fixed:</strong> 1 contract, held entire period — same position size as strategy. "
            f"<strong>B&amp;H Compounding:</strong> starts at same capital, buys floor(equity / notional) "
            f"contracts each session open — the fair comparison against a compounding strategy. "
            f"Both benchmarks pay the same entry slippage and commission as the strategy. "
            f"Neither pays exit costs (held to end of data).</p>"
            f"{cards}"
            f"<table><thead><tr>"
            f"<th>Metric</th><th>{r.strategy_name}</th>"
            f"<th>B&amp;H Fixed</th><th>B&amp;H Compounding</th>"
            f"</tr></thead><tbody>{rows}</tbody></table>"
            f"</div>"
        )

    def _section_trade_log(self, r: "Results") -> str:
        if not r.trade_log:
            return "<h2>Trade Log</h2><div class='chart-box'><p class='section-note'>No trades.</p></div>"

        # Collect unique exit reasons for the filter dropdown
        reasons = sorted({t["reason"] for t in r.trade_log})
        reason_opts = "<option value=''>All exits</option>" + "".join(
            f"<option value='{x}'>{x}</option>" for x in reasons
        )

        trades_json = _j(r.trade_log)

        js = f"""
<script>
(function(){{
  var ALL = {trades_json};
  var PAGE = 30;
  var cur  = 1;
  var sortCol = "num", sortDir = 1;
  var filterText = "", filterExit = "", filterResult = "";

  var cols = [
    {{key:"num",   label:"#",           align:"right"}},
    {{key:"entry", label:"Entry Time",   align:"left"}},
    {{key:"exit",  label:"Exit Time",    align:"left"}},
    {{key:"dir",   label:"Dir",          align:"center"}},
    {{key:"qty",   label:"Qty",          align:"right"}},
    {{key:"ep",    label:"Entry Px",     align:"right"}},
    {{key:"xp",    label:"Exit Px",      align:"right"}},
    {{key:"pts",   label:"PnL pts",      align:"right"}},
    {{key:"gross", label:"Gross $",      align:"right"}},
    {{key:"net",   label:"Net $",        align:"right"}},
    {{key:"reason",label:"Exit Reason",  align:"left"}},
    {{key:"dur",   label:"Bars",         align:"right"}},
  ];

  function filtered(){{
    return ALL.filter(function(t){{
      if(filterText){{
        var q = filterText.toLowerCase();
        if(!t.entry.toLowerCase().includes(q) &&
           !t.exit.toLowerCase().includes(q) &&
           !t.reason.toLowerCase().includes(q) &&
           !String(t.ep).includes(q) &&
           !String(t.xp).includes(q)) return false;
      }}
      if(filterExit && t.reason !== filterExit) return false;
      if(filterResult === "win"  && t.net <= 0) return false;
      if(filterResult === "loss" && t.net >= 0) return false;
      return true;
    }});
  }}

  function sorted(rows){{
    return rows.slice().sort(function(a,b){{
      var av = a[sortCol], bv = b[sortCol];
      if(typeof av === "string") av = av.toLowerCase(), bv = bv.toLowerCase();
      return av < bv ? -sortDir : av > bv ? sortDir : 0;
    }});
  }}

  function fmt$(v){{
    var s = (v < 0 ? "-$" : "$") + Math.abs(v).toLocaleString("en-US",{{minimumFractionDigits:2,maximumFractionDigits:2}});
    return s;
  }}
  function fmtPts(v){{
    return (v > 0 ? "+" : "") + v.toFixed(2);
  }}

  function render(){{
    var rows = sorted(filtered());
    var total = rows.length;
    var pages = Math.max(1, Math.ceil(total / PAGE));
    if(cur > pages) cur = pages;

    // Stats bar
    var wins = rows.filter(function(t){{return t.net>0;}}).length;
    var netSum = rows.reduce(function(s,t){{return s+t.net;}},0);
    document.getElementById("tl-stats").textContent =
      total.toLocaleString() + " trades shown  ·  " +
      (total ? (wins/total*100).toFixed(1) : "0.0") + "% win  ·  net " +
      fmt$(netSum);

    // Table body
    var start = (cur-1)*PAGE, end = Math.min(start+PAGE, total);
    var html = "";
    for(var i=start;i<end;i++){{
      var t = rows[i];
      var nc = t.net > 0 ? "positive" : t.net < 0 ? "negative" : "neutral";
      var gc = t.gross > 0 ? "positive" : t.gross < 0 ? "negative" : "neutral";
      var pc = t.pts > 0 ? "positive" : t.pts < 0 ? "negative" : "neutral";
      html += "<tr>" +
        "<td style='text-align:right;color:#8b949e'>" + t.num + "</td>" +
        "<td>" + t.entry + "</td>" +
        "<td>" + t.exit + "</td>" +
        "<td style='text-align:center'>" + t.dir + "</td>" +
        "<td style='text-align:right'>" + t.qty + "</td>" +
        "<td style='text-align:right'>" + t.ep.toFixed(2) + "</td>" +
        "<td style='text-align:right'>" + t.xp.toFixed(2) + "</td>" +
        "<td style='text-align:right' class='" + pc + "'>" + fmtPts(t.pts) + "</td>" +
        "<td style='text-align:right' class='" + gc + "'>" + fmt$(t.gross) + "</td>" +
        "<td style='text-align:right;font-weight:600' class='" + nc + "'>" + fmt$(t.net) + "</td>" +
        "<td>" + t.reason + "</td>" +
        "<td style='text-align:right;color:#8b949e'>" + t.dur + "</td>" +
        "</tr>";
    }}
    document.getElementById("tl-tbody").innerHTML = html || "<tr><td colspan='12' style='text-align:center;color:#8b949e;padding:24px'>No trades match the current filters.</td></tr>";

    // Column headers with sort indicators
    var hdr = "";
    cols.forEach(function(c){{
      var cls = sortCol===c.key ? (sortDir===1?"sort-asc":"sort-desc") : "";
      hdr += "<th class='" + cls + "' data-col='" + c.key + "' style='text-align:" + c.align + "'>" +
             c.label + "<i class='sort-icon'></i></th>";
    }});
    document.getElementById("tl-thead").innerHTML = hdr;

    // Re-bind sort listeners
    document.querySelectorAll("#tl-table th").forEach(function(th){{
      th.onclick = function(){{
        var k = th.dataset.col;
        if(sortCol===k) sortDir *= -1; else {{ sortCol=k; sortDir=1; }}
        cur=1; render();
      }};
    }});

    // Pagination
    var pgHtml = "<button class='tl-page-btn' id='tl-prev' " + (cur<=1?"disabled":"") + ">&lsaquo; Prev</button>";
    var WING = 2;
    var pnums = [];
    for(var p=1;p<=pages;p++){{
      if(p===1||p===pages||Math.abs(p-cur)<=WING) pnums.push(p);
      else if(pnums[pnums.length-1]!=="…") pnums.push("…");
    }}
    pnums.forEach(function(p){{
      if(p==="…"){{ pgHtml += "<span class='tl-page-info'>…</span>"; return; }}
      pgHtml += "<button class='tl-page-btn" + (p===cur?" active":"") + "' data-pg='" + p + "'>" + p + "</button>";
    }});
    pgHtml += "<button class='tl-page-btn' id='tl-next' " + (cur>=pages?"disabled":"") + ">Next &rsaquo;</button>";
    pgHtml += "<span class='tl-page-info'>" + (total?start+1:0) + "–" + end + " of " + total + "</span>";
    document.getElementById("tl-pages").innerHTML = pgHtml;

    document.getElementById("tl-prev") && (document.getElementById("tl-prev").onclick = function(){{if(cur>1){{cur--;render();}}}});
    document.getElementById("tl-next") && (document.getElementById("tl-next").onclick = function(){{if(cur<pages){{cur++;render();}}}});
    document.querySelectorAll("#tl-pages .tl-page-btn[data-pg]").forEach(function(b){{
      b.onclick = function(){{cur=parseInt(b.dataset.pg);render();}};
    }});
  }}

  // Wire controls
  document.getElementById("tl-search").oninput = function(){{filterText=this.value;cur=1;render();}};
  document.getElementById("tl-exit-filter").onchange = function(){{filterExit=this.value;cur=1;render();}};
  document.getElementById("tl-result-win").onclick  = function(){{filterResult=filterResult==="win"?"":"win";this.classList.toggle("active",filterResult==="win");document.getElementById("tl-result-loss").classList.remove("active");cur=1;render();}};
  document.getElementById("tl-result-loss").onclick = function(){{filterResult=filterResult==="loss"?"":"loss";this.classList.toggle("active",filterResult==="loss");document.getElementById("tl-result-win").classList.remove("active");cur=1;render();}};
  document.getElementById("tl-reset").onclick = function(){{
    filterText="";filterExit="";filterResult="";sortCol="num";sortDir=1;cur=1;
    document.getElementById("tl-search").value="";
    document.getElementById("tl-exit-filter").value="";
    document.getElementById("tl-result-win").classList.remove("active");
    document.getElementById("tl-result-loss").classList.remove("active");
    render();
  }};

  render();
}})();
</script>"""

        return f"""
<h2>Trade Log</h2>
<div class='chart-box'>
  <div class='tl-controls'>
    <input id='tl-search' type='text' placeholder='Search date, price, reason…'>
    <div>
      <label>Exit&nbsp;</label>
      <select id='tl-exit-filter'>{reason_opts}</select>
    </div>
    <button class='tl-btn' id='tl-result-win'>Winners only</button>
    <button class='tl-btn' id='tl-result-loss'>Losers only</button>
    <button class='tl-btn' id='tl-reset'>Reset</button>
    <span class='tl-stats' id='tl-stats'></span>
  </div>
  <table id='tl-table'>
    <thead id='tl-thead'></thead>
    <tbody id='tl-tbody'></tbody>
  </table>
  <div class='tl-pagination' id='tl-pages'></div>
</div>
{js}"""