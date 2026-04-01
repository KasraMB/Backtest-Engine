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


def _downsample_with_dates(arr: np.ndarray, timestamps, max_pts: int):
    """
    Like _downsample_xy but also returns ISO date strings at the same indices.
    equity_curve has length len(df_1m) + 1 (equity[0] = starting capital before bar 0).
    timestamps has length len(df_1m).
    We map: bar x index → timestamps[x-1] for x >= 1, timestamps[0] for x == 0.
    Returns (bar_x, date_x, y).
    """
    n = len(arr)
    if n <= max_pts:
        indices = list(range(n))
    else:
        idx = np.linspace(0, n - 1, max_pts, dtype=int)
        idx[-1] = n - 1
        indices = idx.tolist()

    y_vals = _r([float(arr[i]) for i in indices])
    bar_x  = indices

    # Map bar index to timestamp: equity[0] has no bar → use first timestamp
    ts_len = len(timestamps)
    date_x = []
    for i in indices:
        ts_idx = max(0, min(i - 1, ts_len - 1)) if i > 0 else 0
        date_x.append(str(timestamps[ts_idx])[:10])   # "YYYY-MM-DD"

    return bar_x, date_x, y_vals


def _prebinned_histogram(arr: np.ndarray, n_bins: int = N_HIST_BINS):
    """
    Compute histogram in numpy and return (bin_centers, counts).
    Passing pre-binned data as a bar chart is ~100x smaller than raw float arrays.
    Handles zero-variance arrays (e.g. shuffle final equity which is always constant).
    """
    arr = np.asarray(arr, dtype=np.float64)
    if len(arr) == 0:
        return [], []
    lo, hi = float(arr.min()), float(arr.max())
    if hi == lo:
        return [round(lo, 2)], [int(len(arr))]
    # Reduce bins until numpy can create finite-width bins.
    # Crash occurs when (hi-lo)/n_bins underflows to a subnormal or zero,
    # which happens when hi-lo is extremely small (e.g. MC shuffle on a
    # strategy whose equity has nearly identical final values).
    actual_bins = n_bins
    try:
        counts, edges = np.histogram(arr, bins=actual_bins, range=(lo, hi))
    except ValueError:
        # Binary search downward until it works or we hit 1 bin
        while actual_bins > 1:
            actual_bins //= 2
            try:
                counts, edges = np.histogram(arr, bins=actual_bins, range=(lo, hi))
                break
            except ValueError:
                continue
        else:
            return [round((lo + hi) / 2, 2)], [int(len(arr))]
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
        reversed_results: "Results" = None,
        prop_firm_results: dict = None,
        prop_firm_results_rev: dict = None,
        regime_analysis = None,
        run_result = None,
        run_result_rev = None,
        market_data = None,
    ) -> str:
        """
        Render a self-contained HTML tearsheet.

        Args:
            results:               Computed Results from PerformanceEngine.
            output_path:           Path to save the HTML file.
            auto_open:             Open in default browser immediately.
            reversed_results:      Optional Results from reverse_signals run.
            prop_firm_results:     Optional prop firm grid (normal trades).
            prop_firm_results_rev: Optional prop firm grid (reversed trades).
            regime_analysis:       Optional RegimeAnalysisResult.
            run_result:            Optional RunResult — enables trade examples section.
            run_result_rev:        Optional reversed RunResult — enables trade inspector for reversed view.
            market_data:           Optional MarketData — required alongside run_result.
        """
        html = self._build_html(results, reversed_results,
                                prop_firm_results, prop_firm_results_rev,
                                regime_analysis, run_result, run_result_rev, market_data)

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

    def _build_html(self, r: "Results", r_rev: "Results" = None,
                    prop_firm: dict = None, prop_firm_rev: dict = None,
                    regime_analysis=None, run_result=None, run_result_rev=None, market_data=None) -> str:
        # All sections use _UID_ as a placeholder for the view id.
        # We render once with the placeholder then substitute to scope every
        # Plotly div id and JS global, preventing clashes between normal and
        # reversed views that coexist in the same HTML document.
        has_inspector = run_result is not None and market_data is not None

        def _sections(res, div_id, uid, hidden=False, enable_inspector=False):
            style = " style='display:none'" if hidden else ""
            inner = "\n".join([
                self._section_header(res),
                self._section_core_metrics(res),
                self._section_equity_curve(res),
                self._section_drawdown(res),
                self._section_monte_carlo(res),
                self._section_bootstrap(res),
                self._section_mae_mfe(res),
                self._section_exit_breakdown(res),
                self._section_hourly(res),
                self._section_trade_distribution(res),
                self._section_benchmark(res),
                self._section_trade_log(res, uid=uid, enable_inspector=enable_inspector),
            ])
            # Replace every _UID_ placeholder with the actual uid
            inner = inner.replace("_UID_", uid)
            return f"<div id='{div_id}'{style}>{inner}</div>"

        # Prop firm section — includes normal and reversed datasets if available
        if prop_firm:
            prop_firm_html = self._section_prop_firm(prop_firm, prop_firm_rev)
        else:
            prop_firm_html = """
<div id='propfirm_section' style='margin-top:48px'>
<h2 style='font-size:20px;font-weight:600;color:#e6edf3;margin-bottom:4px'>
  ⚡ Prop Firm Analysis — LucidFlex
</h2>
<div class='chart-box' style='color:#8b949e;font-size:13px;padding:24px'>
  Prop firm analysis requires a strategy with a stop loss on ≥80% of trades.<br>
  The current strategy does not have stop losses set — risk per trade is undefined.<br><br>
  Use <strong style='color:#e6edf3'>EnhancedORBStrategy</strong> or add an SL to your strategy via
  <code style='color:#26a69a'>on_fill → position.set_initial_sl_tp(sl, tp)</code>.
</div>
</div>"""

        normal_html   = _sections(r,     "view_normal",   uid="n", enable_inspector=has_inspector)
        reversed_html = _sections(r_rev, "view_reversed", uid="r", hidden=True, enable_inspector=has_inspector) if r_rev else ""

        if r_rev:
            rev_buttons = """
  <span style='font-size:12px;color:#8b949e;font-weight:600;letter-spacing:.05em;text-transform:uppercase'>View:</span>
  <button id='btn_normal' onclick='toggleReverse(false)'
    style='padding:5px 16px;border-radius:4px;border:1px solid #388bfd;
           background:#388bfd;color:#fff;cursor:pointer;font-size:12px;font-weight:600'>Normal</button>
  <button id='btn_reversed' onclick='toggleReverse(true)'
    style='padding:5px 16px;border-radius:4px;border:1px solid #30363d;
           background:#21262d;color:#e6edf3;cursor:pointer;font-size:12px;font-weight:600'>Reversed</button>
  <span style='width:1px;height:18px;background:#30363d'></span>"""
            rev_script = """
<script>
function toggleReverse(rev) {
  document.getElementById('view_normal').style.display   = rev ? 'none' : '';
  document.getElementById('view_reversed').style.display = rev ? ''     : 'none';
  var bn = document.getElementById('btn_normal');
  var br = document.getElementById('btn_reversed');
  bn.style.background  = rev ? '#21262d' : '#388bfd';
  bn.style.borderColor = rev ? '#30363d' : '#388bfd';
  bn.style.color       = rev ? '#e6edf3' : '#fff';
  br.style.background  = rev ? '#388bfd' : '#21262d';
  br.style.borderColor = rev ? '#388bfd' : '#30363d';
  br.style.color       = rev ? '#fff'    : '#e6edf3';
  if (window.pfSetTrades) window.pfSetTrades(rev ? 'reversed' : 'normal');
  if (window.tcSetReversed) window.tcSetReversed(rev);
  setTimeout(function() { window.dispatchEvent(new Event('resize')); }, 50);
}
</script>"""
        else:
            rev_buttons = ""
            rev_script  = ""

        toggle_btn = f"""
<div id='sticky-bar' style='position:sticky;top:0;z-index:999;background:#0d1117;
     border-bottom:1px solid #30363d;padding:8px 24px;
     display:flex;align-items:center;gap:10px;flex-wrap:wrap'>
{rev_buttons}
  <button onclick='secCollapseAll(true)'
    style='padding:4px 12px;border-radius:4px;border:1px solid #30363d;
           background:#21262d;color:#8b949e;cursor:pointer;font-size:12px'>
    ⊟ Collapse All</button>
  <button onclick='secCollapseAll(false)'
    style='padding:4px 12px;border-radius:4px;border:1px solid #30363d;
           background:#21262d;color:#8b949e;cursor:pointer;font-size:12px'>
    ⊞ Expand All</button>
</div>
{rev_script}"""

        regime_html = self._section_regime(regime_analysis) if regime_analysis else ""

        if has_inspector:
            windows_json, n_total = self._serialize_trade_windows(run_result, market_data)
            windows_json_rev = None
            if run_result_rev is not None:
                windows_json_rev, _ = self._serialize_trade_windows(run_result_rev, market_data)
            trade_inspector_html = self._section_trade_inspector(
                windows_json, n_total, windows_json_rev=windows_json_rev
            )
        else:
            trade_inspector_html = ""

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
  .tl-table th{{cursor:pointer;user-select:none;white-space:nowrap}}
  .tl-table th:hover{{color:#e6edf3}}
  .tl-table th .sort-icon{{margin-left:4px;opacity:.4;font-style:normal}}
  .tl-table th.sort-asc .sort-icon::after{{content:"▲";opacity:1}}
  .tl-table th.sort-desc .sort-icon::after{{content:"▼";opacity:1}}
  .tl-table th .sort-icon::after{{content:"⇅"}}
  .tl-pagination{{display:flex;gap:6px;align-items:center;margin-top:14px;flex-wrap:wrap}}
  .tl-page-btn{{background:#161b22;border:1px solid #30363d;color:#8b949e;border-radius:6px;
    padding:4px 10px;font-size:12px;cursor:pointer;min-width:32px;text-align:center}}
  .tl-page-btn:hover{{border-color:#26a69a;color:#26a69a}}
  .tl-page-btn.active{{border-color:#26a69a;color:#26a69a;background:rgba(38,166,154,.12);font-weight:600}}
  .tl-page-btn:disabled{{opacity:.35;cursor:default;pointer-events:none}}
  .tl-page-info{{font-size:12px;color:#8b949e;margin:0 6px}}
  /* Collapsible sections */
  .sec-chev{{font-size:11px;color:#8b949e;margin-left:8px;display:inline-block;
             vertical-align:middle;transition:transform .22s;flex-shrink:0}}
  h2{{cursor:pointer;user-select:none;display:flex;align-items:center;justify-content:space-between}}
  h2:hover .sec-chev{{color:#e6edf3}}
  .sec-wrap{{overflow:hidden;transition:opacity .15s}}
</style>
</head>
<body>
{toggle_btn}
<div class="container">
{normal_html}
{reversed_html}
{prop_firm_html}
{regime_html}
{trade_inspector_html}
</div>
<script>
(function() {{
  // ── Collapsible sections ────────────────────────────────────────────────────
  // For each container, find direct-child h2 elements and wrap everything
  // between consecutive h2s in a .sec-wrap div that can be toggled.
  function makeCollapsible(container) {{
    if (!container) return;
    var children = Array.from(container.children);
    var groups = [];
    var cur = null;
    children.forEach(function(el) {{
      if (el.tagName === 'H2') {{
        cur = {{h: el, els: []}};
        groups.push(cur);
      }} else if (cur) {{
        cur.els.push(el);
      }}
    }});

    groups.forEach(function(g) {{
      if (!g.els.length) return;

      // Move content into a wrapper div
      var wrap = document.createElement('div');
      wrap.className = 'sec-wrap';
      g.els.forEach(function(el) {{ wrap.appendChild(el); }});
      g.h.insertAdjacentElement('afterend', wrap);

      // Chevron indicator
      var chev = document.createElement('span');
      chev.className = 'sec-chev';
      chev.innerHTML = '&#9650;';
      g.h.appendChild(chev);

      var open = true;
      g.h.addEventListener('click', function() {{
        open = !open;
        wrap.style.display = open ? '' : 'none';
        chev.style.transform = open ? '' : 'rotate(180deg)';
        if (open) {{
          // Nudge Plotly to resize charts that reappeared
          setTimeout(function() {{ window.dispatchEvent(new Event('resize')); }}, 60);
        }}
      }});
    }});
  }}

  // Apply to every section container
  ['view_normal', 'view_reversed',
   'propfirm_section', 'regime_section', 'trade-inspector'
  ].forEach(function(id) {{ makeCollapsible(document.getElementById(id)); }});

  // ── Collapse / Expand All ───────────────────────────────────────────────────
  window.secCollapseAll = function(collapse) {{
    document.querySelectorAll('.sec-wrap').forEach(function(wrap) {{
      var h2 = wrap.previousElementSibling;
      var chev = h2 && h2.querySelector('.sec-chev');
      if (wrap.style.display === 'none' && collapse) return; // already collapsed
      wrap.style.display = collapse ? 'none' : '';
      if (chev) chev.style.transform = collapse ? 'rotate(180deg)' : '';
    }});
    if (!collapse) setTimeout(function() {{ window.dispatchEvent(new Event('resize')); }}, 60);
  }};
}})();
</script>
</body>
</html>"""

    # ── Trade Inspector ───────────────────────────────────────────────────────

    def _serialize_trade_windows(self, run_result, market_data) -> tuple:
        """
        Serialize OHLC windows for every trade into a compact JSON string.
        Returns (windows_json, n_total) where n_total is the full trade count
        and windows_json covers the first MAX_TRADES trades.
        """
        _PAD      = 25
        MAX_TRADES = 500
        df        = market_data.df_1m
        n_bars    = len(df)
        trades    = run_result.trades
        n_total   = len(trades)
        subset    = trades[:MAX_TRADES]

        windows = []
        for idx, trade in enumerate(subset):
            start = max(0, trade.entry_bar - _PAD)
            end   = min(n_bars - 1, trade.exit_bar + _PAD)
            sl    = df.iloc[start : end + 1]

            windows.append({
                "t":   [ts.strftime("%Y-%m-%d %H:%M") for ts in sl.index],
                "o":   _r(sl["open"].tolist()),
                "h":   _r(sl["high"].tolist()),
                "l":   _r(sl["low"].tolist()),
                "c":   _r(sl["close"].tolist()),
                "ei":  trade.entry_bar - start,
                "xi":  trade.exit_bar  - start,
                "entry":  float(round(trade.entry_price, 2)),
                "exit":   float(round(trade.exit_price, 2)),
                "sl":     float(round(trade.sl_price, 2)) if trade.sl_price is not None else None,
                "tp":     float(round(trade.tp_price, 2)) if trade.tp_price is not None else None,
                "isl":    float(round(trade.initial_sl_price, 2)) if trade.initial_sl_price is not None else None,
                "itp":    float(round(trade.initial_tp_price, 2)) if trade.initial_tp_price is not None else None,
                "dir":    int(trade.direction),
                "reason": trade.exit_reason.name,
                "setup":  trade.trade_reason,
                "pnl":    float(round(trade.net_pnl_dollars, 2)),
                "win":    bool(trade.is_winner),
                "num":    idx + 1,
                "qty":    int(trade.contracts),
                "trail":  bool(trade.had_trailing),
                "oi":     (trade.order_placed_bar - start) if trade.order_placed_bar is not None else None,
                "fibs":   [{"p": float(round(f["p"], 2)), "t": f["t"], "d": int(f["d"]),
                             "v": float(f["v"])} for f in trade.fib_levels],
            })

        return _j(windows), n_total

    def _section_trade_inspector(self, windows_json: str, n_total: int,
                                   windows_json_rev: str = None) -> str:
        """
        Interactive trade inspector panel.
        Click a row in the Trade Log to load that trade's candlestick chart.
        Prev/Next buttons and arrow keys for sequential browsing.
        """
        n_shown = n_total if n_total <= 500 else 500
        truncation_note = (
            f"  · first {n_shown} of {n_total} shown"
            if n_total > 500 else ""
        )
        _windows_rev_js = windows_json_rev if windows_json_rev is not None else "null"

        return f"""
<div id='trade-inspector' style='margin-top:48px'>
  <h2 style='font-size:20px;font-weight:600;color:#e6edf3;margin-bottom:4px'>
    🔍 Trade Inspector
  </h2>
  <p style='color:#8b949e;font-size:13px;margin-bottom:16px'>
    Click any row in the Trade Log to inspect it here.
    Use the arrows or ← → keys to browse trades sequentially.{truncation_note}
  </p>
  <div class='chart-box'>
    <div style='display:flex;align-items:center;gap:10px;margin-bottom:12px;flex-wrap:wrap'>
      <button id='tc-prev'
        style='padding:5px 14px;border-radius:4px;border:1px solid #30363d;
               background:#21262d;color:#8b949e;cursor:pointer;font-size:13px'
        disabled>&#8592; Prev</button>
      <button id='tc-next'
        style='padding:5px 14px;border-radius:4px;border:1px solid #30363d;
               background:#21262d;color:#8b949e;cursor:pointer;font-size:13px'
        disabled>Next &#8594;</button>
      <span id='tc-counter'
        style='font-size:12px;color:#8b949e;min-width:80px'>
        — / {n_shown}
      </span>
      <span id='tc-info' style='font-size:12px;color:#8b949e'>
        &#8593; Click any trade row above to inspect it.
      </span>
    </div>
    <div id='tc-chart' style='height:520px'></div>
  </div>
</div>

<script>
(function() {{
  var TC_WINDOWS   = {windows_json};
  var TC_WINDOWS_R = {_windows_rev_js};
  var TC_IS_REV    = false;
  var TC_CUR = -1;

  function tcWin() {{ return (TC_IS_REV && TC_WINDOWS_R) ? TC_WINDOWS_R : TC_WINDOWS; }}

  function tcInspect(idx) {{
    var wins = tcWin();
    if (idx < 0 || idx >= wins.length) return;
    TC_CUR = idx;
    var w = wins[idx];

    var teal = '#26a69a', red = '#ef5350';
    var isWin  = w.win;
    var isLong = w.dir === 1;
    var oc     = isWin ? teal : red;
    var x0     = w.t[0];
    var x1     = w.t[w.t.length - 1];
    var ets    = w.t[w.ei];
    var xts    = w.t[w.xi];

    // ── Shapes ────────────────────────────────────────────────────────────────
    var shapes = [];

    // Initial SL zone (entry ↔ initial SL) — shows original risk
    var riskSl = w.isl !== null ? w.isl : w.sl;   // use isl if available, else fall back to sl
    if (riskSl !== null) {{
      shapes.push({{
        type: 'rect', xref: 'x', yref: 'y',
        x0: ets, x1: xts,
        y0: Math.min(w.entry, riskSl), y1: Math.max(w.entry, riskSl),
        fillcolor: 'rgba(239,83,80,0.08)', line: {{width: 0}}
      }});
    }}

    // Outcome zone (entry ↔ exit price)
    shapes.push({{
      type: 'rect', xref: 'x', yref: 'y',
      x0: ets, x1: xts,
      y0: Math.min(w.entry, w.exit), y1: Math.max(w.entry, w.exit),
      fillcolor: isWin ? 'rgba(38,166,154,0.15)' : 'rgba(239,83,80,0.15)',
      line: {{width: 0}}
    }});

    // Entry line
    shapes.push({{
      type: 'line', xref: 'x', yref: 'y',
      x0: x0, x1: x1, y0: w.entry, y1: w.entry,
      line: {{color: 'rgba(200,200,200,0.6)', width: 1, dash: 'dot'}}
    }});

    // Initial SL line (dashed, lighter — shows original stop)
    if (w.isl !== null) {{
      shapes.push({{
        type: 'line', xref: 'x', yref: 'y',
        x0: x0, x1: x1, y0: w.isl, y1: w.isl,
        line: {{color: 'rgba(239,83,80,0.50)', width: 1, dash: 'dash'}}
      }});
    }} else if (w.sl !== null) {{
      // Fallback: show sl line for trades without initial SL tracking
      shapes.push({{
        type: 'line', xref: 'x', yref: 'y',
        x0: x0, x1: x1, y0: w.sl, y1: w.sl,
        line: {{color: 'rgba(239,83,80,0.75)', width: 1, dash: 'dot'}}
      }});
    }}

    // Initial TP line (dashed, green — shows original target)
    if (w.itp !== null) {{
      shapes.push({{
        type: 'line', xref: 'x', yref: 'y',
        x0: x0, x1: x1, y0: w.itp, y1: w.itp,
        line: {{color: 'rgba(38,166,154,0.50)', width: 1, dash: 'dash'}}
      }});
    }}

    // Final SL line — only for non-trailing trades where the fixed SL was manually moved
    var slChanged = w.sl !== null && w.isl !== null && !w.trail && Math.abs(w.sl - w.isl) > 0.01;
    if (w.sl !== null && slChanged) {{
      shapes.push({{
        type: 'line', xref: 'x', yref: 'y',
        x0: x0, x1: x1, y0: w.sl, y1: w.sl,
        line: {{color: 'rgba(239,83,80,0.85)', width: 1, dash: 'dot'}}
      }});
    }}

    // Exit line
    shapes.push({{
      type: 'line', xref: 'x', yref: 'y',
      x0: x0, x1: x1, y0: w.exit, y1: w.exit,
      line: {{
        color: isWin ? 'rgba(38,166,154,0.70)' : 'rgba(239,83,80,0.70)',
        width: 1, dash: 'dot'
      }}
    }});

    // Fib levels (OTE=gold, STDV=orange, SESSION_OTE=cyan)
    if (w.fibs && w.fibs.length > 0) {{
      var fibColors = {{ 'OTE': 'rgba(56,139,253,0.65)', 'STDV': 'rgba(38,166,154,0.65)', 'SESSION_OTE': 'rgba(38,166,154,0.65)' }};
      w.fibs.forEach(function(fib) {{
        var fc = fibColors[fib.t] || 'rgba(180,180,180,0.45)';
        shapes.push({{
          type: 'line', xref: 'x', yref: 'y',
          x0: x0, x1: x1, y0: fib.p, y1: fib.p,
          line: {{ color: fc, width: 1, dash: 'dashdot' }}
        }});
      }});
    }}

    // ── Annotations ───────────────────────────────────────────────────────────
    var absPnl = Math.abs(w.pnl).toLocaleString('en-US', {{maximumFractionDigits: 0}});
    var pnlStr = (w.pnl >= 0 ? '+$' : '-$') + absPnl;

    var anns = [
      {{
        xref: 'x', yref: 'y', x: x1, y: w.entry,
        text: ' Entry  ' + w.entry.toFixed(2),
        font: {{size: 11, color: 'rgba(210,210,210,0.9)'}},
        showarrow: false, xanchor: 'left',
        bgcolor: 'rgba(22,27,34,0.85)'
      }},
      {{
        xref: 'x', yref: 'y', x: x1, y: w.exit,
        text: ' Exit  ' + w.exit.toFixed(2) + '  (' + pnlStr + ')',
        font: {{size: 11, color: oc}},
        showarrow: false, xanchor: 'left',
        bgcolor: 'rgba(22,27,34,0.85)'
      }}
    ];
    if (w.isl !== null) {{
      anns.push({{
        xref: 'x', yref: 'y', x: x1, y: w.isl,
        text: ' Init SL  ' + w.isl.toFixed(2),
        font: {{size: 11, color: 'rgba(239,83,80,0.70)'}},
        showarrow: false, xanchor: 'left',
        bgcolor: 'rgba(22,27,34,0.85)'
      }});
    }} else if (w.sl !== null) {{
      // Fallback for trades without initial SL tracking
      anns.push({{
        xref: 'x', yref: 'y', x: x1, y: w.sl,
        text: ' SL  ' + w.sl.toFixed(2),
        font: {{size: 11, color: '#ef5350'}},
        showarrow: false, xanchor: 'left',
        bgcolor: 'rgba(22,27,34,0.85)'
      }});
    }}
    if (w.itp !== null) {{
      anns.push({{
        xref: 'x', yref: 'y', x: x1, y: w.itp,
        text: ' Init TP  ' + w.itp.toFixed(2),
        font: {{size: 11, color: 'rgba(38,166,154,0.70)'}},
        showarrow: false, xanchor: 'left',
        bgcolor: 'rgba(22,27,34,0.85)'
      }});
    }}
    if (w.sl !== null && slChanged) {{
      anns.push({{
        xref: 'x', yref: 'y', x: x1, y: w.sl,
        text: ' Final SL  ' + w.sl.toFixed(2),
        font: {{size: 11, color: '#ef5350'}},
        showarrow: false, xanchor: 'left',
        bgcolor: 'rgba(22,27,34,0.85)'
      }});
    }}

    // ── Traces ────────────────────────────────────────────────────────────────
    var candle = {{
      type: 'candlestick',
      x: w.t, open: w.o, high: w.h, low: w.l, close: w.c,
      increasing: {{line: {{color: teal}}, fillcolor: teal}},
      decreasing: {{line: {{color: red}},  fillcolor: red}},
      name: '', showlegend: false, hoverinfo: 'x+y',
      xaxis: 'x', yaxis: 'y'
    }};

    var entryMark = {{
      type: 'scatter', mode: 'markers',
      x: [ets], y: [w.entry],
      marker: {{
        symbol: isLong ? 'triangle-up' : 'triangle-down',
        size: 16, color: '#ffffff',
        line: {{color: '#aaaaaa', width: 1}}
      }},
      showlegend: false,
      hovertemplate: 'Entry: ' + w.entry.toFixed(2) + '<extra></extra>'
    }};

    var exitMark = {{
      type: 'scatter', mode: 'markers',
      x: [xts], y: [w.exit],
      marker: {{symbol: 'circle', size: 12, color: oc, line: {{color: oc, width: 2}}}},
      showlegend: false,
      hovertemplate: 'Exit: ' + w.exit.toFixed(2) + '  ' + pnlStr + '<extra></extra>'
    }};

    // Order-placed marker (AFZ limit placed) — only shown when different from fill bar
    var traces = [candle, entryMark, exitMark];
    if (w.oi !== null && w.oi !== w.ei) {{
      var ots = w.t[w.oi];
      var orderMark = {{
        type: 'scatter', mode: 'markers',
        x: [ots], y: [isLong ? w.l[w.oi] : w.h[w.oi]],
        marker: {{
          symbol: 'diamond', size: 10,
          color: 'rgba(255,200,0,0.9)',
          line: {{color: '#ffcc00', width: 1}}
        }},
        showlegend: false,
        hovertemplate: 'Order placed: ' + ots + '<extra></extra>'
      }};
      traces = [candle, orderMark, entryMark, exitMark];
    }}

    // ── Layout ────────────────────────────────────────────────────────────────
    var layout = {{
      paper_bgcolor: '#161b22', plot_bgcolor: '#161b22',
      font: {{color: '#d0d0d0', family: 'monospace', size: 10}},
      margin: {{l: 10, r: 155, t: 14, b: 40}},
      xaxis: {{
        type: 'date', rangeslider: {{visible: false}},
        showgrid: true, gridcolor: 'rgba(255,255,255,0.05)',
        tickfont: {{size: 9}}
      }},
      yaxis: {{
        showgrid: true, gridcolor: 'rgba(255,255,255,0.05)',
        tickfont: {{size: 9}}, side: 'right', tickformat: '.2f'
      }},
      shapes: shapes, annotations: anns
    }};

    Plotly.react('tc-chart', traces, layout,
                 {{displayModeBar: false, responsive: true}});

    // ── UI updates ────────────────────────────────────────────────────────────
    var dir    = isLong ? 'Long' : 'Short';
    var icon   = isWin  ? '&#9650;' : '&#9660;';
    var icol   = isWin  ? teal : red;
    document.getElementById('tc-counter').textContent =
      'Trade #' + w.num + ' / ' + tcWin().length;
    document.getElementById('tc-info').innerHTML =
      '<span style="color:' + icol + ';font-weight:600">' + icon + ' ' + (isWin ? 'Win' : 'Loss') + '</span>' +
      ' &nbsp;·&nbsp; ' + dir +
      ' &nbsp;·&nbsp; ' + w.reason +
      ' &nbsp;·&nbsp; <span style="color:' + icol + '">' + pnlStr + '</span>' +
      ' &nbsp;·&nbsp; ' + w.qty + ' contract' + (w.qty !== 1 ? 's' : '') +
      ' &nbsp;·&nbsp; ' + ets +
      (w.setup ? '<br><span style="color:#8b949e;font-size:11px;font-family:monospace">' + w.setup + '</span>' : '');

    var prevBtn = document.getElementById('tc-prev');
    var nextBtn = document.getElementById('tc-next');
    var enabledStyle  = 'border:1px solid #388bfd;background:rgba(56,139,253,.12);color:#388bfd;';
    var disabledStyle = 'border:1px solid #30363d;background:#21262d;color:#8b949e;';
    prevBtn.disabled = idx <= 0;
    nextBtn.disabled = idx >= tcWin().length - 1;
    prevBtn.style.cssText = 'padding:5px 14px;border-radius:4px;font-size:13px;cursor:pointer;' +
                            (idx <= 0 ? disabledStyle : enabledStyle);
    nextBtn.style.cssText = 'padding:5px 14px;border-radius:4px;font-size:13px;cursor:pointer;' +
                            (idx >= tcWin().length - 1 ? disabledStyle : enabledStyle);

    // Highlight selected row
    document.querySelectorAll('.tc-row-sel').forEach(function(el) {{
      el.style.background = '';
      el.classList.remove('tc-row-sel');
    }});
    var tr = document.querySelector('[data-trade-idx="' + idx + '"]');
    if (tr) {{
      tr.style.background = 'rgba(56,139,253,0.10)';
      tr.classList.add('tc-row-sel');
    }}

    // Scroll inspector into view
    document.getElementById('trade-inspector').scrollIntoView(
      {{behavior: 'smooth', block: 'start'}}
    );
  }}

  // Expose globally
  window.tcInspect    = tcInspect;
  window.tcPrev       = function() {{ if (TC_CUR > 0) tcInspect(TC_CUR - 1); }};
  window.tcNext       = function() {{ if (TC_CUR < tcWin().length - 1) tcInspect(TC_CUR + 1); }};
  window.tcSetReversed = function(rev) {{ TC_IS_REV = rev; TC_CUR = -1; }};

  document.getElementById('tc-prev').onclick = function() {{ window.tcPrev(); }};
  document.getElementById('tc-next').onclick = function() {{ window.tcNext(); }};

  // Arrow key navigation (only when inspector has a trade loaded and focus is not in an input)
  document.addEventListener('keydown', function(e) {{
    if (TC_CUR < 0) return;
    var tag = document.activeElement && document.activeElement.tagName;
    if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return;
    if (e.key === 'ArrowLeft')  {{ e.preventDefault(); window.tcPrev(); }}
    if (e.key === 'ArrowRight') {{ e.preventDefault(); window.tcNext(); }}
  }});
}})();
</script>"""

    # ── Trade Examples Section ────────────────────────────────────────────────

    def _section_trade_examples(self, run_result, market_data) -> str:
        """
        TradingView-style candlestick chart showing one winning trade and one
        losing trade with position tool overlay (entry, SL, exit lines/zones).
        Embedded inline so the tearsheet stays self-contained.
        """
        from backtest.visualization.trade_chart import _pick_trades, _get_slice, _add_overlay
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go

        winner, loser = _pick_trades(run_result.trades)
        pairs = [(t, lbl) for t, lbl in [(winner, "Winner"), (loser, "Loser")] if t is not None]
        if not pairs:
            return ""

        def _subtitle(trade, label):
            direction = "Long" if trade.direction == 1 else "Short"
            sign = "+" if trade.net_pnl_dollars >= 0 else ""
            ts = market_data.df_1m.index[trade.entry_bar].strftime("%Y-%m-%d  %H:%M")
            icon = "▲" if trade.is_winner else "▼"
            return (f"{icon} {label}  ·  {direction}  ·  "
                    f"{sign}${trade.net_pnl_dollars:,.0f}  ·  "
                    f"{trade.exit_reason.name}  ·  entry {ts}")

        n   = len(pairs)
        fig = make_subplots(
            rows=1, cols=n,
            shared_xaxes=False,
            horizontal_spacing=0.10,
            subplot_titles=[_subtitle(t, lbl) for t, lbl in pairs],
        )

        for col_idx, (trade, lbl) in enumerate(pairs, start=1):
            df_slice, offset = _get_slice(market_data, trade)
            fig.add_trace(go.Candlestick(
                x=df_slice.index,
                open=df_slice["open"], high=df_slice["high"],
                low=df_slice["low"],  close=df_slice["close"],
                increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
                increasing_fillcolor="#26a69a",  decreasing_fillcolor="#ef5350",
                name="", showlegend=False, hoverinfo="x+y",
            ), row=1, col=col_idx)
            _add_overlay(fig, 1, col_idx, df_slice, offset, trade)

        for i in range(1, n + 1):
            s = "" if i == 1 else str(i)
            fig.update_layout(**{
                f"xaxis{s}": dict(type="date", rangeslider=dict(visible=False),
                                  showgrid=True, gridcolor="rgba(255,255,255,0.05)",
                                  tickfont=dict(size=9)),
                f"yaxis{s}": dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)",
                                  tickfont=dict(size=9), side="right", tickformat=".2f"),
            })

        fig.update_layout(
            paper_bgcolor="#161b22",
            plot_bgcolor="#161b22",
            font=dict(color="#d0d0d0", family="monospace"),
            height=560,
            margin=dict(l=10, r=160, t=70, b=40),
        )
        for ann in fig.layout.annotations:
            ann.font = dict(size=12, color="#8b949e")

        chart_html = fig.to_html(include_plotlyjs=False, full_html=False,
                                 config={"displayModeBar": False})

        return f"""
<div id='trade_examples_section' style='margin-top:48px'>
<h2 style='font-size:20px;font-weight:600;color:#e6edf3;margin-bottom:4px'>
  📊 Trade Examples
</h2>
<p style='color:#8b949e;font-size:13px;margin-bottom:16px'>
  One representative winning trade and one losing trade.
  Shows entry, stop-loss, and exit with the TradingView position tool overlay.
</p>
<div class='chart-box'>
{chart_html}
</div>
</div>"""

    # ── Regime Analysis Section ──────────────────────────────────────────────

    def _section_regime(self, ra) -> str:
        """
        Render regime analysis section:
          - Per-regime stats table (win rate, Sharpe, expectancy, n_trades)
          - IS vs OOS Sharpe comparison (filtered vs unfiltered)
          - Permutation test p-value with histogram
          - Regime stability: avg duration + transition matrix heatmap
        """
        import json as _json

        lnames = ra.allowed_regimes or ["bear", "neutral", "bull"]
        colors = {"bear": "#ef5350", "neutral": "#8b949e", "bull": "#26a69a"}
        sig    = ra.perm_pvalue < 0.05
        sig_color = "#26a69a" if sig else "#ef5350"
        sig_text  = "Statistically Significant (p < 0.05)" if sig \
                    else "Not Statistically Significant (p ≥ 0.05)"

        # ── Per-regime stats table ────────────────────────────────────────────
        def fmt_pct(v): return f"{v*100:.1f}%"
        def fmt_sharpe(v): return f"{v:.2f}"
        def fmt_dol(v):
            s = f"${abs(v):,.0f}"
            return f"+{s}" if v >= 0 else f"-{s}"

        rows = ""
        for s in ra.breakdown:
            c = colors.get(s.name, "#8b949e")
            sharpe_c = "#26a69a" if s.sharpe > 0 else "#ef5350"
            rows += f"""
            <tr>
              <td><span style='color:{c};font-weight:700;text-transform:capitalize'>
                {s.name}</span></td>
              <td style='text-align:right'>{s.n_trades}</td>
              <td style='text-align:right'>{fmt_pct(s.win_rate)}</td>
              <td style='text-align:right;color:{"#26a69a" if s.avg_pnl>=0 else "#ef5350"}'>
                {fmt_dol(s.avg_pnl)}</td>
              <td style='text-align:right;color:{sharpe_c}'>{fmt_sharpe(s.sharpe)}</td>
              <td style='text-align:right;color:{"#26a69a" if s.total_pnl>=0 else "#ef5350"}'>
                {fmt_dol(s.total_pnl)}</td>
            </tr>"""

        table_html = f"""
        <table style='width:100%;border-collapse:collapse;font-size:13px'>
          <thead>
            <tr style='border-bottom:1px solid #30363d;color:#8b949e;font-size:11px;
                       text-transform:uppercase;letter-spacing:.05em'>
              <th style='text-align:left;padding:8px 4px'>Regime</th>
              <th style='text-align:right;padding:8px 4px'>Trades</th>
              <th style='text-align:right;padding:8px 4px'>Win Rate</th>
              <th style='text-align:right;padding:8px 4px'>Avg PnL</th>
              <th style='text-align:right;padding:8px 4px'>Sharpe</th>
              <th style='text-align:right;padding:8px 4px'>Total PnL</th>
            </tr>
          </thead>
          <tbody>{rows}</tbody>
        </table>"""

        # ── IS vs OOS comparison cards ────────────────────────────────────────
        filter_label = f"Filtered ({', '.join(ra.allowed_regimes)})" \
                       if ra.allowed_regimes else "No Filter"
        train_end_str = str(ra.train_end_date) if ra.train_end_date else "N/A"

        def sharpe_card(label, val):
            c = "#26a69a" if val > 0 else "#ef5350"
            return f"""<div style='background:#161b22;border:1px solid #21262d;
                           border-radius:6px;padding:12px;flex:1;min-width:120px'>
              <div style='font-size:10px;color:#8b949e;margin-bottom:4px;
                          text-transform:uppercase;letter-spacing:.05em'>{label}</div>
              <div style='font-size:20px;font-weight:700;color:{c}'>{val:.2f}</div>
            </div>"""

        is_oos_html = f"""
        <div style='margin-bottom:6px;font-size:11px;color:#8b949e'>
          Train / IS period ends: <strong style='color:#e6edf3'>{train_end_str}</strong>
          &nbsp;·&nbsp; Filter: <strong style='color:#e6edf3'>{filter_label}</strong>
        </div>
        <div style='display:flex;gap:10px;flex-wrap:wrap;margin-bottom:20px'>
          {sharpe_card("IS Unfiltered Sharpe",  ra.is_unfiltered_sharpe)}
          {sharpe_card("IS Filtered Sharpe",    ra.is_filtered_sharpe)}
          {sharpe_card("OOS Unfiltered Sharpe", ra.oos_unfiltered_sharpe)}
          {sharpe_card("OOS Filtered Sharpe",   ra.oos_filtered_sharpe)}
        </div>"""

        # ── Permutation test data for Plotly histogram ────────────────────────
        perm_gains_json  = _json.dumps([round(float(x), 4)
                                        for x in ra.perm_sharpe_gains])
        actual_gain_json = round(float(ra.actual_sharpe_gain), 4)
        pvalue_json      = round(float(ra.perm_pvalue), 4)

        # ── Transition matrix data ────────────────────────────────────────────
        n_states  = ra.transition_matrix.shape[0]
        state_names = ["Bear", "Neutral", "Bull"][:n_states]
        trans_z   = ra.transition_matrix.tolist()
        trans_text= [[f"{ra.transition_matrix[r][c]*100:.1f}%"
                      for c in range(n_states)] for r in range(n_states)]

        # ── Avg duration bars ─────────────────────────────────────────────────
        dur_names  = list(ra.avg_duration_days.keys())
        dur_values = [round(v, 1) for v in ra.avg_duration_days.values()]
        dur_colors = [colors.get(n, "#8b949e") for n in dur_names]

        return f"""
<div id='regime_section' style='margin-top:48px'>
<h2 style='font-size:20px;font-weight:600;color:#e6edf3;margin-bottom:4px'>
  🔍 Regime Analysis — Hidden Markov Model
</h2>
<p style='color:#8b949e;font-size:13px;margin-bottom:20px'>
  3-state Gaussian HMM fit on daily log returns. States sorted by mean return:
  bear / neutral / bull. Rolling expanding-window predictions are forward-safe.
  Permutation test shuffles regime labels {ra.perm_n:,}× to check if the filter
  improvement is real or due to chance.
</p>

<div class='chart-box' style='margin-bottom:16px'>
  <div style='font-size:13px;font-weight:600;color:#8b949e;margin-bottom:14px'>
    Performance by Regime — All Trades</div>
  {table_html}
</div>

<div class='chart-box' style='margin-bottom:16px'>
  <div style='font-size:13px;font-weight:600;color:#8b949e;margin-bottom:10px'>
    In-Sample vs Out-of-Sample Sharpe</div>
  {is_oos_html}
</div>

<div style='display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px'>

  <div class='chart-box'>
    <div style='font-size:13px;font-weight:600;color:#8b949e;margin-bottom:6px'>
      Permutation Test — Filter Sharpe Gain</div>
    <div style='margin-bottom:10px'>
      <span style='font-size:12px;color:#8b949e'>H₀: any random day-filter gives equal improvement.
        p-value = </span>
      <span style='font-size:14px;font-weight:700;color:{sig_color}'>
        {ra.perm_pvalue:.3f}</span>
      <span style='font-size:12px;color:{sig_color};margin-left:6px'>
        {sig_text}</span>
    </div>
    <div id='regime_perm_hist' style='height:220px'></div>
  </div>

  <div class='chart-box'>
    <div style='font-size:13px;font-weight:600;color:#8b949e;margin-bottom:6px'>
      Regime Stability</div>
    <div style='display:flex;gap:12px'>
      <div style='flex:1' id='regime_duration_bar'></div>
      <div style='flex:1' id='regime_transition_hm'></div>
    </div>
  </div>

</div>

<script>
(function() {{
  var PERM_GAINS   = {perm_gains_json};
  var ACTUAL_GAIN  = {actual_gain_json};
  var PVALUE       = {pvalue_json};
  var SIG_COLOR    = "{sig_color}";
  var TRANS_Z      = {_json.dumps(trans_z)};
  var TRANS_TEXT   = {_json.dumps(trans_text)};
  var STATE_NAMES  = {_json.dumps(state_names)};
  var DUR_NAMES    = {_json.dumps([n.capitalize() for n in dur_names])};
  var DUR_VALUES   = {_json.dumps(dur_values)};
  var DUR_COLORS   = {_json.dumps(dur_colors)};

  var DARK = {{paper_bgcolor:"#161b22",plot_bgcolor:"#161b22",
              font:{{color:"#e6edf3",size:11}},margin:{{l:40,r:20,t:10,b:40}}}};

  // Permutation histogram
  Plotly.newPlot("regime_perm_hist", [
    {{type:"histogram", x:PERM_GAINS, nbinsx:60,
      marker:{{color:"rgba(56,139,253,0.5)",line:{{color:"#388bfd",width:1}}}},
      name:"Random gains"}},
    {{type:"scatter", mode:"lines",
      x:[ACTUAL_GAIN, ACTUAL_GAIN], y:[0, PERM_GAINS.length/10],
      line:{{color:SIG_COLOR, width:2, dash:"dot"}}, name:"Actual gain"}},
  ], Object.assign({{}}, DARK, {{
    xaxis:{{title:{{text:"Sharpe Gain from Filter",font:{{size:10,color:"#8b949e"}}}},
            gridcolor:"#21262d",zerolinecolor:"#30363d"}},
    yaxis:{{title:{{text:"Count",font:{{size:10,color:"#8b949e"}}}},gridcolor:"#21262d"}},
    showlegend:false, height:220,
  }}), {{responsive:true, displayModeBar:false}});

  // Duration bar chart
  Plotly.newPlot("regime_duration_bar", [
    {{type:"bar", x:DUR_NAMES, y:DUR_VALUES,
      marker:{{color:DUR_COLORS}},
      text:DUR_VALUES.map(function(v){{return v+"d"}}),
      textposition:"outside",
      cliponaxis:false}},
  ], Object.assign({{}}, DARK, {{
    yaxis:{{title:{{text:"Avg Days",font:{{size:10,color:"#8b949e"}}}},
            gridcolor:"#21262d",
            range:[0, Math.max.apply(null,DUR_VALUES)*1.3]}},
    xaxis:{{gridcolor:"#21262d"}},
    height:200, margin:{{l:40,r:10,t:24,b:30}},
  }}), {{responsive:true, displayModeBar:false}});

  // Transition matrix heatmap
  Plotly.newPlot("regime_transition_hm", [
    {{type:"heatmap", z:TRANS_Z, x:STATE_NAMES, y:STATE_NAMES,
      text:TRANS_TEXT, texttemplate:"%{{text}}",
      colorscale:[[0,"#0d1117"],[1,"#388bfd"]],
      showscale:false, textfont:{{size:11}}}},
  ], Object.assign({{}}, DARK, {{
    xaxis:{{title:{{text:"To",font:{{size:10,color:"#8b949e"}}}},side:"bottom"}},
    yaxis:{{title:{{text:"From",font:{{size:10,color:"#8b949e"}}}},autorange:"reversed"}},
    height:200, margin:{{l:60,r:10,t:10,b:40}},
  }}), {{responsive:true, displayModeBar:false}});
}})();
</script>
</div>
"""

    # ── Prop Firm Section ────────────────────────────────────────────────────

    def _section_prop_firm(self, prop_firm: dict, prop_firm_rev: dict = None) -> str:
        import json as _json
        from backtest.propfirm.lucidflex import LUCIDFLEX_ACCOUNTS, RISK_PCT_OF_MLL

        def _serialise(pf_data: dict) -> dict:
            """Serialise one prop firm result dict into JS-ready structure."""
            accounts   = list(pf_data.keys())
            first_scheme = next(iter(pf_data[accounts[0]]))
            eval_rps   = [k for k in pf_data[accounts[0]][first_scheme]
                          if isinstance(k, float)]
            funded_rps = [k for k in pf_data[accounts[0]][first_scheme][eval_rps[0]]
                          if isinstance(k, float)]
            schemes    = [k for k in pf_data[accounts[0]]
                          if isinstance(k, str) and k != "optimal_funded_rp"]
            metrics    = ["net_ev", "pass_rate", "survival_rate",
                          "median_withdrawal", "mean_withdrawal", "ev_per_day"]
            all_data   = {}
            for acc in accounts:
                all_data[acc] = {}
                for s in schemes:
                    all_data[acc][s] = {}
                    opt = pf_data[acc][s].get("optimal_funded_rp", {})
                    all_data[acc][s]["optimal_funded_idx"] = [
                        funded_rps.index(opt.get(erp, funded_rps[0]))
                        for erp in eval_rps
                    ]
                    for metric in metrics:
                        z_3d, z_surface, hover_3d = [], [], []
                        for frp in funded_rps:
                            row_z, row_h = [], []
                            for erp in eval_rps:
                                cell  = pf_data[acc][s][erp][frp]
                                val   = cell[metric]
                                row_z.append(round(val, 2))
                                ev    = cell["net_ev"]
                                pr    = cell["pass_rate"]
                                surv  = cell["survival_rate"]
                                med_w = cell["median_withdrawal"]
                                med_n = cell.get("median_n_payouts", 0)
                                pct_f = cell.get("pct_full_payout", 0)
                                dp    = cell.get("median_days_to_pass", 0)
                                dw    = cell.get("median_days_to_withdrawal", 0)
                                row_h.append(
                                    f"Eval {int(erp*100)}% / Funded {int(frp*100)}%<br>"
                                    f"Net EV: ${ev:,.0f}<br>"
                                    f"Pass rate: {pr*100:.1f}%<br>"
                                    f"Survival: {surv*100:.0f}%<br>"
                                    f"Median W: ${med_w:,.0f} | Med payouts: {med_n:.1f}<br>"
                                    f"Full 6 cycles: {pct_f*100:.0f}%<br>"
                                    f"Days eval: {dp:.0f}d / funded: {dw:.0f}d"
                                )
                            z_3d.append(row_z)
                            hover_3d.append(row_h)
                        for ei, erp in enumerate(eval_rps):
                            z_surface.append([z_3d[fi][ei]
                                              for fi in range(len(funded_rps))])
                        all_data[acc][s][metric] = {
                            "z_3d": z_3d, "z_surface": z_surface, "hover": hover_3d,
                        }
                    # Timeline data — must be inside the for s loop
                    timeline = []
                    for frp in funded_rps:
                        row = []
                        for erp in eval_rps:
                            cell = pf_data[acc][s][erp][frp]
                            row.append({
                                "payout_days": cell.get("median_payout_days", []),
                                "gap_days":    cell.get("median_gap_days", []),
                                "eval_days":   cell.get("total_eval_days", 0),
                                "funded_full": cell.get("median_funded_full_days"),
                                "calendar":    cell.get("total_cycle_calendar_days"),
                                "optimal_k":   cell.get("optimal_k", 6),
                                "per_k":       cell.get("per_k", []),
                            })
                        timeline.append(row)
                    all_data[acc][s]["timeline"] = timeline   # [frp_idx][erp_idx]
            return all_data, accounts, eval_rps, funded_rps, schemes

        all_data, accounts, eval_rps, funded_rps, schemes = _serialise(prop_firm)
        has_rev = prop_firm_rev is not None
        if has_rev:
            all_data_rev, _, _, _, _ = _serialise(prop_firm_rev)
        else:
            all_data_rev = {}

        eval_labels   = [f"{int(r*100)}%" for r in eval_rps]
        funded_labels = [f"{int(r*100)}%" for r in funded_rps]
        scheme_labels = {
            "fixed_dollar": "Fixed $",   "pct_balance": "% Balance",
            "frac_dd":      "Frac DD",   "floor_aware": "Floor Aware",
            "max_size":     "Max Size",  "martingale":  "Martingale",
        }
        s_labels = [scheme_labels.get(s, s) for s in schemes]

        account_fees = {
            acc: {
                "eval_fee":  LUCIDFLEX_ACCOUNTS[acc].eval_fee,
                "reset_fee": LUCIDFLEX_ACCOUNTS[acc].reset_fee,
                "mll":       LUCIDFLEX_ACCOUNTS[acc].mll_amount,
            }
            for acc in accounts if acc in LUCIDFLEX_ACCOUNTS
        }

        btn_html = "".join(
            f"""<button id='pf_btn_{acc}' onclick='pfSelectAccount("{acc}")'
              style='padding:6px 18px;border-radius:4px;
                     border:1px solid {"#26a69a" if i==0 else "#30363d"};
                     background:{"rgba(38,166,154,.15)" if i==0 else "#21262d"};
                     color:{"#26a69a" if i==0 else "#e6edf3"};
                     cursor:pointer;font-size:13px;font-weight:600'>{acc}</button>"""
            for i, acc in enumerate(accounts)
        )
        scheme_btn_html = "".join(
            f"""<button id='pf_sbtn_{i}' onclick='pfSelectScheme({i})'
              style='padding:5px 12px;border-radius:4px;
                     border:1px solid {"#388bfd" if i==0 else "#30363d"};
                     background:{"rgba(56,139,253,.15)" if i==0 else "#21262d"};
                     color:{"#388bfd" if i==0 else "#e6edf3"};
                     cursor:pointer;font-size:12px'>{sl}</button>"""
            for i, sl in enumerate(s_labels)
        )
        rev_toggle_html = ""
        if has_rev:
            rev_toggle_html = """
  <span style='font-size:12px;color:#8b949e;margin-left:12px'>Trades:</span>
  <button id='pf_btn_normal' onclick='pfSetTrades("normal")'
    style='padding:5px 12px;border-radius:4px;border:1px solid #26a69a;
           background:rgba(38,166,154,.15);color:#26a69a;cursor:pointer;font-size:12px'>
    Normal</button>
  <button id='pf_btn_reversed' onclick='pfSetTrades("reversed")'
    style='padding:5px 12px;border-radius:4px;border:1px solid #30363d;
           background:#21262d;color:#e6edf3;cursor:pointer;font-size:12px'>
    Reversed</button>"""

        return f"""
<div id='propfirm_section' style='margin-top:48px'>
<h2 style='font-size:20px;font-weight:600;color:#e6edf3;margin-bottom:4px'>
  ⚡ Prop Firm Analysis — LucidFlex
</h2>
<p style='color:#8b949e;font-size:13px;margin-bottom:20px'>
  Net EV = P(pass eval) × E[total withdrawal if funded] − expected fees.
  X-axis: eval risk. Slider: funded risk. Optimal funded risk is pre-selected per cell.
</p>

<div style='display:flex;gap:8px;margin-bottom:12px;flex-wrap:wrap;align-items:center'>
  <span style='font-size:12px;color:#8b949e'>Account:</span>
  {btn_html}
  <span style='font-size:12px;color:#8b949e;margin-left:12px'>Sizing:</span>
  <button id='pf_btn_micros' onclick='pfSelectSizing("micros")'
    style='padding:5px 12px;border-radius:4px;border:1px solid #26a69a;
           background:rgba(38,166,154,.15);color:#26a69a;cursor:pointer;font-size:12px'>
    MNQ Micros</button>
  <button id='pf_btn_auto' onclick='pfSelectSizing("auto")'
    style='padding:5px 12px;border-radius:4px;border:1px solid #30363d;
           background:#21262d;color:#e6edf3;cursor:pointer;font-size:12px'>
    Auto</button>
  <span style='font-size:12px;color:#8b949e;margin-left:12px'>View:</span>
  <button id='pf_btn_2d' onclick='pfSetView("2d")'
    style='padding:5px 12px;border-radius:4px;border:1px solid #26a69a;
           background:rgba(38,166,154,.15);color:#26a69a;cursor:pointer;font-size:12px'>
    2D + Slider</button>
  <button id='pf_btn_3d' onclick='pfSetView("3d")'
    style='padding:5px 12px;border-radius:4px;border:1px solid #30363d;
           background:#21262d;color:#e6edf3;cursor:pointer;font-size:12px'>
    3D Surface</button>
  {rev_toggle_html}
</div>

<div style='display:flex;gap:8px;margin-bottom:16px;flex-wrap:wrap;align-items:center'>
  <span style='font-size:12px;color:#8b949e'>Scheme:</span>
  {scheme_btn_html}
</div>

<!-- 2D view -->
<div id='pf_2d_view'>
  <div style='display:flex;align-items:center;gap:16px;margin-bottom:12px;
              background:#161b22;border:1px solid #30363d;border-radius:8px;padding:14px'>
    <div>
      <div style='font-size:11px;color:#8b949e;margin-bottom:4px;text-transform:uppercase;
                  letter-spacing:.05em'>Funded Risk</div>
      <div style='display:flex;align-items:center;gap:10px'>
        <input type='range' id='pf_funded_slider' min='0' max='{len(funded_rps)-1}'
               value='0' step='1'
               style='width:200px;accent-color:#26a69a'
               oninput='pfSliderChange(this.value)'>
        <span id='pf_slider_label' style='font-size:14px;font-weight:700;
              color:#26a69a;min-width:40px'>{funded_labels[0]}</span>
        <span style='font-size:11px;color:#8b949e'>of MLL</span>
      </div>
    </div>
    <div id='pf_optimal_note' style='font-size:12px;color:#8b949e;line-height:1.5'></div>
  </div>

  <div style='display:flex;flex-direction:column;gap:16px' id='pf_heatmap_grid'>
    <div class='chart-box'>
      <div style='font-size:12px;font-weight:600;color:#8b949e;margin-bottom:6px'>
        Net EV ($) — per cycle</div>
      <div id='pf_hm_net_ev'></div>
    </div>
    <div class='chart-box'>
      <div style='font-size:12px;font-weight:600;color:#8b949e;margin-bottom:6px'>
        Eval Pass Rate (%)</div>
      <div id='pf_hm_pass_rate'></div>
    </div>
    <div class='chart-box'>
      <div style='font-size:12px;font-weight:600;color:#8b949e;margin-bottom:6px'>
        Funded Survival Rate (%)</div>
      <div id='pf_hm_survival_rate'></div>
    </div>
    <div class='chart-box'>
      <div style='font-size:12px;font-weight:600;color:#8b949e;margin-bottom:6px'>
        Median Total Withdrawal ($ | if survive, sum of ≤6 cycles)</div>
      <div id='pf_hm_median_withdrawal'></div>
    </div>
  </div>
</div>

<!-- 3D view -->
<div id='pf_3d_view' style='display:none'>
  <div class='chart-box'>
    <div style='font-size:12px;font-weight:600;color:#8b949e;margin-bottom:6px'>
      Net EV Surface — Eval Risk × Funded Risk</div>
    <div id='pf_surface_net_ev' style='height:520px'></div>
  </div>
</div>

<div id='pf_ev_section' style='margin-top:24px'></div>

<script>
(function() {{
  var ALL_DATA     = {_json.dumps(all_data)};
  var ALL_DATA_REV = {_json.dumps(all_data_rev)};
  var HAS_REV      = {_json.dumps(has_rev)};
  var currentTradeSet = "normal";   // "normal" | "reversed"

  function getActiveData() {{
    return (currentTradeSet === "reversed" && HAS_REV) ? ALL_DATA_REV : ALL_DATA;
  }}
  var SCHEMES      = {_json.dumps(s_labels)};
  var SCHEME_KEYS  = {_json.dumps(schemes)};
  var EVAL_RISKS   = {_json.dumps(eval_labels)};
  var FUNDED_RISKS = {_json.dumps(funded_labels)};
  var EVAL_RPCTS   = {_json.dumps([float(r) for r in eval_rps])};
  var FUNDED_RPCTS = {_json.dumps([float(r) for r in funded_rps])};
  var ACCOUNTS     = {_json.dumps(accounts)};
  var ACCOUNT_FEES = {_json.dumps(account_fees)};

  var currentAccount  = ACCOUNTS[0];
  var currentSchemeIdx= 0;
  var currentFundedIdx= 0;
  var currentView     = "2d";

  var RWG = [
    [0.0,  "rgb(160,0,0)"],
    [0.25, "rgb(220,80,80)"],
    [0.5,  "rgb(255,255,255)"],
    [0.75, "rgb(60,180,60)"],
    [1.0,  "rgb(0,110,0)"],
  ];

  function fmtVal(v, metric) {{
    if (metric === "net_ev") {{
      var abs = Math.abs(v), sign = v < 0 ? "-" : "";
      if (abs >= 1000) return sign + "$" + (abs/1000).toFixed(1) + "k";
      return sign + "$" + Math.round(abs);
    }}
    if (metric === "pass_rate")         return (v*100).toFixed(0)+"%";
    if (metric === "survival_rate")     return (v*100).toFixed(0)+"%";
    if (metric === "median_withdrawal") {{
      var abs = Math.abs(v), sign = v < 0 ? "-" : "";
      if (abs >= 1000) return sign + "$" + (abs/1000).toFixed(1) + "k";
      return sign + "$" + Math.round(abs);
    }}
    return v.toFixed(2);
  }}

  var HM_DIVS    = ["pf_hm_net_ev","pf_hm_pass_rate","pf_hm_survival_rate","pf_hm_median_withdrawal"];
  var HM_METRICS = ["net_ev","pass_rate","survival_rate","median_withdrawal"];

  function getSchemeData() {{
    return getActiveData()[currentAccount][SCHEME_KEYS[currentSchemeIdx]];
  }}

  function render2D(fundedIdx) {{
    var sd   = getSchemeData();
    var opts = sd["optimal_funded_idx"];

    // Mark optimal cells with a star in text
    for (var mi = 0; mi < HM_METRICS.length; mi++) {{
      var metric = HM_METRICS[mi];
      var d      = sd[metric];
      var z_row  = d.z_3d[fundedIdx];    // eval_rp values at this funded slice
      var h_row  = d.hover[fundedIdx];

      // z and text need to be 2D arrays (scheme × eval_risk)
      // We have one scheme at a time — wrap in outer array for proper heatmap
      // Actually schemes are on y-axis, eval_risks on x-axis.
      // We're showing one scheme at a time with schemes on y? No —
      // We show eval_risk on x, funded_risk selected by slider.
      // y-axis: we use all schemes simultaneously for the same funded_risk.
      // Build z as n_schemes × n_eval_risks
      var z_all = [], t_all = [], h_all = [];
      for (var si = 0; si < SCHEME_KEYS.length; si++) {{
        var sd2  = getActiveData()[currentAccount][SCHEME_KEYS[si]];
        var opts2= sd2["optimal_funded_idx"];
        var d2   = sd2[metric];
        var row_z= d2.z_3d[fundedIdx];
        var row_h= d2.hover[fundedIdx];
        var row_t= row_z.map(function(v, ei) {{
          var star = (fundedIdx === opts2[ei]) ? " ★" : "";
          return fmtVal(v, metric) + star;
        }});
        z_all.push(row_z);
        t_all.push(row_t);
        h_all.push(row_h);
      }}

      var flat  = [].concat.apply([], z_all);
      var zmid  = (metric === "net_ev") ? 0 : null;

      var trace = {{
        type: "heatmap",
        z: z_all,
        x: EVAL_RISKS,
        y: SCHEMES,
        text: t_all,
        customdata: h_all,
        hovertemplate: "%{{customdata}}<extra></extra>",
        colorscale: RWG,
        showscale: true,
        colorbar: {{tickfont:{{color:"#e6edf3",size:10}}, outlinecolor:"#30363d", len:0.9}},
        texttemplate: "%{{text}}",
        textfont: {{size:10}},
      }};
      if (zmid !== null) trace.zmid = zmid;

      Plotly.purge(HM_DIVS[mi]);
      Plotly.newPlot(HM_DIVS[mi], [trace], {{
        paper_bgcolor:"#161b22", plot_bgcolor:"#161b22",
        font:{{color:"#e6edf3",size:10}},
        margin:{{l:120,r:80,t:10,b:60}}, height:260,
        xaxis:{{tickfont:{{size:11}}, title:{{text:"Eval Risk (% of MLL)",font:{{size:11,color:"#8b949e"}}}}}},
        yaxis:{{tickfont:{{size:11}}}},
      }}, {{responsive:true, displayModeBar:false}});
    }}

    // Optimal note
    var optFunded = sd["optimal_funded_idx"];
    var allOptSame = optFunded.every(function(v){{return v===optFunded[0];}});
    var noteEl = document.getElementById("pf_optimal_note");
    if (allOptSame) {{
      noteEl.innerHTML = "<strong style='color:#e6edf3'>★ Optimal funded risk for all eval levels:</strong> "
        + FUNDED_RISKS[optFunded[0]] + " of MLL";
    }} else {{
      var parts = optFunded.map(function(fi,ei){{
        return "eval "+EVAL_RISKS[ei]+" → funded "+FUNDED_RISKS[fi];
      }});
      noteEl.innerHTML = "<strong style='color:#e6edf3'>★ Optimal funded risk varies:</strong> "
        + parts.join(" &nbsp;|&nbsp; ");
    }}

    renderEVSummary();
  }}

  function render3D() {{
    var sd  = getSchemeData();
    var d   = sd["net_ev"];
    var trace = {{
      type: "surface",
      z: d.z_surface,   // [erp_idx][frp_idx]
      x: FUNDED_RISKS,
      y: EVAL_RISKS,
      colorscale: RWG,
      cmid: 0,
      showscale: true,
      colorbar: {{tickfont:{{color:"#e6edf3",size:10}}, outlinecolor:"#30363d"}},
      hovertemplate: "Eval: %{{y}}<br>Funded: %{{x}}<br>Net EV: $%{{z:,.0f}}<extra></extra>",
    }};
    Plotly.purge("pf_surface_net_ev");
    Plotly.newPlot("pf_surface_net_ev", [trace], {{
      paper_bgcolor:"#161b22", plot_bgcolor:"#161b22",
      font:{{color:"#e6edf3",size:11}},
      margin:{{l:0,r:0,t:10,b:0}},
      scene:{{
        xaxis:{{title:"Funded Risk",gridcolor:"#30363d",zerolinecolor:"#30363d",
                tickfont:{{color:"#e6edf3"}}}},
        yaxis:{{title:"Eval Risk",  gridcolor:"#30363d",zerolinecolor:"#30363d",
                tickfont:{{color:"#e6edf3"}}}},
        zaxis:{{title:"Net EV ($)", gridcolor:"#30363d",zerolinecolor:"#30363d",
                tickfont:{{color:"#e6edf3"}}}},
        bgcolor:"#161b22",
      }},
    }}, {{responsive:true, displayModeBar:true}});
  }}

  function renderAll() {{
    if (currentView === "2d") {{
      render2D(currentFundedIdx);   // render2D already calls renderEVSummary
    }} else {{
      render3D();
      renderEVSummary();            // render3D doesn't call it — do it here
    }}
  }}

  // ── Pre-select optimal funded risk for current scheme + account ─────────
  function selectOptimalFunded() {{
    var sd  = getSchemeData();
    var opt = sd["optimal_funded_idx"];
    // Find the funded idx that is optimal for the most eval risks
    var counts = Array(FUNDED_RISKS.length).fill(0);
    opt.forEach(function(fi){{ counts[fi]++; }});
    var best = counts.indexOf(Math.max.apply(null, counts));
    currentFundedIdx = best;
    document.getElementById("pf_funded_slider").value = best;
    document.getElementById("pf_slider_label").textContent = FUNDED_RISKS[best];
  }}

  function renderEVSummary() {{
    var fees = ACCOUNT_FEES[currentAccount] || {{eval_fee:0, reset_fee:0, mll:0}};

    // ── Global search across ALL eval_scheme × funded_scheme combinations ─────
    // The eval scheme determines pass_rate (and hence e_cost).
    // The funded scheme determines mean_withdrawal.
    // These are independent — we can mix any eval scheme with any funded scheme.
    // net_ev = optimal-K EV (consistent with ev_per_day); net_ev_6cycle = full 6-payout EV
    // ── Global search: for each K, find the best combination independently ────
    // Structure: best_per_k[k] = {{ ev_per_day, ev, cycle_days, ... }}
    // Then global best = argmax over k of best_per_k[k].ev_per_day

    var MAX_K = 6;
    var best_per_k = [];
    for (var k0 = 1; k0 <= MAX_K; k0++) {{
      best_per_k.push({{
        k:            k0,
        ev_per_day:   -Infinity,
        ev:           0,
        cycle_days:   0,
        p_reach:      0,
        pass_rate:    0,
        survival:     0,
        median_w:     0,
        mean_w:       0,
        eval_scheme_label: "",
        funded_scheme_label: "",
        erp: "", frp: "",
        timeline: null,
      }});
    }}

    for (var esi = 0; esi < SCHEME_KEYS.length; esi++) {{
      var eval_sd = getActiveData()[currentAccount][SCHEME_KEYS[esi]];
      for (var ei = 0; ei < EVAL_RPCTS.length; ei++) {{
        var pr     = eval_sd["pass_rate"].z_3d[0][ei];
        var e_att  = pr > 0 ? 1.0/pr : 10;
        var e_cost = fees.eval_fee + Math.max(0, e_att-1)*fees.reset_fee;
        var e_tl_ref = (eval_sd["timeline"] && eval_sd["timeline"][0])
                       ? eval_sd["timeline"][0][ei] : null;
        var eval_days_ei = e_tl_ref ? (e_tl_ref.eval_days || 0) : 0;

        for (var fsi = 0; fsi < SCHEME_KEYS.length; fsi++) {{
          var funded_sd = getActiveData()[currentAccount][SCHEME_KEYS[fsi]];
          for (var fi = 0; fi < FUNDED_RPCTS.length; fi++) {{
            var mean_w_full = funded_sd["mean_withdrawal"].z_3d[fi][ei];
            var f_tl = (funded_sd["timeline"] && funded_sd["timeline"][fi])
                       ? funded_sd["timeline"][fi][ei] : null;
            if (!f_tl || !f_tl.per_k || !f_tl.per_k.length) continue;

            for (var ki = 0; ki < f_tl.per_k.length; ki++) {{
              var pkr     = f_tl.per_k[ki];
              var k       = pkr.k;
              var k_idx   = k - 1;
              if (pkr.cycle_days == null || pkr.cycle_days <= 0) continue;

              // Recompute cycle_days using this eval scheme's eval_days
              var funded_days_k = pkr.cycle_days - (f_tl.eval_days || 0);
              var cycle_days_k  = eval_days_ei + Math.max(0, funded_days_k);

              // Recompute EV using cross-scheme pass_rate and mean_w_k
              var mean_w_k = pkr.mean_total_w != null ? pkr.mean_total_w : 0;
              var ev_k     = pr * mean_w_k - e_cost;
              var epd_k    = cycle_days_k > 0 ? ev_k / cycle_days_k : -Infinity;

              if (epd_k > best_per_k[k_idx].ev_per_day) {{
                var e_tl = (eval_sd["timeline"] && eval_sd["timeline"][fi])
                           ? eval_sd["timeline"][fi][ei] : null;
                best_per_k[k_idx] = {{
                  k:                   k,
                  ev_per_day:          epd_k,
                  ev:                  ev_k,
                  cycle_days:          cycle_days_k,
                  p_reach:             pkr.p_reach,
                  pass_rate:           pr,
                  survival:            funded_sd["survival_rate"].z_3d[fi][ei],
                  median_w:            funded_sd["median_withdrawal"].z_3d[fi][ei],
                  mean_w:              mean_w_full,
                  mean_w_k:            mean_w_k,
                  eval_scheme_key:     SCHEME_KEYS[esi],
                  eval_scheme_label:   SCHEMES[esi],
                  funded_scheme_key:   SCHEME_KEYS[fsi],
                  funded_scheme_label: SCHEMES[fsi],
                  erp:      EVAL_RISKS[ei],
                  frp:      FUNDED_RISKS[fi],
                  erp_i:    ei, frp_i: fi,
                  timeline: (e_tl && f_tl) ? {{
                    eval_days:   eval_days_ei,
                    payout_days: f_tl.payout_days,
                    gap_days:    f_tl.gap_days,
                    calendar:    (e_tl ? e_tl.calendar : null) || f_tl.calendar,
                    optimal_k:   k,
                    per_k:       null,   // filled after outer loop
                  }} : null,
                }};
              }}
            }}
          }}
        }}
      }}
    }}

    // Global best = argmax over K of ev_per_day
    var best = best_per_k[0];
    for (var ki2 = 1; ki2 < best_per_k.length; ki2++) {{
      if (best_per_k[ki2].ev_per_day > best.ev_per_day) {{
        best = best_per_k[ki2];
      }}
    }}
    // Attach the full per_k comparison table (best combination per K)
    if (best.timeline) {{
      best.timeline.per_k = best_per_k.map(function(bk) {{
        return {{
          k:          bk.k,
          p_reach:    bk.p_reach,
          ev:         bk.ev,
          cycle_days: bk.cycle_days,
          ev_per_day: bk.ev_per_day,
          eval_scheme_label:   bk.eval_scheme_label,
          funded_scheme_label: bk.funded_scheme_label,
          erp: bk.erp,
          frp: bk.frp,
        }};
      }});
    }}

    function fmtD(v) {{
      var s = Math.round(Math.abs(v)).toString().replace(/\\B(?=(\\d{{3}})+(?!\\d))/g,",");
      return (v<0?"-$":"$")+s;
    }}

    var evCol    = best.ev >= 0 ? "#26a69a" : "#ef5350";
    var eAttempts= best.pass_rate > 0 ? (1/best.pass_rate) : 0;
    var eFees    = fees.eval_fee + Math.max(0, eAttempts-1)*fees.reset_fee;
    var samescheme = best.eval_scheme_key === best.funded_scheme_key
                     && best.erp === best.frp;

    document.getElementById("pf_ev_section").innerHTML = `
      <h2 style='font-size:16px;font-weight:600;color:#e6edf3;margin:28px 0 6px'>
        Global Optimal Setting — ${{currentAccount}}
      </h2>
      <p style='font-size:12px;color:#8b949e;margin-bottom:14px'>
        Best combination across <strong style='color:#e6edf3'>all eval schemes × all funded schemes</strong>
        by net EV per cycle. Eval and funded schemes can differ — they are independent.
      </p>
      <div style='background:#0d1117;border:1px solid #30363d;border-radius:8px;padding:20px'>
        <div style='display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:14px'>
          ${{[
            ["EVAL SCHEME",    best.eval_scheme_label],
            ["EVAL RISK",      best.erp+" of MLL"],
            ["FUNDED SCHEME",  best.funded_scheme_label],
            ["FUNDED RISK",    best.frp+" of MLL"],
          ].map(function(p){{
            return "<div style='background:#161b22;border:1px solid #26a69a;border-radius:6px;padding:10px'>"
              +"<div style='font-size:10px;color:#8b949e;margin-bottom:3px'>"+p[0]+"</div>"
              +"<div style='font-size:14px;font-weight:700;color:#26a69a'>"+p[1]+"</div>"
              +"</div>";
          }}).join("")}}
        </div>
        <div style='display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:16px'>
          ${{[
            ["PASS RATE",             (best.pass_rate*100).toFixed(1)+"%"],
            ["SURVIVAL RATE",         (best.survival*100).toFixed(1)+"%"],
            ["MEDIAN W (if survive)", fmtD(best.median_w)],
            ["E[ATTEMPTS]",           eAttempts.toFixed(1)+"×"],
          ].map(function(p){{
            return "<div style='background:#161b22;border:1px solid #21262d;border-radius:6px;padding:10px'>"
              +"<div style='font-size:10px;color:#8b949e;margin-bottom:3px'>"+p[0]+"</div>"
              +"<div style='font-size:14px;font-weight:700;color:#e6edf3'>"+p[1]+"</div>"
              +"</div>";
          }}).join("")}}
        </div>
        <div style='display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:16px'>
          ${{(function(){{
            var tl0  = best.timeline;
            var optK = best.optimal_k || 6;
            var epd  = best.ev_per_day || 0;
            var ann  = epd * 252;
            var cyclesPerYr = best.total_days > 0 ? (252 / best.total_days) : 0;
            var evCol2 = epd >= 0 ? "#26a69a" : "#ef5350";
            return [
              ["OPTIMAL PAYOUTS",  "Target "+optK+" payout"+(optK===1?"":"s")+" then restart"],
              ["EV / TRADING DAY", "<span style='color:"+evCol2+"'>$"+epd.toFixed(2)+"</span>"],
              ["EST. ANNUAL EV",   "<span style='color:"+evCol2+"'>"+fmtD(ann)+"</span> ("+cyclesPerYr.toFixed(1)+"× cycles/yr)"],
            ].map(function(p){{
              return "<div style='background:#161b22;border:1px solid #388bfd33;border-radius:6px;padding:10px'>"
                +"<div style='font-size:10px;color:#8b949e;margin-bottom:3px'>"+p[0]+"</div>"
                +"<div style='font-size:13px;font-weight:700;color:#388bfd'>"+p[1]+"</div>"
                +"</div>";
            }}).join("");
          }})()}}
        </div>
        <div style='border-top:1px solid #30363d;padding-top:14px;margin-bottom:16px'>
          <div style='font-size:11px;color:#8b949e;margin-bottom:8px;line-height:1.8'>
            Net EV = P(pass_eval) × E[total_withdrawal_if_funded] − E[fees]<br>
            = ${{best.pass_rate.toFixed(3)}} × ${{fmtD(best.mean_w)}}
              − (${{fees.eval_fee}} + (${{eAttempts.toFixed(2)}}−1) × ${{fees.reset_fee}})
              = <strong>${{fmtD(best.ev)}}</strong>
              (targeting ${{best.optimal_k}} payout${{best.optimal_k===1?"":"s"}},
              ${{best.cycle_days > 0 ? Math.round(best.cycle_days) : "?"}}d cycle)<br>
            <span style='font-size:10px;color:#636e7b'>
              Optimised on EV/trading day — assumes immediate restart after each cycle.
              E[withdrawal] = unconditional mean across all funded sims incl. blowouts.
            </span>
          </div>
          <div style='display:flex;align-items:baseline;gap:10px'>
            <span style='font-size:11px;color:#8b949e;text-transform:uppercase;letter-spacing:.05em'>
              Max EV / trading day
            </span>
            <span style='font-size:32px;font-weight:700;color:${{evCol}}'>${{(best.ev_per_day||0).toFixed(2)}}/d</span>
          </div>
        </div>

        ${{(function() {{
          // ── Timeline section ───────────────────────────────────────────────
          var tl = best.timeline;
          if (!tl || !tl.payout_days || tl.payout_days.length === 0) return "";

          var gapDays  = tl.gap_days    || [];
          var payDays  = tl.payout_days || [];
          var evalDays = tl.eval_days   || 0;
          var calDays  = tl.calendar;
          var maxPay   = payDays.length;
          var eAtt     = best.pass_rate > 0 ? (1/best.pass_rate) : 0;

          // ── Visual timeline nodes ─────────────────────────────────────────
          var nodes = [];
          nodes.push({{label:"Buy Eval", sub:"Day 0", color:"#8b949e", days:0}});
          nodes.push({{label:"Pass Eval", sub:"~"+Math.round(evalDays)+"d trading", color:"#388bfd", days:evalDays}});
          var cumDays = evalDays;
          for (var p = 0; p < maxPay; p++) {{
            var g = gapDays[p];
            if (g == null) break;
            cumDays += g;
            var col = p === maxPay-1 ? "#f0a500" : "#26a69a";
            nodes.push({{
              label: "Payout "+(p+1),
              sub:   "+"+Math.round(g)+"d",
              color: col,
              days:  cumDays,
            }});
          }}

          var nodeHtml = nodes.map(function(nd, idx) {{
            var isLast = idx === nodes.length - 1;
            return "<div style='display:flex;flex-direction:column;align-items:center;min-width:72px'>"
              + "<div style='width:12px;height:12px;border-radius:50%;background:"+nd.color+
                ";margin-bottom:4px;flex-shrink:0'></div>"
              + "<div style='font-size:10px;font-weight:700;color:"+nd.color+
                ";text-align:center;white-space:nowrap'>"+nd.label+"</div>"
              + "<div style='font-size:9px;color:#8b949e;text-align:center'>"+nd.sub+"</div>"
              + "</div>"
              + (isLast ? "" :
                "<div style='flex:1;height:2px;background:linear-gradient(90deg,#30363d,#444);"+
                "margin:5px 2px 0;min-width:16px'></div>");
          }}).join("");

          // ── Gap breakdown table ───────────────────────────────────────────
          var rows = "";
          rows += "<tr><td style='color:#8b949e;padding:3px 8px 3px 0'>Buy eval → Pass eval</td>"
               +  "<td style='color:#388bfd;font-weight:600;text-align:right'>"
               +  Math.round(evalDays)+" trading days</td>"
               +  "<td style='color:#8b949e;font-size:10px;padding-left:8px'>"
               +  "("+eAtt.toFixed(1)+"× attempts: "
               +  (eAtt-1).toFixed(1)+"× fail + 1× pass)</td></tr>";

          for (var p2 = 0; p2 < maxPay; p2++) {{
            var g2 = gapDays[p2];
            var absDay = payDays[p2];
            if (g2 == null) break;
            var col2 = p2 === maxPay-1 ? "#f0a500" : "#26a69a";
            var from  = p2 === 0 ? "Funded start" : "Payout "+(p2);
            rows += "<tr><td style='color:#8b949e;padding:3px 8px 3px 0'>"+from+" → Payout "+(p2+1)+"</td>"
                 +  "<td style='color:"+col2+";font-weight:600;text-align:right'>+"+Math.round(g2)+" trading days</td>"
                 +  "<td style='color:#8b949e;font-size:10px;padding-left:8px'>"
                 +  "(cumulative from funded start: "+Math.round(absDay||0)+"d)</td></tr>";
          }}

          var totalTradingDays = Math.round(cumDays);
          var calStr = calDays ? calDays+" calendar days (~"+calDays+" incl. weekends)" : "N/A";

          // ── Per-K breakdown table ─────────────────────────────────────────
          var perK = (tl.per_k) || [];
          var optK = tl.optimal_k || maxPay;
          var perKHtml = "";
          if (perK.length > 0) {{
            var pkRows = perK.map(function(r) {{
              var isOpt = r.k === optK;
              var epd   = r.ev_per_day != null ? "$"+r.ev_per_day.toFixed(2)+"/d" : "—";
              var ann   = r.ev_per_day != null ? fmtD(r.ev_per_day*252)+"/yr" : "—";
              var bg    = isOpt ? "rgba(56,139,253,.08)" : "transparent";
              var col   = isOpt ? "#388bfd" : "#8b949e";
              var star  = isOpt ? " ★" : "";
              // Show optimal scheme for this K if different from global best
              var schemeHint = (r.eval_scheme_label && r.funded_scheme_label)
                ? "<span style='font-size:9px;color:#636e7b;display:block'>"
                  +r.eval_scheme_label+" eval → "+r.funded_scheme_label+" funded"
                  +" ("+r.erp+"/"+r.frp+")</span>"
                : "";
              return "<tr style='background:"+bg+"'>"
                + "<td style='color:"+col+";padding:3px 6px;font-weight:"+(isOpt?"700":"400")+"'>"
                + r.k+" payout"+(r.k===1?"":"s")+star+schemeHint+"</td>"
                + "<td style='color:#e6edf3;text-align:right;padding:3px 6px'>"+(r.p_reach*100).toFixed(0)+"%</td>"
                + "<td style='color:"+(r.ev>=0?"#26a69a":"#ef5350")+";text-align:right;padding:3px 6px'>"+fmtD(r.ev)+"</td>"
                + "<td style='color:#8b949e;text-align:right;padding:3px 6px'>"+(r.cycle_days?Math.round(r.cycle_days)+"d":"—")+"</td>"
                + "<td style='color:"+col+";text-align:right;padding:3px 6px;font-weight:"+(isOpt?"700":"400")+"'>"+epd+"</td>"
                + "<td style='color:"+col+";text-align:right;padding:3px 6px;font-size:10px'>"+ann+"</td>"
                + "</tr>";
            }}).join("");
            perKHtml = "<div style='margin-top:16px;border-top:1px solid #30363d;padding-top:14px'>"
              + "<div style='font-size:11px;color:#8b949e;text-transform:uppercase;letter-spacing:.05em;margin-bottom:8px'>"
              + "Strategy by Target Payout Count — ★ = Optimal (max EV/day)</div>"
              + "<table style='width:100%;border-collapse:collapse;font-size:11px'>"
              + "<thead><tr style='border-bottom:1px solid #30363d'>"
              + "<th style='color:#636e7b;text-align:left;padding:3px 6px;font-weight:400'>Target</th>"
              + "<th style='color:#636e7b;text-align:right;padding:3px 6px;font-weight:400'>P(reach)</th>"
              + "<th style='color:#636e7b;text-align:right;padding:3px 6px;font-weight:400'>Net EV</th>"
              + "<th style='color:#636e7b;text-align:right;padding:3px 6px;font-weight:400'>Cycle days</th>"
              + "<th style='color:#636e7b;text-align:right;padding:3px 6px;font-weight:400'>EV/day</th>"
              + "<th style='color:#636e7b;text-align:right;padding:3px 6px;font-weight:400'>Annual EV</th>"
              + "</tr></thead>"
              + "<tbody>" + pkRows + "</tbody></table>"
              + "<div style='font-size:10px;color:#636e7b;margin-top:6px'>"
              + "P(reach) = probability of completing exactly K payouts before MLL breach. "
              + "Net EV = P(pass_eval) × E[withdrawal targeting K] − fees. "
              + "Annual EV assumes continuous redeployment after each cycle."
              + "</div></div>";
          }}

          rows += "<tr style='border-top:1px solid #30363d'>"
               +  "<td style='color:#e6edf3;font-weight:600;padding:6px 8px 3px 0'>Full cycle total</td>"
               +  "<td style='color:#e6edf3;font-weight:700;text-align:right'>"+totalTradingDays+" trading days</td>"
               +  "<td style='color:#8b949e;font-size:10px;padding-left:8px'>≈ "+calStr+"</td></tr>";

          return "<div style='border-top:1px solid #30363d;padding-top:16px'>"
            + "<div style='font-size:11px;color:#8b949e;text-transform:uppercase;"
            + "letter-spacing:.05em;margin-bottom:12px'>Expected Timeline — Full Cycle</div>"
            + "<div style='display:flex;align-items:flex-start;flex-wrap:wrap;gap:0;margin-bottom:16px;overflow-x:auto'>"
            + nodeHtml + "</div>"
            + "<table style='width:100%;border-collapse:collapse;font-size:12px'>"
            + "<tbody>" + rows + "</tbody></table>"
            + perKHtml
            + "<div style='font-size:10px;color:#636e7b;margin-top:8px;line-height:1.5'>"
            + "Timeline: medians conditional on reaching each stage. "
            + "×1.4 applied for calendar day estimate. "
            + "Eval days = E[failed attempts] × median fail duration + median pass duration."
            + "</div></div>";
        }})()}}

      </div>`;
  }}

  window.pfSelectAccount = function(acc) {{
    currentAccount = acc;
    ACCOUNTS.forEach(function(a) {{
      var b = document.getElementById("pf_btn_"+a);
      if (!b) return;
      b.style.borderColor = a===acc?"#26a69a":"#30363d";
      b.style.background  = a===acc?"rgba(38,166,154,.15)":"#21262d";
      b.style.color       = a===acc?"#26a69a":"#e6edf3";
    }});
    selectOptimalFunded();
    renderAll();
  }};

  window.pfSelectScheme = function(idx) {{
    currentSchemeIdx = idx;
    for (var i=0;i<SCHEMES.length;i++) {{
      var b = document.getElementById("pf_sbtn_"+i);
      if (!b) continue;
      b.style.borderColor = i===idx?"#388bfd":"#30363d";
      b.style.background  = i===idx?"rgba(56,139,253,.15)":"#21262d";
      b.style.color       = i===idx?"#388bfd":"#e6edf3";
    }}
    selectOptimalFunded();
    renderAll();
  }};

  window.pfSliderChange = function(val) {{
    currentFundedIdx = parseInt(val);
    document.getElementById("pf_slider_label").textContent = FUNDED_RISKS[currentFundedIdx];
    if (currentView === "2d") render2D(currentFundedIdx);
  }};

  window.pfSetView = function(view) {{
    currentView = view;
    document.getElementById("pf_2d_view").style.display = view==="2d"?"":"none";
    document.getElementById("pf_3d_view").style.display = view==="3d"?"":"none";
    ["2d","3d"].forEach(function(v) {{
      var b = document.getElementById("pf_btn_"+v);
      if (!b) return;
      b.style.borderColor = v===view?"#26a69a":"#30363d";
      b.style.background  = v===view?"rgba(38,166,154,.15)":"#21262d";
      b.style.color       = v===view?"#26a69a":"#e6edf3";
    }});
    renderAll();
  }};

  window.pfSetTrades = function(mode) {{
    currentTradeSet = mode;
    ["normal","reversed"].forEach(function(m) {{
      var b = document.getElementById("pf_btn_"+m);
      if (!b) return;
      b.style.borderColor = m===mode?"#26a69a":"#30363d";
      b.style.background  = m===mode?"rgba(38,166,154,.15)":"#21262d";
      b.style.color       = m===mode?"#26a69a":"#e6edf3";
    }});
    selectOptimalFunded();
    renderAll();
  }};

  window.pfSelectSizing = function(mode) {{
    ["micros","auto"].forEach(function(m) {{
      var b = document.getElementById("pf_btn_"+m);
      if (!b) return;
      b.style.borderColor = m===mode?"#26a69a":"#30363d";
      b.style.background  = m===mode?"rgba(38,166,154,.15)":"#21262d";
      b.style.color       = m===mode?"#26a69a":"#e6edf3";
    }});
    if (mode==="auto") alert("Auto sizing requires re-running with sizing_mode=\\'auto\\'.");
  }};

  selectOptimalFunded();
  renderAll();
}})();
</script>
</div>
"""

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
            card("Commission",    _fmt_dollar(r.total_commission),
                 f"{_fmt_dollar(r.total_commission / r.n_trades if r.n_trades else 0)} / trade", "negative"),
            card("Slippage",      _fmt_dollar(r.total_slippage),
                 f"{_fmt_dollar(r.total_slippage / r.n_trades if r.n_trades else 0)} / trade", "negative"),
        ]

        return f"<h2>Performance Summary</h2><div class='metric-grid'>{''.join(cards)}</div>"

    def _section_equity_curve(self, r: "Results") -> str:
        import json
        import numpy as np

        # ── Strategy equity (trade-resolution) ───────────────────────────────
        # One point per closed trade — avoids flat lines for intraday strategies.
        capital     = r.starting_capital
        trade_pnls  = r.trade_pnls
        trade_dates = [t["exit"][:10] for t in r.trade_log]   # one per trade

        eq_trade_y = [capital]
        running = capital
        for pnl in trade_pnls:
            running += pnl
            eq_trade_y.append(running)

        n_trades   = len(trade_pnls)
        eq_trade_x = list(range(n_trades + 1))
        # Date x: "start" + one date per trade exit — same length as eq_trade_y
        eq_date_x  = ["start"] + trade_dates

        # ── B&H curves sampled at exactly the same x points as strategy ───────
        # Key: all three traces MUST share identical x arrays so Plotly's
        # category axis doesn't create extra categories per trace.
        bm_arr  = np.array(r.benchmark.equity_curve,             dtype=np.float64)
        bmc_arr = np.array(r.benchmark.equity_curve_compounding, dtype=np.float64)
        n_bm    = len(bm_arr)

        sample_idx     = np.linspace(0, n_bm - 1, n_trades + 1, dtype=int)
        sample_idx[-1] = n_bm - 1

        bm_y_sampled  = _r(bm_arr[sample_idx].tolist())
        bmc_y_sampled = _r(bmc_arr[sample_idx].tolist())

        # B&H x = exactly the same as strategy x (trade mode: ints, date mode: dates)
        # No separate bm_date_x — use eq_date_x for all three traces
        traces = [
            {"x": eq_trade_x, "y": _r(eq_trade_y), "type": "scatter", "mode": "lines",
             "name": r.strategy_name,
             "line": {"color": "#26a69a", "width": 2}},
            {"x": eq_trade_x, "y": bmc_y_sampled,  "type": "scatter", "mode": "lines",
             "name": "B&H Compounding",
             "line": {"color": "#ff9800", "width": 1.5, "dash": "dash"}},
            {"x": eq_trade_x, "y": bm_y_sampled,   "type": "scatter", "mode": "lines",
             "name": "B&H Fixed (1 contract)",
             "line": {"color": "#8b949e", "width": 1, "dash": "dot"}},
        ]
        layout = self._base_layout(
            xaxis={"gridcolor": "#21262d", "title": "Trade #"},
            yaxis={"gridcolor": "#21262d", "title": "Equity ($)", "tickformat": "$,.0f"},
            hovermode="x unified", margin={"l": 60, "r": 20, "t": 20, "b": 50},
        )

        # All three traces share the same x — toggle just swaps xaxis type + labels
        all_trade_x = json.dumps([eq_trade_x, eq_trade_x, eq_trade_x])
        all_date_x  = json.dumps([eq_date_x,  eq_date_x,  eq_date_x])

        # ── TradingView-style date ticks ──────────────────────────────────────
        # Pick ~8 evenly-spaced tick positions, label as "Jan '20" or just year
        # when zoomed out. Never rotate — keep horizontal.
        import datetime as _dt
        n_ticks     = min(8, max(4, n_trades // 80))
        tick_pos    = np.linspace(0, n_trades, n_ticks, dtype=int).tolist()

        def _fmt(ds):
            if ds == "start": return "Start"
            try:
                d = _dt.date.fromisoformat(ds)
                # Show just year for multi-year charts, month+year for shorter
                span_years = n_trades / max(252, 1)
                return d.strftime("%Y") if span_years > 3 else d.strftime("%b '%y")
            except Exception:
                return ds[:7]

        tick_vals = [eq_date_x[i] for i in tick_pos]
        tick_text = [_fmt(v) for v in tick_vals]

        date_xaxis_patch = json.dumps({
            "tickvals":  tick_vals,
            "ticktext":  tick_text,
            "tickangle": 0,
            "type":      "category",
            "title":     {"text": "Date", "font": {"size": 11}},
        })
        trade_xaxis_patch = json.dumps({
            "tickvals":  None,
            "ticktext":  None,
            "tickangle": 0,
            "type":      "-",
            "title":     {"text": "Trade #", "font": {"size": 11}},
        })

        controls = """
<div style='display:flex;gap:10px;margin-bottom:8px;align-items:center;flex-wrap:wrap'>
  <div style='display:flex;gap:4px'>
    <button id='_UID_ec_xbar'  onclick='_UID_ecToggleX("trade")'
      style='padding:4px 12px;border-radius:4px;border:1px solid #30363d;
             background:#21262d;color:#e6edf3;cursor:pointer;font-size:12px'>
      Trades
    </button>
    <button id='_UID_ec_xdate' onclick='_UID_ecToggleX("date")'
      style='padding:4px 12px;border-radius:4px;border:1px solid #30363d;
             background:#21262d;color:#e6edf3;cursor:pointer;font-size:12px'>
      Dates
    </button>
  </div>
  <div style='width:1px;height:20px;background:#30363d'></div>
  <button id='_UID_ec_bh_btn' onclick='_UID_ecToggleBH()'
    style='padding:4px 12px;border-radius:4px;border:1px solid #30363d;
           background:#21262d;color:#e6edf3;cursor:pointer;font-size:12px'>
    Hide B&amp;H
  </button>
</div>"""

        script = f"""
<script>
(function() {{
  var TRADE_X      = {all_trade_x};
  var DATE_X       = {all_date_x};
  var DATE_XPATCH  = {date_xaxis_patch};
  var TRADE_XPATCH = {trade_xaxis_patch};
  var bhVisible    = true;
  var xMode        = 'trade';

  function highlight(id, on) {{
    var el = document.getElementById(id);
    if (!el) return;
    el.style.background  = on ? '#388bfd' : '#21262d';
    el.style.borderColor = on ? '#388bfd' : '#30363d';
    el.style.color       = on ? '#ffffff' : '#e6edf3';
  }}

  window._UID_ecToggleX = function(mode) {{
    xMode = mode;
    var xData  = mode === 'date' ? DATE_X  : TRADE_X;
    var xPatch = mode === 'date' ? DATE_XPATCH : TRADE_XPATCH;
    Plotly.restyle('_UID_equity_chart', {{'x': xData}});
    Plotly.relayout('_UID_equity_chart', {{'xaxis.type':      xPatch.type,
                                            'xaxis.title':     xPatch.title,
                                            'xaxis.tickvals':  xPatch.tickvals,
                                            'xaxis.ticktext':  xPatch.ticktext,
                                            'xaxis.tickangle': xPatch.tickangle}});
    highlight('_UID_ec_xbar',  mode === 'trade');
    highlight('_UID_ec_xdate', mode === 'date');
  }};

  window._UID_ecToggleBH = function() {{
    bhVisible = !bhVisible;
    Plotly.restyle('_UID_equity_chart', {{'visible': bhVisible ? true : 'legendonly'}}, [1, 2]);
    var btn = document.getElementById('_UID_ec_bh_btn');
    if (btn) {{
      btn.textContent       = bhVisible ? 'Hide B&H' : 'Show B&H';
      btn.style.background  = bhVisible ? '#21262d' : '#388bfd';
      btn.style.borderColor = bhVisible ? '#30363d' : '#388bfd';
      btn.style.color       = bhVisible ? '#e6edf3' : '#ffffff';
    }}
  }};

  document.addEventListener('DOMContentLoaded', function() {{
    highlight('_UID_ec_xbar', true);
  }});
}})();
</script>"""

        return (
            f"<h2>Equity Curve</h2>"
            f"<div class='chart-box'>"
            f"{controls}"
            f"{self._chart('_UID_equity_chart', traces, layout)}"
            f"{script}"
            f"<p class='section-note'>Strategy equity plotted at trade resolution (one point per closed trade). "
            f"B&H Fixed: 1 contract held entire period. "
            f"B&H Compounding: scales contracts daily as equity grows.</p>"
            f"</div>"
        )

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
                f"{self._chart('_UID_dd_chart', traces, layout)}{stats}</div>")

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

        shuf = fan_chart("_UID_mc_shuffle",   mc.shuffle_percentiles,
                         "Sequence Shuffle (10,000 simulations)",
                         shuf_dd_p5, shuf_dd_p50, shuf_dd_p95,
                         stat_label="Max DD")
        boot = fan_chart("_UID_mc_bootstrap", mc.bootstrap_percentiles,
                         "Bootstrap Resample (10,000 simulations)",
                         mc.bootstrap_p5, mc.bootstrap_p50, mc.bootstrap_p95,
                         stat_label="Final equity")
        feq  = hist_chart("_UID_mc_feq",
                          mc.shuffle_final_equity,    "Shuffle",    "rgba(38,166,154,0.7)",
                          mc.bootstrap_final_equity,  "Bootstrap",  "rgba(239,83,80,0.7)",
                          "Final Equity Distribution", "Final Equity ($)", "$")
        fdd  = hist_chart("_UID_mc_fdd",
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
                f"{self._chart('_UID_mae_mfe_scatter', scatter_traces, scatter_layout)}"
                f"{stats}</div>"
                f"<div class='two-col'>"
                f"{hist_chart('_UID_mae_hist', mae, 'MAE Distribution', 'rgba(239,83,80,0.7)')}"
                f"{hist_chart('_UID_mfe_hist', mfe, 'MFE Distribution', 'rgba(38,166,154,0.7)')}"
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
                f"<div class='chart-box'>{self._chart('_UID_exit_bar', traces, layout)}</div>"
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
                f"{self._chart('_UID_hourly_chart', traces, layout)}"
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
                f"<div class='chart-box'>{self._chart('_UID_pnl_hist', pnl_traces, pnl_layout)}</div>"
                f"<div class='chart-box'>{self._chart('_UID_dur_hist', dur_traces, dur_layout)}</div>"
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

    def _section_trade_log(self, r: "Results", uid: str = "n", enable_inspector: bool = False) -> str:
        if not r.trade_log:
            return "<h2>Trade Log</h2><div class='chart-box'><p class='section-note'>No trades.</p></div>"

        # Collect unique exit reasons for the filter dropdown
        reasons = sorted({t["reason"] for t in r.trade_log})
        reason_opts = "<option value=''>All exits</option>" + "".join(
            f"<option value='{x}'>{x}</option>" for x in reasons
        )

        trades_json = _j(r.trade_log)
        # Prefix all element IDs with uid so normal + reversed don't clash
        p = f"tl{uid}"   # e.g. "tln" or "tlr"

        # Row opening tag — clickable with inspector link when enabled
        if enable_inspector:
            tr_open = (
                "\"<tr data-trade-idx='\" + (t.num-1) + "
                "\"' onclick='tcInspect(\" + (t.num-1) + \")'\" + "
                "\" style='cursor:pointer;transition:background 0.12s'>\""
            )
        else:
            tr_open = '"<tr>"'

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
    document.getElementById("{p}-stats").textContent =
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
      html += {tr_open} +
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
    document.getElementById("{p}-tbody").innerHTML = html || "<tr><td colspan='12' style='text-align:center;color:#8b949e;padding:24px'>No trades match the current filters.</td></tr>";

    // Column headers with sort indicators
    var hdr = "";
    cols.forEach(function(c){{
      var cls = sortCol===c.key ? (sortDir===1?"sort-asc":"sort-desc") : "";
      hdr += "<th class='" + cls + "' data-col='" + c.key + "' style='text-align:" + c.align + "'>" +
             c.label + "<i class='sort-icon'></i></th>";
    }});
    document.getElementById("{p}-thead").innerHTML = hdr;

    // Re-bind sort listeners
    document.querySelectorAll("#{p}-table th").forEach(function(th){{
      th.onclick = function(){{
        var k = th.dataset.col;
        if(sortCol===k) sortDir *= -1; else {{ sortCol=k; sortDir=1; }}
        cur=1; render();
      }};
    }});

    // Pagination
    var pgHtml = "<button class='tl-page-btn' id='{p}-prev' " + (cur<=1?"disabled":"") + ">&lsaquo; Prev</button>";
    var WING = 2;
    var pnums = [];
    for(var p2=1;p2<=pages;p2++){{
      if(p2===1||p2===pages||Math.abs(p2-cur)<=WING) pnums.push(p2);
      else if(pnums[pnums.length-1]!=="…") pnums.push("…");
    }}
    pnums.forEach(function(p2){{
      if(p2==="…"){{ pgHtml += "<span class='tl-page-info'>…</span>"; return; }}
      pgHtml += "<button class='tl-page-btn" + (p2===cur?" active":"") + "' data-pg='" + p2 + "'>" + p2 + "</button>";
    }});
    pgHtml += "<button class='tl-page-btn' id='{p}-next' " + (cur>=pages?"disabled":"") + ">Next &rsaquo;</button>";
    pgHtml += "<span class='tl-page-info'>" + (total?start+1:0) + "–" + end + " of " + total + "</span>";
    document.getElementById("{p}-pages").innerHTML = pgHtml;

    document.getElementById("{p}-prev") && (document.getElementById("{p}-prev").onclick = function(){{if(cur>1){{cur--;render();}}}});
    document.getElementById("{p}-next") && (document.getElementById("{p}-next").onclick = function(){{if(cur<pages){{cur++;render();}}}});
    document.querySelectorAll("#{p}-pages .tl-page-btn[data-pg]").forEach(function(b){{
      b.onclick = function(){{cur=parseInt(b.dataset.pg);render();}};
    }});
  }}

  // Wire controls
  document.getElementById("{p}-search").oninput = function(){{filterText=this.value;cur=1;render();}};
  document.getElementById("{p}-exit-filter").onchange = function(){{filterExit=this.value;cur=1;render();}};
  document.getElementById("{p}-result-win").onclick  = function(){{filterResult=filterResult==="win"?"":"win";this.classList.toggle("active",filterResult==="win");document.getElementById("{p}-result-loss").classList.remove("active");cur=1;render();}};
  document.getElementById("{p}-result-loss").onclick = function(){{filterResult=filterResult==="loss"?"":"loss";this.classList.toggle("active",filterResult==="loss");document.getElementById("{p}-result-win").classList.remove("active");cur=1;render();}};
  document.getElementById("{p}-reset").onclick = function(){{
    filterText="";filterExit="";filterResult="";sortCol="num";sortDir=1;cur=1;
    document.getElementById("{p}-search").value="";
    document.getElementById("{p}-exit-filter").value="";
    document.getElementById("{p}-result-win").classList.remove("active");
    document.getElementById("{p}-result-loss").classList.remove("active");
    render();
  }};

  render();
}})();
</script>"""

        return f"""
<h2>Trade Log</h2>
<div class='chart-box'>
  <div class='tl-controls'>
    <input id='{p}-search' type='text' placeholder='Search date, price, reason…'>
    <div>
      <label>Exit&nbsp;</label>
      <select id='{p}-exit-filter'>{reason_opts}</select>
    </div>
    <button class='tl-btn' id='{p}-result-win'>Winners only</button>
    <button class='tl-btn' id='{p}-result-loss'>Losers only</button>
    <button class='tl-btn' id='{p}-reset'>Reset</button>
    <span class='tl-stats' id='{p}-stats'></span>
  </div>
  <table id='{p}-table' class='tl-table'>
    <thead id='{p}-thead'></thead>
    <tbody id='{p}-tbody'></tbody>
  </table>
  <div class='tl-pagination' id='{p}-pages'></div>
</div>
{js}"""