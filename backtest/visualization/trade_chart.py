"""
trade_chart.py — TradingView-style position visualization for individual trades.

Generates a side-by-side candlestick chart showing one winning trade and one
losing trade with the TradingView position tool overlay:
  • Entry / SL / exit horizontal lines
  • Risk zone (entry → SL, red fill)
  • Outcome zone (entry → exit price, teal or red fill)
  • Entry triangle marker, exit circle marker
  • Price labels on the right edge
"""
from __future__ import annotations

import os
import webbrowser

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Bars shown before entry and after exit in each chart
_PADDING = 25


# ---------------------------------------------------------------------------
# Trade selection
# ---------------------------------------------------------------------------

def _pick_trades(trades):
    """
    Return (winner, loser) — one representative trade of each outcome.

    Prefers trades that have an SL set (so the risk zone renders).
    Picks from the upper third of winners (decent win) and lower third of
    losers (decent loss) to avoid extreme outliers dominating the y-axis.
    """
    with_sl = [t for t in trades if t.sl_price is not None]
    pool    = with_sl if with_sl else trades

    winners = sorted([t for t in pool if     t.is_winner], key=lambda t:  t.net_pnl_dollars, reverse=True)
    losers  = sorted([t for t in pool if not t.is_winner], key=lambda t:  t.net_pnl_dollars)

    winner = winners[len(winners) // 3] if winners else None
    loser  = losers [len(losers)  // 3] if losers  else None
    return winner, loser


# ---------------------------------------------------------------------------
# Data slicing
# ---------------------------------------------------------------------------

def _get_slice(data, trade) -> tuple[pd.DataFrame, int]:
    """Return (df_slice, offset) where offset is the start bar index in data."""
    n     = len(data.df_1m)
    start = max(0, trade.entry_bar - _PADDING)
    end   = min(n - 1, trade.exit_bar + _PADDING)
    return data.df_1m.iloc[start : end + 1], start


# ---------------------------------------------------------------------------
# Position tool overlay
# ---------------------------------------------------------------------------

def _add_overlay(fig, row: int, col: int, df: pd.DataFrame, offset: int, trade) -> None:
    """
    Add TradingView position tool overlay to a subplot:
      1. Red risk zone      (entry ↔ SL, bounded to entry/exit bars)
      2. Outcome zone       (entry ↔ exit_price, teal=win, red=loss)
      3. Entry dotted line  (white/gray, full width)
      4. SL dotted line     (red, full width)
      5. Exit dotted line   (outcome colour, full width)
      6. Entry triangle marker
      7. Exit circle marker
      8. Price labels on right edge
    """
    entry_local = trade.entry_bar - offset
    exit_local  = trade.exit_bar  - offset

    entry_ts = df.index[entry_local]
    exit_ts  = df.index[exit_local]
    x0       = df.index[0]
    x1       = df.index[-1]

    entry   = trade.entry_price
    sl      = trade.sl_price
    exit_p  = trade.exit_price
    is_long = trade.direction == 1
    is_win  = trade.is_winner

    teal = "#26a69a"
    red  = "#ef5350"
    outcome_color = teal if is_win else red

    # ── Shaded zones ──────────────────────────────────────────────────────────

    # Risk zone: entry ↔ SL
    if sl is not None:
        fig.add_shape(
            type="rect",
            x0=entry_ts, x1=exit_ts,
            y0=min(entry, sl), y1=max(entry, sl),
            fillcolor="rgba(239,83,80,0.12)",
            line_width=0,
            row=row, col=col,
        )

    # Outcome zone: entry ↔ exit price
    fig.add_shape(
        type="rect",
        x0=entry_ts, x1=exit_ts,
        y0=min(entry, exit_p), y1=max(entry, exit_p),
        fillcolor="rgba(38,166,154,0.15)" if is_win else "rgba(239,83,80,0.15)",
        line_width=0,
        row=row, col=col,
    )

    # ── Horizontal price lines ────────────────────────────────────────────────

    fig.add_shape(
        type="line", x0=x0, x1=x1, y0=entry, y1=entry,
        line=dict(color="rgba(200,200,200,0.65)", width=1, dash="dot"),
        row=row, col=col,
    )
    if sl is not None:
        fig.add_shape(
            type="line", x0=x0, x1=x1, y0=sl, y1=sl,
            line=dict(color="rgba(239,83,80,0.80)", width=1, dash="dot"),
            row=row, col=col,
        )
    exit_color_rgba = "rgba(38,166,154,0.73)" if is_win else "rgba(239,83,80,0.73)"
    fig.add_shape(
        type="line", x0=x0, x1=x1, y0=exit_p, y1=exit_p,
        line=dict(color=exit_color_rgba, width=1, dash="dot"),
        row=row, col=col,
    )

    # ── Price labels (right edge) ─────────────────────────────────────────────

    fig.add_annotation(
        x=x1, y=entry,
        text=f" Entry  {entry:.2f}",
        font=dict(size=10, color="rgba(210,210,210,0.9)"),
        showarrow=False, xanchor="left",
        bgcolor="rgba(19,23,34,0.80)",
        row=row, col=col,
    )
    if sl is not None:
        fig.add_annotation(
            x=x1, y=sl,
            text=f" SL  {sl:.2f}",
            font=dict(size=10, color=red),
            showarrow=False, xanchor="left",
            bgcolor="rgba(19,23,34,0.80)",
            row=row, col=col,
        )
    pnl_sign = "+" if trade.net_pnl_dollars >= 0 else ""
    fig.add_annotation(
        x=x1, y=exit_p,
        text=f" Exit  {exit_p:.2f}  ({pnl_sign}${trade.net_pnl_dollars:,.0f})",
        font=dict(size=10, color=outcome_color),
        showarrow=False, xanchor="left",
        bgcolor="rgba(19,23,34,0.80)",
        row=row, col=col,
    )

    # ── Entry marker (triangle) ───────────────────────────────────────────────

    fig.add_trace(
        go.Scatter(
            x=[entry_ts], y=[entry],
            mode="markers",
            marker=dict(
                symbol="triangle-up" if is_long else "triangle-down",
                size=16, color="#ffffff",
                line=dict(color="#aaaaaa", width=1),
            ),
            showlegend=False,
            hovertemplate=f"Entry {'Long' if is_long else 'Short'}: {entry:.2f}<extra></extra>",
        ),
        row=row, col=col,
    )

    # ── Exit marker (circle) ──────────────────────────────────────────────────

    fig.add_trace(
        go.Scatter(
            x=[exit_ts], y=[exit_p],
            mode="markers",
            marker=dict(
                symbol="circle",
                size=12, color=outcome_color,
                line=dict(color=outcome_color, width=2),
            ),
            showlegend=False,
            hovertemplate=(
                f"Exit ({trade.exit_reason.name}): {exit_p:.2f}<br>"
                f"PnL: {pnl_sign}${trade.net_pnl_dollars:,.0f}<extra></extra>"
            ),
        ),
        row=row, col=col,
    )


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def plot_trade_examples(
    result,
    data,
    output_path: str = "trade_examples.html",
    auto_open: bool = True,
) -> None:
    """
    Generate a TradingView-style candlestick chart showing one winning trade
    and one losing trade with position tool overlay.

    Args:
        result:      RunResult from run_backtest()
        data:        MarketData used in the same run
        output_path: HTML file path to write
        auto_open:   Open in browser on completion
    """
    winner, loser = _pick_trades(result.trades)
    pairs = [(t, lbl) for t, lbl in [(winner, "Winner"), (loser, "Loser")] if t is not None]

    if not pairs:
        print("  No trades available to plot.")
        return

    def _subtitle(trade, label: str) -> str:
        direction = "Long" if trade.direction == 1 else "Short"
        sign      = "+" if trade.net_pnl_dollars >= 0 else ""
        ts        = data.df_1m.index[trade.entry_bar].strftime("%Y-%m-%d  %H:%M")
        icon      = "▲" if trade.is_winner else "▼"
        return (
            f"{icon} {label}  ·  {direction}  ·  "
            f"{sign}${trade.net_pnl_dollars:,.0f}  ·  "
            f"{trade.exit_reason.name}  ·  entry {ts}"
        )

    n   = len(pairs)
    fig = make_subplots(
        rows=1, cols=n,
        shared_xaxes=False,
        horizontal_spacing=0.10,
        subplot_titles=[_subtitle(t, lbl) for t, lbl in pairs],
    )

    for col_idx, (trade, lbl) in enumerate(pairs, start=1):
        df_slice, offset = _get_slice(data, trade)

        # Candlesticks
        fig.add_trace(
            go.Candlestick(
                x=df_slice.index,
                open=df_slice["open"],
                high=df_slice["high"],
                low=df_slice["low"],
                close=df_slice["close"],
                increasing_line_color="#26a69a",
                decreasing_line_color="#ef5350",
                increasing_fillcolor="#26a69a",
                decreasing_fillcolor="#ef5350",
                name="",
                showlegend=False,
                hoverinfo="x+y",
            ),
            row=1, col=col_idx,
        )

        _add_overlay(fig, 1, col_idx, df_slice, offset, trade)

    # ── Axis styling ──────────────────────────────────────────────────────────

    for i in range(1, n + 1):
        s = "" if i == 1 else str(i)
        fig.update_layout(**{
            f"xaxis{s}": dict(
                type="date",
                rangeslider=dict(visible=False),
                showgrid=True,
                gridcolor="rgba(255,255,255,0.05)",
                tickfont=dict(size=9),
            ),
            f"yaxis{s}": dict(
                showgrid=True,
                gridcolor="rgba(255,255,255,0.05)",
                tickfont=dict(size=9),
                side="right",
                tickformat=".2f",
            ),
        })

    # ── Layout ────────────────────────────────────────────────────────────────

    fig.update_layout(
        title=dict(
            text=f"Trade Examples  ·  {result.strategy_name}",
            font=dict(size=16, color="#e0e0e0"),
            x=0.5,
        ),
        paper_bgcolor="#131722",
        plot_bgcolor="#131722",
        font=dict(color="#d0d0d0", family="monospace"),
        height=620,
        margin=dict(l=20, r=160, t=90, b=50),
    )

    # Style the subplot title annotations (plotly puts them as annotations)
    for ann in fig.layout.annotations:
        ann.font = dict(size=12, color="#b0b0b0")

    fig.write_html(output_path, include_plotlyjs="cdn")
    print(f"  Trade chart → {output_path}")

    if auto_open:
        webbrowser.open(f"file:///{os.path.abspath(output_path)}")
