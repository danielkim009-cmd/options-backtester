"""
Self-contained HTML report generator for the Bull Put Spread backtester.
"""
from __future__ import annotations

import base64
from datetime import datetime
from typing import Dict, List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _iframe(html: str, height: int) -> str:
    """Embed an HTML string as a base64 data-URI iframe."""
    encoded = base64.b64encode(html.encode("utf-8")).decode("ascii")
    return (
        f'<iframe src="data:text/html;base64,{encoded}" '
        f'style="width:100%;height:{height}px;border:0;display:block;"></iframe>'
    )


def _kpi_card(label: str, value: str, color: str = "#d1d4dc") -> str:
    return (
        f'<div class="kpi-card">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value" style="color:{color}">{value}</div>'
        f'</div>'
    )


def _kpi_row_html(items: list) -> str:
    cards = "".join(_kpi_card(label, val, color) for label, val, color in items)
    return f'<div class="kpi-row">{cards}</div>'


def _color_for(val_str: str) -> str:
    """Return a CSS color string based on the sign of a numeric value."""
    num_str = str(val_str).replace("$", "").replace(",", "").replace("%", "")
    try:
        num = float(num_str)
        if num > 0:
            return "color:#26a69a"
        elif num < 0:
            return "color:#ef5350"
    except ValueError:
        pass
    return ""


def _html_table(df: pd.DataFrame, color_cols: list | None = None) -> str:
    """Render a DataFrame as a styled dark HTML table."""
    color_cols = color_cols or []
    rows = ['<table><thead><tr>']
    for col in df.columns:
        rows.append(f"<th>{col}</th>")
    rows.append("</tr></thead><tbody>")
    for _, row in df.iterrows():
        rows.append("<tr>")
        for col in df.columns:
            val = row[col]
            style = _color_for(str(val)) if col in color_cols else ""
            if style:
                rows.append(f'<td style="{style}">{val}</td>')
            else:
                rows.append(f"<td>{val}</td>")
        rows.append("</tr>")
    rows.append("</tbody></table>")
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Main report function
# ---------------------------------------------------------------------------

def generate_html_report(
    *,
    ticker: str,
    name: str,
    config,
    stats: dict,
    trades_df: pd.DataFrame,
    annual_df: pd.DataFrame,
    starting_capital: float,
    strat_final: float,
    strat_cagr: float,
    bh_cagr: float,
    bh_final: float,
    strat_max_dd_usd: float,
    strat_max_dd_pct: float,
    bh_max_dd_pct: float,
    max_loss_usd: float,
    max_loss_pct: float,
    pnls: list,
    exit_reasons: dict,
    eq_html: str,
    tv_html: str,
    cum_html: str,
) -> str:
    """Build and return a self-contained HTML report string."""

    # --- Dark-themed Plotly charts ---
    _dark = dict(
        template="plotly_dark",
        paper_bgcolor="#131722",
        plot_bgcolor="#1e222d",
        font_color="#d1d4dc",
        margin=dict(t=50, b=40, l=40, r=20),
        height=340,
    )
    bar_colors = ["#26a69a" if p > 0 else "#ef5350" for p in pnls]
    fig_bar = go.Figure(go.Bar(
        x=list(range(1, len(pnls) + 1)),
        y=pnls,
        marker_color=bar_colors,
        marker_line_width=0,
    ))
    fig_bar.add_hline(y=0, line_color="#444", line_width=0.8)
    fig_bar.update_layout(title="Per-Trade P&L ($)", xaxis_title="Trade #", yaxis_title="P&L ($)", **_dark)
    bar_html = fig_bar.to_html(include_plotlyjs=False, full_html=False)

    fig_pie = px.pie(
        names=list(exit_reasons.keys()),
        values=list(exit_reasons.values()),
        title="Exit Reasons",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig_pie.update_layout(height=320, margin=dict(t=50, b=10), **{k: v for k, v in _dark.items() if k not in ("height", "margin")})
    pie_html = fig_pie.to_html(include_plotlyjs=False, full_html=False)

    # --- Strategy description ---
    ma = config.ma_type
    ef = config.ema_fast
    es = config.ema_slow
    ep = config.entry_price.capitalize()
    exit_parts = [
        f"profit target ≥ {int(config.profit_target * 100)}% of credit",
        f"DTE ≤ {config.exit_dte}",
    ]
    if config.exit_below_ema:
        exit_parts.append(f"close below {ma}{config.exit_ema_period}")
    if config.use_stop_loss:
        exit_parts.append(f"stop-loss at {config.stop_loss_multiple:.1f}× credit received")

    # --- KPI rows ---
    pos = "#26a69a"
    neg = "#ef5350"
    bh_c = "#4c78a8"
    kpi1 = [
        ("Total Trades",         str(stats["total_trades"]),            "#d1d4dc"),
        ("Strategy P&L",         f"${stats['total_pnl']:,.0f}",        pos if stats["total_pnl"] >= 0 else neg),
        ("Strategy Acct Value",  f"${strat_final:,.0f}",               "#d1d4dc"),
        ("Strategy CAGR",        f"{strat_cagr:.1%}",                  pos if strat_cagr >= 0 else neg),
        ("Strategy Max DD $",    f"${strat_max_dd_usd:,.0f}",          neg),
        ("Strategy Max DD %",    f"{strat_max_dd_pct:.1%}",            neg),
    ]
    kpi2 = [
        ("Win Rate",             f"{stats['win_rate']:.1f}%",          pos if stats["win_rate"] >= 50 else neg),
        ("Avg Win $",            f"${stats['avg_win']:,.0f}",          pos),
        ("Avg Loss $",           f"${stats['avg_loss']:,.0f}",         neg),
        ("Avg Win %",            f"{stats['avg_win_pct']:.2f}%",       pos),
        ("Avg Loss %",           f"{stats['avg_loss_pct']:.2f}%",      neg),
        ("Buy & Hold CAGR",      f"{bh_cagr:.1%}",                     bh_c),
    ]
    kpi3 = [
        ("Buy & Hold Acct Value",f"${bh_final:,.0f}",                  bh_c),
        ("Buy & Hold Max DD %",  f"{bh_max_dd_pct:.1%}",               neg),
        ("Max Loss $",           f"${max_loss_usd:,.0f}",              neg),
        ("Max Loss %",           f"{max_loss_pct:.1%}",                neg),
    ]

    # --- Summary table ---
    summary_df = pd.DataFrame({
        "Metric": [
            "Total Trades", "Winning Trades", "Losing Trades",
            "Total P&L", "Win Rate", "Avg P&L / Trade",
            "Avg Winning Trade", "Avg Losing Trade",
            "Avg Win %", "Avg Loss %", "Profit Factor", "Sharpe Ratio",
            "Max DD $", "Max DD %", "Max Loss $", "Max Loss %", "Avg Trade Duration",
        ],
        "Value": [
            stats["total_trades"], stats["winning_trades"], stats["losing_trades"],
            f"${stats['total_pnl']:,.0f}", f"{stats['win_rate']:.1f}%",
            f"${stats['avg_pnl_per_trade']:.0f}",
            f"${stats['avg_win']:.0f}", f"${stats['avg_loss']:.0f}",
            f"{stats['avg_win_pct']:.2f}%", f"{stats['avg_loss_pct']:.2f}%",
            f"{stats['profit_factor']:.2f}", f"{stats['sharpe']:.2f}",
            f"${strat_max_dd_usd:,.0f}", f"{strat_max_dd_pct:.1%}",
            f"${max_loss_usd:,.0f}", f"{max_loss_pct:.1%}",
            f"{stats['avg_duration_days']:.0f} days",
        ],
    })
    summary_table = _html_table(summary_df)

    bh_col = f"B&H {ticker} P&L %"
    annual_color_cols = ["Annual P&L ($)", "P&L %", bh_col, "Alpha %", "Avg Win %", "Avg Loss %"]
    annual_table = _html_table(annual_df, color_cols=annual_color_cols)

    trade_table = _html_table(
        trades_df.copy().assign(**{"P&L ($)": trades_df["P&L ($)"].apply(lambda v: f"${v:,.2f}"),
                                   "P&L (% Acct)": trades_df["P&L (% Acct)"].apply(lambda v: f"{v:.2f}%")}),
        color_cols=["P&L ($)", "P&L (% Acct)"],
    )

    gen_time = datetime.now().strftime("%Y-%m-%d %H:%M")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Backtest Report — {ticker}</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
  font-family: -apple-system, BlinkMacSystemFont, 'Trebuchet MS', sans-serif;
  background: #131722; color: #d1d4dc; line-height: 1.5; padding: 24px 32px;
}}
h1 {{ font-size: 1.55rem; font-weight: 700; margin-bottom: 4px; }}
h2 {{
  font-size: 1rem; font-weight: 600; margin: 28px 0 10px;
  color: #e0e0e0; border-bottom: 1px solid #2a2e39; padding-bottom: 6px;
}}
.caption {{ font-size: 0.78rem; color: #777; margin-bottom: 8px; }}
.strategy-desc {{ font-size: 0.82rem; color: #aaa; margin: 10px 0 18px; line-height: 1.9; }}
.strategy-desc strong {{ color: #d1d4dc; }}

/* KPI cards */
.kpi-row {{ display: flex; flex-wrap: wrap; gap: 10px; margin: 10px 0; }}
.kpi-card {{
  background: #1e222d; border: 1px solid #2a2e39; border-radius: 6px;
  padding: 11px 16px; flex: 1 1 140px; min-width: 120px;
}}
.kpi-label {{ font-size: 0.7rem; color: #777; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 3px; }}
.kpi-value {{ font-size: 1.15rem; font-weight: 700; }}

/* Charts */
.chart-wrap {{ margin: 12px 0; border-radius: 4px; overflow: hidden; }}
.chart-row {{ display: flex; gap: 16px; align-items: flex-start; margin: 12px 0; }}
.chart-row > .chart-wrap {{ flex: 2; }}
.chart-row > .chart-plotly {{ flex: 1; min-width: 0; }}

/* Tables */
.table-wrap {{ overflow-x: auto; margin: 8px 0; }}
table {{ width: 100%; border-collapse: collapse; font-size: 0.82rem; }}
thead {{ background: #1e222d; }}
th {{
  padding: 8px 10px; text-align: left; font-size: 0.7rem;
  text-transform: uppercase; letter-spacing: 0.04em; color: #888;
  border-bottom: 1px solid #2a2e39; white-space: nowrap;
}}
td {{ padding: 6px 10px; border-bottom: 1px solid #1e222d; white-space: nowrap; }}
tr:hover td {{ background: #1a1e2b; }}

.footer {{ margin-top: 40px; font-size: 0.72rem; color: #444; text-align: center; }}
</style>
</head>
<body>

<h1>📉 Bull Put Spread Backtester — {name} ({ticker})</h1>
<p class="caption">Backtest period: <strong>{config.start_date}</strong> — <strong>{config.end_date}</strong> &nbsp;|&nbsp; Generated: {gen_time}</p>

<div class="strategy-desc">
  <strong>Entry A:</strong> low ≤ {ma}{ef} &nbsp;•&nbsp; {ep} &gt; {ma}{es} &nbsp;•&nbsp;
  {ma}{ef} &gt; {ma}{es} &nbsp;•&nbsp; {ma}{es} &gt; {ma}100 &nbsp;•&nbsp;
  {ma}100 &gt; {ma}200 &nbsp;•&nbsp; {ma}{ef} ≥ {ma}{ef}[7d/14d/21d ago] &nbsp;•&nbsp;
  {ma}{ef} &gt; {ma}{ef}[30d ago] <em>(first day of pullback zone)</em><br>
  <strong>Entry B ({ep}):</strong> close crosses above {ma}200 &nbsp;•&nbsp; {ma}100 &gt; {ma}200 &nbsp;•&nbsp; high &gt; {ma}{ef}<br>
  <strong>Exit:</strong> {" &nbsp;•&nbsp; ".join(exit_parts)}
</div>

<h2>Performance Metrics</h2>
{_kpi_row_html(kpi1)}
{_kpi_row_html(kpi2)}
{_kpi_row_html(kpi3)}

<h2>Account Value Growth</h2>
<div class="chart-wrap">{_iframe(eq_html, 430)}</div>

<h2>Price Chart — Entry / Exit Signals</h2>
<div class="chart-wrap">{_iframe(tv_html, 540)}</div>

<h2>Cumulative P&amp;L &amp; Per-Trade Results</h2>
<div class="chart-row">
  <div class="chart-wrap">{_iframe(cum_html, 370)}</div>
  <div class="chart-plotly">{bar_html}</div>
</div>

<h2>Exit Reasons</h2>
<div style="max-width:480px">{pie_html}</div>

<h2>Summary Statistics</h2>
<div class="table-wrap">{summary_table}</div>

<h2>Annual P&amp;L</h2>
<div class="table-wrap">{annual_table}</div>

<h2>Trade Log</h2>
<p class="caption">{len(trades_df)} trades</p>
<div class="table-wrap">{trade_table}</div>

<div class="footer">Bull Put Spread Backtester &nbsp;|&nbsp; Report generated {gen_time}</div>
</body>
</html>"""
