"""
Streamlit web dashboard for the Bull Put Spread backtester.

Run with:
    streamlit run app.py
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components
from plotly.subplots import make_subplots

# Ensure project root is on sys.path when launched from any directory
sys.path.insert(0, str(Path(__file__).parent))

from config import BacktestConfig
from backtester.data_fetcher import fetch_data
from backtester.engine import run_backtest
from backtester.analytics import compute_stats, trades_to_dataframe


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Bull Put Spread Backtester",
    page_icon="📉",
    layout="wide",
)

st.title("📉 Bull Put Spread Backtester")
st.caption(
    "Entry: low < EMA21, low > EMA50, EMA21 > EMA50, EMA50 > EMA100 (first day of pullback zone)  •  "
    "Exit: profit target **or** DTE stop **or** price below EMA200 (optional)  •  "
    "IV proxy: VIX (SPY) / 30-day HV×1.1 (individual stocks)"
)

# ---------------------------------------------------------------------------
# Sidebar — configuration
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Strategy Parameters")

    ticker = st.text_input("Ticker", value="SPY").upper().strip()

    col1, col2 = st.columns(2)
    start_date = col1.date_input("Start Date", value=pd.Timestamp("2006-04-03"))
    end_date   = col2.date_input("End Date",   value=pd.Timestamp("2026-03-31"))

    st.divider()
    ma_type     = st.radio("Moving Average Type", ["EMA", "SMA"], horizontal=True)
    entry_price = st.radio("Entry Signal Price", ["Close", "Low"], horizontal=True,
                           help="'Close' triggers on a closing price pullback; 'Low' triggers when the intraday low touches the zone.")

    st.divider()
    st.subheader("Options")

    spread_width = st.number_input(
        "Spread Width ($)",
        min_value=1.0, max_value=50.0, value=5.0, step=1.0,
        help="Dollar distance between short and long put strikes.",
    )
    entry_dte = st.slider("Entry DTE (target)", 30, 90, 45)
    exit_dte  = st.slider("Exit DTE (time-stop)", 1, 30, 7)
    exit_below_ema200 = st.checkbox(
        "Exit if price crosses below EMA200",
        value=True,
        help="Close the trade on any day the underlying closes below the 200-day EMA.",
    )
    use_stop_loss = st.checkbox(
        "Exit at stop-loss",
        value=False,
        help="Close the trade when the unrealized loss exceeds a multiple of the credit received.",
    )
    stop_loss_multiple = st.number_input(
        "Stop-loss (× credit received)",
        min_value=0.5, max_value=10.0, value=2.0, step=0.5,
        disabled=not use_stop_loss,
        help="Exit when loss = this multiple × net credit. 2.0 = 200% of credit received.",
    )
    profit_target_pct = st.slider("Profit Target (%)", 25, 90, 50, step=5,
                               help="Close when spread can be bought back for this % of the original credit received.")
    profit_target = profit_target_pct / 100
    short_put_delta = st.slider("Short Put Delta", 0.10, 0.50, 0.40, step=0.05)

    st.divider()
    st.subheader("Capital")
    starting_capital = st.number_input(
        "Starting Capital ($)", min_value=1_000, max_value=10_000_000,
        value=10_000, step=1_000,
    )
    deploy_pct = st.slider(
        "Capital Deployed per Trade (%)", 5, 100, 50, step=5,
        help=(
            "Percentage of current account value to put at risk on each new trade. "
            "Contracts are sized so that max-loss ≤ this amount."
        ),
    )

    st.divider()
    st.subheader("Costs")
    commission = st.number_input("Commission / leg / contract ($)", 0.0, 5.0, 0.65, step=0.05)

    run_btn = st.button("▶  Run Backtest", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Run backtest
# ---------------------------------------------------------------------------

if run_btn:
    config = BacktestConfig(
        ticker=ticker,
        start_date=str(start_date),
        end_date=str(end_date),
        ma_type=ma_type,
        entry_price=entry_price.lower(),
        entry_dte=entry_dte,
        exit_dte=exit_dte,
        profit_target=profit_target,
        short_put_delta=short_put_delta,
        spread_width=spread_width,
        starting_capital=float(starting_capital),
        deploy_pct=deploy_pct / 100,
        commission_per_contract=commission,
        exit_below_ema200=exit_below_ema200,
        use_stop_loss=use_stop_loss,
        stop_loss_multiple=stop_loss_multiple,
    )

    # --- Fetch data ---
    with st.spinner(f"Fetching data for {ticker}…"):
        try:
            df = fetch_data(ticker, str(start_date), str(end_date), 21, 50)
        except Exception as exc:
            st.error(f"Data fetch failed for {ticker}: {exc}")
            st.stop()

    # --- Run backtest ---
    with st.spinner("Running backtest…"):
        trades, equity_curve = run_backtest(df, config)

    # --- Company name ---
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info
        name = info.get("longName") or info.get("shortName") or ticker
    except Exception:
        name = ticker

    st.markdown(f"## {name} ({ticker})")

    if not trades:
        st.warning("No trades were generated. Try a wider date range or different parameters.")
        st.stop()

    stats     = compute_stats(trades)
    trades_df = trades_to_dataframe(trades)

    # -----------------------------------------------------------------------
    # KPI metrics row
    # -----------------------------------------------------------------------

    first_entry  = pd.Timestamp(min(t.entry_date for t in trades))
    last_exit    = pd.Timestamp(max(t.exit_date  for t in trades))
    years        = stats["years"]
    total_capital = float(starting_capital)

    # --- Buy & Hold ---
    idx0     = df.index.searchsorted(first_entry)
    idx1     = min(df.index.searchsorted(last_exit), len(df) - 1)
    entry_px = float(df.iloc[idx0]["close"])
    exit_px  = float(df.iloc[idx1]["close"])
    bh_shares = math.floor(total_capital / entry_px)
    bh_cash   = total_capital - bh_shares * entry_px
    bh_final  = exit_px * bh_shares + bh_cash
    bh_equity = df["close"].loc[first_entry:] * bh_shares + bh_cash

    strat_final = equity_curve.iloc[-1]
    strat_cagr  = (strat_final / total_capital) ** (1 / years) - 1 if years > 0 else 0.0
    bh_cagr     = (bh_final    / total_capital) ** (1 / years) - 1 if years > 0 else 0.0

    # Strategy max drawdown — equity curve peak-to-trough
    eq_arr           = equity_curve.values
    eq_peak          = np.maximum.accumulate(eq_arr)
    strat_dd_arr     = eq_arr - eq_peak
    strat_max_dd_usd = float(np.min(strat_dd_arr))
    strat_max_dd_pct = float(np.min(strat_dd_arr / eq_peak))

    # Strategy max loss — worst single trade
    max_loss_usd = min(t.realized_pnl for t in trades)
    max_loss_pct = min(
        (t.realized_pnl / t.account_value_at_entry)
        for t in trades if t.account_value_at_entry
    )

    # B&H max drawdown
    bh_eq_arr     = bh_equity.values
    bh_peak       = np.maximum.accumulate(bh_eq_arr)
    bh_max_dd_pct = float(np.min((bh_eq_arr - bh_peak) / bh_peak))

    # IV source notice
    from backtester.data_fetcher import CBOE_VOL_INDEX, HV_PREMIUM, DEFAULT_HV_PREMIUM
    iv_src = df["iv_source"].iloc[-1]
    if iv_src == "cboe":
        cboe_sym = CBOE_VOL_INDEX.get(ticker.upper(), "")
        st.success(f"**{ticker} IV source:** CBOE volatility index `{cboe_sym}` — high accuracy.")
    else:
        premium = HV_PREMIUM.get(ticker.upper(), DEFAULT_HV_PREMIUM)
        st.warning(
            f"**{ticker} IV source:** Calibrated HV (30-day EWMA × {premium:.2f}×) — "
            f"approximate. Consider OptionsDX or ORATS for better accuracy."
        )

    st.divider()

    # Row 1 — strategy
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Trades",         stats["total_trades"])
    c2.metric("Strategy P&L",        f"${stats['total_pnl']:,.0f}")
    c3.metric("Strategy Acct Value", f"${strat_final:,.0f}")
    c4.metric("Strategy CAGR",       f"{strat_cagr:.1%}")
    c5.metric("Strategy Max DD $",   f"${strat_max_dd_usd:,.0f}")
    c6.metric("Strategy Max DD %",   f"{strat_max_dd_pct:.1%}")

    st.write("")

    # Row 2 — trade quality + buy & hold
    c7, c8, c9, c10, c11, c12 = st.columns(6)
    c7.metric("Win Rate",                f"{stats['win_rate']:.1f}%")
    c8.metric("Avg Win $",              f"${stats['avg_win']:,.0f}")
    c9.metric("Avg Loss $",             f"${stats['avg_loss']:,.0f}")
    c10.metric("Avg Win %",             f"{stats['avg_win_pct']:.2f}%")
    c11.metric("Avg Loss %",            f"{stats['avg_loss_pct']:.2f}%")
    c12.metric("Buy & Hold CAGR",       f"{bh_cagr:.1%}")

    st.write("")

    # Row 2b — buy & hold + max loss
    c7b, c8b, c9b, c10b, c11b, c12b = st.columns(6)
    c7b.metric("Buy & Hold Acct Value", f"${bh_final:,.0f}")
    c8b.metric("Buy & Hold Max DD %",   f"{bh_max_dd_pct:.1%}")
    c9b.metric("Max Loss $",            f"${max_loss_usd:,.0f}")
    c10b.metric("Max Loss %",           f"{max_loss_pct:.1%}")

    # -----------------------------------------------------------------------
    # Charts
    # -----------------------------------------------------------------------

    exit_dates  = [pd.Timestamp(t.exit_date)  for t in trades]
    pnls        = stats["pnls"]
    cum_pnl     = stats["cum_pnl"]

    # ---- Chart 0: Account value growth — Strategy vs Buy & Hold ----
    fig_equity = go.Figure()
    fig_equity.add_trace(go.Scatter(
        x=bh_equity.index, y=bh_equity.values,
        mode="lines", name=f"{ticker} Buy & Hold",
        line=dict(color="steelblue", width=2),
    ))
    fig_equity.add_trace(go.Scatter(
        x=equity_curve.index, y=equity_curve.values,
        mode="lines", name="Bull Put Spread Strategy",
        line=dict(color="green", width=2),
    ))
    fig_equity.add_hline(
        y=total_capital, line_color="grey",
        line_width=0.8, line_dash="dash",
    )
    fig_equity.update_layout(
        title=f"Account Value Growth — Starting Capital ${total_capital:,.0f}",
        height=380,
        xaxis=dict(
            title="Date",
            rangeslider=dict(visible=True, thickness=0.06),
            rangeselector=dict(
                buttons=[
                    dict(count=1,  label="1M", step="month", stepmode="backward"),
                    dict(count=3,  label="3M", step="month", stepmode="backward"),
                    dict(count=6,  label="6M", step="month", stepmode="backward"),
                    dict(count=1,  label="1Y", step="year",  stepmode="backward"),
                    dict(count=2,  label="2Y", step="year",  stepmode="backward"),
                    dict(step="all", label="All"),
                ],
                bgcolor="#f0f2f6",
                activecolor="#4c78a8",
            ),
        ),
        yaxis_title="Account Value ($)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(t=50, b=10),
    )
    st.plotly_chart(fig_equity, use_container_width=True)

    # ---- Chart 1: TradingView Lightweight Charts ----
    mp = config.ma_type.lower()
    ohlcv_data = [
        {"time": d.strftime("%Y-%m-%d"),
         "open":  round(float(r["open"]),  4),
         "high":  round(float(r["high"]),  4),
         "low":   round(float(r["low"]),   4),
         "close": round(float(r["close"]), 4)}
        for d, r in df.iterrows()
    ]
    volume_data = [
        {"time":  d.strftime("%Y-%m-%d"),
         "value": round(float(r["volume"]), 0),
         "color": "#26a69a" if float(r["close"]) >= float(r["open"]) else "#ef5350"}
        for d, r in df.iterrows()
    ]
    ema_fast_data = [{"time": d.strftime("%Y-%m-%d"), "value": round(float(r[f"{mp}{config.ema_fast}"]), 4)} for d, r in df.iterrows()]
    ema_slow_data = [{"time": d.strftime("%Y-%m-%d"), "value": round(float(r[f"{mp}{config.ema_slow}"]), 4)} for d, r in df.iterrows()]
    ema100_data   = [{"time": d.strftime("%Y-%m-%d"), "value": round(float(r[f"{mp}100"]), 4)} for d, r in df.iterrows()]
    ema200_data   = [{"time": d.strftime("%Y-%m-%d"), "value": round(float(r[f"{mp}200"]), 4)} for d, r in df.iterrows()]

    ohlcv_times   = {r["time"] for r in ohlcv_data}
    entry_dates_  = [pd.Timestamp(t.entry_date) for t in trades]
    exit_dates_tv = [pd.Timestamp(t.exit_date)  for t in trades]
    entry_markers = [{"time": d.strftime("%Y-%m-%d"), "position": "belowBar", "color": "#26a69a", "shape": "arrowUp",   "text": "Entry"} for d in entry_dates_  if d.strftime("%Y-%m-%d") in ohlcv_times]
    exit_markers  = [{"time": d.strftime("%Y-%m-%d"), "position": "aboveBar", "color": "#ef5350", "shape": "arrowDown", "text": "Exit"}  for d in exit_dates_tv if d.strftime("%Y-%m-%d") in ohlcv_times]
    markers_data  = sorted(entry_markers + exit_markers, key=lambda m: m["time"])

    tv_html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"/>
<style>
  html,body{{margin:0;padding:0;background:#131722;}}
  #chart-container{{position:relative;width:100%;}}
  #legend{{position:absolute;top:8px;left:12px;z-index:10;
    font-family:-apple-system,BlinkMacSystemFont,'Trebuchet MS',sans-serif;
    font-size:12px;color:#d1d4dc;background:rgba(19,23,34,0.88);
    padding:5px 10px;border-radius:4px;display:flex;flex-wrap:wrap;gap:14px;align-items:center;pointer-events:none;}}
  .leg-item{{display:flex;align-items:center;gap:4px;white-space:nowrap;}}
  .leg-swatch{{width:10px;height:3px;border-radius:2px;display:inline-block;}}
  .val{{font-weight:600;margin-left:2px;}}
</style></head><body>
<div id="chart-container">
  <div id="legend">
    <span style="font-weight:700;font-size:13px;">{ticker}</span>
    <span class="leg-item">O&nbsp;<span id="val-o" class="val">—</span>&nbsp;H&nbsp;<span id="val-h" class="val">—</span>&nbsp;L&nbsp;<span id="val-l" class="val">—</span>&nbsp;C&nbsp;<span id="val-c" class="val">—</span></span>
    <span class="leg-item"><span class="leg-swatch" style="background:#2962ff"></span>{config.ma_type}{config.ema_fast}&nbsp;<span id="val-ef" class="val" style="color:#2962ff">—</span></span>
    <span class="leg-item"><span class="leg-swatch" style="background:#ff9800"></span>{config.ma_type}{config.ema_slow}&nbsp;<span id="val-es" class="val" style="color:#ff9800">—</span></span>
    <span class="leg-item"><span class="leg-swatch" style="background:#ab47bc"></span>{config.ma_type}100&nbsp;<span id="val-e100" class="val" style="color:#ab47bc">—</span></span>
    <span class="leg-item"><span class="leg-swatch" style="background:#ff7043"></span>{config.ma_type}200&nbsp;<span id="val-e200" class="val" style="color:#ff7043">—</span></span>
    <span class="leg-item" style="color:#26a69a">▲ Entry</span>
    <span class="leg-item" style="color:#ef5350">▼ Exit</span>
  </div>
  <div id="tv-chart"></div>
</div>
<script src="https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>
<script>
const W=document.documentElement.clientWidth;
const chart=LightweightCharts.createChart(document.getElementById('tv-chart'),{{width:W,height:510,
  layout:{{background:{{type:'solid',color:'#131722'}},textColor:'#d1d4dc',fontSize:12}},
  grid:{{vertLines:{{color:'#1e222d'}},horzLines:{{color:'#1e222d'}}}},
  crosshair:{{mode:LightweightCharts.CrosshairMode.Normal}},
  rightPriceScale:{{borderColor:'#2a2e39'}},
  timeScale:{{borderColor:'#2a2e39',timeVisible:true,secondsVisible:false,rightOffset:5,barSpacing:6,minBarSpacing:2}},
  handleScroll:{{mouseWheel:true,pressedMouseMove:true}},handleScale:{{mouseWheel:true,pinch:true}}}});
const candles=chart.addCandlestickSeries({{upColor:'#26a69a',downColor:'#ef5350',borderUpColor:'#26a69a',borderDownColor:'#ef5350',wickUpColor:'#26a69a',wickDownColor:'#ef5350',priceScaleId:'right'}});
candles.setData({json.dumps(ohlcv_data)});
candles.setMarkers({json.dumps(markers_data)});
const volSeries=chart.addHistogramSeries({{priceFormat:{{type:'volume'}},priceScaleId:'volume',scaleMargins:{{top:0.82,bottom:0.00}}}});
volSeries.setData({json.dumps(volume_data)});
chart.priceScale('volume').applyOptions({{scaleMargins:{{top:0.82,bottom:0.00}}}});
const emaFast=chart.addLineSeries({{color:'#2962ff',lineWidth:1,priceScaleId:'right',priceLineVisible:false,lastValueVisible:false,crosshairMarkerVisible:false}});
emaFast.setData({json.dumps(ema_fast_data)});
const emaSlow=chart.addLineSeries({{color:'#ff9800',lineWidth:1,priceScaleId:'right',priceLineVisible:false,lastValueVisible:false,crosshairMarkerVisible:false}});
emaSlow.setData({json.dumps(ema_slow_data)});
const ema100=chart.addLineSeries({{color:'#ab47bc',lineWidth:1,lineStyle:LightweightCharts.LineStyle.Dashed,priceScaleId:'right',priceLineVisible:false,lastValueVisible:false,crosshairMarkerVisible:false}});
ema100.setData({json.dumps(ema100_data)});
const ema200=chart.addLineSeries({{color:'#ff7043',lineWidth:1,priceLineVisible:false,lastValueVisible:false,crosshairMarkerVisible:false}});
ema200.setData({json.dumps(ema200_data)});
chart.timeScale().fitContent();
const ohlcMap={{}};{json.dumps(ohlcv_data)}.forEach(d=>{{ohlcMap[d.time]=d;}});
const efMap={{}};{json.dumps(ema_fast_data)}.forEach(d=>{{efMap[d.time]=d.value;}});
const esMap={{}};{json.dumps(ema_slow_data)}.forEach(d=>{{esMap[d.time]=d.value;}});
const e100Map={{}};{json.dumps(ema100_data)}.forEach(d=>{{e100Map[d.time]=d.value;}});
const e200Map={{}};{json.dumps(ema200_data)}.forEach(d=>{{e200Map[d.time]=d.value;}});
function fmt(v){{return v!==undefined?v.toFixed(2):'—';}}
function fmtC(v,up){{const s=fmt(v);return `<span style="color:${{up?'#26a69a':'#ef5350'}}">${{s}}</span>`;}}
chart.subscribeCrosshairMove(param=>{{
  const t=param.time; if(!t) return;
  const ohlc=ohlcMap[t]; if(!ohlc) return;
  const up=ohlc.close>=ohlc.open;
  document.getElementById('val-o').innerHTML=fmtC(ohlc.open,up);
  document.getElementById('val-h').innerHTML=fmtC(ohlc.high,up);
  document.getElementById('val-l').innerHTML=fmtC(ohlc.low,up);
  document.getElementById('val-c').innerHTML=fmtC(ohlc.close,up);
  document.getElementById('val-ef').textContent=fmt(efMap[t]);
  document.getElementById('val-es').textContent=fmt(esMap[t]);
  document.getElementById('val-e100').textContent=fmt(e100Map[t]);
  document.getElementById('val-e200').textContent=fmt(e200Map[t]);
}});
window.addEventListener('resize',()=>{{chart.applyOptions({{width:document.documentElement.clientWidth}});}});
</script></body></html>"""

    components.html(tv_html, height=530, scrolling=False)

    # ---- Chart 2: Cumulative P&L + per-trade bars ----
    col_l, col_r = st.columns([2, 1])

    with col_l:
        fig_cum = go.Figure()
        fig_cum.add_trace(go.Scatter(
            x=exit_dates,
            y=cum_pnl,
            mode="lines+markers",
            name="Cumulative P&L",
            line=dict(color="green", width=2),
            marker=dict(size=5),
            fill="tozeroy",
            fillcolor="rgba(0,128,0,0.10)",
        ))
        fig_cum.add_hline(y=0, line_color="black", line_width=0.8)
        fig_cum.update_layout(
            title="Cumulative P&L ($)",
            height=340,
            xaxis_title="Trade Exit Date",
            yaxis_title="P&L ($)",
            margin=dict(t=50, b=40),
        )
        st.plotly_chart(fig_cum, use_container_width=True)

    with col_r:
        bar_colors = ["green" if p > 0 else "red" for p in pnls]
        fig_bar = go.Figure(go.Bar(
            x=list(range(1, len(pnls) + 1)),
            y=pnls,
            marker_color=bar_colors,
            marker_line_width=0.3,
        ))
        fig_bar.add_hline(y=0, line_color="black", line_width=0.8)
        fig_bar.update_layout(
            title="Per-Trade P&L ($)",
            height=340,
            xaxis_title="Trade #",
            yaxis_title="P&L ($)",
            margin=dict(t=50, b=40),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # ---- Exit reason pie ----
    col_pie, col_avg = st.columns([1, 2])

    with col_pie:
        reasons = stats["exit_reasons"]
        fig_pie = px.pie(
            names=list(reasons.keys()),
            values=list(reasons.values()),
            title="Exit Reasons",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_pie.update_layout(height=300, margin=dict(t=50, b=10))
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_avg:
        st.subheader("Summary")
        summary_data = {
            "Metric": [
                "Total Trades", "Winning Trades", "Losing Trades",
                "Total P&L", "Win Rate",
                "Avg P&L / Trade", "Avg Winning Trade", "Avg Losing Trade",
                "Avg Win %", "Avg Loss %",
                "Profit Factor", "Sharpe Ratio", "Max DD $", "Max DD %", "Max Loss $", "Max Loss %", "Avg Trade Duration",
            ],
            "Value": [
                stats["total_trades"],
                stats["winning_trades"],
                stats["losing_trades"],
                f"${stats['total_pnl']:,.0f}",
                f"{stats['win_rate']:.1f}%",
                f"${stats['avg_pnl_per_trade']:.0f}",
                f"${stats['avg_win']:.0f}",
                f"${stats['avg_loss']:.0f}",
                f"{stats['avg_win_pct']:.2f}%",
                f"{stats['avg_loss_pct']:.2f}%",
                f"{stats['profit_factor']:.2f}",
                f"{stats['sharpe']:.2f}",
                f"${strat_max_dd_usd:,.0f}",
                f"{strat_max_dd_pct:.1%}",
                f"${max_loss_usd:,.0f}",
                f"{max_loss_pct:.1%}",
                f"{stats['avg_duration_days']:.0f} days",
            ],
        }
        st.table(pd.DataFrame(summary_data).set_index("Metric"))

    # -----------------------------------------------------------------------
    # Trade log
    # -----------------------------------------------------------------------

    st.divider()
    st.subheader("Trade Log")

    def colour_pnl(val):
        if isinstance(val, (int, float)):
            colour = "color: green" if val > 0 else ("color: red" if val < 0 else "")
            return colour
        return ""

    styled = (
        trades_df.style
        .map(colour_pnl, subset=["P&L ($)", "P&L (% Acct)"])
        .format({"P&L ($)": "${:,.2f}", "P&L (% Acct)": "{:.2f}"})
    )
    log_height = min(len(trades_df), 20) * 35 + 38
    st.dataframe(styled, use_container_width=True, hide_index=True, height=log_height)

    csv = trades_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇ Download Trade Log (CSV)",
        data=csv,
        file_name=f"{ticker}_bull_put_spread_backtest.csv",
        mime="text/csv",
    )

else:
    st.info("Configure parameters in the sidebar and click **▶ Run Backtest** to begin.")
