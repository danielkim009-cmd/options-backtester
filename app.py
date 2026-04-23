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
from backtester.report import generate_html_report


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
    "Entry A: low ≤ EMA21×(1+buffer), {Close|Low} > EMA50, EMA21 > EMA50, EMA50 > EMA100, EMA100 > EMA200, EMA21 ≥ EMA21[7d/14d/21d ago] & > EMA21[30d ago] (first day of pullback zone)  •  "
    "Entry B: close crosses above EMA200, EMA100 > EMA200, high > EMA21  •  "
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
    start_date = col1.date_input("Start Date", value=pd.Timestamp("2016-01-04"))
    _today = pd.Timestamp.today().normalize()
    _last_trading_day = _today - pd.Timedelta(days=max(0, _today.weekday() - 4))
    end_date   = col2.date_input("End Date",   value=_last_trading_day)

    st.divider()
    ma_type     = st.radio("Moving Average Type", ["EMA", "SMA"], horizontal=True)
    entry_price = st.radio("Entry Signal Price", ["Close", "Low"], horizontal=True,
                           help="'Close' triggers on a closing price pullback; 'Low' triggers when the intraday low touches the zone.")
    _ema_buffer_opts = {"At or below EMA21 (0%)": 0.0, "Within 1% above EMA21": 0.01, "Within 2% above EMA21": 0.02}
    entry_a_ema_buffer = _ema_buffer_opts[st.selectbox(
        "Entry A — Low threshold",
        list(_ema_buffer_opts.keys()),
        help="How far above EMA21 the intraday low may be and still qualify for Entry A. '0%' = must touch or dip below EMA21.",
    )]

    st.divider()
    st.subheader("Options")

    spread_width = st.number_input(
        "Spread Width ($)",
        min_value=1.0, max_value=50.0, value=5.0, step=1.0,
        help="Dollar distance between short and long put strikes.",
    )
    entry_dte = st.slider("Entry DTE (target)", 30, 90, 45)
    exit_dte  = st.slider("Exit DTE (time-stop)", 1, 30, 7)
    exit_below_ema = st.checkbox(
        "Exit if price crosses below EMA",
        value=True,
        help="Close the trade on any day the underlying closes below the specified EMA.",
    )
    exit_ema_period = st.number_input(
        "Exit EMA Period",
        min_value=10, max_value=500, value=200, step=10,
        disabled=not exit_below_ema,
        help="EMA period for the exit condition above.",
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
        entry_a_ema_buffer=entry_a_ema_buffer,
        entry_dte=entry_dte,
        exit_dte=exit_dte,
        profit_target=profit_target,
        short_put_delta=short_put_delta,
        spread_width=spread_width,
        starting_capital=float(starting_capital),
        deploy_pct=deploy_pct / 100,
        commission_per_contract=commission,
        exit_below_ema=exit_below_ema,
        exit_ema_period=exit_ema_period,
        use_stop_loss=use_stop_loss,
        stop_loss_multiple=stop_loss_multiple,
    )

    # --- Fetch data ---
    with st.spinner(f"Fetching data for {ticker}…"):
        try:
            df = fetch_data(ticker, str(start_date), str(end_date), 21, 50,
                           extra_periods=[exit_ema_period])
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

    # Persist results so filter interactions don't wipe the page
    st.session_state["bt_results"] = dict(
        trades=trades, equity_curve=equity_curve, df=df, config=config, name=name
    )

if "bt_results" in st.session_state:
    r            = st.session_state["bt_results"]
    trades       = r["trades"]
    equity_curve = r["equity_curve"]
    df           = r["df"]
    config       = r["config"]
    name         = r["name"]
    st.markdown(f"## {name} ({ticker})")
    st.caption(f"Backtest period: **{config.start_date}** — **{config.end_date}**")

    ma = config.ma_type
    ef = config.ema_fast
    es = config.ema_slow
    ep = config.entry_price.capitalize()
    exit_ema_line = (
        f"close below {ma}{config.exit_ema_period}"
        if config.exit_below_ema else None
    )
    sl_line = (
        f"stop-loss at {config.stop_loss_multiple:.1f}× credit received"
        if config.use_stop_loss else None
    )
    exit_parts = [
        f"profit target ≥ {int(config.profit_target * 100)}% of credit",
        f"DTE ≤ {config.exit_dte}",
    ]
    if exit_ema_line:
        exit_parts.append(exit_ema_line)
    if sl_line:
        exit_parts.append(sl_line)

    st.markdown(
        f"**Entry A:** low ≤ {ma}{ef} &nbsp;•&nbsp; "
        f"{ep} > {ma}{es} &nbsp;•&nbsp; "
        f"{ma}{ef} > {ma}{es} &nbsp;•&nbsp; "
        f"{ma}{es} > {ma}100 &nbsp;•&nbsp; "
        f"{ma}100 > {ma}200 &nbsp;•&nbsp; "
        f"{ma}{ef} ≥ {ma}{ef}[7d/14d/21d ago] &nbsp;•&nbsp; "
        f"{ma}{ef} > {ma}{ef}[30d ago] &nbsp;*(first day of pullback zone)*"
    )
    st.markdown(
        f"**Entry B ({ep}):** close crosses above {ma}200 &nbsp;•&nbsp; "
        f"{ma}100 > {ma}200 &nbsp;•&nbsp; "
        f"high > {ma}{ef}"
    )
    st.markdown(
        "**Exit:** " + " &nbsp;•&nbsp; ".join(exit_parts)
    )

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

    # ---- Chart 0: Account value growth — TradingView (log scale) ----
    def _resample_tv(series, freq):
        """Resample series to freq ('W'/'ME'/'YE'), forward-fill gaps, return TV list."""
        daily = series.resample("D").last().ffill()
        try:
            resampled = daily.resample(freq).last().dropna()
        except ValueError:
            # older pandas: ME→M, YE→A
            fallback = {"ME": "M", "YE": "A"}.get(freq, freq)
            resampled = daily.resample(fallback).last().dropna()
        seen = {}
        for d, v in resampled.items():
            seen[d.strftime("%Y-%m-%d")] = round(float(v), 2)
        return [{"time": t, "value": v} for t, v in sorted(seen.items())]

    eq_weekly  = _resample_tv(equity_curve, "W")
    eq_monthly = _resample_tv(equity_curve, "ME")
    eq_yearly  = _resample_tv(equity_curve, "YE")
    bh_weekly  = _resample_tv(bh_equity, "W")
    bh_monthly = _resample_tv(bh_equity, "ME")
    bh_yearly  = _resample_tv(bh_equity, "YE")

    eq_html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"/>
<style>
  html,body{{margin:0;padding:0;background:#131722;}}
  #eq-container{{position:relative;width:100%;}}
  #eq-legend{{position:absolute;top:8px;left:12px;z-index:10;
    font-family:-apple-system,BlinkMacSystemFont,'Trebuchet MS',sans-serif;
    font-size:12px;color:#d1d4dc;background:rgba(19,23,34,0.88);
    padding:5px 10px;border-radius:4px;display:flex;flex-wrap:wrap;gap:14px;align-items:center;pointer-events:none;}}
  .leg-item{{display:flex;align-items:center;gap:5px;white-space:nowrap;}}
  .leg-swatch{{width:14px;height:3px;border-radius:2px;display:inline-block;}}
  .val{{font-weight:600;margin-left:2px;}}
  #eq-aggbtns{{position:absolute;top:8px;right:12px;z-index:10;display:flex;gap:4px;}}
  .abtn{{background:#1e222d;color:#d1d4dc;border:1px solid #2a2e39;border-radius:3px;
    padding:2px 8px;font-size:11px;cursor:pointer;font-family:inherit;}}
  .abtn:hover,.abtn.active{{background:#2962ff;color:#fff;border-color:#2962ff;}}
</style></head><body>
<div id="eq-container">
  <div id="eq-legend">
    <span style="font-weight:700;font-size:13px;">Account Value Growth &nbsp;|&nbsp; Starting ${total_capital:,.0f} &nbsp;|&nbsp; Log Scale</span>
    <span class="leg-item"><span class="leg-swatch" style="background:#26a69a"></span>Strategy&nbsp;<span id="eq-strat" class="val" style="color:#26a69a">—</span></span>
    <span class="leg-item"><span class="leg-swatch" style="background:#4c78a8"></span>{ticker} B&amp;H&nbsp;<span id="eq-bh" class="val" style="color:#4c78a8">—</span></span>
  </div>
  <div id="eq-aggbtns">
    <button class="abtn" onclick="setAgg('W',this)">Weekly</button>
    <button class="abtn active" onclick="setAgg('M',this)">Monthly</button>
    <button class="abtn" onclick="setAgg('Y',this)">Yearly</button>
  </div>
  <div id="eq-chart"></div>
</div>
<script src="https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>
<script>
const datasets={{
  W:{{strat:{json.dumps(eq_weekly)}, bh:{json.dumps(bh_weekly)}}},
  M:{{strat:{json.dumps(eq_monthly)},bh:{json.dumps(bh_monthly)}}},
  Y:{{strat:{json.dumps(eq_yearly)}, bh:{json.dumps(bh_yearly)}}},
}};
const W=document.documentElement.clientWidth;
const chart=LightweightCharts.createChart(document.getElementById('eq-chart'),{{
  width:W,height:400,
  layout:{{background:{{type:'solid',color:'#131722'}},textColor:'#d1d4dc',fontSize:12}},
  grid:{{vertLines:{{color:'#1e222d'}},horzLines:{{color:'#1e222d'}}}},
  crosshair:{{mode:LightweightCharts.CrosshairMode.Normal}},
  rightPriceScale:{{borderColor:'#2a2e39',mode:LightweightCharts.PriceScaleMode.Logarithmic}},
  timeScale:{{borderColor:'#2a2e39',timeVisible:true,secondsVisible:false,rightOffset:5,barSpacing:10,minBarSpacing:2}},
  handleScroll:{{mouseWheel:true,pressedMouseMove:true}},
  handleScale:{{mouseWheel:true,pinch:true}},
}});

function fmtAxis(v){{return '$'+Math.round(v).toLocaleString('en-US');}}
const stratSeries=chart.addLineSeries({{
  color:'#26a69a',lineWidth:2,
  priceLineVisible:false,lastValueVisible:false,crosshairMarkerVisible:true,
  priceFormat:{{type:'custom',formatter:fmtAxis}},
}});
stratSeries.createPriceLine({{
  price:{total_capital},color:'#888888',lineWidth:1,
  lineStyle:LightweightCharts.LineStyle.Dashed,axisLabelVisible:true,title:'Start',
}});

const bhSeries=chart.addLineSeries({{
  color:'#4c78a8',lineWidth:2,
  priceLineVisible:false,lastValueVisible:false,crosshairMarkerVisible:true,
  priceFormat:{{type:'custom',formatter:fmtAxis}},
}});

let stratMap={{}}, bhMap={{}};
function fmtDollar(v){{return v!==undefined?'$'+v.toLocaleString('en-US',{{minimumFractionDigits:0,maximumFractionDigits:0}}):'—';}}

function setAgg(key,btn){{
  document.querySelectorAll('.abtn').forEach(b=>b.classList.remove('active'));
  btn.classList.add('active');
  const d=datasets[key];
  stratSeries.setData(d.strat);
  bhSeries.setData(d.bh);
  stratMap={{}};d.strat.forEach(r=>{{stratMap[r.time]=r.value;}});
  bhMap={{}};d.bh.forEach(r=>{{bhMap[r.time]=r.value;}});
  chart.timeScale().fitContent();
}}

// initialise with monthly
setAgg('M', document.querySelector('.abtn.active'));

chart.subscribeCrosshairMove(param=>{{
  const t=param.time; if(!t) return;
  document.getElementById('eq-strat').textContent=fmtDollar(stratMap[t]);
  document.getElementById('eq-bh').textContent=fmtDollar(bhMap[t]);
}});
window.addEventListener('resize',()=>{{chart.applyOptions({{width:document.documentElement.clientWidth}});}});
</script></body></html>"""

    st.session_state["bt_results"]["eq_html"] = eq_html
    components.html(eq_html, height=420, scrolling=False)

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
    <span class="leg-item"><span class="leg-swatch" style="background:#ffe066"></span>{config.ma_type}{config.ema_slow}&nbsp;<span id="val-es" class="val" style="color:#ffe066">—</span></span>
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
const emaSlow=chart.addLineSeries({{color:'#ffe066',lineWidth:1,priceScaleId:'right',priceLineVisible:false,lastValueVisible:false,crosshairMarkerVisible:false}});
emaSlow.setData({json.dumps(ema_slow_data)});
const ema100=chart.addLineSeries({{color:'#ab47bc',lineWidth:1,priceScaleId:'right',priceLineVisible:false,lastValueVisible:false,crosshairMarkerVisible:false}});
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

    st.session_state["bt_results"]["tv_html"] = tv_html
    components.html(tv_html, height=530, scrolling=False)

    # ---- Chart 2: Cumulative P&L + per-trade bars ----
    col_l, col_r = st.columns([2, 1])

    with col_l:
        # Deduplicate by date (last trade wins if multiple exit same day)
        cum_pnl_map = {}
        for d, v in zip(exit_dates, cum_pnl):
            cum_pnl_map[d.strftime("%Y-%m-%d")] = round(float(v), 2)
        cum_pnl_tv = [{"time": t, "value": v} for t, v in sorted(cum_pnl_map.items())]

        cum_pnl_tv_log = [d for d in cum_pnl_tv if d["value"] > 0]

        cum_html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"/>
<style>
  html,body{{margin:0;padding:0;background:#131722;}}
  #cum-container{{position:relative;width:100%;}}
  #cum-legend{{position:absolute;top:8px;left:12px;z-index:10;
    font-family:-apple-system,BlinkMacSystemFont,'Trebuchet MS',sans-serif;
    font-size:12px;color:#d1d4dc;background:rgba(19,23,34,0.88);
    padding:5px 10px;border-radius:4px;display:flex;gap:12px;align-items:center;pointer-events:none;}}
  .val{{font-weight:600;}}
</style></head><body>
<div id="cum-container">
  <div id="cum-legend">
    <span style="font-weight:700;font-size:13px;">Cumulative P&amp;L &nbsp;|&nbsp; Log Scale</span>
    <span><span class="leg-swatch" style="display:inline-block;width:12px;height:3px;background:#26a69a;border-radius:2px;margin-right:4px;"></span><span id="cum-val" class="val" style="color:#26a69a">—</span></span>
    <span style="color:#888;font-size:11px;">(negative values hidden)</span>
  </div>
  <div id="cum-chart"></div>
</div>
<script src="https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>
<script>
const W=document.documentElement.clientWidth;
const chart=LightweightCharts.createChart(document.getElementById('cum-chart'),{{
  width:W,height:340,
  layout:{{background:{{type:'solid',color:'#131722'}},textColor:'#d1d4dc',fontSize:12}},
  grid:{{vertLines:{{color:'#1e222d'}},horzLines:{{color:'#1e222d'}}}},
  crosshair:{{mode:LightweightCharts.CrosshairMode.Normal}},
  rightPriceScale:{{borderColor:'#2a2e39',mode:LightweightCharts.PriceScaleMode.Logarithmic}},
  timeScale:{{borderColor:'#2a2e39',timeVisible:true,secondsVisible:false,rightOffset:5,barSpacing:8,minBarSpacing:2}},
  handleScroll:{{mouseWheel:true,pressedMouseMove:true}},
  handleScale:{{mouseWheel:true,pinch:true}},
}});

const areaSeries=chart.addAreaSeries({{
  lineColor:'#26a69a',
  lineWidth:2,
  topColor:'rgba(38,166,154,0.25)',
  bottomColor:'rgba(38,166,154,0.02)',
  priceLineVisible:false,
  lastValueVisible:false,
  priceFormat:{{type:'custom',formatter:v=>('$'+Math.round(v).toLocaleString('en-US'))}},
  crosshairMarkerVisible:true,
  crosshairMarkerRadius:4,
}});
const data={json.dumps(cum_pnl_tv_log)};
areaSeries.setData(data);
chart.timeScale().fitContent();

const valMap={{}};{json.dumps(cum_pnl_tv)}.forEach(d=>{{valMap[d.time]=d.value;}});
function fmtDollar(v){{
  const sign=v<0?'-':'';
  return sign+'$'+Math.abs(v).toLocaleString('en-US',{{minimumFractionDigits:0,maximumFractionDigits:0}});
}}
chart.subscribeCrosshairMove(param=>{{
  const t=param.time; if(!t) return;
  const v=valMap[t];
  if(v===undefined) return;
  const el=document.getElementById('cum-val');
  el.textContent=fmtDollar(v);
  el.style.color=v>=0?'#26a69a':'#ef5350';
}});
window.addEventListener('resize',()=>{{chart.applyOptions({{width:document.documentElement.clientWidth}});}});
</script></body></html>"""

        st.session_state["bt_results"]["cum_html"] = cum_html
        components.html(cum_html, height=360, scrolling=False)

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
    # Annual P&L table
    # -----------------------------------------------------------------------

    st.divider()
    st.subheader("Annual P&L")

    # Group trades by exit year (P&L is realized when the trade closes)
    annual_df = trades_df.copy()
    annual_df["Exit Year"] = pd.to_datetime(annual_df["Exit Date"]).dt.year
    annual = (
        annual_df.groupby("Exit Year")
        .agg(
            Trades=("P&L ($)", "count"),
            Wins=("P&L ($)", lambda x: (x > 0).sum()),
            Losses=("P&L ($)", lambda x: (x <= 0).sum()),
            Annual_PnL=("P&L ($)", "sum"),
            Avg_Win_Pct=("P&L (% Acct)", lambda x: x[x > 0].mean()),
            Avg_Loss_Pct=("P&L (% Acct)", lambda x: x[x <= 0].mean()),
        )
        .reset_index()
        .rename(columns={"Exit Year": "Year"})
    )

    # Total Account Value: year-end equity from the equity curve
    eq_year_end = equity_curve.groupby(equity_curve.index.year).last()
    annual["Acct_Val"] = annual["Year"].map(eq_year_end)

    # P&L %: annual realized P&L / account equity at start of year
    annual_pct = {}
    for _, row in annual.iterrows():
        yr = row["Year"]
        # Start-of-year equity = prior year-end equity, or starting capital for the first year
        prior_years = eq_year_end.loc[eq_year_end.index < yr]
        start_equity = prior_years.iloc[-1] if len(prior_years) > 0 else equity_curve.iloc[0]
        annual_pct[yr] = row["Annual_PnL"] / start_equity * 100 if start_equity > 0 else 0.0
    annual["P&L %"] = annual["Year"].map(annual_pct).round(1).astype(str) + "%"

    # Buy & Hold annual return %: year-end vs prior year-end equity
    bh_year_end = bh_equity.groupby(bh_equity.index.year).last()
    bh_annual_ret = {}
    bh_years_sorted = sorted(bh_year_end.index)
    for i, yr in enumerate(bh_years_sorted):
        prior_val = bh_year_end.iloc[i - 1] if i > 0 else bh_equity.iloc[0]
        bh_annual_ret[yr] = (bh_year_end[yr] - prior_val) / prior_val * 100
    annual[f"B&H {ticker} P&L %"] = annual["Year"].map(bh_annual_ret).round(1).astype(str) + "%"

    # Alpha: strategy P&L % minus B&H return % (both as raw floats)
    annual["Alpha_raw"] = annual["Year"].map(annual_pct) - annual["Year"].map(bh_annual_ret)
    annual["Alpha %"] = annual["Alpha_raw"].round(1).astype(str) + "%"

    annual["Win Rate"] = (annual["Wins"] / annual["Trades"] * 100).round(1).astype(str) + "%"
    annual["Avg Win %"] = annual["Avg_Win_Pct"].apply(lambda v: f"{v:.2f}%" if pd.notna(v) else "—")
    annual["Avg Loss %"] = annual["Avg_Loss_Pct"].apply(lambda v: f"{v:.2f}%" if pd.notna(v) else "—")
    annual["Annual P&L ($)"] = annual["Annual_PnL"].apply(lambda v: f"${v:,.0f}")
    annual["Total Account Value"] = annual["Acct_Val"].apply(lambda v: f"${v:,.0f}")
    annual = annual.drop(columns=["Annual_PnL", "Acct_Val", "Avg_Win_Pct", "Avg_Loss_Pct", "Alpha_raw"])
    bh_col = f"B&H {ticker} P&L %"
    annual = annual[["Year", "Trades", "Wins", "Losses", "Win Rate", "Avg Win %", "Avg Loss %", "Annual P&L ($)", "P&L %", bh_col, "Alpha %", "Total Account Value"]]

    def colour_annual_pnl(val):
        if isinstance(val, str) and (val.startswith("$") or val.endswith("%")):
            num_str = val.replace("$", "").replace(",", "").replace("%", "")
            try:
                num = float(num_str)
                return "color: green" if num > 0 else ("color: red" if num < 0 else "")
            except ValueError:
                pass
        return ""

    st.session_state["bt_results"]["annual_df"] = annual
    st.session_state["bt_results"].update(
        strat_final=strat_final, strat_cagr=strat_cagr,
        bh_cagr=bh_cagr, bh_final=bh_final,
        strat_max_dd_usd=strat_max_dd_usd, strat_max_dd_pct=strat_max_dd_pct,
        bh_max_dd_pct=bh_max_dd_pct,
        max_loss_usd=max_loss_usd, max_loss_pct=max_loss_pct,
    )

    styled_annual = annual.style.map(colour_annual_pnl, subset=["Annual P&L ($)", "P&L %", "Avg Win %", "Avg Loss %", bh_col, "Alpha %"])
    st.dataframe(styled_annual, use_container_width=True, hide_index=True)

    # -----------------------------------------------------------------------
    # Trade log
    # -----------------------------------------------------------------------

    st.divider()
    st.subheader("Trade Log")

    # --- Inline filter row (collapsed labels so inputs sit flush like column header filters) ---
    log_years   = sorted(pd.to_datetime(trades_df["Exit Date"]).dt.year.unique().tolist())
    log_reasons = sorted(trades_df["Exit Reason"].dropna().unique().tolist())

    kw = dict(label_visibility="collapsed")
    fc1, fc2, fc3, fc4, fc5 = st.columns([1, 1.4, 1.4, 1, 1.4])
    fc1.markdown("<div style='padding-top:4px;font-size:12px;color:grey'>Filters:</div>", unsafe_allow_html=True)
    sel_entry  = fc2.text_input("Entry Date",  placeholder="Entry date (e.g. 2023)", key="log_entry",  **kw)
    sel_exit   = fc3.text_input("Exit Date",   placeholder="Exit date (e.g. 2024)",  key="log_exit",   **kw)
    sel_result = fc4.selectbox("Result",       options=["All", "Wins", "Losses"],                      key="log_result", **kw)
    sel_reason = fc5.selectbox("Exit Reason",  options=["All"] + log_reasons,                          key="log_reason", **kw)

    filtered_df = trades_df.copy()
    if sel_entry:
        filtered_df = filtered_df[filtered_df["Entry Date"].str.contains(sel_entry, case=False, na=False)]
    if sel_exit:
        filtered_df = filtered_df[filtered_df["Exit Date"].str.contains(sel_exit, case=False, na=False)]
    if sel_result == "Wins":
        filtered_df = filtered_df[filtered_df["P&L ($)"] > 0]
    elif sel_result == "Losses":
        filtered_df = filtered_df[filtered_df["P&L ($)"] <= 0]
    if sel_reason != "All":
        filtered_df = filtered_df[filtered_df["Exit Reason"] == sel_reason]

    st.caption(f"{len(filtered_df)} of {len(trades_df)} trades shown")

    def colour_pnl(val):
        if isinstance(val, (int, float)):
            colour = "color: green" if val > 0 else ("color: red" if val < 0 else "")
            return colour
        return ""

    styled = (
        filtered_df.style
        .map(colour_pnl, subset=["P&L ($)", "P&L (% Acct)"])
        .format({"P&L ($)": "${:,.2f}", "P&L (% Acct)": "{:.2f}"})
    )
    log_height = min(len(filtered_df), 20) * 35 + 38
    st.dataframe(styled, use_container_width=True, hide_index=True, height=log_height)

    csv = filtered_df.to_csv(index=False).encode("utf-8")
    col_csv, col_html = st.columns([1, 1])
    col_csv.download_button(
        "⬇ Download Trade Log (CSV)",
        data=csv,
        file_name=f"{ticker}_bull_put_spread_backtest.csv",
        mime="text/csv",
    )

    # HTML report — only available once all charts have been rendered
    _r = st.session_state["bt_results"]
    if all(k in _r for k in ("eq_html", "tv_html", "cum_html", "annual_df")):
        html_report = generate_html_report(
            ticker=ticker,
            name=name,
            config=config,
            stats=stats,
            trades_df=trades_df,
            annual_df=_r["annual_df"],
            starting_capital=starting_capital,
            strat_final=_r["strat_final"],
            strat_cagr=_r["strat_cagr"],
            bh_cagr=_r["bh_cagr"],
            bh_final=_r["bh_final"],
            strat_max_dd_usd=_r["strat_max_dd_usd"],
            strat_max_dd_pct=_r["strat_max_dd_pct"],
            bh_max_dd_pct=_r["bh_max_dd_pct"],
            max_loss_usd=_r["max_loss_usd"],
            max_loss_pct=_r["max_loss_pct"],
            pnls=stats["pnls"],
            exit_reasons=stats["exit_reasons"],
            eq_html=_r["eq_html"],
            tv_html=_r["tv_html"],
            cum_html=_r["cum_html"],
        )
        col_html.download_button(
            "⬇ Download HTML Report",
            data=html_report.encode("utf-8"),
            file_name=f"{ticker}_backtest_report.html",
            mime="text/html",
        )

elif "bt_results" not in st.session_state:
    st.info("Configure parameters in the sidebar and click **▶ Run Backtest** to begin.")
