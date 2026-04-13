# Bull Put Spread Backtester

A fully-featured options strategy backtester for **bull put spreads**, built with Python and Streamlit. Backtest a systematic credit-spread strategy across any US equity or ETF, with real IV data, dynamic position sizing, and an interactive TradingView-style dashboard.

![Python](https://img.shields.io/badge/python-3.9%2B-blue) ![Streamlit](https://img.shields.io/badge/streamlit-1.32%2B-red) ![License](https://img.shields.io/badge/license-MIT-green)

---

## Strategy Overview

The backtester implements a rules-based bull put spread strategy:

**Entry** — all four conditions must be true on the first day of a pullback zone:
1. Entry price (close or intraday low) < Fast MA (EMA/SMA 21)
2. Entry price > Slow MA (EMA/SMA 50)
3. Fast MA > Slow MA
4. Slow MA > MA100 (broader uptrend confirmed)

**Exit** — whichever triggers first:
- 50% of max profit captured (configurable)
- DTE time-stop (default: 7 DTE)
- Price closes below MA200 (optional)
- Stop-loss at N× the credit received (optional)

**Position sizing** — contracts are sized so that the maximum possible loss equals a configurable percentage of the current account value.

---

## Features

- **Real IV data** — uses CBOE volatility indices (VIX for SPY, GVZ for GLD, OVX for USO) where available; falls back to a blended EWMA historical volatility calibrated per-ticker
- **Black-Scholes pricing** — all options are priced with BS; short put strike is found via binary search for a target delta
- **Monthly expiry targeting** — finds the 3rd-Friday expiry closest to a configurable entry DTE
- **Dynamic position sizing** — contracts scale with account equity so compounding is realistic
- **Dual-ticker portfolio** — run two tickers simultaneously with a 50/50 capital split
- **EMA or SMA** — choose moving average type from the sidebar
- **Entry price choice** — trigger on closing price or intraday low
- **TradingView-style chart** — interactive candlestick chart with zoom/scroll, MA overlays, entry/exit markers, and a live crosshair legend
- **Comprehensive analytics** — CAGR, Sharpe ratio, profit factor, max drawdown, win rate, per-trade P&L log, exit reason breakdown
- **Buy & Hold comparison** — normalized to the same starting capital over the same period

---

## Screenshots

The dashboard includes:
- KPI metrics row (CAGR, max drawdown, win rate, buy & hold comparison)
- Account value growth chart (strategy vs buy & hold)
- TradingView candlestick chart with MA lines and trade markers
- Cumulative P&L and per-trade bar chart
- Exit reason pie chart
- Trade summary table
- Colour-coded trade log with CSV export

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/options-backtester.git
cd options-backtester
pip install -r requirements.txt
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## Requirements

```
yfinance>=0.2.18
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
streamlit>=1.32.0
plotly>=5.18.0
```

Python 3.9 or later is required.

---

## Project Structure

```
options-backtester/
├── app.py                      # Streamlit dashboard
├── config.py                   # BacktestConfig dataclass
├── requirements.txt
├── backtester/
│   ├── data_fetcher.py         # OHLCV + IV download and caching
│   ├── engine.py               # Signal generation and simulation loop
│   ├── options_pricer.py       # Black-Scholes pricing, delta search, expiry calendar
│   └── analytics.py           # Performance statistics and trade log
└── data/                       # Parquet cache (auto-created)
```

---

## Configuration

All parameters are adjustable from the Streamlit sidebar:

| Parameter | Default | Description |
|---|---|---|
| Ticker 1 / 2 | SPY / QQQ | Underlying ticker(s) |
| Start / End Date | 2006-04-03 / 2026-03-31 | Backtest window |
| Moving Average Type | EMA | EMA or SMA |
| Entry Signal Price | Close | Close or intraday Low |
| Spread Width | $5 | Dollar distance between strikes |
| Entry DTE | 45 | Target days-to-expiry at entry |
| Exit DTE | 7 | Time-stop: close at or below this DTE |
| Exit below MA200 | On | Close trade if price crosses below 200-period MA |
| Stop-loss | Off | Close at N× credit received loss |
| Profit Target | 50% | Close when spread decays to this % of original credit |
| Short Put Delta | 0.40 | Target delta for the short put |
| Starting Capital | $10,000 | Initial account value |
| Capital Deployed | 50% | Max account % at risk per trade |
| Commission | $0.65 | Per leg per contract |

---

## How It Works

### Signal Generation
The engine scans daily OHLCV data and marks the **first day** a pullback enters the zone defined by the four MA conditions. Repeated signals during the same multi-day pullback are suppressed.

### Options Pricing
A bull put spread is constructed using Black-Scholes:
1. The short put strike is found via binary search to match the target delta
2. The long put strike is `short_strike − spread_width`
3. Net credit = BS(short) − BS(long)

### IV Estimation
- **CBOE indices**: SPY → ^VIX, GLD → ^GVZ, USO → ^OVX
- **All others**: blended EWMA HV (50%×HV10 + 30%×HV21 + 20%×HV30) × ticker-specific premium, floored at VIX

### Position Sizing
```
max_loss_per_contract = (spread_width − net_credit) × 100 + commissions
contracts = floor(account_value × deploy_pct / max_loss_per_contract)
```

### Equity Curve
Account value is tracked daily on a mark-to-market basis (including unrealized P&L from open positions). In dual-ticker mode, the two sub-account curves are forward-filled and summed.

---

## Disclaimer

This software is for educational and research purposes only. It does not constitute financial advice. Past backtested performance is not indicative of future results. Options trading involves substantial risk of loss.

---

## License

MIT License — see [LICENSE](LICENSE) for details.
