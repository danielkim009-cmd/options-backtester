import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path


# ---------------------------------------------------------------------------
# CBOE individual-stock / ETF volatility indices (same methodology as VIX).
# These are the best free source of historical implied vol for specific tickers.
# For all other tickers we fall back to a calibrated HV-based estimate.
# ---------------------------------------------------------------------------
CBOE_VOL_INDEX = {
    # SPX family — VIX is the gold standard
    "SPY":   "^VIX",
    "SPX":   "^VIX",
    "^GSPC": "^VIX",
    "^SPX":  "^VIX",
    "IVV":   "^VIX",
    "VOO":   "^VIX",
    "QQQ":   "^VIX",   # close enough; VXQQ not available
    # Commodity ETFs — these still trade on yfinance
    "GLD":   "^GVZ",   # CBOE Gold Volatility Index
    "IAU":   "^GVZ",
    "USO":   "^OVX",   # CBOE Crude Oil Volatility Index
    "UCO":   "^OVX",
    # Note: ^VXAPL, ^VXAZN, ^VXGOG, ^VXIBM, ^VXGS, ^VXEEM, ^VXEWZ, ^VXFXI
    # are no longer available via yfinance — all fall back to calibrated HV.
}

# Empirical IV-over-HV premium by ticker category.
# Derived from long-run averages of (30d IV / 30d realised vol).
# Used only when no CBOE vol index exists.
HV_PREMIUM = {
    # High-vol tech — IV runs well above realised vol
    "NVDA": 1.45,
    "TSLA": 1.50,
    "META": 1.35,
    "NFLX": 1.35,
    "AMD":  1.40,
    # Mega-cap tech — moderate premium
    "MSFT": 1.25,
    "AAPL": 1.25,   # fallback if VXAPL unavailable
    "AMZN": 1.25,
    "GOOG": 1.25,
    # Financials / industrials
    "JPM":  1.20,
    "BAC":  1.20,
    "XOM":  1.20,
}
DEFAULT_HV_PREMIUM = 1.30   # conservative default for unknown tickers


def _fetch_series(symbol: str, start: str, end: str) -> pd.Series:
    """Download a single price series from yfinance; return Close as a Series."""
    raw = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)
    if raw.empty:
        return pd.Series(dtype=float)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    s = raw["Close"].astype(float)
    s.index = pd.to_datetime(s.index).tz_localize(None)
    return s


def fetch_data(
    ticker: str,
    start: str,
    end: str,
    ema_fast: int = 21,
    ema_slow: int = 50,
    cache_dir: str = "data",
) -> pd.DataFrame:
    """
    Download historical OHLCV data and compute a best-available IV estimate.

    IV priority:
      1. CBOE individual-stock / ETF vol index (e.g. ^VXAPL for AAPL, ^VIX for SPY)
         — same methodology as VIX, most accurate free source.
      2. Calibrated HV: EWMA-30d realised vol × ticker-specific IV premium.
         Used when no CBOE index exists (e.g. NVDA, MSFT).

    The 'iv_source' column records which method was used ('cboe' or 'hv_calibrated').
    """
    Path(cache_dir).mkdir(exist_ok=True)
    cache_file = Path(cache_dir) / f"{ticker}_{start}_{end}.parquet"

    REQUIRED_COLUMNS = {"ema100", "ema200", "sma100", "sma200", "iv"}
    if cache_file.exists():
        cached = pd.read_parquet(cache_file)
        if REQUIRED_COLUMNS.issubset(cached.columns):
            return cached
        # Cache is stale (missing columns) — delete and re-fetch
        cache_file.unlink()

    ticker_upper = ticker.upper()

    # --- Price data ---
    print(f"Downloading {ticker} price data...")
    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if raw.empty:
        raise ValueError(f"No price data returned for {ticker}")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = pd.DataFrame(index=pd.to_datetime(raw.index).tz_localize(None))
    df["open"]   = raw["Open"].values.astype(float)
    df["high"]   = raw["High"].values.astype(float)
    df["low"]    = raw["Low"].values.astype(float)
    df["close"]  = raw["Close"].values.astype(float)
    df["volume"] = raw["Volume"].values.astype(float)

    # --- Always fetch VIX as a market-fear baseline ---
    print("Downloading VIX data...")
    vix = _fetch_series("^VIX", start, end).reindex(df.index).ffill()
    df["vix"] = vix

    # --- Realised vol: multiple windows, EWMA (tracks IV better than rolling) ---
    log_ret     = np.log(df["close"] / df["close"].shift(1))
    df["hv10"]  = log_ret.ewm(span=10,  adjust=False).std() * np.sqrt(252)
    df["hv21"]  = log_ret.ewm(span=21,  adjust=False).std() * np.sqrt(252)
    df["hv30"]  = log_ret.ewm(span=30,  adjust=False).std() * np.sqrt(252)
    # Blend: weight shorter windows more (options market is forward-looking)
    df["hv_blend"] = 0.50 * df["hv10"] + 0.30 * df["hv21"] + 0.20 * df["hv30"]

    # --- IV: use CBOE index if available, else calibrated HV ---
    cboe_symbol = CBOE_VOL_INDEX.get(ticker_upper)
    iv_source   = "hv_calibrated"

    if cboe_symbol:
        print(f"Downloading CBOE vol index {cboe_symbol} for {ticker}...")
        cboe_series = _fetch_series(cboe_symbol, start, end).reindex(df.index).ffill()
        if cboe_series.notna().mean() > 0.80:
            df["iv"]  = cboe_series / 100.0
            iv_source = "cboe"
            print(f"  → Using {cboe_symbol} as IV source (cboe)")
        else:
            print(f"  → {cboe_symbol} has insufficient coverage, falling back to HV")

    if iv_source == "hv_calibrated":
        premium  = HV_PREMIUM.get(ticker_upper, DEFAULT_HV_PREMIUM)
        hv_iv    = df["hv_blend"] * premium
        # Floor: IV should not fall far below VIX-implied market fear
        vix_floor = df["vix"] / 100.0
        df["iv"]  = np.maximum(hv_iv, vix_floor)
        df["iv"]  = df["iv"].fillna(vix_floor * premium)
        print(f"  → Using calibrated HV-blend×{premium:.2f} (floored at VIX) for {ticker}")

    df["iv_source"] = iv_source
    df["iv"] = df["iv"].clip(lower=0.05, upper=3.00)

    # --- Technical indicators (both EMA and SMA computed; engine picks based on config) ---
    for span in [ema_fast, ema_slow, 100, 200]:
        df[f"ema{span}"] = df["close"].ewm(span=span, adjust=False).mean()
        df[f"sma{span}"] = df["close"].rolling(window=span).mean()

    df = df.dropna(subset=[f"ema{ema_fast}", f"ema{ema_slow}", "ema100", "ema200",
                            f"sma{ema_fast}", f"sma{ema_slow}", "sma100", "sma200", "iv"])

    df.to_parquet(cache_file)
    print(f"Cached to {cache_file}\n")
    return df
