"""
Black-Scholes options pricing utilities and monthly expiry date generation.

Note: this backtester simulates options prices using Black-Scholes with a
VIX/HV-based IV estimate.  Real historical options data (e.g. CBOE DataShop,
OptionsDX) would give more accurate results.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Black-Scholes helpers
# ---------------------------------------------------------------------------

def bs_put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """European put price via Black-Scholes. T in years."""
    if T <= 1e-8 or sigma <= 1e-8:
        return float(max(K - S, 0.0))
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return float(max(price, 0.0))


def bs_put_delta(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Put delta (always <= 0). ATM ~-0.50."""
    if T <= 1e-8 or sigma <= 1e-8:
        return -1.0 if K > S else 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return float(norm.cdf(d1) - 1.0)


def find_put_strike_for_delta(
    S: float,
    T: float,
    r: float,
    sigma: float,
    target_delta: float,   # e.g. -0.30
) -> float:
    """
    Binary search for the put strike that achieves *target_delta*.
    Higher strike → more-negative delta (deeper ITM).
    """
    if T <= 1e-8:
        return round(S * (1.0 - abs(target_delta) * 0.10), 2)

    # For puts: K_lo (OTM end) → delta near 0; K_hi (ITM end) → delta near -1
    K_lo, K_hi = S * 0.50, S * 1.05

    delta_lo = bs_put_delta(S, K_lo, T, r, sigma)  # near 0
    delta_hi = bs_put_delta(S, K_hi, T, r, sigma)  # near -1

    # Clamp when target falls outside the achievable range
    if target_delta >= delta_lo:   # target less negative than OTM boundary
        return round(K_lo, 2)
    if target_delta <= delta_hi:   # target more negative than ITM boundary
        return round(K_hi, 2)

    K_mid = K_lo
    for _ in range(100):
        K_mid = (K_lo + K_hi) / 2.0
        delta = bs_put_delta(S, K_mid, T, r, sigma)
        if abs(delta - target_delta) < 5e-4:
            break
        # Higher K → more-negative delta
        if delta > target_delta:
            K_lo = K_mid
        else:
            K_hi = K_mid

    return round(K_mid, 2)


# ---------------------------------------------------------------------------
# Expiry date calendar
# ---------------------------------------------------------------------------

def get_monthly_expiries(start_date, end_date) -> pd.DatetimeIndex:
    """
    Return the 3rd Friday of every month between start_date-4mo and end_date+4mo.
    Standard monthly equity options expire on the 3rd Friday.
    """
    expiries = []
    start = pd.Timestamp(start_date) - pd.DateOffset(months=4)
    end   = pd.Timestamp(end_date)   + pd.DateOffset(months=4)

    cur = pd.Timestamp(start.year, start.month, 1)
    end_ts = pd.Timestamp(end.year, end.month, 1)

    while cur <= end_ts:
        # The 3rd Friday always falls between the 15th and 21st.
        # Start from the 15th and roll forward to the next Friday.
        day15 = cur.replace(day=15)
        days_to_fri = (4 - day15.weekday()) % 7   # 0 if already Friday
        third_fri = day15 + pd.Timedelta(days=days_to_fri)
        expiries.append(third_fri)
        cur += pd.DateOffset(months=1)

    return pd.DatetimeIndex(expiries)


def find_target_expiry(
    entry_date,
    expiry_dates: pd.DatetimeIndex,
    target_dte: int = 45,
) -> Optional[pd.Timestamp]:
    """Return the expiry date whose DTE is closest to *target_dte*."""
    entry = pd.Timestamp(entry_date)
    future = expiry_dates[expiry_dates > entry]
    if future.empty:
        return None
    dtes = (future - entry).days
    return future[int(np.argmin(np.abs(dtes - target_dte)))]


# ---------------------------------------------------------------------------
# Trade data structure
# ---------------------------------------------------------------------------

@dataclass
class PutSpread:
    """A bull put spread (short higher-strike put, long lower-strike put)."""

    entry_date:  pd.Timestamp
    expiry_date: pd.Timestamp
    entry_dte:   int

    short_strike: float
    long_strike:  float

    entry_short_price: float
    entry_long_price:  float
    net_credit:        float   # credit received per share at entry

    max_profit: float          # net_credit * 100 * contracts
    max_loss:   float          # (spread_width - net_credit) * 100 * contracts

    contracts:    int
    spread_width: float

    # Updated daily
    current_short_price:  float = 0.0
    current_long_price:   float = 0.0
    current_spread_value: float = 0.0
    unrealized_pnl:       float = 0.0

    # Set at close
    exit_date:    Optional[pd.Timestamp] = None
    exit_price:   float = 0.0
    realized_pnl: float = 0.0
    exit_reason:  str   = ""
    commissions:  float = 0.0

    # Snapshots at entry
    entry_stock_price:    float = 0.0
    account_value_at_entry: float = 0.0

    @property
    def is_closed(self) -> bool:
        return self.exit_date is not None
