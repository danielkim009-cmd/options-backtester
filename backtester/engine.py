"""
Backtesting engine: signal generation, trade lifecycle, simulation loop.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import pandas as pd

from config import BacktestConfig
from .options_pricer import (
    PutSpread,
    bs_put_price,
    find_put_strike_for_delta,
    get_monthly_expiries,
    find_target_expiry,
)


# ---------------------------------------------------------------------------
# Signal generation
# ---------------------------------------------------------------------------

def add_signals(
    df: pd.DataFrame,
    ema_fast: int,
    ema_slow: int,
    ma_type: str = "EMA",
    entry_price: str = "close",
) -> pd.DataFrame:
    """
    Add a 'signal' column that is True when EITHER entry condition is met.

    Condition A — Pullback zone (first day only):
      1. Intraday low <= fast MA (low touches or dips below EMA21)
      2. Entry price > slow MA   (stays above MA50)
      3. Fast MA > slow MA       (uptrend confirmed)
      4. Slow MA > MA100         (broader uptrend confirmed)
      5. MA100 > MA200           (long-term trend is bullish)
      6. EMA21 >= EMA21[7d/14d/21d ago], EMA21 > EMA21[30d ago]  (EMA21 is rising)

    Condition B — EMA200 crossover (first day only):
      1. Close crosses above EMA200 (previous close was at or below EMA200)
      2. EMA100 > EMA200            (medium-term MA already above long-term trend)
      3. Intraday high > EMA21      (stock pushed above the fast MA intraday)
    """
    df = df.copy()
    p     = ma_type.lower()
    ef    = f"{p}{ema_fast}"
    es    = f"{p}{ema_slow}"
    e200  = f"{p}200"
    price = df[entry_price]

    # Condition A: pullback zone — first bar only
    # Use intraday low for the EMA21 touch so that even a wick into the zone qualifies.
    ema21_rising = (
        (df[ef] >= df[ef].shift(7)) &
        (df[ef] >= df[ef].shift(14)) &
        (df[ef] >= df[ef].shift(21)) &
        (df[ef] > df[ef].shift(30))
    )
    in_zone = (
        (df["low"] <= df[ef]) &        # low touches or dips below EMA21 intraday
        (price > df[es]) &             # entry price stays above EMA50
        (df[ef] > df[es]) &
        (df[es] > df[f"{p}100"]) &
        (df[f"{p}100"] > df[e200]) &   # long-term trend is bullish
        ema21_rising                   # EMA21 is rising over 7, 14, 21 days
    )
    signal_a = in_zone & ~in_zone.shift(1).astype(bool).fillna(False)

    # Condition B: close crosses above EMA200, with EMA100 already above EMA200
    above_e200   = df["close"] > df[e200]
    crossed_e200 = above_e200 & ~above_e200.shift(1).astype(bool).fillna(False)
    signal_b = crossed_e200 & (df[f"{p}100"] > df[e200]) & (df["high"] > df[ef])

    df["signal"] = signal_a | signal_b
    return df


# ---------------------------------------------------------------------------
# Trade lifecycle helpers
# ---------------------------------------------------------------------------

def _commission(contracts: int, config: BacktestConfig) -> float:
    """One-way commission: 2 legs × contracts × rate."""
    return 2 * contracts * config.commission_per_contract


def _contracts_for_capital(
    deploy_capital: float,
    spread_width: float,
    net_credit: float,
    config: BacktestConfig,
) -> int:
    """
    How many contracts can we open given the capital we're willing to put at risk?

    Max loss per contract (before close commissions, which aren't known yet):
        (spread_width - net_credit) * 100 + entry_commission_per_contract
    We include both-way commissions in the at-risk estimate to be conservative.
    """
    entry_comm_per = 2 * config.commission_per_contract   # 2 legs
    close_comm_per = 2 * config.commission_per_contract
    max_loss_per   = (spread_width - net_credit) * 100 + entry_comm_per + close_comm_per
    if max_loss_per <= 0:
        return 0
    return max(1, math.floor(deploy_capital / max_loss_per))


def open_trade(
    entry_date: pd.Timestamp,
    stock_price: float,
    iv: float,
    expiry_dates: pd.DatetimeIndex,
    config: BacktestConfig,
    account_value: float,
) -> Optional[PutSpread]:
    """Construct a new bull put spread; returns None if trade is invalid."""
    expiry = find_target_expiry(entry_date, expiry_dates, config.entry_dte)
    if expiry is None:
        return None

    dte = (expiry - entry_date).days
    if dte <= config.exit_dte:
        return None

    T = dte / 365.0
    r = config.risk_free_rate

    short_strike = find_put_strike_for_delta(
        stock_price, T, r, iv, -config.short_put_delta
    )
    long_strike = round(short_strike - config.spread_width, 2)
    if long_strike <= 0:
        return None

    short_px = bs_put_price(stock_price, short_strike, T, r, iv)
    long_px  = bs_put_price(stock_price, long_strike,  T, r, iv)
    net_credit = short_px - long_px

    if net_credit <= 0.01:
        return None

    deploy_capital = account_value * config.deploy_pct
    contracts = _contracts_for_capital(
        deploy_capital, config.spread_width, net_credit, config
    )
    if contracts < 1:
        return None

    entry_comm = _commission(contracts, config)
    close_comm = _commission(contracts, config)
    total_comm = entry_comm + close_comm

    max_profit = net_credit * 100 * contracts - total_comm
    max_loss   = (config.spread_width - net_credit) * 100 * contracts + total_comm

    return PutSpread(
        entry_date=entry_date,
        expiry_date=expiry,
        entry_dte=dte,
        short_strike=short_strike,
        long_strike=long_strike,
        entry_short_price=short_px,
        entry_long_price=long_px,
        net_credit=net_credit,
        max_profit=max_profit,
        max_loss=max_loss,
        contracts=contracts,
        spread_width=config.spread_width,
        current_short_price=short_px,
        current_long_price=long_px,
        current_spread_value=net_credit,
        commissions=entry_comm,
        entry_stock_price=stock_price,
        account_value_at_entry=account_value,
    )


def update_trade(
    trade: PutSpread,
    current_date: pd.Timestamp,
    stock_price: float,
    iv: float,
    config: BacktestConfig,
) -> None:
    """Reprice the spread and update unrealised P&L."""
    dte = max((trade.expiry_date - current_date).days, 0)
    T = dte / 365.0
    r = config.risk_free_rate

    if dte == 0:
        trade.current_short_price = max(trade.short_strike - stock_price, 0.0)
        trade.current_long_price  = max(trade.long_strike  - stock_price, 0.0)
    else:
        trade.current_short_price = bs_put_price(stock_price, trade.short_strike, T, r, iv)
        trade.current_long_price  = bs_put_price(stock_price, trade.long_strike,  T, r, iv)

    trade.current_spread_value = trade.current_short_price - trade.current_long_price
    trade.unrealized_pnl = (
        (trade.net_credit - trade.current_spread_value) * 100 * trade.contracts
    )


def check_exit(
    trade: PutSpread,
    current_date: pd.Timestamp,
    config: BacktestConfig,
    stock_price: float = 0.0,
    exit_ema: float = 0.0,
) -> Optional[str]:
    """Return exit reason string, or None if trade should remain open."""
    dte = (trade.expiry_date - current_date).days

    if dte <= 0:
        return "expiry"
    if dte <= config.exit_dte:
        return f"{config.exit_dte}_dte"

    buyback_target = trade.net_credit * (1.0 - config.profit_target)
    if trade.current_spread_value <= buyback_target:
        return "profit_target"

    if config.exit_below_ema and exit_ema > 0 and stock_price < exit_ema:
        return f"below_ema{config.exit_ema_period}"

    if config.use_stop_loss:
        stop_trigger = trade.net_credit * (1.0 + config.stop_loss_multiple)
        if trade.current_spread_value >= stop_trigger:
            return "stop_loss"

    return None


def close_trade(
    trade: PutSpread,
    current_date: pd.Timestamp,
    exit_reason: str,
    config: BacktestConfig,
) -> None:
    """Finalise trade: record exit price and realised P&L."""
    close_comm = _commission(trade.contracts, config)
    trade.exit_date    = current_date
    trade.exit_price   = trade.current_spread_value
    trade.exit_reason  = exit_reason
    trade.commissions += close_comm
    trade.realized_pnl = trade.unrealized_pnl - trade.commissions


# ---------------------------------------------------------------------------
# Main simulation loop
# ---------------------------------------------------------------------------

def run_backtest(
    df: pd.DataFrame,
    config: BacktestConfig,
) -> Tuple[List[PutSpread], pd.Series]:
    """
    Simulate the strategy day by day.

    Returns
    -------
    trades      : list of all closed PutSpread objects
    equity_curve: daily pd.Series of strategy account value (indexed by trading date)
    """
    df = add_signals(df, config.ema_fast, config.ema_slow, config.ma_type, config.entry_price)
    expiry_dates = get_monthly_expiries(df.index[0], df.index[-1])

    account_value  = config.starting_capital
    equity: Dict[pd.Timestamp, float] = {}

    open_positions: List[PutSpread] = []
    closed_trades:  List[PutSpread] = []

    for date, row in df.iterrows():
        price = float(row["close"])
        iv    = float(row["iv"])

        # --- update & check exits first ---
        exit_ema_val = float(row.get(f"{config.ma_type.lower()}{config.exit_ema_period}", 0.0))
        to_close = []
        for trade in open_positions:
            update_trade(trade, date, price, iv, config)
            reason = check_exit(trade, date, config, stock_price=price, exit_ema=exit_ema_val)
            if reason:
                close_trade(trade, date, reason, config)
                account_value += trade.realized_pnl
                to_close.append(trade)
                closed_trades.append(trade)

        for t in to_close:
            open_positions.remove(t)

        # --- check entry signal ---
        if row.get("signal", False) and len(open_positions) < config.max_positions:
            trade = open_trade(date, price, iv, expiry_dates, config, account_value)
            if trade is not None:
                open_positions.append(trade)

        # Mark-to-market: include unrealized P&L from open positions
        unrealized = sum(t.unrealized_pnl for t in open_positions)
        equity[date] = account_value + unrealized

    # Close any positions still open at end of data
    last_date  = df.index[-1]
    last_price = float(df.iloc[-1]["close"])
    last_iv    = float(df.iloc[-1]["iv"])
    for trade in open_positions:
        update_trade(trade, last_date, last_price, last_iv, config)
        close_trade(trade, last_date, "end_of_backtest", config)
        account_value += trade.realized_pnl
        closed_trades.append(trade)

    equity[last_date] = account_value

    equity_curve = pd.Series(equity, name="strategy")
    equity_curve.index = pd.to_datetime(equity_curve.index)

    return closed_trades, equity_curve
