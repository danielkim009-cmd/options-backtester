"""
Performance analytics: statistics and chart data for the Streamlit dashboard.
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from .options_pricer import PutSpread


def compute_stats(trades: List[PutSpread]) -> dict:
    """Return a dict of performance metrics for a list of closed trades."""
    if not trades:
        return {}

    pnls = [t.realized_pnl for t in trades]
    total_pnl = sum(pnls)
    wins   = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    win_rate    = len(wins) / len(pnls) * 100
    avg_win     = float(np.mean(wins))   if wins   else 0.0
    avg_loss    = float(np.mean(losses)) if losses else 0.0
    gross_profit = sum(wins)
    gross_loss   = abs(sum(losses))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

    cum_pnl      = np.cumsum(pnls)
    running_max  = np.maximum.accumulate(cum_pnl)
    drawdowns    = cum_pnl - running_max
    max_drawdown = float(drawdowns.min())

    durations = [
        (pd.Timestamp(t.exit_date) - pd.Timestamp(t.entry_date)).days
        for t in trades
        if t.exit_date and t.entry_date
    ]
    avg_duration = float(np.mean(durations)) if durations else 0.0

    exit_reasons: dict[str, int] = {}
    for t in trades:
        exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1

    # Annualised Sharpe (assumes daily P&L sampled per trade; rough)
    if len(pnls) > 1 and np.std(pnls) > 0:
        sharpe = (np.mean(pnls) / np.std(pnls)) * np.sqrt(252 / avg_duration) if avg_duration > 0 else 0.0
    else:
        sharpe = 0.0

    # --- ROCAR: Return on Capital At Risk ---
    # Capital at risk per trade = max_loss (spread_width - credit) × 100 × contracts + commissions
    total_capital_at_risk = sum(t.max_loss for t in trades)
    rocar = (total_pnl / total_capital_at_risk) if total_capital_at_risk > 0 else 0.0

    # --- CAGR ---
    # Annualise ROCAR over the period from first entry to last exit
    first_entry = pd.Timestamp(trades[0].entry_date)
    last_exit   = pd.Timestamp(trades[-1].exit_date)
    years = (last_exit - first_entry).days / 365.25
    if years > 0 and rocar > -1:
        cagr = (1 + rocar) ** (1 / years) - 1
    else:
        cagr = 0.0

    return {
        "total_trades":           len(trades),
        "winning_trades":         len(wins),
        "losing_trades":          len(losses),
        "total_pnl":              total_pnl,
        "win_rate":               win_rate,
        "avg_pnl_per_trade":      total_pnl / len(trades),
        "avg_win":                avg_win,
        "avg_loss":               avg_loss,
        "profit_factor":          profit_factor,
        "max_drawdown":           max_drawdown,
        "avg_duration_days":      avg_duration,
        "sharpe":                 sharpe,
        "exit_reasons":           exit_reasons,
        "pnls":                   pnls,
        "cum_pnl":                list(cum_pnl),
        "rocar":                  rocar,
        "cagr":                   cagr,
        "total_capital_at_risk":  total_capital_at_risk,
        "years":                  years,
    }


def trades_to_dataframe(trades: List[PutSpread], ticker: str = "") -> pd.DataFrame:
    """Convert trade list to a tidy DataFrame for display / export.

    When `ticker` is provided a 'Ticker' column is prepended — used in
    multi-ticker mode to identify which sub-account each trade belongs to.
    """
    rows = []
    for i, t in enumerate(trades, start=1):
        row: dict = {}
        if ticker:
            row["Ticker"] = ticker
        row.update({
            "#":             i,
            "Entry Date":    pd.Timestamp(t.entry_date).strftime("%Y-%m-%d"),
            "Exit Date":     pd.Timestamp(t.exit_date).strftime("%Y-%m-%d") if t.exit_date else "Open",
            "Expiry":        pd.Timestamp(t.expiry_date).strftime("%Y-%m-%d"),
            "Entry DTE":     t.entry_dte,
            "Short Strike":  t.short_strike,
            "Long Strike":   t.long_strike,
            "Net Credit":    round(t.net_credit, 2),
            "Exit Price":    round(t.exit_price, 2),
            "P&L ($)":       round(t.realized_pnl, 2),
            "P&L (% Acct)":  round(t.realized_pnl / t.account_value_at_entry * 100, 2)
                             if t.account_value_at_entry else 0.0,
            "Days Held":     (pd.Timestamp(t.exit_date) - pd.Timestamp(t.entry_date)).days if t.exit_date else None,
            "Exit Reason":   t.exit_reason,
            "Stock @ Entry": round(t.entry_stock_price, 2),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def combine_equity_curves(curve_a: pd.Series, curve_b: pd.Series) -> pd.Series:
    """
    Combine two sub-account equity curves into a single portfolio curve.
    Forward-fills gaps (e.g. different holiday calendars) before summing.
    """
    union_idx = curve_a.index.union(curve_b.index)
    combined  = curve_a.reindex(union_idx).ffill() + curve_b.reindex(union_idx).ffill()
    combined.name = "portfolio"
    return combined
