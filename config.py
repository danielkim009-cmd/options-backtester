from dataclasses import dataclass


@dataclass
class BacktestConfig:
    # Underlying
    ticker: str = "SPY"
    start_date: str = "2006-04-03"
    end_date: str = "2026-03-31"

    # MA settings
    ma_type:        str = "EMA"   # "EMA" or "SMA"
    entry_price:    str = "close" # "close" or "low"
    ema_fast: int = 21
    ema_slow: int = 50
    entry_a_ema_buffer: float = 0.0  # Allow low up to this % above EMA21 (0.0=touch, 0.01=1%, 0.02=2%)

    # Options settings
    entry_dte: int = 45          # Target DTE at entry
    exit_dte: int = 7            # Close at or below this DTE
    profit_target: float = 0.50  # Close at 50% of max profit
    short_put_delta: float = 0.40
    spread_width: float = 5.0    # Dollar width between strikes

    # Position sizing
    starting_capital: float = 10_000.0
    deploy_pct: float = 0.25     # Fraction of current account value to put at risk per trade
    max_positions: int = 1       # Max concurrent open positions

    # Exit rules
    exit_below_ema: bool = True      # Close trade if price crosses below the exit EMA
    exit_ema_period: int = 200       # EMA period for exit condition
    use_stop_loss: bool = False      # Close trade if loss exceeds stop_loss_multiple × credit
    stop_loss_multiple: float = 2.0  # 2.0 = exit when loss = 200% of credit received

    # Costs
    commission_per_contract: float = 0.65  # Per leg, per contract

    # Market params
    risk_free_rate: float = 0.05
    iv_premium: float = 1.10     # IV premium multiplier over HV (individual stocks)
