"""Performance ratio calculations."""

import pandas as pd

from portfolio_manager.metrics.returns import calculate_annualized_return
from portfolio_manager.metrics.risk import calculate_downside_deviation, calculate_volatility


def calculate_sharpe_ratio(
    returns: pd.Series | float,
    risk_free_rate: float,
    volatility: float | None = None,
    annualized: bool = True,
    trading_days: int = 252,
) -> float:
    """Calculate the Sharpe ratio.

    Args:
        returns: Series of returns or annualized return as float.
        risk_free_rate: Annualized risk-free rate.
        volatility: Annualized volatility (calculated if not provided).
        annualized: Whether inputs are already annualized.
        trading_days: Number of trading days per year.

    Returns:
        Sharpe ratio.
    """
    if isinstance(returns, pd.Series):
        if returns.empty:
            return 0.0
        ann_return = calculate_annualized_return(returns, trading_days)
        ann_vol = calculate_volatility(returns, annualize=True, trading_days=trading_days)
    else:
        ann_return = returns
        ann_vol = volatility if volatility is not None else 0.0

    if ann_vol == 0:
        return 0.0

    excess_return = ann_return - risk_free_rate
    return float(excess_return / ann_vol)


def calculate_sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float,
    target: float = 0.0,
    trading_days: int = 252,
) -> float:
    """Calculate the Sortino ratio.

    Args:
        returns: Series of returns.
        risk_free_rate: Annualized risk-free rate.
        target: Target return for downside calculation.
        trading_days: Number of trading days per year.

    Returns:
        Sortino ratio.
    """
    if returns.empty:
        return 0.0

    ann_return = calculate_annualized_return(returns, trading_days)
    downside_dev = calculate_downside_deviation(
        returns, target=target, annualize=True, trading_days=trading_days
    )

    if downside_dev == 0:
        return 0.0

    excess_return = ann_return - risk_free_rate
    return float(excess_return / downside_dev)


def calculate_information_ratio(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    trading_days: int = 252,
) -> float:
    """Calculate the Information ratio.

    Args:
        returns: Series of portfolio returns.
        benchmark_returns: Series of benchmark returns.
        trading_days: Number of trading days per year.

    Returns:
        Information ratio.
    """
    if returns.empty or benchmark_returns.empty:
        return 0.0

    # Align the series
    aligned = pd.concat([returns, benchmark_returns], axis=1).dropna()
    if aligned.empty:
        return 0.0

    active_returns = aligned.iloc[:, 0] - aligned.iloc[:, 1]

    ann_active_return = calculate_annualized_return(active_returns, trading_days)
    tracking_error = calculate_volatility(active_returns, annualize=True, trading_days=trading_days)

    if tracking_error == 0:
        return 0.0

    return float(ann_active_return / tracking_error)


def calculate_calmar_ratio(
    returns: pd.Series,
    max_drawdown: float,
    trading_days: int = 252,
) -> float:
    """Calculate the Calmar ratio.

    Args:
        returns: Series of returns.
        max_drawdown: Maximum drawdown (as negative decimal).
        trading_days: Number of trading days per year.

    Returns:
        Calmar ratio.
    """
    if returns.empty or max_drawdown == 0:
        return 0.0

    ann_return = calculate_annualized_return(returns, trading_days)

    # Max drawdown is negative, so we negate it
    return float(ann_return / abs(max_drawdown))
