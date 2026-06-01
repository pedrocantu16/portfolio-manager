"""Return calculations for portfolio analysis."""

import numpy as np
import pandas as pd


def calculate_returns(prices: pd.DataFrame, method: str = "log") -> pd.DataFrame:
    """Calculate returns from price data.

    Args:
        prices: DataFrame of prices with dates as index and symbols as columns.
        method: Return calculation method - "log" for log returns, "simple" for simple returns.

    Returns:
        DataFrame of returns with same structure as input.
    """
    if prices.empty:
        return pd.DataFrame()

    if method == "log":
        returns = np.log(prices / prices.shift(1))
    else:  # simple
        returns = prices.pct_change()

    return returns.dropna()


def calculate_annualized_return(
    returns: pd.Series | pd.DataFrame, trading_days: int = 252
) -> float | pd.Series:
    """Calculate annualized return from daily returns.

    Args:
        returns: Series or DataFrame of daily returns.
        trading_days: Number of trading days per year.

    Returns:
        Annualized return as a decimal.
    """
    if isinstance(returns, pd.DataFrame):
        return returns.apply(lambda x: calculate_annualized_return(x, trading_days))

    if returns.empty:
        return 0.0

    # For log returns, sum and annualize
    total_return = returns.sum()
    n_days = len(returns)
    annualized = (total_return / n_days) * trading_days

    return float(annualized)


def calculate_cumulative_return(returns: pd.Series | pd.DataFrame) -> float | pd.Series:
    """Calculate cumulative return over the period.

    Args:
        returns: Series or DataFrame of returns.

    Returns:
        Cumulative return as a decimal (e.g., 0.10 for 10% gain).
    """
    if isinstance(returns, pd.DataFrame):
        return returns.apply(calculate_cumulative_return)

    if returns.empty:
        return 0.0

    # For log returns, sum them; for simple returns, compound them
    # Assuming log returns based on our calculation method
    return float(np.exp(returns.sum()) - 1)


def calculate_portfolio_return(
    weights: dict[str, float],
    returns: pd.DataFrame,
    annualize: bool = True,
    trading_days: int = 252,
) -> float:
    """Calculate portfolio return given weights and asset returns.

    Args:
        weights: Dictionary mapping symbol to weight (should sum to 1.0).
        returns: DataFrame of returns with symbols as columns.
        annualize: Whether to annualize the return.
        trading_days: Number of trading days per year.

    Returns:
        Portfolio return (annualized if requested).
    """
    if returns.empty or not weights:
        return 0.0

    # Align weights with available returns
    available_symbols = [s for s in weights.keys() if s in returns.columns]
    if not available_symbols:
        return 0.0

    # Create weight array aligned with returns columns
    weight_series = pd.Series({s: weights[s] for s in available_symbols})
    aligned_returns = returns[available_symbols]

    # Calculate portfolio daily returns
    portfolio_returns = (aligned_returns * weight_series).sum(axis=1)

    if annualize:
        return calculate_annualized_return(portfolio_returns, trading_days)
    else:
        return calculate_cumulative_return(portfolio_returns)
