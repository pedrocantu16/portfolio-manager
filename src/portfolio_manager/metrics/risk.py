"""Risk metrics calculations."""

import numpy as np
import pandas as pd


def calculate_volatility(
    returns: pd.Series | pd.DataFrame,
    annualize: bool = True,
    trading_days: int = 252,
) -> float | pd.Series:
    """Calculate volatility (standard deviation) of returns.

    Args:
        returns: Series or DataFrame of returns.
        annualize: Whether to annualize the volatility.
        trading_days: Number of trading days per year.

    Returns:
        Volatility (annualized if requested).
    """
    if isinstance(returns, pd.DataFrame):
        return returns.apply(lambda x: calculate_volatility(x, annualize, trading_days))

    if returns.empty:
        return 0.0

    vol = returns.std()

    if annualize:
        vol = vol * np.sqrt(trading_days)

    return float(vol)


def calculate_portfolio_volatility(
    weights: dict[str, float],
    returns: pd.DataFrame,
    annualize: bool = True,
    trading_days: int = 252,
) -> float:
    """Calculate portfolio volatility using covariance matrix.

    Args:
        weights: Dictionary mapping symbol to weight.
        returns: DataFrame of returns with symbols as columns.
        annualize: Whether to annualize the volatility.
        trading_days: Number of trading days per year.

    Returns:
        Portfolio volatility (annualized if requested).
    """
    if returns.empty or not weights:
        return 0.0

    # Align weights with available returns
    available_symbols = [s for s in weights.keys() if s in returns.columns]
    if not available_symbols:
        return 0.0

    # Create weight array
    weight_array = np.array([weights[s] for s in available_symbols])
    aligned_returns = returns[available_symbols]

    # Calculate covariance matrix
    cov_matrix = aligned_returns.cov()

    # Portfolio variance: w' * Σ * w
    portfolio_variance = weight_array @ cov_matrix.values @ weight_array
    portfolio_vol = np.sqrt(portfolio_variance)

    if annualize:
        portfolio_vol = portfolio_vol * np.sqrt(trading_days)

    return float(portfolio_vol)


def calculate_var(
    returns: pd.Series | pd.DataFrame,
    confidence: float = 0.95,
    method: str = "historical",
) -> float | pd.Series:
    """Calculate Value at Risk (VaR).

    Args:
        returns: Series or DataFrame of returns.
        confidence: Confidence level (e.g., 0.95 for 95% VaR).
        method: VaR calculation method - "historical" or "parametric".

    Returns:
        VaR as a negative decimal (e.g., -0.05 means 5% loss).
    """
    if isinstance(returns, pd.DataFrame):
        return returns.apply(lambda x: calculate_var(x, confidence, method))

    if returns.empty:
        return 0.0

    if method == "historical":
        # Historical VaR: percentile of actual returns
        var = np.percentile(returns.dropna(), (1 - confidence) * 100)
    else:  # parametric
        # Parametric VaR: assumes normal distribution
        from scipy import stats
        mean = returns.mean()
        std = returns.std()
        var = stats.norm.ppf(1 - confidence, mean, std)

    return float(var)


def calculate_covariance_matrix(
    returns: pd.DataFrame,
    annualize: bool = True,
    trading_days: int = 252,
) -> pd.DataFrame:
    """Calculate covariance matrix of returns.

    Args:
        returns: DataFrame of returns with symbols as columns.
        annualize: Whether to annualize the covariance.
        trading_days: Number of trading days per year.

    Returns:
        Covariance matrix as DataFrame.
    """
    if returns.empty:
        return pd.DataFrame()

    cov = returns.cov()

    if annualize:
        cov = cov * trading_days

    return cov


def calculate_correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Calculate correlation matrix of returns.

    Args:
        returns: DataFrame of returns with symbols as columns.

    Returns:
        Correlation matrix as DataFrame.
    """
    if returns.empty:
        return pd.DataFrame()

    return returns.corr()


def calculate_max_drawdown(prices: pd.Series | pd.DataFrame) -> float | pd.Series:
    """Calculate maximum drawdown from price series.

    Args:
        prices: Series or DataFrame of prices.

    Returns:
        Maximum drawdown as a negative decimal (e.g., -0.20 for 20% drawdown).
    """
    if isinstance(prices, pd.DataFrame):
        return prices.apply(calculate_max_drawdown)

    if prices.empty:
        return 0.0

    # Calculate running maximum
    running_max = prices.expanding().max()

    # Calculate drawdown at each point
    drawdowns = (prices - running_max) / running_max

    # Return maximum drawdown (most negative value)
    return float(drawdowns.min())


def calculate_downside_deviation(
    returns: pd.Series,
    target: float = 0.0,
    annualize: bool = True,
    trading_days: int = 252,
) -> float:
    """Calculate downside deviation (for Sortino ratio).

    Args:
        returns: Series of returns.
        target: Target return (default 0).
        annualize: Whether to annualize.
        trading_days: Number of trading days per year.

    Returns:
        Downside deviation.
    """
    if returns.empty:
        return 0.0

    # Only consider returns below target
    downside_returns = returns[returns < target]

    if len(downside_returns) == 0:
        return 0.0

    downside_dev = np.sqrt(((downside_returns - target) ** 2).mean())

    if annualize:
        downside_dev = downside_dev * np.sqrt(trading_days)

    return float(downside_dev)
