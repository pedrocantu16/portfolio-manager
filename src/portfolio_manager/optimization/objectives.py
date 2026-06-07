"""Objective functions for portfolio optimization."""

import numpy as np


def negative_sharpe_ratio(
    weights: np.ndarray,
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_free_rate: float,
    trading_days: int = 252,
) -> float:
    """Negative Sharpe ratio (for minimization).

    Args:
        weights: Portfolio weights array.
        mean_returns: Mean daily returns for each asset.
        cov_matrix: Covariance matrix of returns.
        risk_free_rate: Annualized risk-free rate.
        trading_days: Trading days per year.

    Returns:
        Negative Sharpe ratio (minimize this to maximize Sharpe).
    """
    portfolio_return = np.sum(mean_returns * weights) * trading_days
    portfolio_vol = np.sqrt(weights @ cov_matrix @ weights) * np.sqrt(trading_days)

    if portfolio_vol == 0:
        return 0.0

    sharpe = (portfolio_return - risk_free_rate) / portfolio_vol
    return -sharpe


def portfolio_volatility(
    weights: np.ndarray,
    cov_matrix: np.ndarray,
    trading_days: int = 252,
) -> float:
    """Portfolio volatility (annualized).

    Args:
        weights: Portfolio weights array.
        cov_matrix: Covariance matrix of returns.
        trading_days: Trading days per year.

    Returns:
        Annualized portfolio volatility.
    """
    return np.sqrt(weights @ cov_matrix @ weights) * np.sqrt(trading_days)


def portfolio_return(
    weights: np.ndarray,
    mean_returns: np.ndarray,
    trading_days: int = 252,
) -> float:
    """Portfolio expected return (annualized).

    Args:
        weights: Portfolio weights array.
        mean_returns: Mean daily returns for each asset.
        trading_days: Trading days per year.

    Returns:
        Annualized expected return.
    """
    return np.sum(mean_returns * weights) * trading_days


def negative_return(
    weights: np.ndarray,
    mean_returns: np.ndarray,
    trading_days: int = 252,
) -> float:
    """Negative portfolio return (for minimization when maximizing return)."""
    return -portfolio_return(weights, mean_returns, trading_days)


def downside_deviation(
    weights: np.ndarray,
    returns: np.ndarray,
    risk_free_rate: float,
    trading_days: int = 252,
) -> float:
    """Calculate annualized downside deviation for a portfolio.

    Args:
        weights: Portfolio weights array.
        returns: Daily returns matrix (rows = days, cols = assets).
        risk_free_rate: Annualized risk-free rate.
        trading_days: Trading days per year.

    Returns:
        Annualized downside deviation.
    """
    daily_rf = risk_free_rate / trading_days
    portfolio_returns = returns @ weights
    downside_returns = np.minimum(portfolio_returns - daily_rf, 0)
    downside_var = np.mean(downside_returns ** 2)
    return np.sqrt(downside_var) * np.sqrt(trading_days)


def negative_sortino_ratio(
    weights: np.ndarray,
    mean_returns: np.ndarray,
    returns: np.ndarray,
    risk_free_rate: float,
    trading_days: int = 252,
) -> float:
    """Negative Sortino ratio (for minimization).

    Sortino ratio uses downside deviation instead of total volatility,
    only penalizing returns below the risk-free rate.

    Args:
        weights: Portfolio weights array.
        mean_returns: Mean daily returns for each asset.
        returns: Daily returns matrix (rows = days, cols = assets).
        risk_free_rate: Annualized risk-free rate.
        trading_days: Trading days per year.

    Returns:
        Negative Sortino ratio (minimize this to maximize Sortino).
    """
    port_return = np.sum(mean_returns * weights) * trading_days
    dd = downside_deviation(weights, returns, risk_free_rate, trading_days)

    if dd == 0:
        return 0.0

    sortino = (port_return - risk_free_rate) / dd
    return -sortino
