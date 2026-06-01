"""Constraint builders for portfolio optimization."""

import numpy as np


def weights_sum_to_one() -> dict:
    """Constraint: weights must sum to 1.0."""
    return {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}


def weight_bounds(
    n_assets: int,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
) -> list[tuple[float, float]]:
    """Bounds for each weight.

    Args:
        n_assets: Number of assets.
        min_weight: Minimum weight per asset (default 0 = no shorting).
        max_weight: Maximum weight per asset (default 1 = no leverage).

    Returns:
        List of (min, max) tuples for each asset.
    """
    return [(min_weight, max_weight) for _ in range(n_assets)]


def target_return_constraint(
    target: float,
    mean_returns: np.ndarray,
    trading_days: int = 252,
) -> dict:
    """Constraint: portfolio return must equal target.

    Args:
        target: Target annualized return.
        mean_returns: Mean daily returns array.
        trading_days: Trading days per year.

    Returns:
        Equality constraint dict for scipy.optimize.
    """
    def constraint_fn(weights: np.ndarray) -> float:
        port_return = np.sum(mean_returns * weights) * trading_days
        return port_return - target

    return {"type": "eq", "fun": constraint_fn}


def min_return_constraint(
    min_return: float,
    mean_returns: np.ndarray,
    trading_days: int = 252,
) -> dict:
    """Constraint: portfolio return must be at least min_return.

    Args:
        min_return: Minimum annualized return.
        mean_returns: Mean daily returns array.
        trading_days: Trading days per year.

    Returns:
        Inequality constraint dict (>= 0 means satisfied).
    """
    def constraint_fn(weights: np.ndarray) -> float:
        port_return = np.sum(mean_returns * weights) * trading_days
        return port_return - min_return

    return {"type": "ineq", "fun": constraint_fn}


def max_volatility_constraint(
    max_vol: float,
    cov_matrix: np.ndarray,
    trading_days: int = 252,
) -> dict:
    """Constraint: portfolio volatility must be at most max_vol.

    Args:
        max_vol: Maximum annualized volatility.
        cov_matrix: Covariance matrix.
        trading_days: Trading days per year.

    Returns:
        Inequality constraint dict (>= 0 means satisfied).
    """
    def constraint_fn(weights: np.ndarray) -> float:
        port_vol = np.sqrt(weights @ cov_matrix @ weights) * np.sqrt(trading_days)
        return max_vol - port_vol

    return {"type": "ineq", "fun": constraint_fn}
