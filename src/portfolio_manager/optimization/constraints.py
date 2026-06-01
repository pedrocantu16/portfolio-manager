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


def sector_max_weight_constraint(
    max_sector_weight: float,
    symbols: list[str],
    sector_map: dict[str, str],
) -> list[dict]:
    """Constraint: maximum weight per sector.

    Args:
        max_sector_weight: Maximum total weight for any single sector.
        symbols: List of symbols in order matching weight array.
        sector_map: Dictionary mapping symbol to sector name.

    Returns:
        List of inequality constraints (one per sector).
    """
    # Group symbols by sector
    sectors: dict[str, list[int]] = {}
    for i, symbol in enumerate(symbols):
        sector = sector_map.get(symbol, "Unknown")
        if sector not in sectors:
            sectors[sector] = []
        sectors[sector].append(i)

    constraints = []
    for sector, indices in sectors.items():
        if sector == "Unknown":
            continue  # Don't constrain unknown sectors

        def make_constraint(idx_list: list[int]) -> dict:
            def constraint_fn(weights: np.ndarray) -> float:
                sector_weight = sum(weights[i] for i in idx_list)
                return max_sector_weight - sector_weight

            return {"type": "ineq", "fun": constraint_fn}

        constraints.append(make_constraint(indices))

    return constraints


def sector_min_weight_constraint(
    min_sector_weight: float,
    symbols: list[str],
    sector_map: dict[str, str],
    sectors_to_constrain: list[str] | None = None,
) -> list[dict]:
    """Constraint: minimum weight per sector.

    Args:
        min_sector_weight: Minimum total weight for specified sectors.
        symbols: List of symbols in order matching weight array.
        sector_map: Dictionary mapping symbol to sector name.
        sectors_to_constrain: List of sectors to apply constraint to.
            If None, applies to all sectors.

    Returns:
        List of inequality constraints.
    """
    # Group symbols by sector
    sectors: dict[str, list[int]] = {}
    for i, symbol in enumerate(symbols):
        sector = sector_map.get(symbol, "Unknown")
        if sector not in sectors:
            sectors[sector] = []
        sectors[sector].append(i)

    constraints = []
    for sector, indices in sectors.items():
        if sectors_to_constrain and sector not in sectors_to_constrain:
            continue
        if sector == "Unknown":
            continue

        def make_constraint(idx_list: list[int]) -> dict:
            def constraint_fn(weights: np.ndarray) -> float:
                sector_weight = sum(weights[i] for i in idx_list)
                return sector_weight - min_sector_weight

            return {"type": "ineq", "fun": constraint_fn}

        constraints.append(make_constraint(indices))

    return constraints
