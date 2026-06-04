"""Monte Carlo simulation for portfolio projections."""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class MonteCarloResult:
    """Results from Monte Carlo simulation."""

    # Simulation paths (each column is a simulation)
    paths: pd.DataFrame

    # Statistics at each time point
    mean_path: pd.Series
    median_path: pd.Series
    percentile_5: pd.Series
    percentile_25: pd.Series
    percentile_75: pd.Series
    percentile_95: pd.Series

    # Final value statistics
    final_mean: float
    final_median: float
    final_std: float
    final_min: float
    final_max: float

    # Probability metrics
    prob_positive_return: float  # P(final > initial)
    prob_double: float  # P(final > 2 * initial)
    prob_loss_10pct: float  # P(final < 0.9 * initial)
    prob_loss_20pct: float  # P(final < 0.8 * initial)

    # Value at Risk
    var_95: float  # 95% VaR (5th percentile of final values)
    cvar_95: float  # Conditional VaR (expected loss beyond VaR)

    def summary(self) -> dict:
        """Return summary as dictionary."""
        return {
            "final_mean": self.final_mean,
            "final_median": self.final_median,
            "final_std": self.final_std,
            "final_min": self.final_min,
            "final_max": self.final_max,
            "prob_positive_return": self.prob_positive_return,
            "prob_double": self.prob_double,
            "prob_loss_10pct": self.prob_loss_10pct,
            "prob_loss_20pct": self.prob_loss_20pct,
            "var_95": self.var_95,
            "cvar_95": self.cvar_95,
        }


def run_monte_carlo(
    weights: dict[str, float],
    returns: pd.DataFrame,
    initial_value: float = 10000.0,
    days: int = 252,
    num_simulations: int = 1000,
    seed: int | None = None,
) -> MonteCarloResult:
    """Run Monte Carlo simulation to project future portfolio values.

    Uses historical return statistics (mean, covariance) to generate
    random future paths assuming multivariate normal returns.

    Args:
        weights: Portfolio weights by symbol.
        returns: Historical daily returns DataFrame.
        initial_value: Starting portfolio value.
        days: Number of days to simulate (252 = 1 year).
        num_simulations: Number of random paths to generate.
        seed: Random seed for reproducibility.

    Returns:
        MonteCarloResult with simulation statistics.
    """
    if seed is not None:
        np.random.seed(seed)

    # Filter to available symbols
    available = [s for s in weights if s in returns.columns]
    if not available:
        raise ValueError("No symbols in weights found in returns")

    # Normalize weights
    total = sum(weights[s] for s in available)
    norm_weights = np.array([weights[s] / total for s in available])

    # Calculate portfolio statistics from historical data
    asset_returns = returns[available]
    mean_returns = asset_returns.mean().values
    cov_matrix = asset_returns.cov().values

    # Portfolio mean and variance
    port_mean = np.dot(norm_weights, mean_returns)
    port_var = np.dot(norm_weights, np.dot(cov_matrix, norm_weights))
    port_std = np.sqrt(port_var)

    # Generate random returns (assuming normal distribution)
    random_returns = np.random.normal(
        port_mean, port_std, size=(days, num_simulations)
    )

    # Calculate cumulative values
    cumulative_returns = np.cumprod(1 + random_returns, axis=0)
    paths = initial_value * cumulative_returns

    # Create DataFrame with dates
    last_date = returns.index[-1]
    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=days)
    paths_df = pd.DataFrame(paths, index=future_dates)

    # Calculate statistics at each time point
    mean_path = paths_df.mean(axis=1)
    median_path = paths_df.median(axis=1)
    percentile_5 = paths_df.quantile(0.05, axis=1)
    percentile_25 = paths_df.quantile(0.25, axis=1)
    percentile_75 = paths_df.quantile(0.75, axis=1)
    percentile_95 = paths_df.quantile(0.95, axis=1)

    # Final value statistics
    final_values = paths_df.iloc[-1]
    final_mean = final_values.mean()
    final_median = final_values.median()
    final_std = final_values.std()
    final_min = final_values.min()
    final_max = final_values.max()

    # Probability metrics
    prob_positive = (final_values > initial_value).mean()
    prob_double = (final_values > 2 * initial_value).mean()
    prob_loss_10 = (final_values < 0.9 * initial_value).mean()
    prob_loss_20 = (final_values < 0.8 * initial_value).mean()

    # Value at Risk
    var_95 = initial_value - np.percentile(final_values, 5)
    losses = initial_value - final_values
    cvar_95 = losses[losses > var_95].mean() if (losses > var_95).any() else var_95

    return MonteCarloResult(
        paths=paths_df,
        mean_path=mean_path,
        median_path=median_path,
        percentile_5=percentile_5,
        percentile_25=percentile_25,
        percentile_75=percentile_75,
        percentile_95=percentile_95,
        final_mean=final_mean,
        final_median=final_median,
        final_std=final_std,
        final_min=final_min,
        final_max=final_max,
        prob_positive_return=prob_positive,
        prob_double=prob_double,
        prob_loss_10pct=prob_loss_10,
        prob_loss_20pct=prob_loss_20,
        var_95=var_95,
        cvar_95=cvar_95,
    )


def run_monte_carlo_correlated(
    weights: dict[str, float],
    returns: pd.DataFrame,
    initial_value: float = 10000.0,
    days: int = 252,
    num_simulations: int = 1000,
    seed: int | None = None,
) -> MonteCarloResult:
    """Run Monte Carlo with correlated asset returns.

    More accurate than simple portfolio variance approach as it
    preserves the correlation structure between assets.

    Args:
        weights: Portfolio weights by symbol.
        returns: Historical daily returns DataFrame.
        initial_value: Starting portfolio value.
        days: Number of days to simulate.
        num_simulations: Number of random paths to generate.
        seed: Random seed for reproducibility.

    Returns:
        MonteCarloResult with simulation statistics.
    """
    if seed is not None:
        np.random.seed(seed)

    # Filter to available symbols
    available = [s for s in weights if s in returns.columns]
    if not available:
        raise ValueError("No symbols in weights found in returns")

    # Normalize weights
    total = sum(weights[s] for s in available)
    norm_weights = np.array([weights[s] / total for s in available])

    # Get historical statistics
    asset_returns = returns[available]
    mean_returns = asset_returns.mean().values
    cov_matrix = asset_returns.cov().values

    # Cholesky decomposition for correlated sampling
    try:
        chol = np.linalg.cholesky(cov_matrix)
    except np.linalg.LinAlgError:
        # If not positive definite, add small diagonal
        cov_matrix += np.eye(len(available)) * 1e-8
        chol = np.linalg.cholesky(cov_matrix)

    # Generate correlated random returns
    all_paths = np.zeros((days, num_simulations))

    for sim in range(num_simulations):
        portfolio_values = [initial_value]

        for _ in range(days):
            # Generate correlated random returns
            z = np.random.standard_normal(len(available))
            asset_returns_sim = mean_returns + chol @ z

            # Portfolio return
            port_return = np.dot(norm_weights, asset_returns_sim)
            new_value = portfolio_values[-1] * (1 + port_return)
            portfolio_values.append(new_value)

        all_paths[:, sim] = portfolio_values[1:]

    # Create DataFrame with dates
    last_date = returns.index[-1]
    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=days)
    paths_df = pd.DataFrame(all_paths, index=future_dates)

    # Calculate statistics (same as simple version)
    mean_path = paths_df.mean(axis=1)
    median_path = paths_df.median(axis=1)
    percentile_5 = paths_df.quantile(0.05, axis=1)
    percentile_25 = paths_df.quantile(0.25, axis=1)
    percentile_75 = paths_df.quantile(0.75, axis=1)
    percentile_95 = paths_df.quantile(0.95, axis=1)

    final_values = paths_df.iloc[-1]
    final_mean = final_values.mean()
    final_median = final_values.median()
    final_std = final_values.std()
    final_min = final_values.min()
    final_max = final_values.max()

    prob_positive = (final_values > initial_value).mean()
    prob_double = (final_values > 2 * initial_value).mean()
    prob_loss_10 = (final_values < 0.9 * initial_value).mean()
    prob_loss_20 = (final_values < 0.8 * initial_value).mean()

    var_95 = initial_value - np.percentile(final_values, 5)
    losses = initial_value - final_values
    cvar_95 = losses[losses > var_95].mean() if (losses > var_95).any() else var_95

    return MonteCarloResult(
        paths=paths_df,
        mean_path=mean_path,
        median_path=median_path,
        percentile_5=percentile_5,
        percentile_25=percentile_25,
        percentile_75=percentile_75,
        percentile_95=percentile_95,
        final_mean=final_mean,
        final_median=final_median,
        final_std=final_std,
        final_min=final_min,
        final_max=final_max,
        prob_positive_return=prob_positive,
        prob_double=prob_double,
        prob_loss_10pct=prob_loss_10,
        prob_loss_20pct=prob_loss_20,
        var_95=var_95,
        cvar_95=cvar_95,
    )
