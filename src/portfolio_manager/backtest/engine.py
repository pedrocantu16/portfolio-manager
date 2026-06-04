"""Backtesting engine for portfolio strategies."""

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import pandas as pd


class RebalanceFrequency(StrEnum):
    """Rebalancing frequency options."""

    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"
    NONE = "none"


@dataclass
class BacktestResult:
    """Results from a portfolio backtest."""

    # Time series
    portfolio_values: pd.Series
    benchmark_values: pd.Series | None
    daily_returns: pd.Series

    # Summary metrics
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_duration: int  # days

    # Comparison (if benchmark provided)
    benchmark_return: float | None
    alpha: float | None
    beta: float | None

    # Trade statistics
    num_rebalances: int
    total_turnover: float  # sum of absolute weight changes

    def summary(self) -> dict:
        """Return summary as dictionary."""
        return {
            "total_return": self.total_return,
            "annualized_return": self.annualized_return,
            "volatility": self.volatility,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_duration_days": self.max_drawdown_duration,
            "benchmark_return": self.benchmark_return,
            "alpha": self.alpha,
            "beta": self.beta,
            "num_rebalances": self.num_rebalances,
            "total_turnover": self.total_turnover,
        }


def _get_rebalance_dates(
    dates: pd.DatetimeIndex,
    frequency: RebalanceFrequency,
) -> set[pd.Timestamp]:
    """Get dates when rebalancing should occur."""
    if frequency == RebalanceFrequency.NONE:
        return set()

    if frequency == RebalanceFrequency.DAILY:
        return set(dates)

    rebalance_dates = set()

    if frequency == RebalanceFrequency.WEEKLY:
        # Rebalance on Mondays (or first trading day of week)
        for date in dates:
            if date.dayofweek == 0:  # Monday
                rebalance_dates.add(date)

    elif frequency == RebalanceFrequency.MONTHLY:
        # Rebalance on first trading day of month
        current_month = None
        for date in dates:
            if date.month != current_month:
                rebalance_dates.add(date)
                current_month = date.month

    elif frequency == RebalanceFrequency.QUARTERLY:
        # Rebalance on first trading day of quarter
        current_quarter = None
        for date in dates:
            quarter = (date.month - 1) // 3
            if quarter != current_quarter:
                rebalance_dates.add(date)
                current_quarter = quarter

    elif frequency == RebalanceFrequency.YEARLY:
        # Rebalance on first trading day of year
        current_year = None
        for date in dates:
            if date.year != current_year:
                rebalance_dates.add(date)
                current_year = date.year

    return rebalance_dates


def _calculate_max_drawdown_duration(prices: pd.Series) -> int:
    """Calculate the longest drawdown duration in days."""
    running_max = prices.expanding().max()
    drawdown = prices / running_max - 1

    # Find drawdown periods
    in_drawdown = drawdown < 0
    if not in_drawdown.any():
        return 0

    # Calculate duration of each drawdown period
    drawdown_starts = in_drawdown & ~in_drawdown.shift(1, fill_value=False)
    drawdown_ends = ~in_drawdown & in_drawdown.shift(1, fill_value=False)

    max_duration = 0
    current_start = None

    for date in prices.index:
        if drawdown_starts.get(date, False):
            current_start = date
        elif drawdown_ends.get(date, False) and current_start is not None:
            duration = (date - current_start).days
            max_duration = max(max_duration, duration)
            current_start = None

    # Handle ongoing drawdown
    if current_start is not None:
        duration = (prices.index[-1] - current_start).days
        max_duration = max(max_duration, duration)

    return max_duration


def run_backtest(
    prices: pd.DataFrame,
    target_weights: dict[str, float],
    initial_value: float = 10000.0,
    rebalance_frequency: RebalanceFrequency = RebalanceFrequency.MONTHLY,
    benchmark_prices: pd.Series | None = None,
    risk_free_rate: float = 0.045,
) -> BacktestResult:
    """Run a historical backtest with the given weights.

    Args:
        prices: DataFrame of daily prices for each asset.
        target_weights: Target portfolio weights (must sum to 1).
        initial_value: Starting portfolio value.
        rebalance_frequency: How often to rebalance to target weights.
        benchmark_prices: Optional benchmark price series for comparison.
        risk_free_rate: Annual risk-free rate for Sharpe calculation.

    Returns:
        BacktestResult with performance metrics.
    """
    # Validate inputs
    available_symbols = [s for s in target_weights if s in prices.columns]
    if not available_symbols:
        raise ValueError("No symbols in target_weights found in prices")

    # Normalize weights to available symbols
    total_weight = sum(target_weights[s] for s in available_symbols)
    weights = {s: target_weights[s] / total_weight for s in available_symbols}

    # Calculate daily returns
    returns = prices[available_symbols].pct_change().dropna()
    dates = returns.index

    # Initialize portfolio
    portfolio_values = [initial_value]
    current_weights = weights.copy()
    rebalance_dates = _get_rebalance_dates(dates, rebalance_frequency)
    num_rebalances = 0
    total_turnover = 0.0

    # Simulate day by day
    for i, date in enumerate(dates):
        # Calculate portfolio return for this day
        daily_return = sum(
            current_weights[s] * returns.loc[date, s] for s in available_symbols
        )
        new_value = portfolio_values[-1] * (1 + daily_return)
        portfolio_values.append(new_value)

        # Update weights based on returns (drift)
        for symbol in available_symbols:
            current_weights[symbol] *= 1 + returns.loc[date, symbol]

        # Normalize weights
        total = sum(current_weights.values())
        if total > 0:
            current_weights = {s: w / total for s, w in current_weights.items()}

        # Rebalance if needed
        if date in rebalance_dates:
            turnover = sum(
                abs(current_weights[s] - weights[s]) for s in available_symbols
            )
            total_turnover += turnover
            current_weights = weights.copy()
            num_rebalances += 1

    # Create portfolio value series
    portfolio_series = pd.Series(
        portfolio_values[1:],  # Skip initial value (before first return)
        index=dates,
        name="portfolio",
    )

    # Calculate daily returns
    daily_returns = portfolio_series.pct_change().dropna()

    # Calculate metrics
    total_return = (portfolio_series.iloc[-1] / initial_value) - 1
    trading_days = len(daily_returns)
    years = trading_days / 252
    annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

    volatility = daily_returns.std() * np.sqrt(252)

    # Sharpe ratio
    excess_return = annualized_return - risk_free_rate
    sharpe_ratio = excess_return / volatility if volatility > 0 else 0

    # Sortino ratio (downside deviation)
    downside_returns = daily_returns[daily_returns < 0]
    downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
    sortino_ratio = excess_return / downside_std if downside_std > 0 else 0

    # Max drawdown
    running_max = portfolio_series.expanding().max()
    drawdown = portfolio_series / running_max - 1
    max_drawdown = drawdown.min()
    max_dd_duration = _calculate_max_drawdown_duration(portfolio_series)

    # Benchmark comparison
    benchmark_return = None
    alpha = None
    beta = None
    benchmark_values = None

    if benchmark_prices is not None:
        # Align benchmark to same dates
        aligned_benchmark = benchmark_prices.reindex(dates).dropna()
        if len(aligned_benchmark) > 0:
            benchmark_values = aligned_benchmark / aligned_benchmark.iloc[0] * initial_value
            benchmark_return = (aligned_benchmark.iloc[-1] / aligned_benchmark.iloc[0]) - 1

            # Calculate alpha and beta
            bench_returns = aligned_benchmark.pct_change().dropna()
            aligned = pd.concat([daily_returns, bench_returns], axis=1).dropna()

            if len(aligned) > 1:
                port_ret = aligned.iloc[:, 0].values
                bench_ret = aligned.iloc[:, 1].values

                covariance = np.cov(port_ret, bench_ret)[0, 1]
                bench_var = np.var(bench_ret, ddof=1)
                beta = covariance / bench_var if bench_var > 0 else 1.0

                # Annualized alpha
                port_annual = np.mean(port_ret) * 252
                bench_annual = np.mean(bench_ret) * 252
                alpha = port_annual - (risk_free_rate + beta * (bench_annual - risk_free_rate))

    return BacktestResult(
        portfolio_values=portfolio_series,
        benchmark_values=benchmark_values,
        daily_returns=daily_returns,
        total_return=total_return,
        annualized_return=annualized_return,
        volatility=volatility,
        sharpe_ratio=sharpe_ratio,
        sortino_ratio=sortino_ratio,
        max_drawdown=max_drawdown,
        max_drawdown_duration=max_dd_duration,
        benchmark_return=benchmark_return,
        alpha=alpha,
        beta=beta,
        num_rebalances=num_rebalances,
        total_turnover=total_turnover,
    )
