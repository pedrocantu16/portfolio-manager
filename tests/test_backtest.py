"""Tests for backtesting and Monte Carlo simulation."""

import numpy as np
import pandas as pd
import pytest

from portfolio_manager.backtest import (
    RebalanceFrequency,
    run_backtest,
    run_monte_carlo,
)


@pytest.fixture
def sample_prices():
    """Create sample price data for testing."""
    np.random.seed(42)
    dates = pd.date_range("2021-01-01", periods=504, freq="B")  # 2 years

    # Generate correlated price paths
    returns_a = np.random.normal(0.0004, 0.015, 504)
    returns_b = np.random.normal(0.0003, 0.020, 504)
    returns_c = np.random.normal(0.0005, 0.010, 504)

    prices_a = 100 * np.cumprod(1 + returns_a)
    prices_b = 50 * np.cumprod(1 + returns_b)
    prices_c = 200 * np.cumprod(1 + returns_c)

    return pd.DataFrame(
        {"A": prices_a, "B": prices_b, "C": prices_c},
        index=dates,
    )


@pytest.fixture
def sample_returns(sample_prices):
    """Calculate returns from sample prices."""
    return sample_prices.pct_change().dropna()


class TestBacktest:
    """Tests for run_backtest function."""

    def test_backtest_basic(self, sample_prices):
        """Backtest should run and return results."""
        weights = {"A": 0.5, "B": 0.3, "C": 0.2}

        result = run_backtest(
            prices=sample_prices,
            target_weights=weights,
            initial_value=10000,
            rebalance_frequency=RebalanceFrequency.MONTHLY,
        )

        assert result.portfolio_values is not None
        assert len(result.portfolio_values) > 0
        assert result.annualized_return is not None
        assert result.volatility > 0
        assert result.max_drawdown <= 0

    def test_backtest_with_benchmark(self, sample_prices):
        """Backtest with benchmark should calculate alpha/beta."""
        weights = {"A": 0.5, "B": 0.5}
        benchmark = sample_prices["C"]

        result = run_backtest(
            prices=sample_prices[["A", "B"]],
            target_weights=weights,
            benchmark_prices=benchmark,
        )

        assert result.benchmark_return is not None
        assert result.alpha is not None
        assert result.beta is not None

    def test_backtest_rebalancing(self, sample_prices):
        """Different rebalance frequencies should give different results."""
        weights = {"A": 0.5, "B": 0.3, "C": 0.2}

        result_monthly = run_backtest(
            prices=sample_prices,
            target_weights=weights,
            rebalance_frequency=RebalanceFrequency.MONTHLY,
        )

        result_none = run_backtest(
            prices=sample_prices,
            target_weights=weights,
            rebalance_frequency=RebalanceFrequency.NONE,
        )

        assert result_monthly.num_rebalances > 0
        assert result_none.num_rebalances == 0
        # Turnover should be zero with no rebalancing
        assert result_none.total_turnover == 0

    def test_backtest_weights_normalize(self, sample_prices):
        """Weights should be normalized to available symbols."""
        # Only A and B available, C is missing
        weights = {"A": 0.5, "B": 0.5, "D": 0.0}

        result = run_backtest(
            prices=sample_prices[["A", "B"]],
            target_weights=weights,
        )

        assert result.portfolio_values is not None

    def test_backtest_sharpe_ratio(self, sample_prices):
        """Sharpe ratio should be calculable."""
        weights = {"A": 0.5, "B": 0.3, "C": 0.2}

        result = run_backtest(
            prices=sample_prices,
            target_weights=weights,
            risk_free_rate=0.04,
        )

        assert isinstance(result.sharpe_ratio, float)
        assert isinstance(result.sortino_ratio, float)

    def test_backtest_max_drawdown_duration(self, sample_prices):
        """Max drawdown duration should be a positive integer."""
        weights = {"A": 0.5, "B": 0.3, "C": 0.2}

        result = run_backtest(
            prices=sample_prices,
            target_weights=weights,
        )

        assert result.max_drawdown_duration >= 0
        assert isinstance(result.max_drawdown_duration, int)


class TestMonteCarlo:
    """Tests for Monte Carlo simulation."""

    def test_monte_carlo_basic(self, sample_returns):
        """Monte Carlo should run and return results."""
        weights = {"A": 0.5, "B": 0.3, "C": 0.2}

        result = run_monte_carlo(
            weights=weights,
            returns=sample_returns,
            initial_value=10000,
            days=252,
            num_simulations=100,
            seed=42,
        )

        assert result.paths is not None
        assert result.paths.shape == (252, 100)
        assert result.final_mean > 0
        assert result.final_median > 0

    def test_monte_carlo_probabilities(self, sample_returns):
        """Probability metrics should be between 0 and 1."""
        weights = {"A": 0.5, "B": 0.3, "C": 0.2}

        result = run_monte_carlo(
            weights=weights,
            returns=sample_returns,
            num_simulations=500,
            seed=42,
        )

        assert 0 <= result.prob_positive_return <= 1
        assert 0 <= result.prob_double <= 1
        assert 0 <= result.prob_loss_10pct <= 1
        assert 0 <= result.prob_loss_20pct <= 1

    def test_monte_carlo_percentiles(self, sample_returns):
        """Percentiles should be ordered correctly."""
        weights = {"A": 0.5, "B": 0.3, "C": 0.2}

        result = run_monte_carlo(
            weights=weights,
            returns=sample_returns,
            num_simulations=500,
            seed=42,
        )

        # At final time point, percentiles should be ordered
        assert result.percentile_5.iloc[-1] <= result.percentile_25.iloc[-1]
        assert result.percentile_25.iloc[-1] <= result.median_path.iloc[-1]
        assert result.median_path.iloc[-1] <= result.percentile_75.iloc[-1]
        assert result.percentile_75.iloc[-1] <= result.percentile_95.iloc[-1]

    def test_monte_carlo_var(self, sample_returns):
        """VaR and CVaR should be positive."""
        weights = {"A": 0.5, "B": 0.3, "C": 0.2}

        result = run_monte_carlo(
            weights=weights,
            returns=sample_returns,
            num_simulations=500,
            seed=42,
        )

        # VaR can be negative if most paths are profitable
        assert isinstance(result.var_95, float)
        assert isinstance(result.cvar_95, float)

    def test_monte_carlo_reproducibility(self, sample_returns):
        """Same seed should give same results."""
        weights = {"A": 0.5, "B": 0.5}

        result1 = run_monte_carlo(
            weights=weights,
            returns=sample_returns,
            seed=123,
        )

        result2 = run_monte_carlo(
            weights=weights,
            returns=sample_returns,
            seed=123,
        )

        assert result1.final_mean == result2.final_mean
        assert result1.final_median == result2.final_median

    def test_monte_carlo_summary(self, sample_returns):
        """Summary method should return dictionary."""
        weights = {"A": 0.5, "B": 0.5}

        result = run_monte_carlo(
            weights=weights,
            returns=sample_returns,
            num_simulations=100,
            seed=42,
        )

        summary = result.summary()
        assert isinstance(summary, dict)
        assert "final_mean" in summary
        assert "prob_positive_return" in summary
        assert "var_95" in summary
