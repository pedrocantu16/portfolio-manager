"""Tests for portfolio metrics calculations."""

import numpy as np
import pandas as pd
import pytest

from portfolio_manager.metrics.returns import (
    calculate_annualized_return,
    calculate_cumulative_return,
    calculate_portfolio_return,
    calculate_returns,
)
from portfolio_manager.metrics.risk import (
    calculate_correlation_matrix,
    calculate_covariance_matrix,
    calculate_max_drawdown,
    calculate_portfolio_volatility,
    calculate_var,
    calculate_volatility,
)
from portfolio_manager.metrics.ratios import (
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
)


@pytest.fixture
def sample_prices():
    """Create sample price data."""
    dates = pd.date_range("2023-01-01", periods=252, freq="B")
    np.random.seed(42)

    # Generate random walk prices
    returns_a = np.random.normal(0.0004, 0.02, 252)  # ~10% annual return
    returns_b = np.random.normal(0.0002, 0.015, 252)  # ~5% annual return

    prices_a = 100 * np.exp(np.cumsum(returns_a))
    prices_b = 50 * np.exp(np.cumsum(returns_b))

    return pd.DataFrame({
        "AAAA": prices_a,
        "BBBB": prices_b,
    }, index=dates)


@pytest.fixture
def sample_returns(sample_prices):
    """Create sample returns from prices."""
    return calculate_returns(sample_prices)


class TestReturns:
    def test_calculate_returns_log(self, sample_prices):
        """Test log return calculation."""
        returns = calculate_returns(sample_prices, method="log")

        assert len(returns) == len(sample_prices) - 1
        assert "AAAA" in returns.columns
        assert "BBBB" in returns.columns

    def test_calculate_returns_simple(self, sample_prices):
        """Test simple return calculation."""
        returns = calculate_returns(sample_prices, method="simple")

        assert len(returns) == len(sample_prices) - 1

    def test_annualized_return(self, sample_returns):
        """Test annualized return calculation."""
        ann_return = calculate_annualized_return(sample_returns["AAAA"])

        # Should be close to 10% based on our generated data
        assert isinstance(ann_return, float)
        assert -1 < ann_return < 1  # Reasonable range

    def test_portfolio_return(self, sample_returns):
        """Test portfolio return calculation."""
        weights = {"AAAA": 0.6, "BBBB": 0.4}
        port_return = calculate_portfolio_return(weights, sample_returns)

        assert isinstance(port_return, float)


class TestRisk:
    def test_volatility(self, sample_returns):
        """Test volatility calculation."""
        vol = calculate_volatility(sample_returns["AAAA"])

        assert isinstance(vol, float)
        assert vol > 0
        # Annualized vol should be reasonable
        assert 0.1 < vol < 0.5

    def test_portfolio_volatility(self, sample_returns):
        """Test portfolio volatility calculation."""
        weights = {"AAAA": 0.6, "BBBB": 0.4}
        port_vol = calculate_portfolio_volatility(weights, sample_returns)

        assert isinstance(port_vol, float)
        assert port_vol > 0

    def test_var(self, sample_returns):
        """Test VaR calculation."""
        var_95 = calculate_var(sample_returns["AAAA"], confidence=0.95)

        assert isinstance(var_95, float)
        assert var_95 < 0  # VaR should be negative (loss)

    def test_covariance_matrix(self, sample_returns):
        """Test covariance matrix calculation."""
        cov = calculate_covariance_matrix(sample_returns)

        assert cov.shape == (2, 2)
        assert cov.loc["AAAA", "AAAA"] > 0  # Variance should be positive
        # Covariance matrix should be symmetric
        assert cov.loc["AAAA", "BBBB"] == pytest.approx(cov.loc["BBBB", "AAAA"])

    def test_correlation_matrix(self, sample_returns):
        """Test correlation matrix calculation."""
        corr = calculate_correlation_matrix(sample_returns)

        assert corr.shape == (2, 2)
        assert corr.loc["AAAA", "AAAA"] == pytest.approx(1.0)
        assert corr.loc["BBBB", "BBBB"] == pytest.approx(1.0)
        assert -1 <= corr.loc["AAAA", "BBBB"] <= 1

    def test_max_drawdown(self, sample_prices):
        """Test max drawdown calculation."""
        max_dd = calculate_max_drawdown(sample_prices["AAAA"])

        assert isinstance(max_dd, float)
        assert max_dd <= 0  # Drawdown should be negative or zero
        assert max_dd >= -1  # Can't lose more than 100%


class TestRatios:
    def test_sharpe_ratio(self, sample_returns):
        """Test Sharpe ratio calculation."""
        sharpe = calculate_sharpe_ratio(
            sample_returns["AAAA"],
            risk_free_rate=0.05,
        )

        assert isinstance(sharpe, float)

    def test_sortino_ratio(self, sample_returns):
        """Test Sortino ratio calculation."""
        sortino = calculate_sortino_ratio(
            sample_returns["AAAA"],
            risk_free_rate=0.05,
        )

        assert isinstance(sortino, float)

    def test_sharpe_with_provided_values(self):
        """Test Sharpe ratio with pre-calculated values."""
        sharpe = calculate_sharpe_ratio(
            returns=0.12,  # 12% annual return
            risk_free_rate=0.05,  # 5% risk-free rate
            volatility=0.20,  # 20% volatility
        )

        expected = (0.12 - 0.05) / 0.20
        assert sharpe == pytest.approx(expected)
