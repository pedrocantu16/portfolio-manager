"""Tests for portfolio optimization."""

import numpy as np
import pandas as pd
import pytest

from portfolio_manager.optimization.optimizer import Objective, PortfolioOptimizer


@pytest.fixture
def sample_returns():
    """Create sample return data for 3 assets."""
    np.random.seed(42)
    n_days = 252

    # Generate correlated returns
    # Asset A: high return, high vol
    # Asset B: medium return, medium vol
    # Asset C: low return, low vol (like bonds)
    returns_a = np.random.normal(0.0008, 0.02, n_days)  # ~20% annual, 32% vol
    returns_b = np.random.normal(0.0004, 0.012, n_days)  # ~10% annual, 19% vol
    returns_c = np.random.normal(0.0002, 0.005, n_days)  # ~5% annual, 8% vol

    return pd.DataFrame({
        "HIGH": returns_a,
        "MED": returns_b,
        "LOW": returns_c,
    })


class TestPortfolioOptimizer:
    def test_max_sharpe(self, sample_returns):
        """Test max Sharpe optimization."""
        optimizer = PortfolioOptimizer(sample_returns, risk_free_rate=0.04)
        result = optimizer.optimize(Objective.MAX_SHARPE)

        assert result.success
        assert sum(result.weights.values()) == pytest.approx(1.0, rel=1e-4)
        assert result.sharpe_ratio > 0
        assert all(0 <= w <= 1 for w in result.weights.values())

    def test_min_volatility(self, sample_returns):
        """Test min volatility optimization."""
        optimizer = PortfolioOptimizer(sample_returns, risk_free_rate=0.04)
        result = optimizer.optimize(Objective.MIN_VOLATILITY)

        assert result.success
        assert sum(result.weights.values()) == pytest.approx(1.0, rel=1e-4)
        # Should favor lower volatility assets
        assert result.weights["LOW"] > result.weights["HIGH"]

    def test_max_return(self, sample_returns):
        """Test max return optimization."""
        optimizer = PortfolioOptimizer(sample_returns, risk_free_rate=0.04)
        result = optimizer.optimize(Objective.MAX_RETURN)

        assert result.success
        # Should put most weight in highest return asset
        assert result.weights["HIGH"] > 0.5

    def test_max_position_constraint(self, sample_returns):
        """Test that max position constraint is respected."""
        optimizer = PortfolioOptimizer(sample_returns, risk_free_rate=0.04)
        result = optimizer.optimize(
            Objective.MAX_SHARPE,
            max_weight=0.4,
        )

        assert result.success
        assert all(w <= 0.41 for w in result.weights.values())  # small tolerance

    def test_weights_sum_to_one(self, sample_returns):
        """Test that weights always sum to 1."""
        optimizer = PortfolioOptimizer(sample_returns, risk_free_rate=0.04)

        for obj in [Objective.MAX_SHARPE, Objective.MIN_VOLATILITY, Objective.MAX_RETURN]:
            result = optimizer.optimize(obj)
            total = sum(result.weights.values())
            assert total == pytest.approx(1.0, rel=1e-4)

    def test_no_negative_weights(self, sample_returns):
        """Test that weights are non-negative (no shorting)."""
        optimizer = PortfolioOptimizer(sample_returns, risk_free_rate=0.04)
        result = optimizer.optimize(Objective.MAX_SHARPE)

        assert all(w >= -1e-6 for w in result.weights.values())

    def test_target_return(self, sample_returns):
        """Test target return optimization."""
        optimizer = PortfolioOptimizer(sample_returns, risk_free_rate=0.04)

        # First get bounds
        min_vol = optimizer.optimize(Objective.MIN_VOLATILITY)
        max_ret = optimizer.optimize(Objective.MAX_RETURN)

        # Target somewhere in between
        target = (min_vol.expected_return + max_ret.expected_return) / 2

        result = optimizer.optimize(
            Objective.TARGET_RETURN,
            target_return=target,
        )

        assert result.success
        assert result.expected_return == pytest.approx(target, rel=0.05)

    def test_efficient_frontier(self, sample_returns):
        """Test efficient frontier calculation."""
        optimizer = PortfolioOptimizer(sample_returns, risk_free_rate=0.04)
        frontier = optimizer.efficient_frontier(n_points=10)

        assert len(frontier) > 0
        # Returns should be monotonically increasing (roughly)
        returns = [r.expected_return for r in frontier]
        assert returns[-1] >= returns[0]
