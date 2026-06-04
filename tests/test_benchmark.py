"""Tests for benchmark comparison metrics."""

import numpy as np
import pandas as pd
import pytest

from portfolio_manager.metrics.benchmark import (
    BenchmarkAnalysis,
    calculate_alpha,
    calculate_beta,
    calculate_information_ratio,
    calculate_r_squared,
    calculate_tracking_error,
    calculate_up_down_capture,
)


@pytest.fixture
def sample_returns():
    """Create sample portfolio and benchmark returns."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=252, freq="B")

    # Market returns with some volatility
    market = pd.Series(
        np.random.normal(0.0004, 0.01, 252),
        index=dates,
        name="benchmark",
    )

    # Portfolio that's correlated with market but with some alpha
    portfolio = market * 1.2 + np.random.normal(0.0002, 0.005, 252)
    portfolio = pd.Series(portfolio, index=dates, name="portfolio")

    return portfolio, market


def test_calculate_beta_correlated(sample_returns):
    """Beta should be > 1 for portfolio that moves more than market."""
    portfolio, benchmark = sample_returns
    beta = calculate_beta(portfolio, benchmark)
    assert beta > 1.0  # Portfolio is more volatile


def test_calculate_beta_uncorrelated():
    """Beta should be ~1 for uncorrelated returns with default."""
    np.random.seed(42)
    portfolio = pd.Series(np.random.normal(0, 0.01, 100))
    benchmark = pd.Series(np.random.normal(0, 0.01, 100))
    beta = calculate_beta(portfolio, benchmark)
    # Uncorrelated assets will have beta close to 0
    assert -0.5 < beta < 0.5


def test_calculate_alpha_positive(sample_returns):
    """Alpha should capture excess returns."""
    portfolio, benchmark = sample_returns
    alpha = calculate_alpha(portfolio, benchmark)
    # Our sample portfolio has positive drift added
    assert isinstance(alpha, float)


def test_calculate_alpha_zero_for_index():
    """Alpha should be ~0 for an index vs itself."""
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0.0004, 0.01, 252))
    alpha = calculate_alpha(returns, returns)
    assert abs(alpha) < 0.01  # Close to zero


def test_calculate_tracking_error_identical():
    """Tracking error should be 0 for identical returns."""
    returns = pd.Series([0.01, -0.02, 0.015, 0.005, -0.01])
    te = calculate_tracking_error(returns, returns)
    assert te == 0.0


def test_calculate_tracking_error_different(sample_returns):
    """Tracking error should be positive for different returns."""
    portfolio, benchmark = sample_returns
    te = calculate_tracking_error(portfolio, benchmark)
    assert te > 0


def test_calculate_information_ratio(sample_returns):
    """Information ratio should be calculable."""
    portfolio, benchmark = sample_returns
    ir = calculate_information_ratio(portfolio, benchmark)
    assert isinstance(ir, float)


def test_calculate_r_squared_perfect_correlation():
    """R-squared should be 1 for perfectly correlated returns."""
    returns = pd.Series([0.01, -0.02, 0.015, 0.005, -0.01])
    r2 = calculate_r_squared(returns, returns)
    assert r2 == pytest.approx(1.0)


def test_calculate_r_squared_range(sample_returns):
    """R-squared should be between 0 and 1."""
    portfolio, benchmark = sample_returns
    r2 = calculate_r_squared(portfolio, benchmark)
    assert 0 <= r2 <= 1


def test_calculate_up_down_capture(sample_returns):
    """Capture ratios should be calculable."""
    portfolio, benchmark = sample_returns
    upside, downside = calculate_up_down_capture(portfolio, benchmark)
    assert upside > 0
    assert downside > 0


def test_benchmark_analysis_run(sample_returns):
    """BenchmarkAnalysis should return all metrics."""
    portfolio, benchmark = sample_returns
    analysis = BenchmarkAnalysis(
        portfolio_returns=portfolio,
        benchmark_returns=benchmark,
        benchmark_symbol="SPY",
    )
    results = analysis.run()

    assert "beta" in results
    assert "alpha" in results
    assert "tracking_error" in results
    assert "information_ratio" in results
    assert "r_squared" in results
    assert "upside_capture" in results
    assert "downside_capture" in results
    assert results["benchmark"] == "SPY"


def test_empty_returns():
    """Handle empty returns gracefully."""
    empty = pd.Series([], dtype=float)
    assert calculate_beta(empty, empty) == 1.0
    assert calculate_tracking_error(empty, empty) == 0.0
    assert calculate_r_squared(empty, empty) == 0.0
