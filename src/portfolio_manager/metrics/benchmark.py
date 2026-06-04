"""Benchmark comparison metrics."""

import numpy as np
import pandas as pd


def calculate_beta(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> float:
    """Calculate portfolio beta relative to benchmark.

    Beta measures systematic risk - how much the portfolio moves
    relative to the market.

    Args:
        portfolio_returns: Daily portfolio returns.
        benchmark_returns: Daily benchmark returns.

    Returns:
        Beta coefficient. >1 means more volatile than market,
        <1 means less volatile.
    """
    # Align the series
    aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
    if len(aligned) < 2:
        return 1.0

    port_ret = aligned.iloc[:, 0].values
    bench_ret = aligned.iloc[:, 1].values

    covariance = np.cov(port_ret, bench_ret)[0, 1]
    benchmark_variance = np.var(bench_ret, ddof=1)

    if benchmark_variance == 0:
        return 1.0

    return covariance / benchmark_variance


def calculate_alpha(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float = 0.045,
    trading_days: int = 252,
) -> float:
    """Calculate Jensen's alpha (annualized).

    Alpha measures excess return above what CAPM predicts.
    Positive alpha = outperformance, negative = underperformance.

    Args:
        portfolio_returns: Daily portfolio returns.
        benchmark_returns: Daily benchmark returns.
        risk_free_rate: Annual risk-free rate.
        trading_days: Trading days per year.

    Returns:
        Annualized alpha as decimal (0.02 = 2%).
    """
    beta = calculate_beta(portfolio_returns, benchmark_returns)

    # Annualize returns
    port_annual = portfolio_returns.mean() * trading_days
    bench_annual = benchmark_returns.mean() * trading_days

    # Jensen's alpha: R_p - [R_f + β(R_m - R_f)]
    expected_return = risk_free_rate + beta * (bench_annual - risk_free_rate)
    alpha = port_annual - expected_return

    return alpha


def calculate_tracking_error(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    trading_days: int = 252,
) -> float:
    """Calculate tracking error (annualized).

    Tracking error measures how closely the portfolio follows
    the benchmark. Lower = more similar to benchmark.

    Args:
        portfolio_returns: Daily portfolio returns.
        benchmark_returns: Daily benchmark returns.
        trading_days: Trading days per year.

    Returns:
        Annualized tracking error as decimal.
    """
    # Align the series
    aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
    if len(aligned) < 2:
        return 0.0

    # Tracking error = std of excess returns
    excess_returns = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    tracking_error = excess_returns.std() * np.sqrt(trading_days)

    return tracking_error


def calculate_information_ratio(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    trading_days: int = 252,
) -> float:
    """Calculate information ratio.

    Information ratio = alpha / tracking error.
    Measures risk-adjusted excess return vs benchmark.

    Args:
        portfolio_returns: Daily portfolio returns.
        benchmark_returns: Daily benchmark returns.
        trading_days: Trading days per year.

    Returns:
        Information ratio. Higher is better.
    """
    alpha = calculate_alpha(portfolio_returns, benchmark_returns)
    tracking_error = calculate_tracking_error(
        portfolio_returns, benchmark_returns, trading_days
    )

    if tracking_error == 0:
        return 0.0

    return alpha / tracking_error


def calculate_r_squared(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> float:
    """Calculate R-squared (coefficient of determination).

    R-squared measures how much of the portfolio's movement
    is explained by the benchmark. 1.0 = perfectly correlated.

    Args:
        portfolio_returns: Daily portfolio returns.
        benchmark_returns: Daily benchmark returns.

    Returns:
        R-squared between 0 and 1.
    """
    # Align the series
    aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
    if len(aligned) < 2:
        return 0.0

    correlation = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
    return correlation ** 2


def calculate_up_down_capture(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> tuple[float, float]:
    """Calculate upside and downside capture ratios.

    Upside capture: How much of benchmark gains the portfolio captures.
    Downside capture: How much of benchmark losses the portfolio captures.

    Ideal: High upside capture, low downside capture.

    Args:
        portfolio_returns: Daily portfolio returns.
        benchmark_returns: Daily benchmark returns.

    Returns:
        Tuple of (upside_capture, downside_capture) as percentages.
    """
    # Align the series
    aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
    if len(aligned) < 2:
        return 100.0, 100.0

    port_ret = aligned.iloc[:, 0]
    bench_ret = aligned.iloc[:, 1]

    # Up days
    up_days = bench_ret > 0
    if up_days.sum() > 0:
        upside_capture = (port_ret[up_days].mean() / bench_ret[up_days].mean()) * 100
    else:
        upside_capture = 100.0

    # Down days
    down_days = bench_ret < 0
    if down_days.sum() > 0:
        downside_capture = (port_ret[down_days].mean() / bench_ret[down_days].mean()) * 100
    else:
        downside_capture = 100.0

    return upside_capture, downside_capture


class BenchmarkAnalysis:
    """Complete benchmark comparison analysis."""

    def __init__(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        benchmark_symbol: str = "SPY",
        risk_free_rate: float = 0.045,
    ):
        self.portfolio_returns = portfolio_returns
        self.benchmark_returns = benchmark_returns
        self.benchmark_symbol = benchmark_symbol
        self.risk_free_rate = risk_free_rate

    def run(self) -> dict:
        """Run full benchmark analysis.

        Returns:
            Dictionary with all benchmark metrics.
        """
        upside, downside = calculate_up_down_capture(
            self.portfolio_returns, self.benchmark_returns
        )

        return {
            "benchmark": self.benchmark_symbol,
            "beta": calculate_beta(self.portfolio_returns, self.benchmark_returns),
            "alpha": calculate_alpha(
                self.portfolio_returns,
                self.benchmark_returns,
                self.risk_free_rate,
            ),
            "tracking_error": calculate_tracking_error(
                self.portfolio_returns, self.benchmark_returns
            ),
            "information_ratio": calculate_information_ratio(
                self.portfolio_returns, self.benchmark_returns
            ),
            "r_squared": calculate_r_squared(
                self.portfolio_returns, self.benchmark_returns
            ),
            "upside_capture": upside,
            "downside_capture": downside,
        }
