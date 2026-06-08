"""Value-Add Analysis - Compare optimized portfolio vs S&P 500 benchmark."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from portfolio_manager.metrics import calculate_returns
from portfolio_manager.metrics.ratios import calculate_sharpe_ratio, calculate_sortino_ratio
from portfolio_manager.optimization import PortfolioOptimizer
from portfolio_manager.optimization.optimizer import Objective, ReturnEstimator


@dataclass
class ValueAddWindow:
    """Results for a single test window."""

    window_num: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    portfolio_return: float
    spy_return: float
    excess_return: float
    portfolio_sharpe: float
    spy_sharpe: float
    portfolio_sortino: float
    spy_sortino: float
    portfolio_volatility: float
    spy_volatility: float
    portfolio_downside_vol: float
    spy_downside_vol: float
    beat_benchmark: bool
    weights: dict[str, float]


@dataclass
class ValueAddResult:
    """Complete value-add analysis results."""

    windows: list[ValueAddWindow]
    objective: Objective
    total_portfolio_return: float
    total_spy_return: float
    avg_portfolio_sharpe: float
    avg_spy_sharpe: float
    avg_portfolio_sortino: float
    avg_spy_sortino: float
    alpha: float
    tracking_error: float
    information_ratio: float
    win_rate: float
    confidence_pct: float
    p_value: float
    last_weights: dict[str, float]
    verdict: str

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Windows analyzed: {len(self.windows)}",
            f"",
            f"Total Out-of-Sample Return:",
            f"  Portfolio: {self.total_portfolio_return*100:+.1f}%",
            f"  S&P 500:   {self.total_spy_return*100:+.1f}%",
            f"  Excess:    {(self.total_portfolio_return - self.total_spy_return)*100:+.1f}%",
            f"",
            f"Risk-Adjusted Performance (avg across windows):",
            f"  Sharpe  - Portfolio: {self.avg_portfolio_sharpe:.2f}, SPY: {self.avg_spy_sharpe:.2f}",
            f"  Sortino - Portfolio: {self.avg_portfolio_sortino:.2f}, SPY: {self.avg_spy_sortino:.2f}",
            f"",
            f"Value-Add Metrics:",
            f"  Alpha:             {self.alpha*100:+.2f}% annualized",
            f"  Tracking Error:    {self.tracking_error*100:.1f}%",
            f"  Information Ratio: {self.information_ratio:.2f}",
            f"  Win Rate:          {self.win_rate*100:.0f}% of periods beat SPY",
            f"",
            f"Statistical Significance:",
            f"  Confidence: {self.confidence_pct:.0f}%",
            f"  p-value:    {self.p_value:.3f}",
            f"",
            f"Verdict: {self.verdict}",
        ]
        return "\n".join(lines)


def run_value_add_analysis(
    prices: pd.DataFrame,
    symbols: list[str],
    train_months: int = 12,
    test_months: int = 3,
    objective: Objective = Objective.MAX_SHARPE,
    max_weight: float = 0.30,
    risk_free_rate: float = 0.045,
    return_method: ReturnEstimator = ReturnEstimator.HISTORICAL,
) -> ValueAddResult:
    """Run value-add analysis comparing optimized portfolio vs S&P 500.

    Args:
        prices: DataFrame with price data including SPY and portfolio symbols.
        symbols: List of portfolio symbols (excluding SPY).
        train_months: Training window in months.
        test_months: Test window in months.
        objective: Optimization objective.
        max_weight: Maximum weight per position.
        risk_free_rate: Annual risk-free rate.
        return_method: Return estimation method.

    Returns:
        ValueAddResult with complete analysis.
    """
    if "SPY" not in prices.columns:
        raise ValueError("SPY must be included in price data for benchmark comparison")

    # Filter to available symbols
    available = [s for s in symbols if s in prices.columns and s != "SPY"]
    if len(available) < 2:
        raise ValueError("Need at least 2 portfolio symbols for optimization")

    # Calculate trading days per window
    train_days = train_months * 21
    test_days = test_months * 21
    window_size = train_days + test_days

    total_days = len(prices)
    if total_days < window_size:
        raise ValueError(
            f"Insufficient data: need {window_size} days, have {total_days}"
        )

    windows: list[ValueAddWindow] = []
    portfolio_test_returns: list[pd.Series] = []
    spy_test_returns: list[pd.Series] = []

    window_num = 0
    start_idx = 0

    while start_idx + window_size <= total_days:
        train_start_idx = start_idx
        train_end_idx = start_idx + train_days
        test_start_idx = train_end_idx
        test_end_idx = train_end_idx + test_days

        # Get train and test data
        train_prices = prices.iloc[train_start_idx:train_end_idx]
        test_prices = prices.iloc[test_start_idx:test_end_idx]

        train_returns = calculate_returns(train_prices[available])
        test_returns = calculate_returns(test_prices[available + ["SPY"]])

        # Get market returns for CAPM if needed
        market_returns = None
        if return_method == ReturnEstimator.CAPM:
            market_returns = calculate_returns(train_prices[["SPY"]])["SPY"]

        # Optimize on training data
        try:
            optimizer = PortfolioOptimizer(
                train_returns,
                risk_free_rate,
                max_expected_return=0.25,
                shrinkage=0.3 if return_method == ReturnEstimator.HISTORICAL else 0.0,
                return_method=return_method,
                market_returns=market_returns,
            )
            result = optimizer.optimize(objective=objective, max_weight=max_weight)

            if not result.success:
                start_idx += test_days
                continue

            weights = result.weights
        except Exception:
            start_idx += test_days
            continue

        # Calculate test period returns
        port_daily_returns = sum(
            test_returns[s] * weights.get(s, 0)
            for s in available
            if s in test_returns.columns
        )
        spy_daily_returns = test_returns["SPY"]

        # Store for aggregate calculations
        portfolio_test_returns.append(port_daily_returns)
        spy_test_returns.append(spy_daily_returns)

        # Calculate window metrics
        port_total = (1 + port_daily_returns).prod() - 1
        spy_total = (1 + spy_daily_returns).prod() - 1

        port_annual_ret = port_daily_returns.mean() * 252
        port_annual_vol = port_daily_returns.std() * np.sqrt(252)
        spy_annual_ret = spy_daily_returns.mean() * 252
        spy_annual_vol = spy_daily_returns.std() * np.sqrt(252)

        # Calculate downside volatility (std of returns below risk-free rate)
        daily_rf = risk_free_rate / 252
        port_downside = port_daily_returns[port_daily_returns < daily_rf]
        spy_downside = spy_daily_returns[spy_daily_returns < daily_rf]
        port_downside_vol = port_downside.std() * np.sqrt(252) if len(port_downside) > 1 else 0.0
        spy_downside_vol = spy_downside.std() * np.sqrt(252) if len(spy_downside) > 1 else 0.0

        port_sharpe = calculate_sharpe_ratio(port_annual_ret, port_annual_vol, risk_free_rate)
        spy_sharpe = calculate_sharpe_ratio(spy_annual_ret, spy_annual_vol, risk_free_rate)
        port_sortino = calculate_sortino_ratio(port_daily_returns, risk_free_rate)
        spy_sortino = calculate_sortino_ratio(spy_daily_returns, risk_free_rate)

        # Determine if portfolio beat SPY based on objective
        if objective == Objective.MAX_SHARPE:
            beat = port_sharpe > spy_sharpe
        elif objective == Objective.MAX_SORTINO:
            beat = port_sortino > spy_sortino
        elif objective == Objective.MIN_VOLATILITY:
            beat = port_annual_vol < spy_annual_vol
        else:
            beat = port_total > spy_total

        window = ValueAddWindow(
            window_num=window_num,
            train_start=str(train_prices.index[0].date()),
            train_end=str(train_prices.index[-1].date()),
            test_start=str(test_prices.index[0].date()),
            test_end=str(test_prices.index[-1].date()),
            portfolio_return=port_total,
            spy_return=spy_total,
            excess_return=port_total - spy_total,
            portfolio_sharpe=port_sharpe,
            spy_sharpe=spy_sharpe,
            portfolio_sortino=port_sortino,
            spy_sortino=spy_sortino,
            portfolio_volatility=port_annual_vol,
            spy_volatility=spy_annual_vol,
            portfolio_downside_vol=port_downside_vol,
            spy_downside_vol=spy_downside_vol,
            beat_benchmark=beat,
            weights=weights,
        )
        windows.append(window)

        window_num += 1
        start_idx += test_days

    if not windows:
        raise ValueError("No valid windows could be computed")

    # Aggregate metrics
    all_port_returns = pd.concat(portfolio_test_returns)
    all_spy_returns = pd.concat(spy_test_returns)

    # Total returns
    total_port_return = (1 + all_port_returns).prod() - 1
    total_spy_return = (1 + all_spy_returns).prod() - 1

    # Average Sharpe/Sortino across windows
    avg_port_sharpe = np.mean([w.portfolio_sharpe for w in windows])
    avg_spy_sharpe = np.mean([w.spy_sharpe for w in windows])
    avg_port_sortino = np.mean([w.portfolio_sortino for w in windows])
    avg_spy_sortino = np.mean([w.spy_sortino for w in windows])

    # Alpha and tracking error
    excess_returns = all_port_returns - all_spy_returns
    alpha = excess_returns.mean() * 252  # Annualized
    tracking_error = excess_returns.std() * np.sqrt(252)
    information_ratio = alpha / tracking_error if tracking_error > 0 else 0.0

    # Win rate
    win_rate = sum(1 for w in windows if w.beat_benchmark) / len(windows)

    # Statistical significance (t-test on excess returns)
    t_stat, p_value = stats.ttest_1samp(excess_returns.dropna(), 0)
    confidence_pct = (1 - p_value) * 100 if p_value < 1 else 0

    # Verdict
    if alpha > 0 and confidence_pct >= 95:
        verdict = "Strong evidence: Diversification adds significant value over SPY"
    elif alpha > 0 and confidence_pct >= 80:
        verdict = "Moderate evidence: Portfolio likely adds value, but not conclusive"
    elif alpha > 0 and win_rate >= 0.5:
        verdict = "Weak evidence: Portfolio shows some outperformance, consider longer test"
    elif alpha <= 0 and confidence_pct >= 80:
        verdict = "Consider SPY-only: Diversification not adding value"
    else:
        verdict = "Inconclusive: Not enough evidence to determine value-add"

    # Get last optimized weights
    last_weights = windows[-1].weights if windows else {}

    return ValueAddResult(
        windows=windows,
        objective=objective,
        total_portfolio_return=total_port_return,
        total_spy_return=total_spy_return,
        avg_portfolio_sharpe=avg_port_sharpe,
        avg_spy_sharpe=avg_spy_sharpe,
        avg_portfolio_sortino=avg_port_sortino,
        avg_spy_sortino=avg_spy_sortino,
        alpha=alpha,
        tracking_error=tracking_error,
        information_ratio=information_ratio,
        win_rate=win_rate,
        confidence_pct=confidence_pct,
        p_value=p_value,
        last_weights=last_weights,
        verdict=verdict,
    )
