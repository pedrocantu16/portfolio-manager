"""Walk-forward optimization for validating portfolio strategies."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from portfolio_manager.backtest.engine import RebalanceFrequency, run_backtest
from portfolio_manager.optimization.optimizer import (
    Objective,
    OptimizationResult,
    PortfolioOptimizer,
    ReturnEstimator,
)


@dataclass
class WalkForwardWindow:
    """Results for a single walk-forward window."""

    window_num: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    # Training period metrics (in-sample)
    train_return: float
    train_volatility: float
    train_sharpe: float

    # Test period metrics (out-of-sample)
    test_return: float
    test_volatility: float
    test_sharpe: float

    # Optimal weights from training
    optimal_weights: dict[str, float]

    # Decay (difference between train and test)
    sharpe_decay: float
    return_decay: float


@dataclass
class WalkForwardResult:
    """Aggregate results from walk-forward optimization."""

    windows: list[WalkForwardWindow]

    # Aggregate metrics
    avg_train_sharpe: float
    avg_test_sharpe: float
    avg_sharpe_decay: float
    avg_sharpe_decay_pct: float

    avg_train_return: float
    avg_test_return: float
    avg_return_decay: float

    # Out-of-sample performance
    total_oos_return: float
    oos_sharpe: float
    oos_volatility: float

    # Strategy assessment
    overfitting_score: float  # Higher = more overfitting (0-1)
    consistency_score: float  # Higher = more consistent (0-1)

    def summary(self) -> dict:
        """Return summary as dictionary."""
        return {
            "num_windows": len(self.windows),
            "avg_train_sharpe": self.avg_train_sharpe,
            "avg_test_sharpe": self.avg_test_sharpe,
            "avg_sharpe_decay": self.avg_sharpe_decay,
            "avg_sharpe_decay_pct": self.avg_sharpe_decay_pct,
            "avg_train_return": self.avg_train_return,
            "avg_test_return": self.avg_test_return,
            "total_oos_return": self.total_oos_return,
            "oos_sharpe": self.oos_sharpe,
            "overfitting_score": self.overfitting_score,
            "consistency_score": self.consistency_score,
        }


def run_walk_forward(
    prices: pd.DataFrame,
    train_months: int = 12,
    test_months: int = 3,
    objective: Objective = Objective.MAX_SHARPE,
    min_weight: float = 0.0,
    max_weight: float = 0.3,
    risk_free_rate: float = 0.045,
    return_method: ReturnEstimator = ReturnEstimator.HISTORICAL,
    shrinkage: float = 0.3,
    max_expected_return: float = 0.25,
    market_prices: pd.Series | None = None,
) -> WalkForwardResult:
    """Run walk-forward optimization to validate strategy.

    Args:
        prices: DataFrame of daily prices for each asset.
        train_months: Number of months for training window.
        test_months: Number of months for test window.
        objective: Optimization objective.
        min_weight: Minimum weight per asset.
        max_weight: Maximum weight per asset.
        risk_free_rate: Annual risk-free rate.
        return_method: Method for estimating expected returns.
        shrinkage: Shrinkage factor for historical returns.
        max_expected_return: Cap on expected return per asset.
        market_prices: Market index prices (for CAPM method).

    Returns:
        WalkForwardResult with per-window and aggregate metrics.
    """
    windows: list[WalkForwardWindow] = []
    oos_values: list[float] = []  # Track cumulative OOS performance
    oos_returns: list[float] = []

    # Calculate window sizes in trading days (approx)
    train_days = train_months * 21  # ~21 trading days per month
    test_days = test_months * 21
    step_days = test_days  # Non-overlapping test windows

    dates = prices.index
    start_idx = 0
    window_num = 0

    while start_idx + train_days + test_days <= len(dates):
        window_num += 1

        # Define window boundaries
        train_start_idx = start_idx
        train_end_idx = start_idx + train_days
        test_start_idx = train_end_idx
        test_end_idx = min(test_start_idx + test_days, len(dates))

        train_start = dates[train_start_idx]
        train_end = dates[train_end_idx - 1]
        test_start = dates[test_start_idx]
        test_end = dates[test_end_idx - 1]

        # Get train and test data
        train_prices = prices.iloc[train_start_idx:train_end_idx]
        test_prices = prices.iloc[test_start_idx:test_end_idx]
        train_returns = train_prices.pct_change().dropna()

        # Prepare market returns for CAPM if needed
        market_returns = None
        if return_method == ReturnEstimator.CAPM and market_prices is not None:
            market_returns = market_prices.iloc[train_start_idx:train_end_idx].pct_change().dropna()

        # Optimize on training data
        try:
            optimizer = PortfolioOptimizer(
                returns=train_returns,
                risk_free_rate=risk_free_rate,
                max_expected_return=max_expected_return,
                shrinkage=shrinkage,
                return_method=return_method,
                market_returns=market_returns,
            )
            opt_result = optimizer.optimize(
                objective=objective,
                min_weight=min_weight,
                max_weight=max_weight,
            )
        except Exception:
            # Skip window if optimization fails
            start_idx += step_days
            continue

        # Get training period performance (in-sample backtest)
        train_backtest = run_backtest(
            prices=train_prices,
            target_weights=opt_result.weights,
            rebalance_frequency=RebalanceFrequency.MONTHLY,
            risk_free_rate=risk_free_rate,
        )

        # Get test period performance (out-of-sample backtest)
        test_backtest = run_backtest(
            prices=test_prices,
            target_weights=opt_result.weights,
            rebalance_frequency=RebalanceFrequency.MONTHLY,
            risk_free_rate=risk_free_rate,
        )

        # Calculate decay
        sharpe_decay = train_backtest.sharpe_ratio - test_backtest.sharpe_ratio
        return_decay = train_backtest.annualized_return - test_backtest.annualized_return

        window = WalkForwardWindow(
            window_num=window_num,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            train_return=train_backtest.annualized_return,
            train_volatility=train_backtest.volatility,
            train_sharpe=train_backtest.sharpe_ratio,
            test_return=test_backtest.annualized_return,
            test_volatility=test_backtest.volatility,
            test_sharpe=test_backtest.sharpe_ratio,
            optimal_weights=opt_result.weights,
            sharpe_decay=sharpe_decay,
            return_decay=return_decay,
        )
        windows.append(window)

        # Track OOS performance
        oos_returns.append(test_backtest.total_return)

        # Move to next window
        start_idx += step_days

    if not windows:
        raise ValueError("Not enough data for walk-forward optimization")

    # Calculate aggregate metrics
    avg_train_sharpe = np.mean([w.train_sharpe for w in windows])
    avg_test_sharpe = np.mean([w.test_sharpe for w in windows])
    avg_sharpe_decay = avg_train_sharpe - avg_test_sharpe
    avg_sharpe_decay_pct = (avg_sharpe_decay / avg_train_sharpe * 100) if avg_train_sharpe != 0 else 0

    avg_train_return = np.mean([w.train_return for w in windows])
    avg_test_return = np.mean([w.test_return for w in windows])
    avg_return_decay = avg_train_return - avg_test_return

    # Total OOS return (compounded)
    total_oos_return = np.prod([1 + r for r in oos_returns]) - 1

    # OOS Sharpe (simplified: avg return / std of returns)
    if len(oos_returns) > 1:
        oos_vol = np.std([w.test_return for w in windows])
        oos_sharpe = (avg_test_return - risk_free_rate) / oos_vol if oos_vol > 0 else 0
    else:
        oos_vol = windows[0].test_volatility if windows else 0
        oos_sharpe = avg_test_sharpe

    # Overfitting score (0-1, higher = more overfitting)
    # Based on how much performance decays out-of-sample
    if avg_train_sharpe > 0:
        overfitting_score = min(1.0, max(0.0, avg_sharpe_decay / avg_train_sharpe))
    else:
        overfitting_score = 0.5  # Neutral if train Sharpe is negative

    # Consistency score (0-1, higher = more consistent)
    # Based on how often test Sharpe is positive
    positive_tests = sum(1 for w in windows if w.test_sharpe > 0)
    consistency_score = positive_tests / len(windows) if windows else 0

    return WalkForwardResult(
        windows=windows,
        avg_train_sharpe=avg_train_sharpe,
        avg_test_sharpe=avg_test_sharpe,
        avg_sharpe_decay=avg_sharpe_decay,
        avg_sharpe_decay_pct=avg_sharpe_decay_pct,
        avg_train_return=avg_train_return,
        avg_test_return=avg_test_return,
        avg_return_decay=avg_return_decay,
        total_oos_return=total_oos_return,
        oos_sharpe=oos_sharpe,
        oos_volatility=oos_vol,
        overfitting_score=overfitting_score,
        consistency_score=consistency_score,
    )
