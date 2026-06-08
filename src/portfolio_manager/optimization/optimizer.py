"""Portfolio optimizer using mean-variance optimization."""

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from portfolio_manager.optimization.constraints import (
    target_return_constraint,
    weight_bounds,
    weights_sum_to_one,
)
from portfolio_manager.optimization.objectives import (
    downside_deviation,
    negative_sharpe_ratio,
    negative_sortino_ratio,
    portfolio_return,
    portfolio_volatility,
)


class Objective(StrEnum):
    """Optimization objectives."""

    MAX_SHARPE = "max_sharpe"
    MAX_SORTINO = "max_sortino"
    MIN_VOLATILITY = "min_volatility"
    TARGET_RETURN = "target_return"
    MAX_RETURN = "max_return"


class ReturnEstimator(StrEnum):
    """Method for estimating expected returns."""

    HISTORICAL = "historical"
    CAPM = "capm"


@dataclass
class OptimizationResult:
    """Result of portfolio optimization."""

    weights: dict[str, float]
    expected_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    objective: Objective
    success: bool
    message: str

    def __repr__(self) -> str:
        return (
            f"OptimizationResult(objective={self.objective.value}, "
            f"return={self.expected_return:.2%}, vol={self.volatility:.2%}, "
            f"sharpe={self.sharpe_ratio:.2f}, sortino={self.sortino_ratio:.2f})"
        )


class PortfolioOptimizer:
    """Mean-variance portfolio optimizer."""

    def __init__(
        self,
        returns: pd.DataFrame,
        risk_free_rate: float = 0.05,
        trading_days: int = 252,
        max_expected_return: float | None = None,
        shrinkage: float = 0.0,
        return_method: ReturnEstimator = ReturnEstimator.HISTORICAL,
        market_returns: pd.Series | None = None,
        market_risk_premium: float = 0.05,
    ):
        """Initialize optimizer with return data.

        Args:
            returns: DataFrame of daily returns (columns = assets).
            risk_free_rate: Annualized risk-free rate.
            trading_days: Trading days per year.
            max_expected_return: Cap on annualized expected return per asset.
                If None, no cap is applied. Recommended: 0.20-0.30 (20-30%).
            shrinkage: Shrinkage factor (0.0-1.0) to blend historical returns
                toward the grand mean. 0.0 = pure historical, 1.0 = all assets
                have same expected return. Recommended: 0.3-0.5.
            return_method: Method for estimating expected returns.
                HISTORICAL = use historical mean returns (with shrinkage/cap).
                CAPM = use Capital Asset Pricing Model (beta-based).
            market_returns: Daily returns of market index (required for CAPM).
            market_risk_premium: Expected market excess return (default 5%).
        """
        self.returns = returns
        self.risk_free_rate = risk_free_rate
        self.trading_days = trading_days

        self.symbols = list(returns.columns)
        self.n_assets = len(self.symbols)

        # Calculate expected returns based on method
        if return_method == ReturnEstimator.CAPM:
            if market_returns is None:
                raise ValueError("market_returns required for CAPM method")
            raw_mean_returns = self._calculate_capm_returns(
                returns, market_returns, market_risk_premium
            )
        else:
            raw_mean_returns = returns.mean().values

            # Apply shrinkage toward grand mean (only for historical)
            if shrinkage > 0:
                grand_mean = raw_mean_returns.mean()
                raw_mean_returns = (
                    (1 - shrinkage) * raw_mean_returns + shrinkage * grand_mean
                )

        # Apply return cap (convert to daily, cap, keep as daily)
        if max_expected_return is not None:
            daily_cap = max_expected_return / trading_days
            raw_mean_returns = np.clip(raw_mean_returns, -daily_cap, daily_cap)

        self.mean_returns = raw_mean_returns
        self.cov_matrix = returns.cov().values
        self.returns_matrix = returns.values

    def _calculate_capm_returns(
        self,
        returns: pd.DataFrame,
        market_returns: pd.Series,
        market_risk_premium: float,
    ) -> np.ndarray:
        """Calculate expected returns using CAPM.

        E(R) = Rf + β × Market Risk Premium

        Args:
            returns: Asset returns DataFrame.
            market_returns: Market index returns Series.
            market_risk_premium: Expected annual excess return of market.

        Returns:
            Array of daily expected returns for each asset.
        """
        # Align market returns with asset returns
        aligned = pd.concat([returns, market_returns.rename("_market")], axis=1).dropna()
        market = aligned["_market"]
        assets = aligned.drop(columns=["_market"])

        # Calculate beta for each asset: Cov(asset, market) / Var(market)
        market_var = market.var()
        betas = []
        for col in assets.columns:
            cov = assets[col].cov(market)
            beta = cov / market_var if market_var > 0 else 1.0
            betas.append(beta)

        betas = np.array(betas)

        # CAPM expected return (daily)
        daily_rf = self.risk_free_rate / self.trading_days
        daily_mrp = market_risk_premium / self.trading_days

        expected_returns = daily_rf + betas * daily_mrp

        return expected_returns

    def _obj_max_sharpe(self, weights: np.ndarray) -> float:
        return negative_sharpe_ratio(
            weights, self.mean_returns, self.cov_matrix,
            self.risk_free_rate, self.trading_days
        )

    def _obj_max_sortino(self, weights: np.ndarray) -> float:
        return negative_sortino_ratio(
            weights, self.mean_returns, self.returns_matrix,
            self.risk_free_rate, self.trading_days
        )

    def _obj_min_vol(self, weights: np.ndarray) -> float:
        return portfolio_volatility(weights, self.cov_matrix, self.trading_days)

    def _obj_max_return(self, weights: np.ndarray) -> float:
        return -portfolio_return(weights, self.mean_returns, self.trading_days)

    def optimize(
        self,
        objective: Objective = Objective.MAX_SHARPE,
        min_weight: float = 0.0,
        max_weight: float = 1.0,
        target_return: float | None = None,
        ticker_max: dict[str, float] | None = None,
        ticker_min: dict[str, float] | None = None,
    ) -> OptimizationResult:
        """Run portfolio optimization.

        Args:
            objective: Optimization objective.
            min_weight: Minimum weight per asset (0 = no shorting).
            max_weight: Maximum weight per asset (1 = no single asset > 100%).
            target_return: Required for TARGET_RETURN objective.
            ticker_max: Per-ticker maximum weights (e.g., {"BND": 0.10}).
            ticker_min: Per-ticker minimum weights (e.g., {"VOO": 0.20}).

        Returns:
            OptimizationResult with optimal weights and metrics.
        """
        # Initial guess: equal weights
        x0 = np.array([1.0 / self.n_assets] * self.n_assets)

        # Build per-asset bounds
        bounds = []
        for i, symbol in enumerate(self.symbols):
            asset_min = min_weight
            asset_max = max_weight
            if ticker_min and symbol in ticker_min:
                asset_min = max(asset_min, ticker_min[symbol])
            if ticker_max and symbol in ticker_max:
                asset_max = min(asset_max, ticker_max[symbol])
            bounds.append((asset_min, asset_max))

        # Base constraint: weights sum to 1
        constraints = [weights_sum_to_one()]

        # Select objective function
        obj_fn: Callable[[np.ndarray], float]
        if objective == Objective.MAX_SHARPE:
            obj_fn = self._obj_max_sharpe
        elif objective == Objective.MAX_SORTINO:
            obj_fn = self._obj_max_sortino
        elif objective == Objective.MIN_VOLATILITY:
            obj_fn = self._obj_min_vol
        elif objective == Objective.TARGET_RETURN:
            if target_return is None:
                raise ValueError("target_return required for TARGET_RETURN objective")
            constraints.append(
                target_return_constraint(
                    target_return, self.mean_returns, self.trading_days
                )
            )
            obj_fn = self._obj_min_vol
        elif objective == Objective.MAX_RETURN:
            obj_fn = self._obj_max_return
        else:
            raise ValueError(f"Unknown objective: {objective}")

        # Run optimization
        result = minimize(
            obj_fn,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-9},
        )

        # Extract results
        optimal_weights = result.x

        # Clean up tiny weights (numerical noise)
        optimal_weights = np.where(
            np.abs(optimal_weights) < 1e-6, 0.0, optimal_weights
        )
        # Renormalize
        optimal_weights = optimal_weights / optimal_weights.sum()

        # Calculate metrics for optimal portfolio
        exp_return = portfolio_return(
            optimal_weights, self.mean_returns, self.trading_days
        )
        vol = portfolio_volatility(
            optimal_weights, self.cov_matrix, self.trading_days
        )
        sharpe = (exp_return - self.risk_free_rate) / vol if vol > 0 else 0.0

        dd = downside_deviation(
            optimal_weights, self.returns_matrix, self.risk_free_rate, self.trading_days
        )
        sortino = (exp_return - self.risk_free_rate) / dd if dd > 0 else 0.0

        return OptimizationResult(
            weights={s: w for s, w in zip(self.symbols, optimal_weights)},
            expected_return=exp_return,
            volatility=vol,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            objective=objective,
            success=result.success,
            message=result.message,
        )

    def efficient_frontier(
        self,
        n_points: int = 50,
        min_weight: float = 0.0,
        max_weight: float = 1.0,
    ) -> list[OptimizationResult]:
        """Calculate points on the efficient frontier.

        Args:
            n_points: Number of points to calculate.
            min_weight: Minimum weight per asset.
            max_weight: Maximum weight per asset.

        Returns:
            List of OptimizationResults along the frontier.
        """
        # Find min and max return portfolios
        min_vol = self.optimize(
            Objective.MIN_VOLATILITY, min_weight, max_weight
        )
        max_ret = self.optimize(
            Objective.MAX_RETURN, min_weight, max_weight
        )

        # Generate target returns
        target_returns = np.linspace(
            min_vol.expected_return,
            max_ret.expected_return,
            n_points,
        )

        frontier = []
        for target in target_returns:
            try:
                result = self.optimize(
                    Objective.TARGET_RETURN,
                    min_weight=min_weight,
                    max_weight=max_weight,
                    target_return=target,
                )
                if result.success:
                    frontier.append(result)
            except Exception:
                continue

        return frontier
