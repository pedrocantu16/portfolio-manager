"""Backtesting and simulation tools."""

from portfolio_manager.backtest.engine import (
    BacktestResult,
    RebalanceFrequency,
    run_backtest,
)
from portfolio_manager.backtest.montecarlo import (
    MonteCarloResult,
    run_monte_carlo,
    run_monte_carlo_correlated,
)
from portfolio_manager.backtest.walkforward import (
    WalkForwardResult,
    WalkForwardWindow,
    run_walk_forward,
)
from portfolio_manager.backtest.valueadd import (
    ValueAddResult,
    ValueAddWindow,
    run_value_add_analysis,
)

__all__ = [
    "BacktestResult",
    "MonteCarloResult",
    "RebalanceFrequency",
    "ValueAddResult",
    "ValueAddWindow",
    "WalkForwardResult",
    "WalkForwardWindow",
    "run_backtest",
    "run_monte_carlo",
    "run_monte_carlo_correlated",
    "run_value_add_analysis",
    "run_walk_forward",
]
