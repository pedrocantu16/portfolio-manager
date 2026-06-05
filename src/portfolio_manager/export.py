"""Export utilities for saving results to files."""

import csv
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import pandas as pd


def _serialize(obj: Any) -> Any:
    """Convert objects to JSON-serializable format."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: _serialize(v) for k, v in asdict(obj).items()}
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    if hasattr(obj, "__dict__"):
        return {k: _serialize(v) for k, v in obj.__dict__.items()}
    return obj


def export_to_json(data: dict, path: Path) -> None:
    """Export data to JSON file."""
    with open(path, "w") as f:
        json.dump(_serialize(data), f, indent=2, default=str)


def export_to_csv(rows: list[dict], path: Path) -> None:
    """Export rows to CSV file."""
    if not rows:
        return

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def export_trades(
    trades: list[tuple[str, float, float, float]],
    path: Path,
) -> None:
    """Export rebalancing trades to CSV.

    Args:
        trades: List of (symbol, current_value, target_value, trade_value) tuples.
        path: Output file path.
    """
    rows = [
        {
            "symbol": symbol,
            "action": "BUY" if trade > 0 else "SELL",
            "current_value": round(current, 2),
            "target_value": round(target, 2),
            "trade_value": round(trade, 2),
        }
        for symbol, current, target, trade in trades
    ]
    export_to_csv(rows, path)


def export_weights(
    weights: dict[str, float],
    path: Path,
    current_weights: dict[str, float] | None = None,
) -> None:
    """Export portfolio weights to CSV or JSON.

    Args:
        weights: Optimal weights by symbol.
        path: Output file path (.csv or .json).
        current_weights: Optional current weights for comparison.
    """
    if path.suffix.lower() == ".json":
        data = {"optimal_weights": weights}
        if current_weights:
            data["current_weights"] = current_weights
            data["changes"] = {
                s: weights.get(s, 0) - current_weights.get(s, 0)
                for s in set(weights) | set(current_weights)
            }
        export_to_json(data, path)
    else:
        rows = [
            {
                "symbol": symbol,
                "current_weight": round(current_weights.get(symbol, 0) * 100, 2) if current_weights else None,
                "optimal_weight": round(weight * 100, 2),
                "change": round((weight - current_weights.get(symbol, 0)) * 100, 2) if current_weights else None,
            }
            for symbol, weight in sorted(weights.items(), key=lambda x: -x[1])
        ]
        export_to_csv(rows, path)


def export_backtest(
    result: Any,
    path: Path,
    benchmark_symbol: str | None = None,
) -> None:
    """Export backtest results to CSV or JSON.

    Args:
        result: BacktestResult object.
        path: Output file path.
        benchmark_symbol: Benchmark ticker for labeling.
    """
    if path.suffix.lower() == ".json":
        data = {
            "summary": {
                "total_return": result.total_return,
                "annualized_return": result.annualized_return,
                "volatility": result.volatility,
                "sharpe_ratio": result.sharpe_ratio,
                "sortino_ratio": result.sortino_ratio,
                "max_drawdown": result.max_drawdown,
                "max_drawdown_duration_days": result.max_drawdown_duration,
                "num_rebalances": result.num_rebalances,
                "total_turnover": result.total_turnover,
            },
            "benchmark": {
                "symbol": benchmark_symbol,
                "return": result.benchmark_return,
                "alpha": result.alpha,
                "beta": result.beta,
            } if result.benchmark_return is not None else None,
            "daily_values": {
                str(date.date()): value
                for date, value in result.portfolio_values.items()
            },
        }
        export_to_json(data, path)
    else:
        # CSV: daily values
        rows = [
            {
                "date": str(date.date()),
                "portfolio_value": round(value, 2),
                "benchmark_value": round(result.benchmark_values[date], 2) if result.benchmark_values is not None and date in result.benchmark_values.index else None,
            }
            for date, value in result.portfolio_values.items()
        ]
        export_to_csv(rows, path)


def export_walk_forward(
    result: Any,
    path: Path,
) -> None:
    """Export walk-forward optimization results to CSV or JSON.

    Args:
        result: WalkForwardResult object.
        path: Output file path.
    """
    if path.suffix.lower() == ".json":
        data = {
            "summary": {
                "num_windows": len(result.windows),
                "avg_train_sharpe": result.avg_train_sharpe,
                "avg_test_sharpe": result.avg_test_sharpe,
                "avg_sharpe_decay": result.avg_sharpe_decay,
                "avg_sharpe_decay_pct": result.avg_sharpe_decay_pct,
                "avg_train_return": result.avg_train_return,
                "avg_test_return": result.avg_test_return,
                "total_oos_return": result.total_oos_return,
                "oos_sharpe": result.oos_sharpe,
                "overfitting_score": result.overfitting_score,
                "consistency_score": result.consistency_score,
            },
            "windows": [
                {
                    "window": w.window_num,
                    "train_period": f"{w.train_start.date()} to {w.train_end.date()}",
                    "test_period": f"{w.test_start.date()} to {w.test_end.date()}",
                    "train_sharpe": w.train_sharpe,
                    "test_sharpe": w.test_sharpe,
                    "sharpe_decay": w.sharpe_decay,
                    "train_return": w.train_return,
                    "test_return": w.test_return,
                    "weights": w.optimal_weights,
                }
                for w in result.windows
            ],
        }
        export_to_json(data, path)
    else:
        rows = [
            {
                "window": w.window_num,
                "train_start": str(w.train_start.date()),
                "train_end": str(w.train_end.date()),
                "test_start": str(w.test_start.date()),
                "test_end": str(w.test_end.date()),
                "train_sharpe": round(w.train_sharpe, 3),
                "test_sharpe": round(w.test_sharpe, 3),
                "sharpe_decay": round(w.sharpe_decay, 3),
                "train_return": round(w.train_return, 4),
                "test_return": round(w.test_return, 4),
            }
            for w in result.windows
        ]
        export_to_csv(rows, path)


def export_simulation(
    result: Any,
    path: Path,
    cash_end: float,
    initial_total: float,
) -> None:
    """Export Monte Carlo simulation results to CSV or JSON.

    Args:
        result: MonteCarloResult object.
        path: Output file path.
        cash_end: Projected cash value.
        initial_total: Initial total portfolio value.
    """
    scenarios = [
        ("best_case_95th", result.percentile_95.iloc[-1]),
        ("optimistic_75th", result.percentile_75.iloc[-1]),
        ("median_50th", result.final_median),
        ("mean", result.final_mean),
        ("pessimistic_25th", result.percentile_25.iloc[-1]),
        ("worst_case_5th", result.percentile_5.iloc[-1]),
    ]

    if path.suffix.lower() == ".json":
        data = {
            "scenarios": {
                name: {
                    "invested": invested,
                    "cash": cash_end,
                    "total": invested + cash_end,
                    "return_pct": (invested + cash_end) / initial_total - 1,
                }
                for name, invested in scenarios
            },
            "probabilities": {
                "gain": result.prob_positive_return,
                "double": result.prob_double,
                "lose_10pct": result.prob_loss_10pct,
                "lose_20pct": result.prob_loss_20pct,
            },
            "risk_metrics": {
                "var_95": result.var_95,
                "cvar_95": result.cvar_95,
                "std_dev": result.final_std,
            },
        }
        export_to_json(data, path)
    else:
        rows = [
            {
                "scenario": name,
                "invested": round(invested, 2),
                "cash": round(cash_end, 2),
                "total": round(invested + cash_end, 2),
                "return_pct": round((invested + cash_end) / initial_total - 1, 4),
            }
            for name, invested in scenarios
        ]
        export_to_csv(rows, path)
