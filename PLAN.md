# Portfolio Manager Implementation Plan

## Overview
Python CLI application for portfolio management with risk/return analysis and optimization capabilities. Architecture supports future web UI (FastAPI + React/Streamlit).

## Tech Stack
- **Python 3.11+** with **uv** for package management
- **Typer** for CLI
- **pandas/numpy** for data manipulation
- **yfinance** for market data (Yahoo Finance)
- **scipy.optimize** for portfolio optimization
- **pydantic** for data validation
- **pytest** for testing
- **ruff** for linting/formatting

---

## Phase 1: Foundation ✅ COMPLETE

### Implemented
- [x] Project setup with `pyproject.toml`
- [x] Position dataclass (`core/position.py`)
- [x] Portfolio class (`core/portfolio.py`)
- [x] Fidelity CSV parser (`data/parsers/fidelity.py`)
- [x] Market data fetcher via yfinance (`data/market.py`)
- [x] Return calculations (`metrics/returns.py`)
- [x] Risk metrics: volatility, VaR, max drawdown, covariance (`metrics/risk.py`)
- [x] Ratios: Sharpe, Sortino (`metrics/ratios.py`)
- [x] CLI commands: `load`, `summary`, `metrics`, `holdings`
- [x] Unit tests (21 passing)

---

## Phase 2: Optimization & Analysis ✅ COMPLETE

### Optimization
- [x] Mean-variance optimization (`optimization/optimizer.py`)
- [x] Objectives: max_sharpe, min_volatility, max_return
- [x] Position constraints (min/max weight per position)
- [x] Return estimation: historical (with shrinkage/cap) and CAPM
- [x] CLI commands: `optimize`, `rebalance`

### Additional Features
- [x] Multiple account support (`core/account.py`)
- [x] Sector classification (`core/sectors.py`)
- [x] Sector constraints for optimizer
- [x] Transaction cost modeling (`core/costs.py`)
- [x] Tax-loss harvesting analysis (`core/tax.py`)
- [x] CLI commands: `sectors`, `tax-harvest`

### CLI Usage
```bash
# Optimization
uv run portfolio optimize <csv> --max-position 0.3 --method capm
uv run portfolio rebalance <csv>

# Analysis
uv run portfolio sectors <csv>
uv run portfolio tax-harvest <csv> --tax-rate 0.25
```

---

## Phase 3: Advanced Features (In Progress)

### Benchmarking ✅
- [x] Benchmark comparison (vs S&P 500, custom benchmark)
- [x] Alpha/Beta calculations
- [x] Tracking error
- [x] R-squared, Information ratio
- [x] Upside/Downside capture ratios

### Backtesting ✅
- [x] Historical portfolio backtesting
- [x] Monte Carlo simulation
- [x] Walk-forward optimization

### Data & Export ✅
- [x] Export recommendations to CSV/JSON
- [x] Generic CSV parser (user-provided format with quantity, weight, or value)
- [ ] Support for specific brokerages (Schwab, Vanguard, Interactive Brokers)
- [ ] Import transaction history

### CLI Commands (Planned)
```bash
portfolio benchmark <csv> --vs SPY
portfolio backtest <csv> --period 5y
portfolio export <csv> --format csv
```

---

## Phase 4: Web UI (Future)

### Backend
- [ ] FastAPI backend exposing core functionality
- [ ] REST API endpoints for portfolio operations
- [ ] WebSocket for real-time updates
- [ ] User authentication

### Frontend
- [ ] React or Streamlit dashboard
- [ ] Interactive charts (allocation pie, performance line)
- [ ] Optimization visualization (efficient frontier)
- [ ] Real-time price updates

---

## Project Structure
```
portfolio-manager/
├── pyproject.toml
├── src/portfolio_manager/
│   ├── cli/main.py              # CLI entry point
│   ├── core/
│   │   ├── position.py          # Position dataclass
│   │   ├── portfolio.py         # Portfolio class
│   │   ├── account.py           # Multi-account support
│   │   ├── sectors.py           # Sector classification
│   │   ├── costs.py             # Transaction cost modeling
│   │   └── tax.py               # Tax-loss harvesting
│   ├── data/
│   │   ├── parsers/
│   │   │   ├── base.py          # Abstract parser
│   │   │   └── fidelity.py      # Fidelity CSV parser
│   │   └── market.py            # yfinance wrapper
│   ├── metrics/
│   │   ├── returns.py           # Return calculations
│   │   ├── risk.py              # Risk metrics
│   │   └── ratios.py            # Sharpe, Sortino, etc.
│   ├── optimization/
│   │   ├── objectives.py        # Objective functions
│   │   ├── constraints.py       # Constraint builders
│   │   └── optimizer.py         # Mean-variance optimizer
│   └── config.py                # Settings
├── tests/
│   ├── test_parser.py
│   ├── test_metrics.py
│   └── test_optimizer.py
└── PLAN.md
```

---

## Quick Start
```bash
# Install dependencies
source ~/.local/bin/env
uv sync

# Run tests
uv run pytest tests/ -v

# CLI commands
uv run portfolio --help
uv run portfolio summary <csv>
uv run portfolio metrics <csv>
uv run portfolio optimize <csv> --max-position 0.3
uv run portfolio rebalance <csv>
uv run portfolio sectors <csv>
uv run portfolio tax-harvest <csv>
```
