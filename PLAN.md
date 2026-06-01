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

### CLI Usage
```bash
uv run portfolio load Portfolio_Positions.csv
uv run portfolio summary Portfolio_Positions.csv
uv run portfolio metrics Portfolio_Positions.csv --period 1y
uv run portfolio holdings Portfolio_Positions.csv
```

---

## Phase 2: Optimization (Next)

### 2.1 Optimizer (`src/portfolio_manager/optimization/`)
- [ ] `objectives.py` - Objective functions (max return, min risk, max Sharpe)
- [ ] `constraints.py` - Position/sector constraints
- [ ] `optimizer.py` - Mean-variance optimization using scipy.optimize

### 2.2 CLI Commands
```bash
portfolio optimize --objective sharpe --max-position 0.3
portfolio rebalance --target-weights <file>
```

### 2.3 Additional Features
- [ ] Multiple account support (`core/account.py`)
- [ ] Sector classification and constraints
- [ ] Transaction cost modeling
- [ ] Tax-loss harvesting suggestions

---

## Phase 3: Web UI (Future)

### 3.1 Backend
- [ ] FastAPI backend exposing core functionality
- [ ] REST API endpoints for portfolio operations
- [ ] WebSocket for real-time updates

### 3.2 Frontend
- [ ] React or Streamlit dashboard
- [ ] Interactive charts (portfolio allocation, performance)
- [ ] Optimization visualization (efficient frontier)

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
│   │   └── account.py           # Account class (TODO)
│   ├── data/
│   │   ├── parsers/
│   │   │   ├── base.py          # Abstract parser
│   │   │   └── fidelity.py      # Fidelity CSV parser
│   │   └── market.py            # yfinance wrapper
│   ├── metrics/
│   │   ├── returns.py           # Return calculations
│   │   ├── risk.py              # Risk metrics
│   │   └── ratios.py            # Sharpe, Sortino, etc.
│   ├── optimization/            # TODO
│   │   ├── objectives.py
│   │   ├── constraints.py
│   │   └── optimizer.py
│   └── config.py                # Settings
├── tests/
│   ├── test_parser.py
│   └── test_metrics.py
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

# Use CLI
uv run portfolio --help
uv run portfolio metrics <your-portfolio.csv>
```
