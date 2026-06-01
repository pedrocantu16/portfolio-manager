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

## Phase 2: Optimization ✅ COMPLETE

### Implemented
- [x] `objectives.py` - Objective functions (max Sharpe, min volatility, max return)
- [x] `constraints.py` - Position size constraints, weights sum to 1
- [x] `optimizer.py` - Mean-variance optimization using scipy.optimize
- [x] CLI command: `optimize` - Find optimal allocation
- [x] CLI command: `rebalance` - Show trades needed to reach optimal
- [x] Return estimation methods: historical (with shrinkage/cap) and CAPM
- [x] Unit tests for optimizer (8 tests)

### Return Estimation Methods
| Method | Description | Use Case |
|--------|-------------|----------|
| `historical` | Historical mean returns with shrinkage and cap | Default, balanced |
| `capm` | Beta-based (E(R) = Rf + β × MRP) | Conservative, theory-based |

### CLI Usage
```bash
# Historical with adjustments (default)
uv run portfolio optimize Portfolio_Positions.csv --max-position 0.3

# CAPM-based returns
uv run portfolio optimize Portfolio_Positions.csv --method capm

# Minimum volatility
uv run portfolio optimize Portfolio_Positions.csv --objective min_volatility

# Rebalancing analysis
uv run portfolio rebalance Portfolio_Positions.csv
```

### Future Enhancements
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
│   ├── optimization/
│   │   ├── objectives.py        # Objective functions
│   │   ├── constraints.py       # Constraint builders
│   │   └── optimizer.py         # Portfolio optimizer
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

# Run tests (29 total)
uv run pytest tests/ -v

# Use CLI
uv run portfolio --help
uv run portfolio metrics <your-portfolio.csv>
uv run portfolio optimize <your-portfolio.csv> --max-position 0.3
uv run portfolio rebalance <your-portfolio.csv>
```
