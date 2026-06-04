# Portfolio Manager

A Python CLI for portfolio analysis and optimization. Fetches real market data from Yahoo Finance, calculates risk/return metrics, and finds optimal allocations using mean-variance optimization.

## Features

- **Portfolio Analysis**: Load Fidelity CSV exports and view holdings, gains/losses
- **Risk Metrics**: Volatility, VaR, max drawdown, Sharpe/Sortino ratios
- **Portfolio Optimization**: Mean-variance optimization with multiple objectives
- **Return Estimation**: Historical returns, CAPM, or shrinkage estimators
- **Benchmark Comparison**: Compare vs S&P 500 or custom benchmark (alpha, beta, tracking error)
- **Sector Analysis**: View sector allocation and concentration
- **Tax-Loss Harvesting**: Identify opportunities to harvest losses and reduce taxes
- **Transaction Costs**: Estimate spread and commission costs for rebalancing

## Installation

Requires Python 3.11+ and [uv](https://github.com/astral-sh/uv).

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env

# Clone and install
git clone https://github.com/pedrocantu16/portfolio-manager.git
cd portfolio-manager
uv sync
```

## Quick Start

```bash
# View portfolio summary
uv run portfolio summary Portfolio_Positions.csv

# Calculate risk/return metrics
uv run portfolio metrics Portfolio_Positions.csv

# Find optimal allocation
uv run portfolio optimize Portfolio_Positions.csv --max-position 0.3

# Show rebalancing trades
uv run portfolio rebalance Portfolio_Positions.csv
```

## Commands

### `portfolio summary <csv>`

Display portfolio positions with current values and gains/losses.

```bash
uv run portfolio summary Portfolio_Positions.csv
```

### `portfolio metrics <csv>`

Calculate risk/return metrics using historical market data.

```bash
uv run portfolio metrics Portfolio_Positions.csv --period 1y
```

**Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--period` | `1y` | Historical data period (1mo, 3mo, 6mo, 1y, 2y, 3y, 5y) |

**Output includes:**
- Expected annual return
- Volatility (standard deviation)
- Value at Risk (VaR) at 95% confidence
- Maximum drawdown
- Sharpe and Sortino ratios

### `portfolio optimize <csv>`

Find optimal portfolio allocation using mean-variance optimization.

```bash
# Basic optimization (max Sharpe ratio)
uv run portfolio optimize Portfolio_Positions.csv --max-position 0.3

# Minimum volatility portfolio
uv run portfolio optimize Portfolio_Positions.csv --objective min_volatility

# Using CAPM for expected returns
uv run portfolio optimize Portfolio_Positions.csv --method capm
```

**Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--objective` | `max_sharpe` | Optimization goal: `max_sharpe`, `min_volatility`, `max_return` |
| `--max-position` | `1.0` | Maximum weight per position (e.g., 0.3 = 30%) |
| `--min-position` | `0.0` | Minimum weight per position |
| `--period` | `2y` | Historical data period |
| `--method` | `historical` | Return estimation: `historical` or `capm` |
| `--max-return` | `0.25` | Cap expected return per asset (25%) |
| `--shrinkage` | `0.3` | Blend returns toward mean (0-1, historical only) |
| `--market-premium` | `0.05` | Market risk premium for CAPM (5%) |

### `portfolio rebalance <csv>`

Show trades needed to reach optimal allocation, with before/after comparison.

```bash
uv run portfolio rebalance Portfolio_Positions.csv --max-position 0.3
```

**Options:** Same as `optimize` command.

### `portfolio holdings <csv>`

List all holdings with descriptions.

```bash
uv run portfolio holdings Portfolio_Positions.csv
```

### `portfolio sectors <csv>`

Show sector allocation of the portfolio.

```bash
uv run portfolio sectors Portfolio_Positions.csv
```

**Output includes:**
- Weight and value per sector
- Number of positions per sector
- List of holdings by sector

### `portfolio benchmark <csv>`

Compare portfolio performance against a benchmark (default: S&P 500).

```bash
# Compare vs S&P 500
uv run portfolio benchmark Portfolio_Positions.csv

# Compare vs NASDAQ (QQQ)
uv run portfolio benchmark Portfolio_Positions.csv --vs QQQ --period 2y
```

**Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--vs` | `SPY` | Benchmark ticker symbol |
| `--period` | `1y` | Historical data period |

**Output includes:**
- Annual return and volatility comparison
- Beta (market sensitivity)
- Alpha (risk-adjusted excess return)
- R-squared (how much movement explained by benchmark)
- Tracking error
- Information ratio
- Upside/downside capture ratios

### `portfolio tax-harvest <csv>`

Analyze tax-loss harvesting opportunities.

```bash
uv run portfolio tax-harvest Portfolio_Positions.csv --tax-rate 0.25 --min-loss 100
```

**Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--tax-rate` | `0.25` | Combined federal + state tax rate |
| `--min-loss` | `100` | Minimum loss in dollars to consider |

**Output includes:**
- Total unrealized gains and harvestable losses
- Potential tax savings
- Candidates for harvesting with alternative investments
- Wash sale rule reminders

## Return Estimation Methods

### Historical (default)
Uses historical mean returns with optional adjustments:
- **Shrinkage**: Blends individual returns toward the grand mean (reduces extreme estimates)
- **Return cap**: Limits maximum expected return per asset

```bash
uv run portfolio optimize <csv> --method historical --shrinkage 0.3 --max-return 0.25
```

### CAPM (Capital Asset Pricing Model)
Estimates returns based on market sensitivity (beta):

```
E(R) = Risk-free rate + β × Market Risk Premium
```

More conservative estimates grounded in market fundamentals.

```bash
uv run portfolio optimize <csv> --method capm --market-premium 0.05
```

## Example Output

```
Portfolio Optimization (max_sharpe, historical)
──────────────────────────────────────────────────
Optimal Allocation:
┏━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┓
┃ Symbol ┃ Current ┃ Optimal ┃ Change ┃
┡━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━┩
│ IVV    │    3.4% │   30.0% │ +26.6% │
│ VOO    │    4.1% │   30.0% │ +25.9% │
│ VXUS   │    1.7% │   30.0% │ +28.3% │
│ AMZN   │   82.9% │    0.0% │ -82.9% │
└────────┴─────────┴─────────┴────────┘

Expected Performance:
  Expected Return:  +19.1%
  Volatility:       16.2%
  Sharpe Ratio:     0.96
```

## Development

```bash
# Run tests
uv run pytest tests/ -v

# Lint
uv run ruff check src/

# Format
uv run ruff format src/
```

## Project Structure

```
src/portfolio_manager/
├── cli/main.py              # CLI commands
├── core/
│   ├── position.py          # Position dataclass
│   ├── portfolio.py         # Portfolio class
│   ├── account.py           # Multi-account support
│   ├── sectors.py           # Sector classification
│   ├── costs.py             # Transaction cost modeling
│   └── tax.py               # Tax-loss harvesting
├── data/
│   ├── parsers/fidelity.py  # Fidelity CSV parser
│   └── market.py            # Yahoo Finance wrapper
├── metrics/
│   ├── returns.py           # Return calculations
│   ├── risk.py              # Risk metrics
│   └── ratios.py            # Sharpe, Sortino
└── optimization/
    ├── objectives.py        # Objective functions
    ├── constraints.py       # Constraint builders (incl. sector)
    └── optimizer.py         # Mean-variance optimizer
```

## License

MIT
