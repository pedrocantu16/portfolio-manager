"""CLI entry point for portfolio manager."""

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from portfolio_manager.core.costs import CostModel
from portfolio_manager.core.portfolio import Portfolio
from portfolio_manager.core.sectors import Sector, get_sectors_for_portfolio
from portfolio_manager.core.tax import analyze_tax_loss_harvesting, get_wash_sale_alternatives
from portfolio_manager.data.market import MarketDataFetcher
from portfolio_manager.data.parsers.fidelity import FidelityParser
from portfolio_manager.metrics import (
    calculate_max_drawdown,
    calculate_portfolio_return,
    calculate_portfolio_volatility,
    calculate_returns,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_var,
)
from portfolio_manager.metrics.benchmark import BenchmarkAnalysis
from portfolio_manager.optimization import PortfolioOptimizer, ReturnEstimator
from portfolio_manager.optimization.optimizer import Objective

app = typer.Typer(
    name="portfolio",
    help="Portfolio management CLI with risk/return analysis.",
    no_args_is_help=True,
)
console = Console()

# Store loaded portfolio in memory (simple approach for CLI)
_loaded_portfolio: Portfolio | None = None


def _get_portfolio() -> Portfolio:
    """Get the currently loaded portfolio or raise an error."""
    if _loaded_portfolio is None:
        console.print("[red]No portfolio loaded. Use 'portfolio load <file>' first.[/red]")
        raise typer.Exit(1)
    return _loaded_portfolio


@app.command()
def load(
    file_path: Annotated[Path, typer.Argument(help="Path to the CSV file to load")],
    account: Annotated[str | None, typer.Option(help="Account name")] = None,
) -> None:
    """Load a portfolio from a CSV file."""
    global _loaded_portfolio

    if not file_path.exists():
        console.print(f"[red]File not found: {file_path}[/red]")
        raise typer.Exit(1)

    parser = FidelityParser()

    if not parser.can_parse(file_path):
        console.print(
            "[red]Unable to parse file. Only Fidelity CSV exports are supported.[/red]"
        )
        raise typer.Exit(1)

    try:
        portfolio = parser.parse(file_path)
        if account:
            portfolio.name = account
        _loaded_portfolio = portfolio

        console.print(f"[green]Loaded portfolio: {portfolio.name}[/green]")
        console.print(f"  Positions: {len(portfolio.get_investable_positions())}")
        console.print(f"  Total Value: ${portfolio.total_value:,.2f}")

    except Exception as e:
        console.print(f"[red]Error loading portfolio: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def summary(
    file_path: Annotated[
        Path | None, typer.Argument(help="Path to CSV file (optional if loaded)")
    ] = None,
) -> None:
    """Show portfolio summary with positions."""
    global _loaded_portfolio

    if file_path:
        load(file_path)

    portfolio = _get_portfolio()

    # Header
    console.print()
    console.print(f"[bold]Portfolio Summary: {portfolio.name}[/bold]")
    console.print("─" * 60)

    # Value summary
    console.print(f"  Total Value:      ${portfolio.total_value:,.2f}")
    console.print(f"  Invested Value:   ${portfolio.invested_value:,.2f}")
    console.print(f"  Cash:             ${portfolio.cash:,.2f}")
    gl_pct = portfolio.total_gain_loss_percent
    console.print(f"  Total Gain/Loss:  ${portfolio.total_gain_loss:,.2f} ({gl_pct:+.2f}%)")
    console.print()

    # Positions table
    table = Table(title="Positions")
    table.add_column("Symbol", style="cyan")
    table.add_column("Type", style="dim")
    table.add_column("Quantity", justify="right")
    table.add_column("Price", justify="right")
    table.add_column("Value", justify="right")
    table.add_column("Weight", justify="right")
    table.add_column("Gain/Loss", justify="right")
    table.add_column("G/L %", justify="right")

    # Sort by value descending
    positions = sorted(
        portfolio.get_investable_positions(),
        key=lambda p: p.current_value,
        reverse=True,
    )

    weights = portfolio.get_weights(include_cash=False)

    for pos in positions:
        gain_style = "green" if pos.gain_loss_dollar >= 0 else "red"
        weight = weights.get(pos.symbol, 0) * 100

        table.add_row(
            pos.symbol,
            pos.asset_type.value,
            f"{pos.quantity:,.2f}",
            f"${pos.current_price:,.2f}",
            f"${pos.current_value:,.2f}",
            f"{weight:.1f}%",
            f"[{gain_style}]${pos.gain_loss_dollar:+,.2f}[/{gain_style}]",
            f"[{gain_style}]{pos.gain_loss_percent:+.2f}%[/{gain_style}]",
        )

    console.print(table)


@app.command()
def metrics(
    file_path: Annotated[
        Path | None, typer.Argument(help="Path to CSV file (optional if loaded)")
    ] = None,
    period: Annotated[
        str, typer.Option(help="Historical data period (1mo, 3mo, 6mo, 1y, 2y)")
    ] = "1y",
) -> None:
    """Calculate and display portfolio metrics."""
    global _loaded_portfolio

    if file_path:
        load(file_path)

    portfolio = _get_portfolio()

    console.print()
    console.print(f"[bold]Portfolio Metrics ({period} historical data)[/bold]")
    console.print("─" * 50)

    # Get symbols for market data
    symbols = portfolio.get_symbols(include_cash=False)

    if not symbols:
        console.print("[yellow]No investable positions to analyze.[/yellow]")
        return

    # Fetch market data
    console.print("[dim]Fetching market data...[/dim]")
    fetcher = MarketDataFetcher()

    try:
        prices = fetcher.get_historical_prices(symbols, period=period)
        risk_free_rate = fetcher.get_risk_free_rate()
    except Exception as e:
        console.print(f"[red]Error fetching market data: {e}[/red]")
        raise typer.Exit(1)

    if prices.empty:
        console.print("[red]No price data available for portfolio symbols.[/red]")
        raise typer.Exit(1)

    # Filter to symbols we have data for
    available_symbols = [s for s in symbols if s in prices.columns]
    missing_symbols = set(symbols) - set(available_symbols)

    if missing_symbols:
        console.print(f"[yellow]Warning: No data for: {', '.join(missing_symbols)}[/yellow]")

    # Calculate returns
    returns = calculate_returns(prices[available_symbols])

    # Get weights for available symbols only
    all_weights = portfolio.get_weights(include_cash=False)
    weights = {s: all_weights[s] for s in available_symbols if s in all_weights}

    # Renormalize weights
    total_weight = sum(weights.values())
    if total_weight > 0:
        weights = {s: w / total_weight for s, w in weights.items()}

    # Portfolio metrics
    portfolio_return = calculate_portfolio_return(weights, returns, annualize=True)
    portfolio_vol = calculate_portfolio_volatility(weights, returns, annualize=True)

    # Calculate portfolio daily returns for VaR and ratios
    weight_series = {s: weights[s] for s in available_symbols if s in weights}
    portfolio_daily_returns = sum(
        returns[s] * weight_series.get(s, 0) for s in available_symbols if s in returns.columns
    )

    var_95 = calculate_var(portfolio_daily_returns, confidence=0.95)
    var_dollar = var_95 * portfolio.invested_value

    # Calculate max drawdown from portfolio prices
    portfolio_prices = (1 + portfolio_daily_returns).cumprod()
    max_dd = calculate_max_drawdown(portfolio_prices)

    # Ratios
    sharpe = calculate_sharpe_ratio(portfolio_return, risk_free_rate, portfolio_vol)
    sortino = calculate_sortino_ratio(portfolio_daily_returns, risk_free_rate)

    # Display results
    console.print()
    console.print(f"  Total Value:        ${portfolio.total_value:,.2f}")
    console.print(f"  Invested Value:     ${portfolio.invested_value:,.2f}")
    console.print(f"  Cash:               ${portfolio.cash:,.2f}")
    console.print()

    console.print("[bold]Returns:[/bold]")
    ret_style = "green" if portfolio_return >= 0 else "red"
    console.print(f"  Expected Annual:    [{ret_style}]{portfolio_return*100:+.1f}%[/{ret_style}]")
    console.print()

    console.print("[bold]Risk:[/bold]")
    console.print(f"  Volatility (σ):     {portfolio_vol*100:.1f}%")
    console.print(f"  VaR (95%, daily):   ${var_dollar:,.2f}")
    console.print(f"  Max Drawdown:       {max_dd*100:.1f}%")
    console.print()

    console.print("[bold]Ratios:[/bold]")
    console.print(f"  Risk-Free Rate:     {risk_free_rate*100:.2f}%")
    console.print(f"  Sharpe:             {sharpe:.2f}")
    console.print(f"  Sortino:            {sortino:.2f}")
    console.print()


@app.command()
def holdings(
    file_path: Annotated[
        Path | None, typer.Argument(help="Path to CSV file (optional if loaded)")
    ] = None,
) -> None:
    """List all holdings in the portfolio."""
    global _loaded_portfolio

    if file_path:
        load(file_path)

    portfolio = _get_portfolio()

    table = Table(title=f"Holdings - {portfolio.name}")
    table.add_column("Symbol", style="cyan")
    table.add_column("Description")
    table.add_column("Quantity", justify="right")
    table.add_column("Value", justify="right")

    for symbol, pos in sorted(portfolio.positions.items()):
        table.add_row(
            symbol,
            pos.description[:40] + "..." if len(pos.description) > 40 else pos.description,
            f"{pos.quantity:,.2f}",
            f"${pos.current_value:,.2f}",
        )

    console.print(table)


@app.command()
def optimize(
    file_path: Annotated[
        Path | None, typer.Argument(help="Path to CSV file (optional if loaded)")
    ] = None,
    objective: Annotated[
        str, typer.Option(help="Objective: max_sharpe, min_volatility, max_return")
    ] = "max_sharpe",
    max_position: Annotated[
        float, typer.Option(help="Maximum weight per position (0.0-1.0)")
    ] = 1.0,
    min_position: Annotated[
        float, typer.Option(help="Minimum weight per position (0.0-1.0)")
    ] = 0.0,
    period: Annotated[
        str, typer.Option(help="Historical data period (1y, 2y, 3y, 5y)")
    ] = "2y",
    method: Annotated[
        str, typer.Option(help="Return estimate: historical or capm")
    ] = "historical",
    max_return: Annotated[
        float, typer.Option(help="Cap expected return per asset (e.g., 0.25 = 25%)")
    ] = 0.25,
    shrinkage: Annotated[
        float, typer.Option(help="Shrinkage factor 0-1 (historical only)")
    ] = 0.3,
    market_premium: Annotated[
        float, typer.Option(help="Market risk premium for CAPM (e.g., 0.05 = 5%)")
    ] = 0.05,
) -> None:
    """Find optimal portfolio allocation."""
    global _loaded_portfolio

    if file_path:
        load(file_path)

    portfolio = _get_portfolio()

    # Map string to Objective enum
    obj_map = {
        "max_sharpe": Objective.MAX_SHARPE,
        "min_volatility": Objective.MIN_VOLATILITY,
        "min_vol": Objective.MIN_VOLATILITY,
        "max_return": Objective.MAX_RETURN,
    }
    if objective not in obj_map:
        console.print(f"[red]Unknown objective: {objective}[/red]")
        console.print("Options: max_sharpe, min_volatility, max_return")
        raise typer.Exit(1)

    obj = obj_map[objective]

    symbols = portfolio.get_symbols(include_cash=False)
    if not symbols:
        console.print("[yellow]No investable positions to optimize.[/yellow]")
        return

    # Validate method
    method_map = {
        "historical": ReturnEstimator.HISTORICAL,
        "capm": ReturnEstimator.CAPM,
    }
    if method not in method_map:
        console.print(f"[red]Unknown method: {method}[/red]")
        console.print("Options: historical, capm")
        raise typer.Exit(1)

    return_method = method_map[method]

    console.print()
    console.print(f"[bold]Portfolio Optimization ({obj.value}, {method})[/bold]")
    console.print("─" * 50)
    console.print("[dim]Fetching market data...[/dim]")

    fetcher = MarketDataFetcher()
    try:
        # Fetch asset prices
        prices = fetcher.get_historical_prices(symbols, period=period)
        risk_free_rate = fetcher.get_risk_free_rate()

        # Fetch market returns for CAPM
        market_returns = None
        if return_method == ReturnEstimator.CAPM:
            market_prices = fetcher.get_historical_prices(["SPY"], period=period)
            if not market_prices.empty:
                market_returns = calculate_returns(market_prices)["SPY"]
    except Exception as e:
        console.print(f"[red]Error fetching market data: {e}[/red]")
        raise typer.Exit(1)

    if prices.empty:
        console.print("[red]No price data available.[/red]")
        raise typer.Exit(1)

    available_symbols = [s for s in symbols if s in prices.columns]
    missing_symbols = set(symbols) - set(available_symbols)
    if missing_symbols:
        console.print(f"[yellow]Warning: No data for: {', '.join(missing_symbols)}[/yellow]")

    returns = calculate_returns(prices[available_symbols])

    console.print("[dim]Running optimization...[/dim]")
    optimizer = PortfolioOptimizer(
        returns,
        risk_free_rate,
        max_expected_return=max_return,
        shrinkage=shrinkage if return_method == ReturnEstimator.HISTORICAL else 0.0,
        return_method=return_method,
        market_returns=market_returns,
        market_risk_premium=market_premium,
    )
    result = optimizer.optimize(
        objective=obj,
        min_weight=min_position,
        max_weight=max_position,
    )

    if not result.success:
        console.print(f"[yellow]Optimization warning: {result.message}[/yellow]")

    # Display results
    console.print()
    console.print("[bold]Optimal Allocation:[/bold]")

    table = Table()
    table.add_column("Symbol", style="cyan")
    table.add_column("Current", justify="right")
    table.add_column("Optimal", justify="right")
    table.add_column("Change", justify="right")

    current_weights = portfolio.get_weights(include_cash=False)
    # Renormalize current weights to available symbols
    total_current = sum(current_weights.get(s, 0) for s in available_symbols)
    if total_current > 0:
        current_weights = {
            s: current_weights.get(s, 0) / total_current for s in available_symbols
        }

    # Sort by optimal weight descending
    sorted_symbols = sorted(
        available_symbols, key=lambda s: result.weights.get(s, 0), reverse=True
    )

    for symbol in sorted_symbols:
        current = current_weights.get(symbol, 0) * 100
        optimal = result.weights.get(symbol, 0) * 100
        change = optimal - current

        if abs(optimal) < 0.1 and abs(current) < 0.1:
            continue

        change_style = "green" if change > 0 else "red" if change < 0 else "dim"
        table.add_row(
            symbol,
            f"{current:.1f}%",
            f"{optimal:.1f}%",
            f"[{change_style}]{change:+.1f}%[/{change_style}]",
        )

    console.print(table)

    # Calculate current portfolio metrics for comparison
    current_return = calculate_portfolio_return(current_weights, returns, annualize=True)
    current_vol = calculate_portfolio_volatility(current_weights, returns, annualize=True)
    current_sharpe = (
        (current_return - risk_free_rate) / current_vol if current_vol > 0 else 0.0
    )

    # Comparison table
    console.print()
    console.print("[bold]Performance Comparison:[/bold]")

    def fmt_change(curr: float, opt: float, pct: bool = True) -> str:
        diff = opt - curr
        style = "green" if diff > 0 else "red" if diff < 0 else "dim"
        if pct:
            return f"[{style}]{diff*100:+.1f}%[/{style}]"
        return f"[{style}]{diff:+.2f}[/{style}]"

    comp_table = Table(show_header=True)
    comp_table.add_column("Metric")
    comp_table.add_column("Current", justify="right")
    comp_table.add_column("Optimal", justify="right")
    comp_table.add_column("Change", justify="right")

    comp_table.add_row(
        "Expected Return",
        f"{current_return*100:.1f}%",
        f"{result.expected_return*100:.1f}%",
        fmt_change(current_return, result.expected_return),
    )
    comp_table.add_row(
        "Volatility",
        f"{current_vol*100:.1f}%",
        f"{result.volatility*100:.1f}%",
        fmt_change(current_vol, result.volatility),
    )
    comp_table.add_row(
        "Sharpe Ratio",
        f"{current_sharpe:.2f}",
        f"{result.sharpe_ratio:.2f}",
        fmt_change(current_sharpe, result.sharpe_ratio, pct=False),
    )

    console.print(comp_table)
    console.print()


@app.command()
def rebalance(
    file_path: Annotated[
        Path | None, typer.Argument(help="Path to CSV file (optional if loaded)")
    ] = None,
    max_position: Annotated[
        float, typer.Option(help="Maximum weight per position (0.0-1.0)")
    ] = 0.30,
    period: Annotated[
        str, typer.Option(help="Historical data period (1y, 2y, 3y, 5y)")
    ] = "2y",
    method: Annotated[
        str, typer.Option(help="Return estimate: historical or capm")
    ] = "historical",
    max_return: Annotated[
        float, typer.Option(help="Cap expected return per asset (e.g., 0.25 = 25%)")
    ] = 0.25,
    shrinkage: Annotated[
        float, typer.Option(help="Shrinkage factor 0-1 (historical only)")
    ] = 0.3,
    market_premium: Annotated[
        float, typer.Option(help="Market risk premium for CAPM (e.g., 0.05 = 5%)")
    ] = 0.05,
) -> None:
    """Show rebalancing trades to reach optimal allocation."""
    global _loaded_portfolio

    if file_path:
        load(file_path)

    portfolio = _get_portfolio()
    symbols = portfolio.get_symbols(include_cash=False)

    if not symbols:
        console.print("[yellow]No investable positions.[/yellow]")
        return

    # Validate method
    method_map = {
        "historical": ReturnEstimator.HISTORICAL,
        "capm": ReturnEstimator.CAPM,
    }
    return_method = method_map.get(method, ReturnEstimator.HISTORICAL)

    console.print()
    console.print(f"[bold]Rebalancing Analysis ({method})[/bold]")
    console.print("─" * 50)
    console.print("[dim]Fetching market data...[/dim]")

    fetcher = MarketDataFetcher()
    try:
        prices = fetcher.get_historical_prices(symbols, period=period)
        risk_free_rate = fetcher.get_risk_free_rate()

        market_returns = None
        if return_method == ReturnEstimator.CAPM:
            market_prices = fetcher.get_historical_prices(["SPY"], period=period)
            if not market_prices.empty:
                market_returns = calculate_returns(market_prices)["SPY"]
    except Exception as e:
        console.print(f"[red]Error fetching market data: {e}[/red]")
        raise typer.Exit(1)

    available_symbols = [s for s in symbols if s in prices.columns]
    returns = calculate_returns(prices[available_symbols])

    optimizer = PortfolioOptimizer(
        returns,
        risk_free_rate,
        max_expected_return=max_return,
        shrinkage=shrinkage if return_method == ReturnEstimator.HISTORICAL else 0.0,
        return_method=return_method,
        market_returns=market_returns,
        market_risk_premium=market_premium,
    )
    result = optimizer.optimize(
        objective=Objective.MAX_SHARPE,
        max_weight=max_position,
    )

    # Calculate current metrics for comparison
    current_weights = portfolio.get_weights(include_cash=False)
    total_current = sum(current_weights.get(s, 0) for s in available_symbols)
    if total_current > 0:
        current_weights = {
            s: current_weights.get(s, 0) / total_current for s in available_symbols
        }

    current_return = calculate_portfolio_return(current_weights, returns, annualize=True)
    current_vol = calculate_portfolio_volatility(current_weights, returns, annualize=True)
    current_sharpe = (
        (current_return - risk_free_rate) / current_vol if current_vol > 0 else 0.0
    )

    invested_value = portfolio.invested_value

    # Show trades needed
    console.print()
    console.print("[bold]Suggested Trades:[/bold]")

    table = Table()
    table.add_column("Symbol", style="cyan")
    table.add_column("Action", justify="center")
    table.add_column("Current $", justify="right")
    table.add_column("Target $", justify="right")
    table.add_column("Trade $", justify="right")

    trades = []
    for symbol in available_symbols:
        current_pct = current_weights.get(symbol, 0)
        optimal_pct = result.weights.get(symbol, 0)

        current_val = current_pct * invested_value
        target_val = optimal_pct * invested_value
        trade_val = target_val - current_val

        if abs(trade_val) < 10:
            continue

        trades.append((symbol, current_val, target_val, trade_val))

    # Sort by absolute trade size
    trades.sort(key=lambda x: abs(x[3]), reverse=True)

    for symbol, current_val, target_val, trade_val in trades:
        action = "[green]BUY[/green]" if trade_val > 0 else "[red]SELL[/red]"
        trade_style = "green" if trade_val > 0 else "red"
        table.add_row(
            symbol,
            action,
            f"${current_val:,.0f}",
            f"${target_val:,.0f}",
            f"[{trade_style}]${trade_val:+,.0f}[/{trade_style}]",
        )

    console.print(table)

    # Comparison
    console.print()
    console.print("[bold]Impact:[/bold]")

    comp_table = Table(show_header=True)
    comp_table.add_column("Metric")
    comp_table.add_column("Current", justify="right")
    comp_table.add_column("Optimal", justify="right")
    comp_table.add_column("Change", justify="right")

    def fmt_change(curr: float, opt: float, pct: bool = True) -> str:
        diff = opt - curr
        style = "green" if diff > 0 else "red" if diff < 0 else "dim"
        if pct:
            return f"[{style}]{diff*100:+.1f}%[/{style}]"
        return f"[{style}]{diff:+.2f}[/{style}]"

    comp_table.add_row(
        "Expected Return",
        f"{current_return*100:.1f}%",
        f"{result.expected_return*100:.1f}%",
        fmt_change(current_return, result.expected_return),
    )
    comp_table.add_row(
        "Volatility",
        f"{current_vol*100:.1f}%",
        f"{result.volatility*100:.1f}%",
        fmt_change(current_vol, result.volatility),
    )
    comp_table.add_row(
        "Sharpe Ratio",
        f"{current_sharpe:.2f}",
        f"{result.sharpe_ratio:.2f}",
        fmt_change(current_sharpe, result.sharpe_ratio, pct=False),
    )

    console.print(comp_table)

    # Transaction costs
    cost_model = CostModel()
    trade_dict = {symbol: trade_val for symbol, _, _, trade_val in trades}
    costs = cost_model.estimate_rebalance_costs(trade_dict)

    if costs.total_cost > 0:
        console.print()
        console.print("[bold]Estimated Transaction Costs:[/bold]")
        console.print(f"  Total Trades:    ${costs.total_trade_value:,.0f}")
        console.print(f"  Spread Cost:     ${costs.total_spread_cost:,.2f}")
        console.print(f"  Total Cost:      ${costs.total_cost:,.2f} ({costs.cost_percent:.2f}%)")
    console.print()


@app.command()
def sectors(
    file_path: Annotated[
        Path | None, typer.Argument(help="Path to CSV file (optional if loaded)")
    ] = None,
) -> None:
    """Show sector allocation of portfolio."""
    global _loaded_portfolio

    if file_path:
        load(file_path)

    portfolio = _get_portfolio()

    console.print()
    console.print("[bold]Sector Allocation[/bold]")
    console.print("─" * 50)

    # Get sectors for all symbols
    symbols = portfolio.get_symbols(include_cash=False)
    sector_map = get_sectors_for_portfolio(symbols, use_api=False)

    # Calculate sector weights
    weights = portfolio.get_weights(include_cash=False)
    sector_weights: dict[Sector, float] = {}
    sector_values: dict[Sector, float] = {}

    for symbol, weight in weights.items():
        sector = sector_map.get(symbol, Sector.UNKNOWN)
        sector_weights[sector] = sector_weights.get(sector, 0.0) + weight
        pos = portfolio.get_position(symbol)
        if pos:
            sector_values[sector] = sector_values.get(sector, 0.0) + pos.current_value

    # Sort by weight
    sorted_sectors = sorted(sector_weights.items(), key=lambda x: x[1], reverse=True)

    table = Table()
    table.add_column("Sector")
    table.add_column("Weight", justify="right")
    table.add_column("Value", justify="right")
    table.add_column("Positions", justify="right")

    for sector, weight in sorted_sectors:
        # Count positions in sector
        count = sum(1 for s in symbols if sector_map.get(s) == sector)
        value = sector_values.get(sector, 0)

        table.add_row(
            sector.value,
            f"{weight*100:.1f}%",
            f"${value:,.0f}",
            str(count),
        )

    console.print(table)

    # Show positions by sector
    console.print()
    console.print("[bold]Positions by Sector:[/bold]")

    for sector, _ in sorted_sectors:
        sector_symbols = [s for s in symbols if sector_map.get(s) == sector]
        if sector_symbols:
            positions_str = ", ".join(sector_symbols)
            console.print(f"  {sector.value}: {positions_str}")

    console.print()


@app.command()
def tax_harvest(
    file_path: Annotated[
        Path | None, typer.Argument(help="Path to CSV file (optional if loaded)")
    ] = None,
    tax_rate: Annotated[
        float, typer.Option(help="Combined tax rate (e.g., 0.25 = 25%)")
    ] = 0.25,
    min_loss: Annotated[
        float, typer.Option(help="Minimum loss in dollars to consider")
    ] = 100.0,
) -> None:
    """Analyze tax-loss harvesting opportunities."""
    global _loaded_portfolio

    if file_path:
        load(file_path)

    portfolio = _get_portfolio()

    console.print()
    console.print("[bold]Tax-Loss Harvesting Analysis[/bold]")
    console.print("─" * 50)

    analysis = analyze_tax_loss_harvesting(
        portfolio,
        tax_rate=tax_rate,
        min_loss_threshold=min_loss,
    )

    if not analysis.candidates:
        console.print("[green]No tax-loss harvesting opportunities found.[/green]")
        console.print(f"  Total Gains: ${analysis.total_gains:,.2f}")
        console.print()
        return

    # Summary
    console.print(f"  Total Unrealized Gains:  ${analysis.total_gains:,.2f}")
    console.print(f"  Total Harvestable Loss:  ${analysis.total_harvestable_loss:,.2f}")
    console.print(f"  Net Gain/Loss:           ${analysis.net_gain_loss:,.2f}")
    console.print(f"  Potential Tax Savings:   [green]${analysis.total_tax_savings:,.2f}[/green]")
    console.print()

    # Candidates table
    console.print("[bold]Harvesting Candidates:[/bold]")

    table = Table()
    table.add_column("Symbol", style="cyan")
    table.add_column("Value", justify="right")
    table.add_column("Loss", justify="right")
    table.add_column("Loss %", justify="right")
    table.add_column("Tax Savings", justify="right")
    table.add_column("Alternatives")

    for candidate in analysis.candidates:
        pos = candidate.position
        alternatives = get_wash_sale_alternatives(pos.symbol)
        alt_str = ", ".join(alternatives[:3])

        table.add_row(
            pos.symbol,
            f"${pos.current_value:,.0f}",
            f"[red]${candidate.unrealized_loss:,.0f}[/red]",
            f"[red]-{candidate.loss_percent:.1f}%[/red]",
            f"[green]${candidate.tax_savings_estimate:,.0f}[/green]",
            alt_str,
        )

    console.print(table)

    console.print()
    console.print("[dim]Note: Alternatives are suggestions to avoid wash sale rule.[/dim]")
    console.print("[dim]Wait 30 days before repurchasing the same security.[/dim]")
    console.print()


@app.command()
def benchmark(
    file_path: Annotated[
        Path | None, typer.Argument(help="Path to CSV file (optional if loaded)")
    ] = None,
    vs: Annotated[
        str, typer.Option(help="Benchmark ticker symbol")
    ] = "SPY",
    period: Annotated[
        str, typer.Option(help="Historical data period (1y, 2y, 3y, 5y)")
    ] = "1y",
) -> None:
    """Compare portfolio performance against a benchmark."""
    global _loaded_portfolio

    if file_path:
        load(file_path)

    portfolio = _get_portfolio()

    console.print()
    console.print(f"[bold]Benchmark Comparison vs {vs}[/bold]")
    console.print("─" * 50)

    symbols = portfolio.get_symbols(include_cash=False)
    if not symbols:
        console.print("[yellow]No investable positions to analyze.[/yellow]")
        return

    console.print("[dim]Fetching market data...[/dim]")
    fetcher = MarketDataFetcher()

    try:
        # Fetch portfolio assets and benchmark
        all_symbols = symbols + [vs]
        prices = fetcher.get_historical_prices(all_symbols, period=period)
        risk_free_rate = fetcher.get_risk_free_rate()
    except Exception as e:
        console.print(f"[red]Error fetching market data: {e}[/red]")
        raise typer.Exit(1)

    if vs not in prices.columns:
        console.print(f"[red]Benchmark {vs} not found.[/red]")
        raise typer.Exit(1)

    # Filter to available symbols
    available_symbols = [s for s in symbols if s in prices.columns]
    missing_symbols = set(symbols) - set(available_symbols)
    if missing_symbols:
        console.print(f"[yellow]Warning: No data for: {', '.join(missing_symbols)}[/yellow]")

    returns = calculate_returns(prices)

    # Calculate portfolio returns
    weights = portfolio.get_weights(include_cash=False)
    total_weight = sum(weights.get(s, 0) for s in available_symbols)
    if total_weight > 0:
        weights = {s: weights.get(s, 0) / total_weight for s in available_symbols}

    portfolio_returns = sum(
        returns[s] * weights.get(s, 0)
        for s in available_symbols
        if s in returns.columns
    )
    benchmark_returns = returns[vs]

    # Run benchmark analysis
    analysis = BenchmarkAnalysis(
        portfolio_returns=portfolio_returns,
        benchmark_returns=benchmark_returns,
        benchmark_symbol=vs,
        risk_free_rate=risk_free_rate,
    )
    results = analysis.run()

    # Portfolio vs benchmark returns
    port_annual = portfolio_returns.mean() * 252
    bench_annual = benchmark_returns.mean() * 252
    port_vol = portfolio_returns.std() * (252 ** 0.5)
    bench_vol = benchmark_returns.std() * (252 ** 0.5)

    console.print()
    console.print("[bold]Performance:[/bold]")

    perf_table = Table(show_header=True)
    perf_table.add_column("Metric")
    perf_table.add_column("Portfolio", justify="right")
    perf_table.add_column(vs, justify="right")
    perf_table.add_column("Difference", justify="right")

    ret_diff = port_annual - bench_annual
    ret_style = "green" if ret_diff > 0 else "red"
    perf_table.add_row(
        "Annual Return",
        f"{port_annual*100:.1f}%",
        f"{bench_annual*100:.1f}%",
        f"[{ret_style}]{ret_diff*100:+.1f}%[/{ret_style}]",
    )

    vol_diff = port_vol - bench_vol
    vol_style = "red" if vol_diff > 0 else "green"
    perf_table.add_row(
        "Volatility",
        f"{port_vol*100:.1f}%",
        f"{bench_vol*100:.1f}%",
        f"[{vol_style}]{vol_diff*100:+.1f}%[/{vol_style}]",
    )

    console.print(perf_table)

    # Benchmark metrics
    console.print()
    console.print("[bold]Risk Metrics:[/bold]")

    beta = results["beta"]
    alpha = results["alpha"]
    tracking_error = results["tracking_error"]
    info_ratio = results["information_ratio"]
    r_squared = results["r_squared"]
    upside, downside = results["upside_capture"], results["downside_capture"]

    # Beta interpretation
    if beta > 1.1:
        beta_note = "more volatile than market"
    elif beta < 0.9:
        beta_note = "less volatile than market"
    else:
        beta_note = "moves with market"

    # Alpha interpretation
    alpha_style = "green" if alpha > 0 else "red"
    alpha_note = "outperforming" if alpha > 0 else "underperforming"

    console.print(f"  Beta:              {beta:.2f} ({beta_note})")
    console.print(f"  Alpha:             [{alpha_style}]{alpha*100:+.2f}%[/{alpha_style}] ({alpha_note})")
    console.print(f"  R-squared:         {r_squared:.2f} ({r_squared*100:.0f}% explained by {vs})")
    console.print(f"  Tracking Error:    {tracking_error*100:.1f}%")
    console.print(f"  Information Ratio: {info_ratio:.2f}")
    console.print()

    console.print("[bold]Capture Ratios:[/bold]")
    up_style = "green" if upside > 100 else "yellow"
    down_style = "green" if downside < 100 else "yellow"
    console.print(f"  Upside Capture:    [{up_style}]{upside:.0f}%[/{up_style}]")
    console.print(f"  Downside Capture:  [{down_style}]{downside:.0f}%[/{down_style}]")
    console.print()


if __name__ == "__main__":
    app()
