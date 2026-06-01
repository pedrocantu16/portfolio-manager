"""CLI entry point for portfolio manager."""

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from portfolio_manager.core.portfolio import Portfolio
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
    console.print()


if __name__ == "__main__":
    app()
