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


if __name__ == "__main__":
    app()
