"""Streamlit web application for Portfolio Manager."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import tempfile

from portfolio_manager.core.portfolio import Portfolio
from portfolio_manager.data.parsers.fidelity import FidelityParser
from portfolio_manager.data.parsers.generic import GenericParser
from portfolio_manager.data.market import MarketDataFetcher
from portfolio_manager.metrics import (
    calculate_returns,
    calculate_portfolio_return,
    calculate_portfolio_volatility,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_var,
    calculate_max_drawdown,
)
from portfolio_manager.optimization import PortfolioOptimizer
from portfolio_manager.optimization.optimizer import Objective
from portfolio_manager.backtest import run_backtest, run_monte_carlo, run_walk_forward, RebalanceFrequency
from portfolio_manager.core.sectors import Sector, get_sectors_for_portfolio
from portfolio_manager.core.tax import analyze_tax_loss_harvesting, get_wash_sale_alternatives
from portfolio_manager.metrics.benchmark import BenchmarkAnalysis

st.set_page_config(
    page_title="Portfolio Manager",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Portfolio Manager")


def parse_uploaded_file(uploaded_file) -> Portfolio | None:
    """Parse an uploaded CSV file into a Portfolio."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = Path(tmp.name)

    parsers = [
        ("Fidelity", FidelityParser()),
        ("Generic", GenericParser()),
    ]

    for name, parser in parsers:
        if parser.can_parse(tmp_path):
            portfolio = parser.parse(tmp_path)
            tmp_path.unlink()
            return portfolio

    tmp_path.unlink()
    return None


@st.cache_data(ttl=300)
def fetch_market_data(symbols: list[str], period: str = "1y"):
    """Fetch market data with caching."""
    fetcher = MarketDataFetcher()
    prices = fetcher.get_historical_prices(symbols, period=period)
    risk_free_rate = fetcher.get_risk_free_rate()
    return prices, risk_free_rate


def show_portfolio_summary(portfolio: Portfolio):
    """Display portfolio summary metrics."""
    st.subheader("Portfolio Summary")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Value", f"${portfolio.total_value:,.0f}")
    with col2:
        st.metric("Invested", f"${portfolio.invested_value:,.0f}")
    with col3:
        st.metric("Cash", f"${portfolio.cash:,.0f}")
    with col4:
        st.metric("Positions", len(portfolio.get_symbols(include_cash=False)))


def show_allocation_chart(portfolio: Portfolio):
    """Display allocation pie chart."""
    st.subheader("Allocation")

    positions = portfolio.get_equity_positions()
    if not positions:
        st.warning("No equity positions found.")
        return

    df = pd.DataFrame([
        {"Symbol": p.symbol, "Value": p.current_value}
        for p in positions
    ])

    if portfolio.cash > 0:
        df = pd.concat([df, pd.DataFrame([{"Symbol": "Cash", "Value": portfolio.cash}])])

    fig = px.pie(
        df,
        values="Value",
        names="Symbol",
        hole=0.4,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(showlegend=False, margin=dict(t=0, b=0, l=0, r=0))

    st.plotly_chart(fig, use_container_width=True)


def show_holdings_table(portfolio: Portfolio):
    """Display holdings table."""
    st.subheader("Holdings")

    positions = list(portfolio.positions.values())
    df = pd.DataFrame([
        {
            "Symbol": p.symbol,
            "Shares": p.quantity,
            "Price": p.current_price,
            "Value": p.current_value,
            "Cost Basis": p.cost_basis,
            "Gain/Loss": p.gain_loss_dollar,
            "Gain %": p.gain_loss_percent,
            "Weight %": p.percent_of_account,
        }
        for p in positions
    ])

    df = df.sort_values("Value", ascending=False)

    st.dataframe(
        df.style.format({
            "Shares": "{:.2f}",
            "Price": "${:.2f}",
            "Value": "${:,.0f}",
            "Cost Basis": "${:,.0f}",
            "Gain/Loss": "${:+,.0f}",
            "Gain %": "{:+.1f}%",
            "Weight %": "{:.1f}%",
        }).map(
            lambda x: "color: green" if isinstance(x, (int, float)) and x > 0 else "color: red" if isinstance(x, (int, float)) and x < 0 else "",
            subset=["Gain/Loss", "Gain %"]
        ),
        use_container_width=True,
        hide_index=True,
    )


def show_risk_metrics(portfolio: Portfolio, prices: pd.DataFrame, risk_free_rate: float):
    """Display risk/return metrics."""
    st.subheader("Risk Metrics")

    symbols = portfolio.get_symbols(include_cash=False)
    available = [s for s in symbols if s in prices.columns]

    if not available:
        st.warning("No price data available for risk metrics.")
        return

    returns = calculate_returns(prices[available])
    weights = portfolio.get_weights(include_cash=False)
    total_weight = sum(weights.get(s, 0) for s in available)
    if total_weight > 0:
        weights = {s: weights.get(s, 0) / total_weight for s in available}

    port_return = calculate_portfolio_return(weights, returns, annualize=True)
    port_vol = calculate_portfolio_volatility(weights, returns, annualize=True)
    sharpe = calculate_sharpe_ratio(port_return, port_vol, risk_free_rate)

    port_returns = sum(returns[s] * weights.get(s, 0) for s in available)
    sortino = calculate_sortino_ratio(port_returns, risk_free_rate)
    var_95 = calculate_var(port_returns) * portfolio.invested_value
    max_dd = calculate_max_drawdown(port_returns)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Expected Return", f"{port_return*100:.1f}%")
        st.metric("Volatility", f"{port_vol*100:.1f}%")

    with col2:
        st.metric("Sharpe Ratio", f"{sharpe:.2f}")
        st.metric("Sortino Ratio", f"{sortino:.2f}")

    with col3:
        st.metric("VaR (95%)", f"${var_95:,.0f}")
        st.metric("Max Drawdown", f"{max_dd*100:.1f}%")


def show_optimization(portfolio: Portfolio, prices: pd.DataFrame, risk_free_rate: float):
    """Interactive portfolio optimization."""
    st.subheader("Portfolio Optimization")

    symbols = portfolio.get_symbols(include_cash=False)
    available = [s for s in symbols if s in prices.columns]

    if len(available) < 2:
        st.warning("Need at least 2 assets for optimization.")
        return

    col1, col2 = st.columns(2)

    with col1:
        objective = st.selectbox(
            "Objective",
            ["Max Sharpe", "Min Volatility"],
            index=0,
        )
        max_position_pct = st.slider(
            "Max Position Size (%)",
            min_value=1,
            max_value=100,
            value=30,
            step=1,
            help="Maximum weight for any single position",
        )
        max_position = max_position_pct / 100

    with col2:
        period = st.selectbox(
            "Historical Period",
            ["1y", "2y", "3y", "5y"],
            index=1,
        )

    if st.button("Run Optimization", type="primary"):
        with st.spinner("Optimizing..."):
            returns = calculate_returns(prices[available])

            obj = Objective.MAX_SHARPE if "Sharpe" in objective else Objective.MIN_VOLATILITY

            optimizer = PortfolioOptimizer(
                returns,
                risk_free_rate,
                max_expected_return=0.25,
                shrinkage=0.3,
            )
            result = optimizer.optimize(
                objective=obj,
                max_weight=max_position,
            )

            current_weights = portfolio.get_weights(include_cash=False)
            total = sum(current_weights.get(s, 0) for s in available)
            if total > 0:
                current_weights = {s: current_weights.get(s, 0) / total for s in available}

            # Results table
            data = []
            for symbol in sorted(available, key=lambda s: result.weights.get(s, 0), reverse=True):
                current = current_weights.get(symbol, 0) * 100
                optimal = result.weights.get(symbol, 0) * 100
                change = optimal - current
                if abs(optimal) > 0.1 or abs(current) > 0.1:
                    data.append({
                        "Symbol": symbol,
                        "Current": f"{current:.1f}%",
                        "Optimal": f"{optimal:.1f}%",
                        "Change": change,
                    })

            df = pd.DataFrame(data)

            col1, col2 = st.columns(2)

            with col1:
                st.write("**Optimal Allocation**")
                st.dataframe(
                    df.style.format({"Change": "{:+.1f}%"}).map(
                        lambda x: "color: green" if x > 0 else "color: red" if x < 0 else "",
                        subset=["Change"]
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

            with col2:
                st.write("**Expected Performance**")
                current_ret = calculate_portfolio_return(current_weights, returns, annualize=True)
                current_vol = calculate_portfolio_volatility(current_weights, returns, annualize=True)
                current_sharpe = (current_ret - risk_free_rate) / current_vol if current_vol > 0 else 0

                perf_df = pd.DataFrame([
                    {"Metric": "Return", "Current": f"{current_ret*100:.1f}%", "Optimal": f"{result.expected_return*100:.1f}%"},
                    {"Metric": "Volatility", "Current": f"{current_vol*100:.1f}%", "Optimal": f"{result.volatility*100:.1f}%"},
                    {"Metric": "Sharpe", "Current": f"{current_sharpe:.2f}", "Optimal": f"{result.sharpe_ratio:.2f}"},
                ])
                st.dataframe(perf_df, use_container_width=True, hide_index=True)


def show_backtest(portfolio: Portfolio, prices: pd.DataFrame, risk_free_rate: float):
    """Backtest visualization."""
    st.subheader("Backtest")

    symbols = portfolio.get_symbols(include_cash=False)
    available = [s for s in symbols if s in prices.columns]

    if not available:
        st.warning("No price data available for backtesting.")
        return

    col1, col2 = st.columns(2)

    with col1:
        rebalance = st.selectbox(
            "Rebalance Frequency",
            ["Monthly", "Quarterly", "Yearly", "None"],
            index=0,
        )

    with col2:
        benchmark = st.text_input("Benchmark", value="SPY")

    if st.button("Run Backtest", type="primary", key="backtest_btn"):
        with st.spinner("Running backtest..."):
            rebal_map = {
                "Monthly": RebalanceFrequency.MONTHLY,
                "Quarterly": RebalanceFrequency.QUARTERLY,
                "Yearly": RebalanceFrequency.YEARLY,
                "None": RebalanceFrequency.NONE,
            }

            weights = portfolio.get_weights(include_cash=False)
            total = sum(weights.get(s, 0) for s in available)
            if total > 0:
                weights = {s: weights.get(s, 0) / total for s in available}

            # Fetch benchmark
            fetcher = MarketDataFetcher()
            bench_prices = fetcher.get_historical_prices([benchmark], period="3y")
            bench_series = bench_prices[benchmark] if benchmark in bench_prices.columns else None

            result = run_backtest(
                prices=prices[available],
                target_weights=weights,
                initial_value=portfolio.invested_value,
                rebalance_frequency=rebal_map[rebalance],
                benchmark_prices=bench_series,
                risk_free_rate=risk_free_rate,
            )

            # Performance chart
            fig = go.Figure()

            # Normalize to 100
            port_norm = result.portfolio_values / result.portfolio_values.iloc[0] * 100
            fig.add_trace(go.Scatter(
                x=port_norm.index,
                y=port_norm.values,
                name="Portfolio",
                line=dict(color="blue", width=2),
            ))

            if result.benchmark_values is not None:
                bench_norm = result.benchmark_values / result.benchmark_values.iloc[0] * 100
                fig.add_trace(go.Scatter(
                    x=bench_norm.index,
                    y=bench_norm.values,
                    name=benchmark,
                    line=dict(color="gray", width=1, dash="dash"),
                ))

            fig.update_layout(
                title="Performance (indexed to 100)",
                xaxis_title="Date",
                yaxis_title="Value",
                hovermode="x unified",
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            )

            st.plotly_chart(fig, use_container_width=True)

            # Metrics
            col1, col2, col3 = st.columns(3)

            with col1:
                color = "normal" if result.total_return >= 0 else "inverse"
                st.metric("Total Return", f"{result.total_return*100:+.1f}%")
                st.metric("Annualized", f"{result.annualized_return*100:+.1f}%")

            with col2:
                st.metric("Volatility", f"{result.volatility*100:.1f}%")
                st.metric("Sharpe Ratio", f"{result.sharpe_ratio:.2f}")

            with col3:
                st.metric("Max Drawdown", f"{result.max_drawdown*100:.1f}%")
                if result.alpha is not None:
                    st.metric("Alpha", f"{result.alpha*100:+.2f}%")


def show_simulation(portfolio: Portfolio, prices: pd.DataFrame, risk_free_rate: float):
    """Monte Carlo simulation."""
    st.subheader("Monte Carlo Simulation")

    symbols = portfolio.get_symbols(include_cash=False)
    available = [s for s in symbols if s in prices.columns]

    if not available:
        st.warning("No price data available for simulation.")
        return

    col1, col2 = st.columns(2)

    with col1:
        years = st.slider("Years to Simulate", 1, 5, 1)

    with col2:
        num_sims = st.selectbox("Simulations", [500, 1000, 5000], index=1)

    if st.button("Run Simulation", type="primary", key="sim_btn"):
        with st.spinner(f"Running {num_sims:,} simulations..."):
            returns = calculate_returns(prices[available])

            weights = portfolio.get_weights(include_cash=False)
            total = sum(weights.get(s, 0) for s in available)
            if total > 0:
                weights = {s: weights.get(s, 0) / total for s in available}

            result = run_monte_carlo(
                weights=weights,
                returns=returns,
                initial_value=portfolio.invested_value,
                days=years * 252,
                num_simulations=num_sims,
                seed=42,
            )

            # Fan chart
            fig = go.Figure()

            # Percentile bands
            fig.add_trace(go.Scatter(
                x=list(range(len(result.percentile_95))),
                y=result.percentile_95.values,
                mode="lines",
                line=dict(width=0),
                showlegend=False,
            ))
            fig.add_trace(go.Scatter(
                x=list(range(len(result.percentile_5))),
                y=result.percentile_5.values,
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(0, 100, 200, 0.2)",
                name="5th-95th percentile",
            ))

            fig.add_trace(go.Scatter(
                x=list(range(len(result.percentile_75))),
                y=result.percentile_75.values,
                mode="lines",
                line=dict(width=0),
                showlegend=False,
            ))
            fig.add_trace(go.Scatter(
                x=list(range(len(result.percentile_25))),
                y=result.percentile_25.values,
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(0, 100, 200, 0.3)",
                name="25th-75th percentile",
            ))

            fig.add_trace(go.Scatter(
                x=list(range(len(result.median_path))),
                y=result.median_path.values,
                mode="lines",
                line=dict(color="blue", width=2),
                name="Median",
            ))

            fig.update_layout(
                title=f"Projected Portfolio Value ({years} year{'s' if years > 1 else ''})",
                xaxis_title="Trading Days",
                yaxis_title="Value ($)",
                hovermode="x unified",
            )

            st.plotly_chart(fig, use_container_width=True)

            # Probability metrics
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Median Final Value", f"${result.final_median:,.0f}")
                st.metric("Mean Final Value", f"${result.final_mean:,.0f}")

            with col2:
                st.metric("P(Gain)", f"{result.prob_positive_return*100:.0f}%")
                st.metric("P(Lose 10%+)", f"{result.prob_loss_10pct*100:.0f}%")

            with col3:
                st.metric("VaR (95%)", f"${result.var_95:,.0f}")
                st.metric("Expected Shortfall", f"${result.cvar_95:,.0f}")


@st.cache_data(ttl=3600)
def get_cached_sectors(symbols: tuple[str, ...]) -> dict[str, str]:
    """Fetch sectors with caching (1 hour TTL)."""
    sector_map = get_sectors_for_portfolio(list(symbols), use_api=True)
    return {s: sector.value for s, sector in sector_map.items()}


def show_sectors(portfolio: Portfolio):
    """Display sector allocation."""
    st.subheader("Sector Allocation")

    symbols = portfolio.get_symbols(include_cash=False)
    if not symbols:
        st.warning("No equity positions found.")
        return

    with st.spinner("Fetching sector data..."):
        sector_values = get_cached_sectors(tuple(symbols))
    sector_map = {s: Sector(v) for s, v in sector_values.items()}
    weights = portfolio.get_weights(include_cash=False)

    sector_data: dict[str, float] = {}
    for symbol, weight in weights.items():
        sector = sector_map.get(symbol, Sector.UNKNOWN)
        sector_data[sector.value] = sector_data.get(sector.value, 0) + weight

    df = pd.DataFrame([
        {"Sector": sector, "Weight": weight}
        for sector, weight in sorted(sector_data.items(), key=lambda x: -x[1])
    ])

    col1, col2 = st.columns(2)

    with col1:
        fig = px.pie(df, values="Weight", names="Sector", hole=0.4)
        fig.update_traces(textposition="inside", textinfo="percent+label")
        fig.update_layout(showlegend=False, margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.dataframe(
            df.style.format({"Weight": "{:.1%}"}),
            use_container_width=True,
            hide_index=True,
        )

        st.write("**Positions by Sector:**")
        for sector in df["Sector"]:
            sector_symbols = [s for s in symbols if sector_map.get(s, Sector.UNKNOWN).value == sector]
            if sector_symbols:
                st.write(f"- **{sector}**: {', '.join(sector_symbols)}")


def show_rebalance(portfolio: Portfolio, prices: pd.DataFrame, risk_free_rate: float):
    """Show rebalancing trades."""
    st.subheader("Rebalancing Trades")

    symbols = portfolio.get_symbols(include_cash=False)
    available = [s for s in symbols if s in prices.columns]

    if len(available) < 2:
        st.warning("Need at least 2 assets for optimization.")
        return

    col1, col2 = st.columns(2)

    with col1:
        objective = st.selectbox(
            "Objective",
            ["Max Sharpe", "Min Volatility"],
            index=0,
            key="rebal_obj",
        )

    with col2:
        max_position_pct = st.slider(
            "Max Position Size (%)",
            min_value=1,
            max_value=100,
            value=30,
            step=1,
            key="rebal_max_pos",
        )
        max_position = max_position_pct / 100

    if st.button("Calculate Trades", type="primary", key="rebal_btn"):
        with st.spinner("Calculating optimal trades..."):
            returns = calculate_returns(prices[available])
            obj = Objective.MAX_SHARPE if "Sharpe" in objective else Objective.MIN_VOLATILITY

            optimizer = PortfolioOptimizer(
                returns,
                risk_free_rate,
                max_expected_return=0.25,
                shrinkage=0.3,
            )
            result = optimizer.optimize(objective=obj, max_weight=max_position)

            current_weights = portfolio.get_weights(include_cash=False)
            total = sum(current_weights.get(s, 0) for s in available)
            if total > 0:
                current_weights = {s: current_weights.get(s, 0) / total for s in available}

            total_value = portfolio.invested_value
            trades = []

            for symbol in available:
                current_val = current_weights.get(symbol, 0) * total_value
                target_val = result.weights.get(symbol, 0) * total_value
                trade_val = target_val - current_val

                if abs(trade_val) > 10:
                    trades.append({
                        "Symbol": symbol,
                        "Action": "BUY" if trade_val > 0 else "SELL",
                        "Current": current_val,
                        "Target": target_val,
                        "Trade": abs(trade_val),
                    })

            if not trades:
                st.success("Portfolio is already optimally allocated!")
                return

            trades_df = pd.DataFrame(trades).sort_values("Trade", ascending=False)

            st.write("**Trades to Execute:**")
            st.dataframe(
                trades_df.style.format({
                    "Current": "${:,.0f}",
                    "Target": "${:,.0f}",
                    "Trade": "${:,.0f}",
                }).map(
                    lambda x: "color: green" if x == "BUY" else "color: red",
                    subset=["Action"]
                ),
                use_container_width=True,
                hide_index=True,
            )

            total_buys = sum(t["Trade"] for t in trades if t["Action"] == "BUY")
            total_sells = sum(t["Trade"] for t in trades if t["Action"] == "SELL")

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Buys", f"${total_buys:,.0f}")
            with col2:
                st.metric("Total Sells", f"${total_sells:,.0f}")


def show_benchmark(portfolio: Portfolio, prices: pd.DataFrame, risk_free_rate: float):
    """Benchmark comparison analysis."""
    st.subheader("Benchmark Comparison")

    symbols = portfolio.get_symbols(include_cash=False)
    available = [s for s in symbols if s in prices.columns]

    if not available:
        st.warning("No price data available.")
        return

    col1, col2 = st.columns(2)

    with col1:
        benchmark = st.text_input("Benchmark Ticker", value="SPY", key="bench_ticker")

    with col2:
        period = st.selectbox("Period", ["1y", "2y", "3y"], index=0, key="bench_period")

    if st.button("Compare", type="primary", key="bench_btn"):
        with st.spinner("Analyzing..."):
            fetcher = MarketDataFetcher()
            bench_prices = fetcher.get_historical_prices([benchmark], period=period)

            if benchmark not in bench_prices.columns:
                st.error(f"Could not fetch data for {benchmark}")
                return

            returns = calculate_returns(prices[available])
            bench_returns = bench_prices[benchmark].pct_change().dropna()

            weights = portfolio.get_weights(include_cash=False)
            total = sum(weights.get(s, 0) for s in available)
            if total > 0:
                weights = {s: weights.get(s, 0) / total for s in available}

            portfolio_returns = sum(
                returns[s] * weights.get(s, 0)
                for s in available if s in returns.columns
            )

            analysis = BenchmarkAnalysis(
                portfolio_returns=portfolio_returns,
                benchmark_returns=bench_returns,
                benchmark_symbol=benchmark,
                risk_free_rate=risk_free_rate,
            )
            results = analysis.run()

            port_annual = portfolio_returns.mean() * 252
            bench_annual = bench_returns.mean() * 252
            port_vol = portfolio_returns.std() * (252 ** 0.5)
            bench_vol = bench_returns.std() * (252 ** 0.5)

            st.write("**Performance Comparison:**")
            perf_df = pd.DataFrame([
                {"Metric": "Annual Return", "Portfolio": f"{port_annual*100:.1f}%", benchmark: f"{bench_annual*100:.1f}%"},
                {"Metric": "Volatility", "Portfolio": f"{port_vol*100:.1f}%", benchmark: f"{bench_vol*100:.1f}%"},
            ])
            st.dataframe(perf_df, use_container_width=True, hide_index=True)

            st.write("**Risk Metrics:**")
            col1, col2, col3 = st.columns(3)

            with col1:
                beta = results["beta"]
                beta_note = "more volatile" if beta > 1.1 else "less volatile" if beta < 0.9 else "market-like"
                st.metric("Beta", f"{beta:.2f}", help=beta_note)

            with col2:
                alpha = results["alpha"]
                st.metric("Alpha", f"{alpha*100:+.2f}%", help="Risk-adjusted excess return")

            with col3:
                r_sq = results["r_squared"]
                st.metric("R-squared", f"{r_sq:.2f}", help=f"{r_sq*100:.0f}% explained by {benchmark}")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Tracking Error", f"{results['tracking_error']*100:.1f}%")

            with col2:
                st.metric("Information Ratio", f"{results['information_ratio']:.2f}")

            with col3:
                st.metric("Upside Capture", f"{results['upside_capture']:.0f}%")
                st.metric("Downside Capture", f"{results['downside_capture']:.0f}%")


def show_tax_harvest(portfolio: Portfolio):
    """Tax-loss harvesting analysis."""
    st.subheader("Tax-Loss Harvesting")

    col1, col2 = st.columns(2)

    with col1:
        tax_rate = st.slider("Tax Rate (%)", 10, 50, 25, key="tax_rate") / 100

    with col2:
        min_loss = st.number_input("Min Loss ($)", value=100, step=50, key="min_loss")

    if st.button("Analyze", type="primary", key="tax_btn"):
        analysis = analyze_tax_loss_harvesting(
            portfolio,
            tax_rate=tax_rate,
            min_loss_threshold=min_loss,
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Gains", f"${analysis.total_gains:,.0f}")

        with col2:
            st.metric("Harvestable Losses", f"${analysis.total_harvestable_loss:,.0f}")

        with col3:
            st.metric("Potential Tax Savings", f"${analysis.total_tax_savings:,.0f}")

        if not analysis.candidates:
            st.success("No tax-loss harvesting opportunities found.")
            return

        st.write("**Harvesting Candidates:**")

        data = []
        for candidate in analysis.candidates:
            pos = candidate.position
            alternatives = get_wash_sale_alternatives(pos.symbol)
            data.append({
                "Symbol": pos.symbol,
                "Value": pos.current_value,
                "Loss": candidate.unrealized_loss,
                "Loss %": candidate.loss_percent,
                "Tax Savings": candidate.tax_savings_estimate,
                "Alternatives": ", ".join(alternatives[:3]),
            })

        df = pd.DataFrame(data)
        st.dataframe(
            df.style.format({
                "Value": "${:,.0f}",
                "Loss": "${:,.0f}",
                "Loss %": "{:.1f}%",
                "Tax Savings": "${:,.0f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

        st.info("Wait 30 days before repurchasing the same security to avoid wash sale rules.")


def show_walkforward(portfolio: Portfolio, prices: pd.DataFrame, risk_free_rate: float):
    """Walk-forward optimization analysis."""
    st.subheader("Walk-Forward Optimization")

    symbols = portfolio.get_symbols(include_cash=False)
    available = [s for s in symbols if s in prices.columns]

    if len(available) < 2:
        st.warning("Need at least 2 assets for walk-forward analysis.")
        return

    col1, col2, col3 = st.columns(3)

    with col1:
        train_months = st.slider("Train Window (months)", 6, 24, 12, key="wf_train")

    with col2:
        test_months = st.slider("Test Window (months)", 1, 6, 3, key="wf_test")

    with col3:
        objective = st.selectbox(
            "Objective",
            ["Max Sharpe", "Min Volatility"],
            key="wf_obj",
        )

    if st.button("Run Walk-Forward", type="primary", key="wf_btn"):
        with st.spinner("Running walk-forward analysis (this may take a minute)..."):
            obj = Objective.MAX_SHARPE if "Sharpe" in objective else Objective.MIN_VOLATILITY

            try:
                result = run_walk_forward(
                    prices=prices[available],
                    train_months=train_months,
                    test_months=test_months,
                    objective=obj,
                    max_weight=0.3,
                    risk_free_rate=risk_free_rate,
                )
            except ValueError as e:
                st.error(str(e))
                return

            st.write(f"**Results ({len(result.windows)} windows):**")

            window_data = []
            for w in result.windows:
                window_data.append({
                    "Window": w.window_num,
                    "Test Period": f"{w.test_start.strftime('%Y-%m')} to {w.test_end.strftime('%Y-%m')}",
                    "Train Sharpe": w.train_sharpe,
                    "Test Sharpe": w.test_sharpe,
                    "Decay": w.sharpe_decay,
                })

            df = pd.DataFrame(window_data)
            st.dataframe(
                df.style.format({
                    "Train Sharpe": "{:.2f}",
                    "Test Sharpe": "{:.2f}",
                    "Decay": "{:+.2f}",
                }).map(
                    lambda x: "color: green" if x < 0.2 else "color: orange" if x < 0.5 else "color: red",
                    subset=["Decay"]
                ),
                use_container_width=True,
                hide_index=True,
            )

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Avg Train Sharpe", f"{result.avg_train_sharpe:.2f}")
                st.metric("Avg Test Sharpe", f"{result.avg_test_sharpe:.2f}")

            with col2:
                decay_color = "normal" if result.avg_sharpe_decay_pct < 30 else "inverse"
                st.metric("Sharpe Decay", f"{result.avg_sharpe_decay_pct:.0f}%")
                st.metric("OOS Return", f"{result.total_oos_return*100:+.1f}%")

            with col3:
                overfit_label = "Low" if result.overfitting_score < 0.3 else "Moderate" if result.overfitting_score < 0.5 else "High"
                st.metric("Overfitting Risk", overfit_label)
                st.metric("Consistency", f"{result.consistency_score*100:.0f}%")


# Template downloads
TEMPLATES = {
    "quantity": """symbol,quantity,cost_basis,account
AAPL,100,15000.00,Main
VOO,50,20000.00,Main
MSFT,25,8000.00,IRA
GOOGL,10,12000.00,Main""",
    "weight": """symbol,weight,account
VOO,40,Main
VXUS,20,Main
BND,20,Main
VNQ,10,Main
GLD,10,Main""",
    "value": """symbol,value,cost_basis,account
AAPL,25000,20000,Main
VOO,30000,28000,Main
MSFT,15000,12000,IRA""",
}

# Main app
st.sidebar.subheader("Download Template")
template_type = st.sidebar.selectbox(
    "Template type",
    ["quantity", "weight", "value"],
    format_func=lambda x: x.capitalize(),
    label_visibility="collapsed",
)
st.sidebar.download_button(
    label=f"Download {template_type} template",
    data=TEMPLATES[template_type],
    file_name=f"portfolio_template_{template_type}.csv",
    mime="text/csv",
)

st.sidebar.divider()

uploaded_file = st.sidebar.file_uploader(
    "Upload Portfolio CSV",
    type=["csv"],
    help="Upload a Fidelity export or generic CSV file",
)

if uploaded_file is not None:
    portfolio = parse_uploaded_file(uploaded_file)

    if portfolio is None:
        st.error("Could not parse the uploaded file. Please check the format.")
    else:
        st.sidebar.success(f"Loaded: {portfolio.name}")
        st.sidebar.write(f"**{len(portfolio.positions)}** positions")
        st.sidebar.write(f"**${portfolio.total_value:,.0f}** total value")

        # Fetch market data
        symbols = portfolio.get_symbols(include_cash=False)

        with st.spinner("Fetching market data..."):
            try:
                prices, risk_free_rate = fetch_market_data(symbols, period="3y")
            except Exception as e:
                st.error(f"Error fetching market data: {e}")
                prices, risk_free_rate = pd.DataFrame(), 0.045

        # Tabs for different views
        tabs = st.tabs([
            "📈 Summary",
            "📊 Holdings",
            "🏢 Sectors",
            "🎯 Optimize",
            "💱 Rebalance",
            "📉 Backtest",
            "🎲 Simulate",
            "📊 Benchmark",
            "💰 Tax Harvest",
            "🔄 Walk-Forward",
        ])

        with tabs[0]:  # Summary
            show_portfolio_summary(portfolio)
            col1, col2 = st.columns(2)
            with col1:
                show_allocation_chart(portfolio)
            with col2:
                if not prices.empty:
                    show_risk_metrics(portfolio, prices, risk_free_rate)

        with tabs[1]:  # Holdings
            show_holdings_table(portfolio)

        with tabs[2]:  # Sectors
            show_sectors(portfolio)

        with tabs[3]:  # Optimize
            if not prices.empty:
                show_optimization(portfolio, prices, risk_free_rate)
            else:
                st.warning("No price data available.")

        with tabs[4]:  # Rebalance
            if not prices.empty:
                show_rebalance(portfolio, prices, risk_free_rate)
            else:
                st.warning("No price data available.")

        with tabs[5]:  # Backtest
            if not prices.empty:
                show_backtest(portfolio, prices, risk_free_rate)
            else:
                st.warning("No price data available.")

        with tabs[6]:  # Simulate
            if not prices.empty:
                show_simulation(portfolio, prices, risk_free_rate)
            else:
                st.warning("No price data available.")

        with tabs[7]:  # Benchmark
            if not prices.empty:
                show_benchmark(portfolio, prices, risk_free_rate)
            else:
                st.warning("No price data available.")

        with tabs[8]:  # Tax Harvest
            show_tax_harvest(portfolio)

        with tabs[9]:  # Walk-Forward
            if not prices.empty:
                show_walkforward(portfolio, prices, risk_free_rate)
            else:
                st.warning("No price data available.")

else:
    st.info("👈 Upload a portfolio CSV file to get started")

    st.markdown("""
    ### Supported Formats

    **Fidelity CSV** - Direct export from Fidelity portfolio view

    **Generic CSV** - Any CSV with these columns:
    - `symbol` (required)
    - `quantity`, `weight`, or `value` (one required)
    - `cost_basis`, `current_price`, `account` (optional)

    ### Features
    - 📈 Portfolio summary and allocation charts
    - 📊 Risk/return metrics (Sharpe, Sortino, VaR)
    - 🏢 Sector allocation breakdown
    - 🎯 Mean-variance optimization
    - 💱 Rebalancing trade recommendations
    - 📉 Historical backtesting
    - 🎲 Monte Carlo simulation
    - 📊 Benchmark comparison (alpha, beta, tracking error)
    - 💰 Tax-loss harvesting analysis
    - 🔄 Walk-forward strategy validation
    """)
