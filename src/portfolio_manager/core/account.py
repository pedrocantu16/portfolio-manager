"""Account class for multi-account support."""

from dataclasses import dataclass, field

from portfolio_manager.core.portfolio import Portfolio
from portfolio_manager.core.position import Position


@dataclass
class Account:
    """Represents a brokerage account containing one or more portfolios."""

    name: str
    account_number: str = ""
    portfolios: dict[str, Portfolio] = field(default_factory=dict)

    @property
    def total_value(self) -> float:
        """Total value across all portfolios."""
        return sum(p.total_value for p in self.portfolios.values())

    @property
    def total_cash(self) -> float:
        """Total cash across all portfolios."""
        return sum(p.cash for p in self.portfolios.values())

    @property
    def total_invested(self) -> float:
        """Total invested value across all portfolios."""
        return sum(p.invested_value for p in self.portfolios.values())

    def add_portfolio(self, portfolio: Portfolio) -> None:
        """Add a portfolio to this account."""
        self.portfolios[portfolio.name] = portfolio

    def get_portfolio(self, name: str) -> Portfolio | None:
        """Get a portfolio by name."""
        return self.portfolios.get(name)

    def get_consolidated_portfolio(self) -> Portfolio:
        """Create a consolidated portfolio combining all positions.

        Positions with the same symbol are aggregated.
        """
        consolidated = Portfolio(name=f"{self.name} (Consolidated)")

        for portfolio in self.portfolios.values():
            for symbol, position in portfolio.positions.items():
                if symbol in consolidated.positions:
                    # Aggregate position
                    existing = consolidated.positions[symbol]
                    new_quantity = existing.quantity + position.quantity
                    new_value = existing.current_value + position.current_value
                    new_cost = existing.cost_basis + position.cost_basis
                    new_gain = existing.gain_loss_dollar + position.gain_loss_dollar

                    consolidated.positions[symbol] = Position(
                        symbol=symbol,
                        quantity=new_quantity,
                        current_price=position.current_price,
                        current_value=new_value,
                        cost_basis=new_cost,
                        avg_cost_basis=new_cost / new_quantity if new_quantity > 0 else 0,
                        gain_loss_dollar=new_gain,
                        gain_loss_percent=(new_gain / new_cost * 100) if new_cost > 0 else 0,
                        percent_of_account=0,  # Will be recalculated
                        asset_type=position.asset_type,
                        description=position.description,
                    )
                else:
                    consolidated.add_position(position)

        # Recalculate percent_of_account
        total = consolidated.total_value
        if total > 0:
            for symbol, pos in consolidated.positions.items():
                consolidated.positions[symbol] = Position(
                    symbol=pos.symbol,
                    quantity=pos.quantity,
                    current_price=pos.current_price,
                    current_value=pos.current_value,
                    cost_basis=pos.cost_basis,
                    avg_cost_basis=pos.avg_cost_basis,
                    gain_loss_dollar=pos.gain_loss_dollar,
                    gain_loss_percent=pos.gain_loss_percent,
                    percent_of_account=(pos.current_value / total) * 100,
                    asset_type=pos.asset_type,
                    description=pos.description,
                )

        return consolidated

    def __len__(self) -> int:
        return len(self.portfolios)

    def __repr__(self) -> str:
        return (
            f"Account(name={self.name!r}, portfolios={len(self.portfolios)}, "
            f"value=${self.total_value:,.2f})"
        )
