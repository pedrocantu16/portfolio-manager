"""Portfolio class representing a collection of positions."""

from dataclasses import dataclass, field

from portfolio_manager.core.position import AssetType, Position

# Known ETF symbols for classification
ETF_SYMBOLS = {
    "IVV", "VOO", "SPY", "VTI", "QQQ", "VXUS", "VEA", "VWO", "BND", "AGG",
    "IBIT", "GBTC", "ETHE", "ARKK", "ARKW", "XLF", "XLE", "XLK", "XLV",
    "GLD", "SLV", "USO", "VNQ", "SCHD", "VYM", "JEPI", "JEPQ",
}


@dataclass
class Portfolio:
    """A collection of positions with aggregation and analysis methods."""

    positions: dict[str, Position] = field(default_factory=dict)
    name: str = "default"
    account_number: str = ""

    @property
    def cash(self) -> float:
        """Total cash value (money market positions)."""
        return sum(
            pos.current_value
            for pos in self.positions.values()
            if pos.is_cash
        )

    @property
    def invested_value(self) -> float:
        """Total value of non-cash positions."""
        return sum(
            pos.current_value
            for pos in self.positions.values()
            if not pos.is_cash
        )

    @property
    def total_value(self) -> float:
        """Total portfolio value including cash."""
        return sum(pos.current_value for pos in self.positions.values())

    @property
    def total_cost_basis(self) -> float:
        """Total cost basis of all positions."""
        return sum(pos.cost_basis for pos in self.positions.values())

    @property
    def total_gain_loss(self) -> float:
        """Total unrealized gain/loss."""
        return sum(pos.gain_loss_dollar for pos in self.positions.values())

    @property
    def total_gain_loss_percent(self) -> float:
        """Total gain/loss as a percentage of cost basis."""
        if self.total_cost_basis == 0:
            return 0.0
        return (self.total_gain_loss / self.total_cost_basis) * 100

    def add_position(self, position: Position) -> None:
        """Add a position to the portfolio."""
        self.positions[position.symbol] = position

    def get_position(self, symbol: str) -> Position | None:
        """Get a position by symbol."""
        return self.positions.get(symbol)

    def get_weights(self, include_cash: bool = False) -> dict[str, float]:
        """Get position weights as a fraction of portfolio value.

        Args:
            include_cash: If True, include cash positions in weights.

        Returns:
            Dictionary mapping symbol to weight (0.0 to 1.0).
        """
        if include_cash:
            total = self.total_value
            positions = self.positions.values()
        else:
            total = self.invested_value
            positions = [p for p in self.positions.values() if not p.is_cash]

        if total == 0:
            return {}

        return {pos.symbol: pos.current_value / total for pos in positions}

    def get_equity_positions(self) -> list[Position]:
        """Get all equity (non-ETF, non-cash) positions."""
        return [
            pos for pos in self.positions.values()
            if pos.asset_type == AssetType.EQUITY
        ]

    def get_etf_positions(self) -> list[Position]:
        """Get all ETF positions."""
        return [
            pos for pos in self.positions.values()
            if pos.asset_type == AssetType.ETF
        ]

    def get_investable_positions(self) -> list[Position]:
        """Get all non-cash positions (equities and ETFs)."""
        return [
            pos for pos in self.positions.values()
            if not pos.is_cash
        ]

    def get_symbols(self, include_cash: bool = False) -> list[str]:
        """Get list of symbols in the portfolio."""
        if include_cash:
            return list(self.positions.keys())
        return [s for s, p in self.positions.items() if not p.is_cash]

    def __len__(self) -> int:
        return len(self.positions)

    def __repr__(self) -> str:
        return (
            f"Portfolio(name={self.name!r}, positions={len(self.positions)}, "
            f"value=${self.total_value:,.2f})"
        )
