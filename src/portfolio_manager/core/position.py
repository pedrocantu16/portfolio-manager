"""Position dataclass representing a single holding in a portfolio."""

from dataclasses import dataclass
from enum import StrEnum


class AssetType(StrEnum):
    """Type of asset in a position."""

    EQUITY = "equity"
    ETF = "etf"
    CASH = "cash"
    CRYPTO = "crypto"


@dataclass
class Position:
    """Represents a single position/holding in a portfolio."""

    symbol: str
    quantity: float
    current_price: float
    current_value: float
    cost_basis: float
    avg_cost_basis: float
    gain_loss_dollar: float
    gain_loss_percent: float
    percent_of_account: float
    asset_type: AssetType
    description: str = ""

    @property
    def is_cash(self) -> bool:
        """Check if this position is a cash/money market position."""
        return self.asset_type == AssetType.CASH

    @property
    def is_etf(self) -> bool:
        """Check if this position is an ETF."""
        return self.asset_type == AssetType.ETF

    @property
    def is_equity(self) -> bool:
        """Check if this position is an equity/stock."""
        return self.asset_type == AssetType.EQUITY

    def __repr__(self) -> str:
        return (
            f"Position(symbol={self.symbol!r}, quantity={self.quantity}, "
            f"value=${self.current_value:,.2f}, type={self.asset_type.value})"
        )
