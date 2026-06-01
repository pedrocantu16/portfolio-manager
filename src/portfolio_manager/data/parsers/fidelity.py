"""Parser for Fidelity brokerage CSV exports."""

import csv
from pathlib import Path

from portfolio_manager.core.portfolio import ETF_SYMBOLS, Portfolio
from portfolio_manager.core.position import AssetType, Position
from portfolio_manager.data.parsers.base import BaseParser

# Money market symbols treated as cash
MONEY_MARKET_SYMBOLS = {"SPAXX", "FDRXX", "FZFXX", "SPRXX", "FTEXX"}


def _parse_money(value: str | None) -> float:
    """Parse a money string like '$1,234.56' or '-$1,234.56' to float."""
    if value is None or value == "":
        return 0.0
    # Remove $, commas, and handle negatives
    cleaned = value.replace("$", "").replace(",", "").replace("+", "")
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


def _parse_percent(value: str | None) -> float:
    """Parse a percentage string like '12.34%' or '-12.34%' to float."""
    if value is None or value == "":
        return 0.0
    cleaned = value.replace("%", "").replace("+", "")
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


def _classify_asset(symbol: str, description: str) -> AssetType:
    """Classify an asset based on symbol and description."""
    # Clean symbol (remove ** suffix for money market)
    clean_symbol = symbol.rstrip("*")

    if clean_symbol in MONEY_MARKET_SYMBOLS:
        return AssetType.CASH

    if clean_symbol in ETF_SYMBOLS:
        return AssetType.ETF

    # Check description for ETF keywords
    desc_upper = description.upper()
    if any(kw in desc_upper for kw in ["ETF", "INDEX FUND", "TRUST ETF"]):
        return AssetType.ETF

    # Check for crypto-related
    if any(kw in desc_upper for kw in ["BITCOIN", "ETHEREUM", "CRYPTO"]):
        return AssetType.CRYPTO

    return AssetType.EQUITY


class FidelityParser(BaseParser):
    """Parser for Fidelity portfolio CSV exports."""

    def can_parse(self, file_path: Path | str) -> bool:
        """Check if file looks like a Fidelity export."""
        path = Path(file_path)
        if not path.exists() or path.suffix.lower() != ".csv":
            return False

        try:
            with open(path, encoding="utf-8-sig") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if header is None:
                    return False
                # Check for Fidelity-specific columns
                return "Account Number" in header and "Symbol" in header
        except Exception:
            return False

    def parse(self, file_path: Path | str) -> Portfolio:
        """Parse a Fidelity CSV export into a Portfolio."""
        path = Path(file_path)

        portfolio = Portfolio()

        with open(path, encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)

            for row in reader:
                symbol = row.get("Symbol") or ""
                symbol = symbol.strip()

                # Skip empty rows or pending activity
                if not symbol or symbol.lower() == "pending activity":
                    continue

                # Clean symbol (remove ** for money market display)
                clean_symbol = symbol.rstrip("*")

                # Get account info (from first valid row)
                if not portfolio.account_number:
                    portfolio.account_number = row.get("Account Number") or ""
                    portfolio.name = row.get("Account Name") or "default"

                description = row.get("Description") or ""
                asset_type = _classify_asset(symbol, description)

                # Parse values - handle money market specially
                if asset_type == AssetType.CASH:
                    # Money market: value is in Current Value, no quantity/price
                    current_value = _parse_money(row.get("Current Value", "0"))
                    position = Position(
                        symbol=clean_symbol,
                        quantity=current_value,  # Use value as quantity for cash
                        current_price=1.0,
                        current_value=current_value,
                        cost_basis=current_value,
                        avg_cost_basis=1.0,
                        gain_loss_dollar=0.0,
                        gain_loss_percent=0.0,
                        percent_of_account=_parse_percent(row.get("Percent Of Account", "0")),
                        asset_type=asset_type,
                        description=description,
                    )
                else:
                    # Regular position
                    quantity_str = row.get("Quantity", "0")
                    try:
                        quantity = float(quantity_str) if quantity_str else 0.0
                    except ValueError:
                        quantity = 0.0

                    position = Position(
                        symbol=clean_symbol,
                        quantity=quantity,
                        current_price=_parse_money(row.get("Last Price", "0")),
                        current_value=_parse_money(row.get("Current Value", "0")),
                        cost_basis=_parse_money(row.get("Cost Basis Total", "0")),
                        avg_cost_basis=_parse_money(row.get("Average Cost Basis", "0")),
                        gain_loss_dollar=_parse_money(row.get("Total Gain/Loss Dollar", "0")),
                        gain_loss_percent=_parse_percent(row.get("Total Gain/Loss Percent", "0")),
                        percent_of_account=_parse_percent(row.get("Percent Of Account", "0")),
                        asset_type=asset_type,
                        description=description,
                    )

                portfolio.add_position(position)

        return portfolio
