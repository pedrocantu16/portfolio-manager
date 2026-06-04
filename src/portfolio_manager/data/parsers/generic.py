"""Generic CSV parser for user-provided portfolio data."""

import csv
from pathlib import Path

from portfolio_manager.core.portfolio import Portfolio
from portfolio_manager.core.position import AssetType, Position
from portfolio_manager.data.parsers.base import BaseParser


# Common ETF symbols for classification
ETF_SYMBOLS = {
    "SPY", "VOO", "IVV", "VTI", "QQQ", "VEA", "VWO", "VXUS", "BND", "AGG",
    "VNQ", "GLD", "SLV", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP",
    "XLU", "XLRE", "XLC", "VIG", "VYM", "SCHD", "ARKK", "IWM", "EFA", "EEM",
    "TLT", "LQD", "HYG", "VGT", "IBIT", "GBTC", "VUG", "VTV", "IJH", "IJR",
}


def _classify_asset(symbol: str) -> AssetType:
    """Classify asset type based on symbol."""
    symbol = symbol.upper()
    if symbol in ETF_SYMBOLS:
        return AssetType.ETF
    if symbol.endswith("XX"):  # Money market funds like SPAXX
        return AssetType.CASH
    return AssetType.EQUITY


class GenericParser(BaseParser):
    """Parser for generic CSV portfolio format.

    Expected columns:
        - symbol (required): Ticker symbol
        - quantity OR weight/percent OR value (one required)
        - cost_basis (optional): Total cost basis in dollars
        - current_price (optional): Current price per share
        - account (optional): Account name

    If using weight/percent, quantities are calculated from prices.
    If current_price is not provided, it will be fetched from Yahoo Finance.
    """

    REQUIRED_COLUMNS = {"symbol"}
    QUANTITY_COLUMNS = {"quantity", "shares", "qty"}
    WEIGHT_COLUMNS = {"weight", "percent", "pct", "%", "allocation"}
    OPTIONAL_COLUMNS = {"cost_basis", "current_price", "account", "name", "description", "value"}

    def can_parse(self, file_path: Path) -> bool:
        """Check if file is a valid generic CSV format."""
        if not file_path.suffix.lower() == ".csv":
            return False

        try:
            with open(file_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames is None:
                    return False

                # Normalize column names
                columns = {col.lower().strip() for col in reader.fieldnames}

                # Must have symbol column
                if not self.REQUIRED_COLUMNS.issubset(columns):
                    return False

                # Must have either quantity, weight, or value column
                has_quantity = bool(columns & self.QUANTITY_COLUMNS)
                has_weight = bool(columns & self.WEIGHT_COLUMNS)
                has_value = "value" in columns

                return has_quantity or has_weight or has_value
        except Exception:
            return False

    def _parse_number(self, value: str | None) -> float | None:
        """Parse a number from string, handling currency and percent symbols."""
        if not value:
            return None
        cleaned = value.replace(",", "").replace("$", "").replace("%", "").strip()
        try:
            return float(cleaned)
        except ValueError:
            return None

    def _get_column(self, row: dict, candidates: set[str]) -> str | None:
        """Get value from first matching column."""
        for col in candidates:
            if col in row and row[col]:
                return row[col]
        return None

    def parse(self, file_path: Path, total_value: float = 100000.0) -> Portfolio:
        """Parse generic CSV file into Portfolio.

        Args:
            file_path: Path to CSV file.
            total_value: Total portfolio value (used when weights are provided).

        Returns:
            Portfolio object with positions.
        """
        positions: dict[str, Position] = {}
        account_name = "Portfolio"
        raw_data: list[dict] = []

        with open(file_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            # Normalize column names in header
            if reader.fieldnames:
                fieldname_map = {col: col.lower().strip() for col in reader.fieldnames}
            else:
                fieldname_map = {}

            # Determine which columns are available
            columns = set(fieldname_map.values())
            has_quantity = bool(columns & self.QUANTITY_COLUMNS)
            has_weight = bool(columns & self.WEIGHT_COLUMNS)
            has_value = "value" in columns

            for row in reader:
                # Normalize keys
                row = {fieldname_map.get(k, k.lower().strip()): v for k, v in row.items()}

                symbol = (row.get("symbol") or "").strip().upper()
                if not symbol:
                    continue

                # Get quantity, weight, or value
                quantity = None
                weight = None
                value = None

                if has_quantity:
                    qty_str = self._get_column(row, self.QUANTITY_COLUMNS)
                    quantity = self._parse_number(qty_str)

                if has_weight:
                    wt_str = self._get_column(row, self.WEIGHT_COLUMNS)
                    weight = self._parse_number(wt_str)
                    # Convert from percentage if > 1 (e.g., 50 -> 0.50)
                    if weight is not None and weight > 1:
                        weight = weight / 100

                if has_value:
                    value = self._parse_number(row.get("value", ""))

                # Must have at least one
                if quantity is None and weight is None and value is None:
                    continue

                if quantity is not None and quantity <= 0:
                    continue

                # Parse other fields
                cost_basis = self._parse_number(row.get("cost_basis", "")) or 0.0
                current_price = self._parse_number(row.get("current_price", "")) or 0.0

                # Account name
                acct = (row.get("account") or "").strip()
                if acct and account_name == "Portfolio":
                    account_name = acct

                # Description
                description = (row.get("name") or row.get("description") or symbol).strip()

                raw_data.append({
                    "symbol": symbol,
                    "quantity": quantity,
                    "weight": weight,
                    "value": value,
                    "cost_basis": cost_basis,
                    "current_price": current_price,
                    "description": description,
                })

        # Fetch prices for symbols that need them
        symbols_needing_prices = list({
            d["symbol"] for d in raw_data
            if d["current_price"] == 0
        })

        price_map: dict[str, float] = {}
        if symbols_needing_prices:
            from portfolio_manager.data.market import MarketDataFetcher
            fetcher = MarketDataFetcher()
            try:
                prices = fetcher.get_historical_prices(symbols_needing_prices, period="5d")
                if not prices.empty:
                    for symbol in symbols_needing_prices:
                        if symbol in prices.columns:
                            price_map[symbol] = prices[symbol].dropna().iloc[-1]
            except Exception:
                pass

        # Process each row and create positions
        for data in raw_data:
            symbol = data["symbol"]
            quantity = data["quantity"]
            weight = data["weight"]
            value = data["value"]
            cost_basis = data["cost_basis"]
            current_price = data["current_price"] or price_map.get(symbol, 0.0)
            description = data["description"]

            # Calculate quantity from weight or value if needed
            if quantity is None:
                if value is not None and current_price > 0:
                    quantity = value / current_price
                elif weight is not None and current_price > 0:
                    position_value = weight * total_value
                    quantity = position_value / current_price
                elif value is not None:
                    # Use value directly if no price available
                    quantity = 1.0
                    current_price = value
                else:
                    continue

            if quantity <= 0:
                continue

            # Calculate current value
            current_value = quantity * current_price if current_price > 0 else (value or 0.0)

            # Calculate gain/loss
            avg_cost = cost_basis / quantity if quantity > 0 and cost_basis > 0 else 0.0
            gain_loss = current_value - cost_basis if cost_basis > 0 else 0.0
            gain_loss_pct = (gain_loss / cost_basis * 100) if cost_basis > 0 else 0.0

            position = Position(
                symbol=symbol,
                description=description,
                quantity=quantity,
                current_price=current_price,
                current_value=current_value,
                cost_basis=cost_basis,
                avg_cost_basis=avg_cost,
                gain_loss_dollar=gain_loss,
                gain_loss_percent=gain_loss_pct,
                percent_of_account=0.0,  # Calculated later
                asset_type=_classify_asset(symbol),
            )

            # Aggregate if same symbol appears multiple times
            if symbol in positions:
                existing = positions[symbol]
                new_qty = existing.quantity + quantity
                new_cost = existing.cost_basis + cost_basis
                new_value = existing.current_value + current_value
                positions[symbol] = Position(
                    symbol=symbol,
                    description=description,
                    quantity=new_qty,
                    current_price=current_price or existing.current_price,
                    current_value=new_value,
                    cost_basis=new_cost,
                    avg_cost_basis=new_cost / new_qty if new_qty > 0 else 0.0,
                    gain_loss_dollar=new_value - new_cost if new_cost > 0 else 0.0,
                    gain_loss_percent=((new_value - new_cost) / new_cost * 100) if new_cost > 0 else 0.0,
                    percent_of_account=0.0,
                    asset_type=_classify_asset(symbol),
                )
            else:
                positions[symbol] = position

        # Calculate total value and percentages
        total_val = sum(p.current_value for p in positions.values())

        if total_val > 0:
            for symbol, pos in positions.items():
                positions[symbol] = Position(
                    symbol=pos.symbol,
                    description=pos.description,
                    quantity=pos.quantity,
                    current_price=pos.current_price,
                    current_value=pos.current_value,
                    cost_basis=pos.cost_basis,
                    avg_cost_basis=pos.avg_cost_basis,
                    gain_loss_dollar=pos.gain_loss_dollar,
                    gain_loss_percent=pos.gain_loss_percent,
                    percent_of_account=pos.current_value / total_val * 100,
                    asset_type=pos.asset_type,
                )

        return Portfolio(name=account_name, positions=positions)
