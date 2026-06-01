"""Sector classification for portfolio positions."""

from enum import StrEnum

import yfinance as yf


class Sector(StrEnum):
    """GICS sectors."""

    TECHNOLOGY = "Technology"
    HEALTHCARE = "Healthcare"
    FINANCIALS = "Financials"
    CONSUMER_DISCRETIONARY = "Consumer Discretionary"
    CONSUMER_STAPLES = "Consumer Staples"
    INDUSTRIALS = "Industrials"
    ENERGY = "Energy"
    UTILITIES = "Utilities"
    MATERIALS = "Materials"
    REAL_ESTATE = "Real Estate"
    COMMUNICATION = "Communication Services"
    UNKNOWN = "Unknown"


# Static mapping for common symbols (faster than API calls)
SECTOR_MAP: dict[str, Sector] = {
    # Technology
    "AAPL": Sector.TECHNOLOGY,
    "MSFT": Sector.TECHNOLOGY,
    "GOOGL": Sector.TECHNOLOGY,
    "GOOG": Sector.TECHNOLOGY,
    "NVDA": Sector.TECHNOLOGY,
    "AMD": Sector.TECHNOLOGY,
    "INTC": Sector.TECHNOLOGY,
    "CRM": Sector.TECHNOLOGY,
    "ORCL": Sector.TECHNOLOGY,
    "ADBE": Sector.TECHNOLOGY,
    "CSCO": Sector.TECHNOLOGY,
    "IBM": Sector.TECHNOLOGY,
    "NOW": Sector.TECHNOLOGY,
    "CRWD": Sector.TECHNOLOGY,
    "ARM": Sector.TECHNOLOGY,
    # Consumer Discretionary
    "AMZN": Sector.CONSUMER_DISCRETIONARY,
    "TSLA": Sector.CONSUMER_DISCRETIONARY,
    "HD": Sector.CONSUMER_DISCRETIONARY,
    "MCD": Sector.CONSUMER_DISCRETIONARY,
    "NKE": Sector.CONSUMER_DISCRETIONARY,
    "SBUX": Sector.CONSUMER_DISCRETIONARY,
    "TGT": Sector.CONSUMER_DISCRETIONARY,
    "LOW": Sector.CONSUMER_DISCRETIONARY,
    "EXPE": Sector.CONSUMER_DISCRETIONARY,
    "TRIP": Sector.CONSUMER_DISCRETIONARY,
    # Communication Services
    "META": Sector.COMMUNICATION,
    "NFLX": Sector.COMMUNICATION,
    "DIS": Sector.COMMUNICATION,
    "CMCSA": Sector.COMMUNICATION,
    "T": Sector.COMMUNICATION,
    "VZ": Sector.COMMUNICATION,
    # Financials
    "JPM": Sector.FINANCIALS,
    "BAC": Sector.FINANCIALS,
    "WFC": Sector.FINANCIALS,
    "GS": Sector.FINANCIALS,
    "MS": Sector.FINANCIALS,
    "V": Sector.FINANCIALS,
    "MA": Sector.FINANCIALS,
    "PYPL": Sector.FINANCIALS,
    "COIN": Sector.FINANCIALS,
    # Healthcare
    "JNJ": Sector.HEALTHCARE,
    "UNH": Sector.HEALTHCARE,
    "PFE": Sector.HEALTHCARE,
    "ABBV": Sector.HEALTHCARE,
    "MRK": Sector.HEALTHCARE,
    "LLY": Sector.HEALTHCARE,
    # Consumer Staples
    "PG": Sector.CONSUMER_STAPLES,
    "KO": Sector.CONSUMER_STAPLES,
    "PEP": Sector.CONSUMER_STAPLES,
    "WMT": Sector.CONSUMER_STAPLES,
    "COST": Sector.CONSUMER_STAPLES,
    # Energy
    "XOM": Sector.ENERGY,
    "CVX": Sector.ENERGY,
    "COP": Sector.ENERGY,
    "SLB": Sector.ENERGY,
    "EOG": Sector.ENERGY,
    # Industrials
    "CAT": Sector.INDUSTRIALS,
    "BA": Sector.INDUSTRIALS,
    "HON": Sector.INDUSTRIALS,
    "UPS": Sector.INDUSTRIALS,
    "RTX": Sector.INDUSTRIALS,
    "DE": Sector.INDUSTRIALS,
    "GE": Sector.INDUSTRIALS,
    "AAL": Sector.INDUSTRIALS,
    "DAL": Sector.INDUSTRIALS,
    "UAL": Sector.INDUSTRIALS,
    "MRTN": Sector.INDUSTRIALS,
    # Utilities
    "NEE": Sector.UTILITIES,
    "DUK": Sector.UTILITIES,
    "SO": Sector.UTILITIES,
    "EVGO": Sector.UTILITIES,
    # Real Estate
    "AMT": Sector.REAL_ESTATE,
    "PLD": Sector.REAL_ESTATE,
    "CCI": Sector.REAL_ESTATE,
    # Materials
    "LIN": Sector.MATERIALS,
    "APD": Sector.MATERIALS,
    "ECL": Sector.MATERIALS,
}

# ETF sector mappings (broad market ETFs are diversified)
ETF_SECTOR_MAP: dict[str, Sector | None] = {
    "SPY": None,  # Diversified
    "VOO": None,
    "IVV": None,
    "VTI": None,
    "QQQ": Sector.TECHNOLOGY,  # Tech-heavy
    "VXUS": None,  # International diversified
    "XLK": Sector.TECHNOLOGY,
    "XLF": Sector.FINANCIALS,
    "XLE": Sector.ENERGY,
    "XLV": Sector.HEALTHCARE,
    "XLI": Sector.INDUSTRIALS,
    "XLY": Sector.CONSUMER_DISCRETIONARY,
    "XLP": Sector.CONSUMER_STAPLES,
    "XLU": Sector.UTILITIES,
    "XLRE": Sector.REAL_ESTATE,
    "XLC": Sector.COMMUNICATION,
    "IBIT": Sector.FINANCIALS,  # Bitcoin ETF
    "GBTC": Sector.FINANCIALS,
}


def get_sector(symbol: str, use_api: bool = False) -> Sector:
    """Get sector for a symbol.

    Args:
        symbol: Stock ticker symbol.
        use_api: If True, fetch from Yahoo Finance when not in static map.

    Returns:
        Sector enum value.
    """
    # Check static map first
    if symbol in SECTOR_MAP:
        return SECTOR_MAP[symbol]

    if symbol in ETF_SECTOR_MAP:
        sector = ETF_SECTOR_MAP[symbol]
        return sector if sector else Sector.UNKNOWN

    # Try Yahoo Finance API
    if use_api:
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            sector_str = info.get("sector", "")

            # Map Yahoo Finance sector names to our enum
            sector_mapping = {
                "Technology": Sector.TECHNOLOGY,
                "Healthcare": Sector.HEALTHCARE,
                "Financial Services": Sector.FINANCIALS,
                "Consumer Cyclical": Sector.CONSUMER_DISCRETIONARY,
                "Consumer Defensive": Sector.CONSUMER_STAPLES,
                "Industrials": Sector.INDUSTRIALS,
                "Energy": Sector.ENERGY,
                "Utilities": Sector.UTILITIES,
                "Basic Materials": Sector.MATERIALS,
                "Real Estate": Sector.REAL_ESTATE,
                "Communication Services": Sector.COMMUNICATION,
            }
            return sector_mapping.get(sector_str, Sector.UNKNOWN)
        except Exception:
            pass

    return Sector.UNKNOWN


def get_sectors_for_portfolio(
    symbols: list[str], use_api: bool = False
) -> dict[str, Sector]:
    """Get sectors for all symbols in a portfolio.

    Args:
        symbols: List of ticker symbols.
        use_api: If True, fetch from Yahoo Finance for unknown symbols.

    Returns:
        Dictionary mapping symbol to sector.
    """
    return {symbol: get_sector(symbol, use_api) for symbol in symbols}


def calculate_sector_weights(
    weights: dict[str, float], sectors: dict[str, Sector]
) -> dict[Sector, float]:
    """Calculate weight allocated to each sector.

    Args:
        weights: Dictionary mapping symbol to portfolio weight.
        sectors: Dictionary mapping symbol to sector.

    Returns:
        Dictionary mapping sector to total weight.
    """
    sector_weights: dict[Sector, float] = {}

    for symbol, weight in weights.items():
        sector = sectors.get(symbol, Sector.UNKNOWN)
        sector_weights[sector] = sector_weights.get(sector, 0.0) + weight

    return sector_weights
