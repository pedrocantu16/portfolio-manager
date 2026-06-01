"""Market data fetcher using yfinance."""


import pandas as pd
import yfinance as yf


class MarketDataFetcher:
    """Fetches market data from Yahoo Finance."""

    def __init__(self, cache_enabled: bool = True):
        """Initialize the market data fetcher.

        Args:
            cache_enabled: Whether to cache fetched data in memory.
        """
        self.cache_enabled = cache_enabled
        self._price_cache: dict[str, pd.DataFrame] = {}

    def get_historical_prices(
        self,
        symbols: list[str],
        period: str = "1y",
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Fetch historical adjusted close prices for multiple symbols.

        Args:
            symbols: List of ticker symbols.
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max).
            interval: Data interval (1d, 1wk, 1mo).

        Returns:
            DataFrame with dates as index and symbols as columns.
        """
        if not symbols:
            return pd.DataFrame()

        # Create cache key
        cache_key = f"{','.join(sorted(symbols))}_{period}_{interval}"
        if self.cache_enabled and cache_key in self._price_cache:
            return self._price_cache[cache_key]

        # Fetch data
        try:
            data = yf.download(
                tickers=symbols,
                period=period,
                interval=interval,
                auto_adjust=True,
                progress=False,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to fetch market data: {e}") from e

        # Extract close prices
        if len(symbols) == 1:
            # Single symbol returns Series, need to convert to DataFrame
            if isinstance(data, pd.DataFrame) and "Close" in data.columns:
                prices = data[["Close"]].copy()
                prices.columns = symbols
            else:
                prices = pd.DataFrame({symbols[0]: data["Close"]})
        else:
            # Multiple symbols returns MultiIndex DataFrame
            if isinstance(data.columns, pd.MultiIndex):
                prices = data["Close"].copy()
            else:
                prices = data[["Close"]].copy()
                prices.columns = symbols

        # Drop rows with all NaN
        prices = prices.dropna(how="all")

        if self.cache_enabled:
            self._price_cache[cache_key] = prices

        return prices

    def get_risk_free_rate(self) -> float:
        """Fetch the current risk-free rate (3-month T-bill rate).

        Returns:
            Annualized risk-free rate as a decimal (e.g., 0.05 for 5%).
        """
        try:
            # ^IRX is the 13-week T-bill rate
            tbill = yf.Ticker("^IRX")
            hist = tbill.history(period="5d")

            if hist.empty:
                # Fallback to a reasonable default
                return 0.05

            # ^IRX is quoted as a percentage (e.g., 5.0 for 5%)
            rate = hist["Close"].iloc[-1] / 100
            return float(rate)
        except Exception:
            # Fallback if fetch fails
            return 0.05

    def get_current_prices(self, symbols: list[str]) -> dict[str, float]:
        """Fetch current prices for symbols.

        Args:
            symbols: List of ticker symbols.

        Returns:
            Dictionary mapping symbol to current price.
        """
        prices = {}
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.fast_info
                prices[symbol] = info.get("lastPrice", info.get("regularMarketPrice", 0.0))
            except Exception:
                prices[symbol] = 0.0
        return prices

    def clear_cache(self) -> None:
        """Clear the price cache."""
        self._price_cache.clear()
