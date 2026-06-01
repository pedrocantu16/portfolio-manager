"""Transaction cost modeling for rebalancing."""

from dataclasses import dataclass


@dataclass
class TradeCost:
    """Cost breakdown for a single trade."""

    symbol: str
    trade_value: float
    commission: float
    spread_cost: float
    total_cost: float

    @property
    def cost_percent(self) -> float:
        """Cost as percentage of trade value."""
        if self.trade_value == 0:
            return 0.0
        return (self.total_cost / abs(self.trade_value)) * 100


@dataclass
class RebalanceCosts:
    """Total costs for rebalancing a portfolio."""

    trades: list[TradeCost]
    total_commission: float
    total_spread_cost: float
    total_cost: float
    total_trade_value: float

    @property
    def cost_percent(self) -> float:
        """Total cost as percentage of trade value."""
        if self.total_trade_value == 0:
            return 0.0
        return (self.total_cost / self.total_trade_value) * 100


class CostModel:
    """Model for estimating transaction costs."""

    def __init__(
        self,
        commission_per_trade: float = 0.0,
        commission_per_share: float = 0.0,
        spread_percent: float = 0.05,
        min_spread: float = 0.01,
    ):
        """Initialize cost model.

        Args:
            commission_per_trade: Fixed commission per trade (default $0 for most brokers).
            commission_per_share: Commission per share (default $0).
            spread_percent: Estimated bid-ask spread as percent (default 0.05%).
            min_spread: Minimum spread cost in dollars (default $0.01).
        """
        self.commission_per_trade = commission_per_trade
        self.commission_per_share = commission_per_share
        self.spread_percent = spread_percent / 100  # Convert to decimal
        self.min_spread = min_spread

    def estimate_trade_cost(
        self,
        symbol: str,
        trade_value: float,
        shares: float = 0,
    ) -> TradeCost:
        """Estimate cost for a single trade.

        Args:
            symbol: Ticker symbol.
            trade_value: Dollar value of trade (positive for buy, negative for sell).
            shares: Number of shares (for per-share commission).

        Returns:
            TradeCost breakdown.
        """
        abs_value = abs(trade_value)

        # Commission
        commission = self.commission_per_trade
        if shares > 0:
            commission += self.commission_per_share * shares

        # Spread cost (half the spread, since we cross it)
        spread_cost = max(abs_value * self.spread_percent / 2, self.min_spread)

        total = commission + spread_cost

        return TradeCost(
            symbol=symbol,
            trade_value=trade_value,
            commission=commission,
            spread_cost=spread_cost,
            total_cost=total,
        )

    def estimate_rebalance_costs(
        self,
        trades: dict[str, float],
        shares: dict[str, float] | None = None,
    ) -> RebalanceCosts:
        """Estimate total costs for rebalancing.

        Args:
            trades: Dictionary mapping symbol to trade value (positive=buy, negative=sell).
            shares: Optional dictionary mapping symbol to number of shares.

        Returns:
            RebalanceCosts with total breakdown.
        """
        if shares is None:
            shares = {}

        trade_costs = []
        for symbol, value in trades.items():
            if abs(value) < 1:  # Skip tiny trades
                continue
            cost = self.estimate_trade_cost(symbol, value, shares.get(symbol, 0))
            trade_costs.append(cost)

        total_commission = sum(t.commission for t in trade_costs)
        total_spread = sum(t.spread_cost for t in trade_costs)
        total_cost = sum(t.total_cost for t in trade_costs)
        total_value = sum(abs(t.trade_value) for t in trade_costs)

        return RebalanceCosts(
            trades=trade_costs,
            total_commission=total_commission,
            total_spread_cost=total_spread,
            total_cost=total_cost,
            total_trade_value=total_value,
        )


# Default cost model (commission-free broker with typical spreads)
DEFAULT_COST_MODEL = CostModel(
    commission_per_trade=0.0,
    commission_per_share=0.0,
    spread_percent=0.05,  # 5 basis points
)
