"""Tax-loss harvesting analysis."""

from dataclasses import dataclass
from datetime import date, timedelta

from portfolio_manager.core.portfolio import Portfolio
from portfolio_manager.core.position import Position


@dataclass
class HarvestCandidate:
    """A position that's a candidate for tax-loss harvesting."""

    position: Position
    unrealized_loss: float
    loss_percent: float
    tax_savings_estimate: float
    wash_sale_warning: bool = False

    def __repr__(self) -> str:
        return (
            f"HarvestCandidate({self.position.symbol}, "
            f"loss=${self.unrealized_loss:,.2f}, "
            f"tax_savings=${self.tax_savings_estimate:,.2f})"
        )


@dataclass
class HarvestingAnalysis:
    """Results of tax-loss harvesting analysis."""

    candidates: list[HarvestCandidate]
    total_harvestable_loss: float
    total_tax_savings: float
    total_gains: float
    net_gain_loss: float

    @property
    def can_offset_gains(self) -> bool:
        """Whether losses can offset existing gains."""
        return self.total_gains > 0 and self.total_harvestable_loss > 0


def analyze_tax_loss_harvesting(
    portfolio: Portfolio,
    tax_rate: float = 0.25,
    min_loss_threshold: float = 100.0,
    min_loss_percent: float = 5.0,
) -> HarvestingAnalysis:
    """Analyze portfolio for tax-loss harvesting opportunities.

    Args:
        portfolio: Portfolio to analyze.
        tax_rate: Combined federal + state tax rate (default 25%).
        min_loss_threshold: Minimum dollar loss to consider (default $100).
        min_loss_percent: Minimum loss percentage to consider (default 5%).

    Returns:
        HarvestingAnalysis with candidates and totals.
    """
    candidates = []
    total_gains = 0.0
    total_losses = 0.0

    for position in portfolio.get_investable_positions():
        gain_loss = position.gain_loss_dollar

        if gain_loss >= 0:
            total_gains += gain_loss
        else:
            # This is a loss
            loss = abs(gain_loss)
            loss_pct = abs(position.gain_loss_percent)

            # Check thresholds
            if loss >= min_loss_threshold and loss_pct >= min_loss_percent:
                tax_savings = loss * tax_rate

                candidate = HarvestCandidate(
                    position=position,
                    unrealized_loss=loss,
                    loss_percent=loss_pct,
                    tax_savings_estimate=tax_savings,
                    wash_sale_warning=False,
                )
                candidates.append(candidate)
                total_losses += loss

    # Sort by tax savings (largest first)
    candidates.sort(key=lambda c: c.tax_savings_estimate, reverse=True)

    return HarvestingAnalysis(
        candidates=candidates,
        total_harvestable_loss=total_losses,
        total_tax_savings=total_losses * tax_rate,
        total_gains=total_gains,
        net_gain_loss=total_gains - total_losses,
    )


def get_wash_sale_alternatives(
    symbol: str,
    sector: str | None = None,
) -> list[str]:
    """Suggest alternative investments to avoid wash sale rule.

    The wash sale rule prevents claiming a loss if you buy a
    "substantially identical" security within 30 days.

    Args:
        symbol: Symbol being sold.
        sector: Sector of the symbol (for finding alternatives).

    Returns:
        List of alternative symbols that are similar but not identical.
    """
    # Common alternatives (similar exposure, different security)
    alternatives: dict[str, list[str]] = {
        # S&P 500 alternatives
        "SPY": ["VOO", "IVV", "SPLG"],
        "VOO": ["SPY", "IVV", "SPLG"],
        "IVV": ["SPY", "VOO", "SPLG"],
        # Total market alternatives
        "VTI": ["ITOT", "SPTM", "SCHB"],
        "ITOT": ["VTI", "SPTM", "SCHB"],
        # International alternatives
        "VXUS": ["IXUS", "VEU", "ACWX"],
        "VEA": ["IEFA", "SCHF", "EFA"],
        "VWO": ["IEMG", "EEM", "SCHE"],
        # Bond alternatives
        "BND": ["AGG", "SCHZ", "IUSB"],
        "AGG": ["BND", "SCHZ", "IUSB"],
        # Tech ETF alternatives
        "QQQ": ["VGT", "XLK", "QQQM"],
        "VGT": ["QQQ", "XLK", "FTEC"],
        # Individual stocks - suggest sector ETFs
        "AAPL": ["XLK", "VGT", "QQQ"],
        "MSFT": ["XLK", "VGT", "QQQ"],
        "GOOGL": ["XLC", "VOX", "FCOM"],
        "AMZN": ["XLY", "VCR", "FDIS"],
        "TSLA": ["XLY", "VCR", "CARZ"],
    }

    if symbol in alternatives:
        return alternatives[symbol]

    # Default: suggest broad market if no specific alternative
    return ["VTI", "VOO", "SPY"]


@dataclass
class WashSaleCheck:
    """Result of wash sale rule check."""

    symbol: str
    is_at_risk: bool
    days_until_safe: int
    safe_date: date
    message: str


def check_wash_sale_risk(
    symbol: str,
    last_purchase_date: date | None = None,
    last_sale_date: date | None = None,
) -> WashSaleCheck:
    """Check if a trade would trigger wash sale rule.

    The wash sale rule applies if you sell at a loss and buy the same
    or substantially identical security within 30 days before or after.

    Args:
        symbol: Symbol being traded.
        last_purchase_date: Date of most recent purchase (if any).
        last_sale_date: Date of loss sale (if selling).

    Returns:
        WashSaleCheck with risk assessment.
    """
    today = date.today()
    wash_sale_window = timedelta(days=30)

    if last_purchase_date:
        days_since_purchase = (today - last_purchase_date).days
        if days_since_purchase < 30:
            safe_date = last_purchase_date + wash_sale_window
            return WashSaleCheck(
                symbol=symbol,
                is_at_risk=True,
                days_until_safe=30 - days_since_purchase,
                safe_date=safe_date,
                message=f"Wash sale risk: purchased {days_since_purchase} days ago",
            )

    if last_sale_date:
        days_since_sale = (today - last_sale_date).days
        if days_since_sale < 30:
            safe_date = last_sale_date + wash_sale_window
            return WashSaleCheck(
                symbol=symbol,
                is_at_risk=True,
                days_until_safe=30 - days_since_sale,
                safe_date=safe_date,
                message=f"Wash sale risk: sold at loss {days_since_sale} days ago",
            )

    return WashSaleCheck(
        symbol=symbol,
        is_at_risk=False,
        days_until_safe=0,
        safe_date=today,
        message="No wash sale risk",
    )
