"""Tests for Fidelity CSV parser."""

import tempfile
from pathlib import Path

import pytest

from portfolio_manager.core.position import AssetType
from portfolio_manager.data.parsers.fidelity import FidelityParser


SAMPLE_CSV = """\
Account Number,Account Name,Symbol,Description,Quantity,Last Price,Last Price Change,Current Value,Today's Gain/Loss Dollar,Today's Gain/Loss Percent,Total Gain/Loss Dollar,Total Gain/Loss Percent,Percent Of Account,Cost Basis Total,Average Cost Basis,Type
Z27202803,Individual - TOD,SPAXX**,HELD IN MONEY MARKET,,,,$55788.53,,,,,52.10%,,,Cash,
Z27202803,Individual - TOD,AMZN,AMAZON.COM INC,157,$270.64,-$3.36,$42490.48,-$527.52,-1.23%,+$9249.00,+27.82%,39.68%,$33241.48,$211.73,Cash,
Z27202803,Individual - TOD,IVV,ISHARES CORE S&P 500 ETF,2.327,$760.05,+$1.88,$1768.63,+$4.37,+0.24%,+$250.87,+16.52%,1.65%,$1517.76,$652.24,Cash,
Z27202803,Individual - TOD,Pending activity,,,,,$2.28,,,,,,
"""


@pytest.fixture
def sample_csv_file():
    """Create a temporary CSV file with sample data."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(SAMPLE_CSV)
        f.flush()
        yield Path(f.name)


def test_can_parse_valid_file(sample_csv_file):
    """Test that parser recognizes valid Fidelity CSV."""
    parser = FidelityParser()
    assert parser.can_parse(sample_csv_file) is True


def test_can_parse_invalid_file():
    """Test that parser rejects non-existent files."""
    parser = FidelityParser()
    assert parser.can_parse(Path("/nonexistent/file.csv")) is False


def test_parse_creates_portfolio(sample_csv_file):
    """Test that parsing creates a valid portfolio."""
    parser = FidelityParser()
    portfolio = parser.parse(sample_csv_file)

    assert portfolio is not None
    assert portfolio.account_number == "Z27202803"
    assert portfolio.name == "Individual - TOD"


def test_parse_cash_position(sample_csv_file):
    """Test that money market is parsed as cash."""
    parser = FidelityParser()
    portfolio = parser.parse(sample_csv_file)

    cash_pos = portfolio.get_position("SPAXX")
    assert cash_pos is not None
    assert cash_pos.asset_type == AssetType.CASH
    assert cash_pos.current_value == pytest.approx(55788.53)


def test_parse_equity_position(sample_csv_file):
    """Test that equities are parsed correctly."""
    parser = FidelityParser()
    portfolio = parser.parse(sample_csv_file)

    amzn = portfolio.get_position("AMZN")
    assert amzn is not None
    assert amzn.asset_type == AssetType.EQUITY
    assert amzn.quantity == 157
    assert amzn.current_price == pytest.approx(270.64)
    assert amzn.current_value == pytest.approx(42490.48)
    assert amzn.cost_basis == pytest.approx(33241.48)
    assert amzn.gain_loss_dollar == pytest.approx(9249.00)
    assert amzn.gain_loss_percent == pytest.approx(27.82)


def test_parse_etf_position(sample_csv_file):
    """Test that ETFs are classified correctly."""
    parser = FidelityParser()
    portfolio = parser.parse(sample_csv_file)

    ivv = portfolio.get_position("IVV")
    assert ivv is not None
    assert ivv.asset_type == AssetType.ETF
    assert ivv.quantity == pytest.approx(2.327)


def test_parse_skips_pending_activity(sample_csv_file):
    """Test that pending activity rows are skipped."""
    parser = FidelityParser()
    portfolio = parser.parse(sample_csv_file)

    # Should have 3 positions (SPAXX, AMZN, IVV), not 4
    assert len(portfolio.positions) == 3
    assert "Pending activity" not in portfolio.positions


def test_portfolio_totals(sample_csv_file):
    """Test portfolio aggregate calculations."""
    parser = FidelityParser()
    portfolio = parser.parse(sample_csv_file)

    expected_total = 55788.53 + 42490.48 + 1768.63
    assert portfolio.total_value == pytest.approx(expected_total)

    expected_cash = 55788.53
    assert portfolio.cash == pytest.approx(expected_cash)

    expected_invested = 42490.48 + 1768.63
    assert portfolio.invested_value == pytest.approx(expected_invested)
