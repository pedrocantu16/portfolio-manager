"""Base parser interface for brokerage CSV exports."""

from abc import ABC, abstractmethod
from pathlib import Path

from portfolio_manager.core.portfolio import Portfolio


class BaseParser(ABC):
    """Abstract base class for brokerage CSV parsers."""

    @abstractmethod
    def parse(self, file_path: Path | str) -> Portfolio:
        """Parse a CSV file and return a Portfolio object.

        Args:
            file_path: Path to the CSV file.

        Returns:
            Portfolio object with positions loaded.
        """
        ...

    @abstractmethod
    def can_parse(self, file_path: Path | str) -> bool:
        """Check if this parser can handle the given file.

        Args:
            file_path: Path to the CSV file.

        Returns:
            True if this parser can handle the file format.
        """
        ...
