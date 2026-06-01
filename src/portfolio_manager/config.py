"""Configuration settings for portfolio manager."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with defaults."""

    # Risk-free rate (annualized)
    risk_free_rate: float = 0.05

    # Trading days per year
    trading_days: int = 252

    # Default analysis period
    default_period: str = "1y"

    # VaR confidence level
    var_confidence: float = 0.95

    # Cache settings
    cache_enabled: bool = True

    class Config:
        env_prefix = "PORTFOLIO_"


# Global settings instance
settings = Settings()
