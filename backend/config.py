"""Configuration settings for the BTC 5-min trading bot."""
import os
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Database (SQLite for Phase 1, PostgreSQL for production)
    DATABASE_URL: str = "sqlite:///./tradingbot.db"

    # API Keys (optional)
    POLYMARKET_API_KEY: Optional[str] = None

    # Kalshi API
    KALSHI_API_KEY_ID: Optional[str] = None
    KALSHI_PRIVATE_KEY_PATH: Optional[str] = None
    KALSHI_ENABLED: bool = True

    # AI API Keys
    GROQ_API_KEY: Optional[str] = None

    # AI Model Configuration
    GROQ_MODEL: str = "llama-3.1-8b-instant"

    # AI Feature Flags
    AI_LOG_ALL_CALLS: bool = True
    AI_DAILY_BUDGET_USD: float = 1.0

    # Bot settings - BTC 5-MIN TRADING
    SIMULATION_MODE: bool = True
    INITIAL_BANKROLL: float = 10000.0
    KELLY_FRACTION: float = 0.15  # Fractional Kelly

    # BTC 5-min specific settings
    SCAN_INTERVAL_SECONDS: int = 60  # Scan every minute
    SETTLEMENT_INTERVAL_SECONDS: int = 120  # Check settlements every 2 min
    BTC_PRICE_SOURCE: str = "coinbase"
    MIN_EDGE_THRESHOLD: float = 0.02  # 2% edge required — these are 50/50 markets
    MAX_ENTRY_PRICE: float = 0.55  # Enter up to 55c
    MAX_TRADES_PER_WINDOW: int = 1
    MAX_TOTAL_PENDING_TRADES: int = 20

    # Multi-timeframe BTC Up/Down trading (comma lists, "tf:value")
    BTC_TIMEFRAMES: str = "5m,15m,1h,1d"
    BTC_TF_MIN_EDGE: str = "5m:0.02,15m:0.04,1h:0.05,1d:0.05"
    BTC_TF_MAX_ENTRY: str = "5m:0.80,15m:0.80,1h:0.80,1d:0.80"
    BTC_TF_MIN_TIME: str = "5m:60,15m:120,1h:300,1d:1800"
    BTC_TF_MAX_TIME: str = "5m:1800,15m:1800,1h:7200,1d:129600"
    MAX_TRADES_PER_SCAN: int = 4
    # Rules model ("rules-v4": end price vs start price on each settlement feed). 5m can fall back to the old indicator model with "micro".
    BTC_5M_MODEL: str = "rules"
    # Predict-only by default: record one prediction per window (Polymarket order-book probability,
    # which beat the price model in a backtest) and score it after settlement; no paper trades.
    # Set true to paper-trade model-vs-market edges again (the backtest says those lose on average).
    BTC_PAPER_TRADING: bool = False
    # Only trade once this fraction of the window has elapsed (most of the outcome is locked in)
    BTC_TF_MIN_ELAPSED: str = "5m:0.6,15m:0.75,1h:0.75,1d:0.5"
    BTC_MAX_SPREAD: float = 0.05        # skip windows whose Up book is wider than this

    # Survival mode (paper strategies on Polymarket US books, BRTI settlement)
    SURVIVAL_ENABLED: bool = True
    SURVIVAL_INTERVAL_SECONDS: int = 20
    # Keep testing these past the 10% drawdown pause (comma-separated; daily stop still applies)
    SURVIVAL_NO_PAUSE: str = "model_gap"
    # No new trades for these; history stays on the scoreboard as RETIRED (comma-separated)
    SURVIVAL_RETIRED: str = "underdog_late,late_gap"

    # Real money on Polymarket US - OFF unless every one of these is set. Orders are only placed
    # for strategies listed in US_LIVE_STRATEGIES that the survival scoreboard marks PROVEN.
    US_LIVE_TRADING: bool = False
    POLYMARKET_US_KEY_ID: str = ""
    POLYMARKET_US_SECRET_KEY: str = ""
    US_LIVE_STRATEGIES: str = ""        # comma-separated, e.g. "favorite_late"
    US_LIVE_MAX_STAKE: float = 5.0      # dollars per trade
    US_LIVE_DAILY_LOSS: float = 20.0    # stop for the day after losing this much
    US_LIVE_MAX_OPEN: int = 3           # open live positions at once

    # Risk management
    DAILY_LOSS_LIMIT: float = 300.0
    MAX_TRADE_SIZE: float = 75.0
    MIN_TIME_REMAINING: int = 60  # Don't trade windows closing in < 60s
    MAX_TIME_REMAINING: int = 1800  # Trade windows up to 30min out

    # Indicator weights for composite signal (must sum to ~1.0)
    WEIGHT_RSI: float = 0.20
    WEIGHT_MOMENTUM: float = 0.35
    WEIGHT_VWAP: float = 0.20
    WEIGHT_SMA: float = 0.15
    WEIGHT_MARKET_SKEW: float = 0.10

    # Volume filter
    MIN_MARKET_VOLUME: float = 100.0  # Low volume for 5-min markets

    # Weather trading settings
    WEATHER_ENABLED: bool = True
    WEATHER_SCAN_INTERVAL_SECONDS: int = 300  # 5 min
    WEATHER_SETTLEMENT_INTERVAL_SECONDS: int = 1800  # 30 min
    WEATHER_MIN_EDGE_THRESHOLD: float = 0.08  # 8% — weather has more signal than 5-min BTC
    WEATHER_MAX_ENTRY_PRICE: float = 0.70
    WEATHER_MAX_TRADE_SIZE: float = 100.0
    WEATHER_CITIES: str = "nyc,chicago,miami,los_angeles,denver"

    class Config:
        env_file = ".env"


settings = Settings()


def tf_setting(name: str, timeframe: str, default: float) -> float:
    """Read a per-timeframe value from a "5m:0.02,1h:0.05" style setting."""
    raw = getattr(settings, name, "") or ""
    for part in raw.split(","):
        k, _, v = part.partition(":")
        if k.strip() == timeframe:
            try:
                return float(v)
            except ValueError:
                break
    return default
