"""Common data schema for normalized prediction markets."""

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


class Platform(str, Enum):
    POLYMARKET = "polymarket"
    MANIFOLD = "manifold"
    KALSHI = "kalshi"


class NormalizedMarket(BaseModel):
    """Unified market representation across all platforms."""

    id: str
    platform: Platform
    question: str
    description: str = ""
    market_probability: float  # current consensus probability (0-1)
    volume: float  # trading volume
    liquidity: float = 0.0
    resolution_date: datetime
    category: str = "unknown"
    url: str = ""
    raw_data: dict = Field(default_factory=dict, repr=False)
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_row(self) -> dict:
        """Convert to flat dict for CSV/DataFrame export (no raw_data)."""
        return {
            "id": self.id,
            "platform": self.platform.value,
            "question": self.question,
            "market_probability": self.market_probability,
            "volume": self.volume,
            "liquidity": self.liquidity,
            "resolution_date": self.resolution_date.isoformat(),
            "category": self.category,
            "url": self.url,
            "fetched_at": self.fetched_at.isoformat(),
        }
