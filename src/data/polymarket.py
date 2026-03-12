"""Polymarket Gamma API client."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone

import httpx

from src.data.normalizer import NormalizedMarket, Platform
from src.utils.logger import log

BASE_URL = "https://gamma-api.polymarket.com/markets"


async def fetch_polymarket_markets(
    end_date_min: str = "2026-03-08",
    end_date_max: str = "2026-03-15",
    max_markets: int = 800,
    rate_limit_delay: float = 0.5,
) -> list[dict]:
    """Fetch all active Polymarket markets in the given date range."""
    markets: list[dict] = []
    offset = 0
    batch_size = 100

    # Polymarket API returns 422 when min == max; bump max by 1 day
    if end_date_min == end_date_max:
        from datetime import datetime as _dt, timedelta
        end_date_max = (_dt.fromisoformat(end_date_max) + timedelta(days=1)).strftime("%Y-%m-%d")

    async with httpx.AsyncClient(timeout=30) as client:
        while len(markets) < max_markets:
            params = {
                "active": "true",
                "closed": "false",
                "limit": batch_size,
                "offset": offset,
                "end_date_min": end_date_min,
                "end_date_max": end_date_max,
            }
            try:
                resp = await client.get(BASE_URL, params=params)
                resp.raise_for_status()
                batch = resp.json()
            except httpx.HTTPError as e:
                log.error(f"Polymarket fetch error at offset {offset}: {e}")
                break

            if not batch:
                break

            markets.extend(batch)
            log.info(f"Polymarket: fetched {len(markets)} markets (offset={offset})")
            offset += batch_size
            await asyncio.sleep(rate_limit_delay)

    return markets[:max_markets]


def normalize_polymarket(raw: dict) -> NormalizedMarket | None:
    """Convert raw Polymarket API response to NormalizedMarket."""
    try:
        outcome_prices = json.loads(raw.get("outcomePrices", "[]"))
        if not outcome_prices:
            return None

        yes_price = float(outcome_prices[0])

        end_date_str = raw.get("endDate", "")
        if not end_date_str:
            return None

        resolution_date = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))

        slug = raw.get("slug", raw.get("id", ""))
        url = f"https://polymarket.com/event/{slug}" if slug else ""

        return NormalizedMarket(
            id=str(raw.get("id", "")),
            platform=Platform.POLYMARKET,
            question=raw.get("question", ""),
            description=raw.get("description", "")[:2000],
            market_probability=yes_price,
            volume=float(raw.get("volumeNum", 0)),
            liquidity=float(raw.get("liquidityNum", 0)),
            resolution_date=resolution_date,
            category=_infer_category(raw),
            url=url,
            raw_data=raw,
        )
    except (ValueError, KeyError, IndexError) as e:
        log.warning(f"Failed to normalize Polymarket market {raw.get('id')}: {e}")
        return None


def _infer_category(raw: dict) -> str:
    """Simple keyword-based category inference from question text."""
    question = (raw.get("question", "") + " " + raw.get("groupItemTitle", "")).lower()
    tags = " ".join(raw.get("tags", []) if raw.get("tags") else []).lower()
    text = question + " " + tags

    categories = {
        "sports": ["nhl", "nba", "nfl", "mlb", "soccer", "cricket", "tennis", "game", "match", "score", "win", "championship"],
        "politics": ["election", "president", "senate", "congress", "vote", "party", "democrat", "republican", "governor", "prime minister"],
        "crypto": ["bitcoin", "ethereum", "crypto", "btc", "eth", "token", "blockchain"],
        "finance": ["stock", "s&p", "nasdaq", "gdp", "inflation", "fed", "interest rate", "market cap"],
        "entertainment": ["oscar", "grammy", "movie", "album", "billboard", "streaming", "box office"],
        "weather": ["temperature", "tornado", "hurricane", "weather", "climate", "snow", "rainfall"],
        "science": ["ai", "spacex", "nasa", "fda", "vaccine", "study", "research"],
    }

    for cat, keywords in categories.items():
        if any(kw in text for kw in keywords):
            return cat

    return "other"
