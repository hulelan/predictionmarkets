"""Manifold Markets API client."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import httpx

from src.data.normalizer import NormalizedMarket, Platform
from src.utils.logger import log

SEARCH_URL = "https://api.manifold.markets/v0/search-markets"


async def fetch_manifold_markets(
    close_date_min: datetime | None = None,
    close_date_max: datetime | None = None,
    max_markets: int = 500,
    rate_limit_delay: float = 0.3,
) -> list[dict]:
    """Fetch binary Manifold markets sorted by volume, filtered to date range."""
    all_markets: list[dict] = []
    offset = 0
    batch_size = 100
    # Fetch more than needed since we filter by date locally
    max_scan = max_markets * 10

    async with httpx.AsyncClient(timeout=30) as client:
        while len(all_markets) < max_markets and offset < max_scan:
            params = {
                "term": "",
                "sort": "24-hour-vol",
                "filter": "open",
                "contractType": "BINARY",
                "limit": batch_size,
                "offset": offset,
            }

            try:
                resp = await client.get(SEARCH_URL, params=params)
                resp.raise_for_status()
                batch = resp.json()
            except httpx.HTTPError as e:
                log.error(f"Manifold fetch error at offset {offset}: {e}")
                break

            if not batch:
                break

            for m in batch:
                close_time_ms = m.get("closeTime")
                if close_time_ms is None:
                    continue

                # Guard against absurd timestamps (year > 9999)
                if close_time_ms > 253402300000000:
                    continue

                close_dt = datetime.fromtimestamp(close_time_ms / 1000, tz=timezone.utc)

                # Skip markets outside our date window
                if close_date_min and close_dt < close_date_min:
                    continue
                if close_date_max and close_dt > close_date_max:
                    continue

                all_markets.append(m)

            log.info(f"Manifold: {len(all_markets)} markets in window after scanning {offset + batch_size} (sorted by 24h volume)")
            offset += batch_size
            await asyncio.sleep(rate_limit_delay)

    return all_markets[:max_markets]


def normalize_manifold(raw: dict) -> NormalizedMarket | None:
    """Convert raw Manifold API response to NormalizedMarket."""
    try:
        close_time_ms = raw.get("closeTime")
        if close_time_ms is None:
            return None

        close_dt = datetime.fromtimestamp(close_time_ms / 1000, tz=timezone.utc)
        probability = raw.get("probability")
        if probability is None:
            return None

        market_url = raw.get("url", "")
        if not market_url and raw.get("slug"):
            market_url = f"https://manifold.markets/{raw.get('creatorUsername', '')}/{raw['slug']}"

        return NormalizedMarket(
            id=raw.get("id", ""),
            platform=Platform.MANIFOLD,
            question=raw.get("question", ""),
            description=raw.get("textDescription", "")[:2000],
            market_probability=float(probability),
            volume=float(raw.get("volume", 0)),
            liquidity=float(raw.get("totalLiquidity", 0)),
            resolution_date=close_dt,
            category=_infer_manifold_category(raw),
            url=market_url,
            raw_data=raw,
        )
    except (ValueError, KeyError) as e:
        log.warning(f"Failed to normalize Manifold market {raw.get('id')}: {e}")
        return None


def _infer_manifold_category(raw: dict) -> str:
    """Infer category from Manifold market data."""
    question = raw.get("question", "").lower()
    groups = " ".join(raw.get("groupSlugs", []) if raw.get("groupSlugs") else []).lower()
    text = question + " " + groups

    if any(w in text for w in ["ai", "gpt", "claude", "llm", "agi", "machine-learning"]):
        return "science"
    if any(w in text for w in ["election", "president", "congress", "trump", "biden", "politics"]):
        return "politics"
    if any(w in text for w in ["crypto", "bitcoin", "ethereum"]):
        return "crypto"
    if any(w in text for w in ["nba", "nfl", "nhl", "sports", "game", "match"]):
        return "sports"
    if any(w in text for w in ["stock", "s&p", "market", "economic"]):
        return "finance"

    return "other"
