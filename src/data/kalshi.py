"""Kalshi API client."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone

import httpx

from src.data.normalizer import NormalizedMarket, Platform
from src.utils.logger import log

BASE_URL = "https://api.elections.kalshi.com/trade-api/v2/markets"


async def fetch_kalshi_markets(
    expiration_min: datetime | None = None,
    expiration_max: datetime | None = None,
    max_markets: int = 5000,
    min_volume: int = 0,
    rate_limit_delay: float = 0.3,
) -> list[dict]:
    """Fetch open Kalshi markets (excluding MVE parlays) with volume, filtered to date range."""
    all_markets: list[dict] = []
    cursor = ""

    async with httpx.AsyncClient(timeout=30) as client:
        while len(all_markets) < max_markets:
            params = {
                "status": "open",
                "limit": 1000,
                "mve_filter": "exclude",
            }
            if cursor:
                params["cursor"] = cursor

            try:
                resp = await client.get(BASE_URL, params=params)
                resp.raise_for_status()
                # Handle invalid control characters in Kalshi responses
                raw_text = resp.text.replace("\r", "").replace("\x00", "")
                data = json.loads(raw_text)
            except (httpx.HTTPError, json.JSONDecodeError) as e:
                log.error(f"Kalshi fetch error: {e}")
                break

            markets = data.get("markets", [])
            cursor = data.get("cursor", "")

            if not markets:
                break

            for m in markets:
                vol = m.get("volume", 0)
                if vol < min_volume:
                    continue

                # Use expected_expiration_time (actual resolution) over close_time
                exp_str = m.get("expected_expiration_time") or m.get("close_time", "")
                if not exp_str:
                    continue

                try:
                    exp_dt = datetime.fromisoformat(exp_str.replace("Z", "+00:00"))
                except ValueError:
                    continue

                if expiration_min and exp_dt < expiration_min:
                    continue
                if expiration_max and exp_dt > expiration_max:
                    continue

                all_markets.append(m)

            log.info(f"Kalshi: {len(all_markets)} markets in window after scanning {len(markets)} (page cursor={'...' if cursor else 'none'})")

            if not cursor:
                break

            await asyncio.sleep(rate_limit_delay)

    return all_markets[:max_markets]


def normalize_kalshi(raw: dict) -> NormalizedMarket | None:
    """Convert raw Kalshi API response to NormalizedMarket."""
    try:
        # Compute probability from yes_bid/yes_ask (cents, 0-100)
        yes_bid = raw.get("yes_bid", 0) or 0
        yes_ask = raw.get("yes_ask", 0) or 0

        if yes_bid > 0 and yes_ask > 0:
            # Midpoint of bid-ask spread
            probability = (yes_bid + yes_ask) / 200.0
        elif yes_bid > 0:
            probability = yes_bid / 100.0
        elif yes_ask > 0:
            probability = yes_ask / 100.0
        else:
            # Use last_price as fallback
            last_price = raw.get("last_price", 0) or 0
            if last_price > 0:
                probability = last_price / 100.0
            else:
                return None

        exp_str = raw.get("expected_expiration_time") or raw.get("close_time", "")
        if not exp_str:
            return None
        resolution_date = datetime.fromisoformat(exp_str.replace("Z", "+00:00"))

        volume = raw.get("volume", 0) or 0

        ticker = raw.get("ticker", "")
        title = raw.get("title", "")
        subtitle = raw.get("subtitle", "")
        question = f"{title}: {subtitle}" if subtitle else title
        if not question:
            question = ticker

        # Build a readable URL
        event_ticker = raw.get("event_ticker", "")
        url = f"https://kalshi.com/markets/{event_ticker}" if event_ticker else ""

        return NormalizedMarket(
            id=ticker,
            platform=Platform.KALSHI,
            question=question,
            description=raw.get("rules_primary", "")[:2000],
            market_probability=probability,
            volume=float(volume),
            liquidity=float(raw.get("liquidity", 0) or 0),
            resolution_date=resolution_date,
            category=_infer_kalshi_category(ticker),
            url=url,
            raw_data=raw,
        )
    except (ValueError, KeyError) as e:
        log.warning(f"Failed to normalize Kalshi market {raw.get('ticker')}: {e}")
        return None


def _infer_kalshi_category(ticker: str) -> str:
    """Infer category from Kalshi ticker prefix."""
    t = ticker.upper()
    # Sports
    sports_kw = [
        "NCAAMB", "NCAAWB", "NCAABB", "NBA", "NHL", "NFL", "MLS",
        "UFC", "ATP", "WTA", "WBC", "INDYCAR", "NASCAR", "KHL",
        "SHL", "SOCCER", "CRICKET", "EPL", "LALIGA", "SERIEA",
        "BUNDESLIGA", "LIGUE1", "UCL", "CHAMPIONS",
    ]
    for kw in sports_kw:
        if kw in t:
            return "sports"
    # Crypto
    crypto_kw = ["BTC", "ETH", "SOL", "XRP", "DOGE", "SHIBA"]
    for kw in crypto_kw:
        if kw in t:
            return "crypto"
    # Finance / economics
    finance_kw = [
        "KXINX", "KXNASDAQ", "KXWTI", "KXTNOTE", "KXUSD", "KXEUR",
        "KXGOLD", "KXAAGAS", "KXFED", "KXGDP", "KXCPI", "KXRATE",
        "KXDJIA", "KXSPY", "KXBOND", "KXYIELD", "KXJOBS", "KXNFP",
        "KXRETAIL", "KXHOUSING", "SPX", "NASDAQ", "DJIA",
    ]
    for kw in finance_kw:
        if kw in t:
            return "finance"
    # Weather
    if "KXTEMP" in t or "KXHIGH" in t or "KXLOW" in t:
        return "weather"
    # Politics
    if "KXPRES" in t or "KXSENATE" in t or "KXHOUSE" in t or "KXGOV" in t:
        return "politics"
    return "other"
