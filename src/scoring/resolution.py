"""Fetch resolution outcomes from prediction market APIs."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import httpx

from src.data.normalizer import NormalizedMarket, Platform
from src.utils.logger import log


async def check_resolution(market: NormalizedMarket, client: httpx.AsyncClient) -> int | None:
    """Check if a market has resolved. Returns 1 (Yes), 0 (No), or None (unresolved)."""
    if market.platform == Platform.POLYMARKET:
        return await _check_polymarket(market.id, client)
    elif market.platform == Platform.KALSHI:
        return await _check_kalshi(market.id, client)
    elif market.platform == Platform.MANIFOLD:
        return await _check_manifold(market.id, client)
    return None


async def _check_polymarket(market_id: str, client: httpx.AsyncClient) -> int | None:
    """Check Polymarket resolution status."""
    try:
        resp = await client.get(f"https://gamma-api.polymarket.com/markets/{market_id}")
        resp.raise_for_status()
        data = resp.json()

        if data.get("resolved"):
            outcome = data.get("outcome", "")
            if outcome == "Yes":
                return 1
            elif outcome == "No":
                return 0
    except Exception as e:
        log.warning(f"Polymarket resolution check failed for {market_id}: {e}")
    return None


async def _check_kalshi(market_ticker: str, client: httpx.AsyncClient) -> int | None:
    """Check Kalshi resolution status with retry on 429."""
    import json as _json

    for attempt in range(3):
        try:
            resp = await client.get(f"https://api.elections.kalshi.com/trade-api/v2/markets/{market_ticker}")
            if resp.status_code == 429:
                await asyncio.sleep(2 ** attempt)
                continue
            resp.raise_for_status()
            raw_text = resp.text.replace("\r", "").replace("\x00", "")
            data = _json.loads(raw_text)
            market = data.get("market", data)

            result = market.get("result", "")
            if result == "yes":
                return 1
            elif result == "no":
                return 0
            if market.get("status") == "settled":
                exp_val = market.get("expiration_value", "")
                if exp_val == "Yes":
                    return 1
                elif exp_val == "No":
                    return 0
            return None
        except Exception as e:
            log.warning(f"Kalshi resolution check failed for {market_ticker}: {e}")
            return None
    log.warning(f"Kalshi rate limit exhausted for {market_ticker}")
    return None


async def _check_manifold(market_id: str, client: httpx.AsyncClient) -> int | None:
    """Check Manifold resolution status."""
    try:
        resp = await client.get(f"https://api.manifold.markets/v0/market/{market_id}")
        resp.raise_for_status()
        data = resp.json()

        if data.get("isResolved"):
            resolution = data.get("resolution", "")
            if resolution == "YES":
                return 1
            elif resolution == "NO":
                return 0
    except Exception as e:
        log.warning(f"Manifold resolution check failed for {market_id}: {e}")
    return None


async def check_all_resolutions(
    markets: list[NormalizedMarket],
    concurrency: int = 5,
) -> list[int | None]:
    """Check resolution status for all markets."""
    sem = asyncio.Semaphore(concurrency)
    results: list[int | None] = []

    async with httpx.AsyncClient(timeout=15) as client:
        async def check_one(market: NormalizedMarket) -> int | None:
            async with sem:
                result = await check_resolution(market, client)
                await asyncio.sleep(0.3)  # rate limiting
                return result

        tasks = [check_one(m) for m in markets]
        results = await asyncio.gather(*tasks)

    resolved_count = sum(1 for r in results if r is not None)
    log.info(f"Resolution check: {resolved_count}/{len(markets)} resolved")
    return results
