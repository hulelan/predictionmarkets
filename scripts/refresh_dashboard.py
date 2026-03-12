#!/usr/bin/env python3
"""Refresh market prices and resolution status for the dashboard.

Fetches current prices from Polymarket, Kalshi, and Manifold APIs,
checks resolution status, and updates data/processed/markets.json.
Run on a schedule (e.g. every 30 minutes) to keep the dashboard current.
"""

import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

import httpx

from src.data.normalizer import Platform
from src.scoring.resolution import check_resolution
from src.utils.logger import log

MARKETS_PATH = Path("data/processed/markets.json")


async def fetch_current_price(
    market_id: str,
    platform: str,
    client: httpx.AsyncClient,
) -> dict | None:
    """Fetch current price + resolution status for a single market."""
    try:
        if platform == "polymarket":
            resp = await client.get(
                f"https://gamma-api.polymarket.com/markets/{market_id}"
            )
            resp.raise_for_status()
            data = resp.json()
            prices = json.loads(data.get("outcomePrices", "[]"))
            return {
                "market_probability": float(prices[0]) if prices else None,
                "resolved": data.get("resolved", False),
                "outcome": data.get("outcome"),
                "volume": float(data.get("volume", 0) or 0),
            }

        elif platform == "kalshi":
            resp = await client.get(
                f"https://api.elections.kalshi.com/trade-api/v2/markets/{market_id}"
            )
            if resp.status_code == 429:
                await asyncio.sleep(2)
                resp = await client.get(
                    f"https://api.elections.kalshi.com/trade-api/v2/markets/{market_id}"
                )
            resp.raise_for_status()
            raw_text = resp.text.replace("\r", "").replace("\x00", "")
            data = json.loads(raw_text)
            m = data.get("market", data)
            yes_bid = m.get("yes_bid", 0) or 0
            yes_ask = m.get("yes_ask", 0) or 0
            prob = (yes_bid + yes_ask) / 200.0 if (yes_bid + yes_ask) > 0 else None
            resolved = m.get("result") in ("yes", "no") or m.get("status") == "settled"
            outcome = None
            if m.get("result") == "yes":
                outcome = "Yes"
            elif m.get("result") == "no":
                outcome = "No"
            return {
                "market_probability": prob,
                "resolved": resolved,
                "outcome": outcome,
                "volume": float(m.get("volume", 0) or 0),
            }

        elif platform == "manifold":
            resp = await client.get(
                f"https://api.manifold.markets/v0/market/{market_id}"
            )
            resp.raise_for_status()
            data = resp.json()
            resolved = data.get("isResolved", False)
            outcome = None
            if resolved:
                res = data.get("resolution", "")
                outcome = "Yes" if res == "YES" else "No" if res == "NO" else res
            return {
                "market_probability": data.get("probability"),
                "resolved": resolved,
                "outcome": outcome,
                "volume": float(data.get("volume", 0) or 0),
            }

    except Exception as e:
        log.warning(f"Failed to fetch {platform}/{market_id}: {e}")
        return None


async def refresh_all(concurrency: int = 10):
    """Refresh all market prices and save updated markets.json."""
    if not MARKETS_PATH.exists():
        log.error("No markets.json found. Run fetch_markets.py first.")
        return

    with open(MARKETS_PATH) as f:
        markets = json.load(f)

    log.info(f"Refreshing prices for {len(markets)} markets...")

    sem = asyncio.Semaphore(concurrency)
    updated = 0
    resolved = 0

    async with httpx.AsyncClient(timeout=15) as client:

        async def refresh_one(market: dict) -> None:
            nonlocal updated, resolved
            async with sem:
                result = await fetch_current_price(
                    market["id"], market["platform"], client
                )
                await asyncio.sleep(0.2)  # rate limiting

            if result is None:
                return

            if result["market_probability"] is not None:
                market["market_probability"] = result["market_probability"]
                updated += 1

            if result["resolved"]:
                market["resolved"] = True
                market["outcome"] = result["outcome"]
                resolved += 1
            elif "resolved" not in market:
                market["resolved"] = False

            market["last_refreshed"] = datetime.now(timezone.utc).isoformat()

        tasks = [refresh_one(m) for m in markets]
        await asyncio.gather(*tasks)

    # Save
    with open(MARKETS_PATH, "w") as f:
        json.dump(markets, f, indent=2, default=str)

    log.info(
        f"Refreshed {updated}/{len(markets)} prices, "
        f"{resolved} resolved. Saved to {MARKETS_PATH}"
    )


if __name__ == "__main__":
    asyncio.run(refresh_all())
