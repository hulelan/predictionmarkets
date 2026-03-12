"""Fetch trade-level data from prediction market platforms."""

import asyncio
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

import httpx

from src.utils.logger import log

KALSHI_TRADES_URL = "https://api.elections.kalshi.com/trade-api/v2/markets/trades"
MANIFOLD_BETS_URL = "https://api.manifold.markets/v0/bets"


@dataclass
class TradeRecord:
    """Unified trade record across platforms."""

    timestamp: datetime
    size: float  # dollar value or contract count
    direction: str  # "yes" or "no"
    price: float  # price at time of trade (0-1)
    platform: str


async def fetch_kalshi_trades(
    ticker: str,
    max_trades: int = 1000,
) -> list[TradeRecord]:
    """Fetch trade history for a Kalshi market."""
    trades: list[TradeRecord] = []

    async with httpx.AsyncClient(timeout=30) as client:
        cursor = ""
        while len(trades) < max_trades:
            params = {"ticker": ticker, "limit": 1000}
            if cursor:
                params["cursor"] = cursor

            try:
                resp = await client.get(KALSHI_TRADES_URL, params=params)
                resp.raise_for_status()
                raw_text = resp.text.replace("\r", "").replace("\x00", "")
                data = json.loads(raw_text)
            except (httpx.HTTPError, json.JSONDecodeError) as e:
                log.debug(f"Kalshi trades fetch error for {ticker}: {e}")
                break

            raw_trades = data.get("trades", [])
            cursor = data.get("cursor", "")

            if not raw_trades:
                break

            for t in raw_trades:
                try:
                    ts = datetime.fromisoformat(
                        t.get("created_time", "").replace("Z", "+00:00")
                    )
                    trades.append(
                        TradeRecord(
                            timestamp=ts,
                            size=float(t.get("count", 1)),
                            direction=t.get("taker_side", "yes").lower(),
                            price=float(t.get("yes_price", 50)) / 100.0,
                            platform="kalshi",
                        )
                    )
                except (ValueError, TypeError):
                    continue

            if not cursor:
                break

    return trades[:max_trades]


async def fetch_manifold_bets(
    contract_id: str,
    max_bets: int = 1000,
) -> list[TradeRecord]:
    """Fetch bet history for a Manifold market."""
    trades: list[TradeRecord] = []

    async with httpx.AsyncClient(timeout=30) as client:
        params = {"contractId": contract_id, "limit": min(max_bets, 1000)}

        try:
            resp = await client.get(MANIFOLD_BETS_URL, params=params)
            resp.raise_for_status()
            bets = resp.json()
        except (httpx.HTTPError, json.JSONDecodeError) as e:
            log.debug(f"Manifold bets fetch error for {contract_id}: {e}")
            return []

        for b in bets:
            try:
                ts = datetime.fromtimestamp(
                    b.get("createdTime", 0) / 1000.0, tz=timezone.utc
                )
                amount = abs(float(b.get("amount", 0)))
                if amount == 0:
                    continue

                outcome = b.get("outcome", "YES").upper()
                prob_after = float(b.get("probAfter", 0.5))

                trades.append(
                    TradeRecord(
                        timestamp=ts,
                        size=amount,
                        direction="yes" if outcome == "YES" else "no",
                        price=prob_after,
                        platform="manifold",
                    )
                )
            except (ValueError, TypeError):
                continue

    return trades[:max_bets]


async def fetch_trades_for_market(market_dict: dict) -> list[TradeRecord]:
    """Fetch trades for a market based on its platform.

    Args:
        market_dict: A market dict with at least 'platform', 'id', and 'raw_data'.
    """
    platform = market_dict.get("platform", "")
    market_id = market_dict.get("id", "")
    raw = market_dict.get("raw_data", {})

    if platform == "kalshi":
        ticker = raw.get("ticker", market_id)
        return await fetch_kalshi_trades(ticker)

    elif platform == "manifold":
        contract_id = raw.get("id", market_id)
        return await fetch_manifold_bets(contract_id)

    # Polymarket CLOB often returns 403 — skip for now
    return []
