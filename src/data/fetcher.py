"""Orchestrator: fetch markets from all platforms, normalize, filter, select top N."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.data.filters import apply_filters, deduplicate_markets, sort_by_volume
from src.data.kalshi import fetch_kalshi_markets, normalize_kalshi
from src.data.manifold import fetch_manifold_markets, normalize_manifold
from src.data.normalizer import NormalizedMarket
from src.data.polymarket import fetch_polymarket_markets, normalize_polymarket
from src.utils.logger import log


async def fetch_all_markets(
    end_date_min: str = "2026-03-08",
    end_date_max: str = "2026-03-15",
    target_total: int = 1000,
    min_probability: float = 0.02,
    max_probability: float = 0.98,
    min_volume: float = 0,
    resolution_after: str | None = None,
    resolution_before: str | None = None,
) -> list[NormalizedMarket]:
    """Main pipeline: fetch from all platforms → normalize → filter → dedupe → top N by volume.

    Args:
        resolution_after: ISO datetime string. Only include markets resolving after this time.
        resolution_before: ISO datetime string. Only include markets resolving before this time.
    """

    # 1. Fetch raw data from all platforms in parallel
    log.info(f"Fetching markets from Polymarket, Kalshi, and Manifold ({end_date_min} to {end_date_max})...")

    close_min = datetime.fromisoformat(f"{end_date_min}T00:00:00+00:00")
    close_max = datetime.fromisoformat(f"{end_date_max}T23:59:59+00:00")

    poly_raw, kalshi_raw, manifold_raw = await asyncio.gather(
        fetch_polymarket_markets(
            end_date_min=end_date_min,
            end_date_max=end_date_max,
            max_markets=3000,
        ),
        fetch_kalshi_markets(
            expiration_min=close_min,
            expiration_max=close_max,
            max_markets=10000,
            min_volume=0,
        ),
        fetch_manifold_markets(
            close_date_min=close_min,
            close_date_max=close_max,
            max_markets=500,
        ),
    )

    log.info(f"Raw: {len(poly_raw)} Polymarket, {len(kalshi_raw)} Kalshi, {len(manifold_raw)} Manifold")

    # Save raw data
    _save_raw(poly_raw, "data/raw/polymarket.json")
    _save_raw(kalshi_raw, "data/raw/kalshi.json")
    _save_raw(manifold_raw, "data/raw/manifold.json")

    # 2. Normalize
    poly_markets = [m for raw in poly_raw if (m := normalize_polymarket(raw)) is not None]
    kalshi_markets = [m for raw in kalshi_raw if (m := normalize_kalshi(raw)) is not None]
    manifold_markets = [m for raw in manifold_raw if (m := normalize_manifold(raw)) is not None]
    all_markets = poly_markets + kalshi_markets + manifold_markets

    log.info(
        f"Normalized: {len(poly_markets)} Polymarket, {len(kalshi_markets)} Kalshi, "
        f"{len(manifold_markets)} Manifold = {len(all_markets)} total"
    )

    # 2b. Time window filter
    if resolution_after or resolution_before:
        res_after = datetime.fromisoformat(resolution_after) if resolution_after else None
        res_before = datetime.fromisoformat(resolution_before) if resolution_before else None
        before_count = len(all_markets)
        all_markets = [
            m for m in all_markets
            if (not res_after or m.resolution_date >= res_after)
            and (not res_before or m.resolution_date <= res_before)
        ]
        log.info(f"Time window filter: {before_count} → {len(all_markets)} markets")

    # 3. Filter
    filtered = apply_filters(
        all_markets,
        min_probability=min_probability,
        max_probability=max_probability,
        min_volume=min_volume,
    )

    # 4. Deduplicate
    deduped = deduplicate_markets(filtered)

    # 5. Sort by volume and take top N
    sorted_markets = sort_by_volume(deduped)
    top = sorted_markets[:target_total]

    log.info(f"Final selection: {len(top)} markets")

    # 6. Save processed data
    save_processed(top)

    return top


def save_processed(markets: list[NormalizedMarket]) -> None:
    """Save processed markets to JSON and CSV."""
    Path("data/processed").mkdir(parents=True, exist_ok=True)

    # JSON (full data for scripts)
    json_data = [m.model_dump(mode="json", exclude={"raw_data"}) for m in markets]
    with open("data/processed/markets.json", "w") as f:
        json.dump(json_data, f, indent=2, default=str)

    # CSV (for quick inspection)
    rows = [m.to_row() for m in markets]
    df = pd.DataFrame(rows)
    df.to_csv("data/processed/markets.csv", index=False)

    log.info(f"Saved {len(markets)} markets to data/processed/")

    # Print summary
    _print_summary(markets)


def load_processed_markets(include_raw: bool = True) -> list[NormalizedMarket]:
    """Load markets from processed JSON file.

    Args:
        include_raw: If True (default), re-attach raw platform data from
            data/raw/ files. This provides price movement context
            (oneDayPriceChange, bid/ask, etc.) for the agent.
    """
    path = Path("data/processed/markets.json")
    if not path.exists():
        raise FileNotFoundError(f"No processed markets at {path}. Run fetch_markets.py first.")

    with open(path) as f:
        data = json.load(f)

    markets = [NormalizedMarket(**m) for m in data]

    if include_raw:
        _attach_raw_data(markets)

    return markets


def _attach_raw_data(markets: list[NormalizedMarket]) -> None:
    """Re-attach raw platform data from data/raw/ files to markets."""
    raw_dir = Path("data/raw")

    # Build lookup: platform -> {market_id: raw_dict}
    platform_lookups: dict[str, dict[str, dict]] = {}

    for platform_file, id_key in [
        ("polymarket.json", "condition_id"),
        ("kalshi.json", "ticker"),
        ("manifold.json", "id"),
    ]:
        fpath = raw_dir / platform_file
        if not fpath.exists():
            continue
        with open(fpath) as f:
            raw_items = json.load(f)
        lookup = {}
        for item in raw_items:
            key = item.get(id_key, "")
            if key:
                lookup[key] = item
        platform_lookups[platform_file.replace(".json", "")] = lookup

    # Attach raw_data to each market
    for m in markets:
        lookup = platform_lookups.get(m.platform.value, {})
        raw = lookup.get(m.id, {})
        if raw:
            m.raw_data = raw


def _save_raw(data: list[dict], path: str) -> None:
    """Save raw API response to JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def _print_summary(markets: list[NormalizedMarket]) -> None:
    """Print a summary of the fetched markets."""
    from collections import Counter

    platform_counts = Counter(m.platform.value for m in markets)
    category_counts = Counter(m.category for m in markets)
    volumes = [m.volume for m in markets]

    log.info("=== Market Summary ===")
    log.info(f"Total: {len(markets)}")
    log.info(f"By platform: {dict(platform_counts)}")
    log.info(f"By category: {dict(category_counts.most_common(10))}")
    if volumes:
        log.info(f"Volume: min={min(volumes):.0f}, median={sorted(volumes)[len(volumes)//2]:.0f}, max={max(volumes):.0f}")
    log.info(f"Probability range: {min(m.market_probability for m in markets):.3f} - {max(m.market_probability for m in markets):.3f}")
