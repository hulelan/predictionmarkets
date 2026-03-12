#!/usr/bin/env python3
"""Fetch and store the most liquid prediction markets resolving tomorrow/day after."""

import asyncio
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

from src.data.fetcher import fetch_all_markets
from src.utils.logger import log


async def main():
    # Dynamic window: tomorrow through day-after-tomorrow
    now = datetime.now(timezone.utc)
    tomorrow = (now + timedelta(days=1)).strftime("%Y-%m-%d")
    day_after = (now + timedelta(days=2)).strftime("%Y-%m-%d")

    log.info(f"Fetching most liquid markets resolving {tomorrow} to {day_after}...")

    markets = await fetch_all_markets(
        end_date_min="2026-03-12",
        end_date_max="2026-03-12",
        target_total=2000,
        min_probability=0.05,
        max_probability=0.95,
        min_volume=100,
        # 8am–8pm ET on Mar 12 = 12:00–00:00 UTC (EDT = UTC-4)
        resolution_after="2026-03-12T12:00:00+00:00",
        resolution_before="2026-03-13T03:59:00+00:00",
    )
    log.info(f"Done! {len(markets)} markets saved to data/processed/")


if __name__ == "__main__":
    asyncio.run(main())
