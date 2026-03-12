"""Filtering logic for normalized markets."""

from src.data.normalizer import NormalizedMarket
from src.utils.logger import log


def apply_filters(
    markets: list[NormalizedMarket],
    min_probability: float = 0.02,
    max_probability: float = 0.98,
    min_volume: float = 0,
) -> list[NormalizedMarket]:
    """Filter markets by probability range and volume threshold."""
    filtered = []
    for m in markets:
        if m.market_probability < min_probability or m.market_probability > max_probability:
            continue
        if m.volume < min_volume:
            continue
        if not m.question.strip():
            continue
        filtered.append(m)

    log.info(f"Filtered {len(markets)} → {len(filtered)} markets")
    return filtered


def sort_by_volume(markets: list[NormalizedMarket]) -> list[NormalizedMarket]:
    """Sort markets by volume descending."""
    return sorted(markets, key=lambda m: m.volume, reverse=True)


def deduplicate_markets(markets: list[NormalizedMarket]) -> list[NormalizedMarket]:
    """Remove duplicate markets based on similar question text.

    Simple approach: normalize question text and check for exact matches.
    Keeps the one with higher volume.
    """
    seen: dict[str, NormalizedMarket] = {}
    for m in markets:
        key = _normalize_question(m.question)
        if key in seen:
            if m.volume > seen[key].volume:
                seen[key] = m
        else:
            seen[key] = m

    deduped = list(seen.values())
    if len(deduped) < len(markets):
        log.info(f"Deduplicated {len(markets)} → {len(deduped)} markets")
    return deduped


def _normalize_question(q: str) -> str:
    """Normalize question text for deduplication."""
    return q.lower().strip().rstrip("?").strip()
