"""Abstract base class for LLM-based market evaluators."""

import asyncio
import json
import re
from abc import ABC, abstractmethod
from datetime import datetime, timezone

from src.data.normalizer import NormalizedMarket
from src.utils.cache import PredictionCache
from src.utils.logger import log


class BaseEvaluator(ABC):
    """Abstract interface for LLM-based market evaluators."""

    def __init__(self):
        self.cache = PredictionCache()

    @abstractmethod
    async def predict(self, market: NormalizedMarket) -> dict:
        """Return {"probability": float, "confidence": str, "reasoning": str}."""
        pass

    @abstractmethod
    def model_name(self) -> str:
        pass

    async def evaluate_batch(
        self,
        markets: list[NormalizedMarket],
        concurrency: int = 5,
    ) -> list[dict]:
        """Evaluate all markets with bounded concurrency and disk caching."""
        name = self.model_name()
        existing = self.cache.load_existing(name)
        log.info(f"[{name}] Cache has {len(existing)} existing predictions")

        sem = asyncio.Semaphore(concurrency)
        results = []

        async def process_one(market: NormalizedMarket) -> dict:
            # Check cache
            if market.id in existing:
                return existing[market.id]

            async with sem:
                try:
                    result = await self.predict(market)
                    pred = {
                        "market_id": market.id,
                        "platform": market.platform.value,
                        "question": market.question,
                        "model": name,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        **result,
                    }
                except Exception as e:
                    log.error(f"[{name}] Failed on {market.id}: {e}")
                    pred = {
                        "market_id": market.id,
                        "platform": market.platform.value,
                        "question": market.question,
                        "model": name,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "probability": market.market_probability,
                        "confidence": "failed",
                        "reasoning": f"API error: {str(e)[:200]}",
                    }

                # Save incrementally
                self.cache.append(name, pred)
                return pred

        tasks = [process_one(m) for m in markets]
        results = await asyncio.gather(*tasks)

        log.info(f"[{name}] Completed {len(results)} predictions")
        return results


def extract_json(text: str) -> dict:
    """Robustly extract JSON from LLM response text."""
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find JSON block in markdown (greedy to capture full object)
    match = re.search(r"```(?:json)?\s*(\{.*\})\s*(?:```|$)", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find any JSON object containing "probability" (greedy)
    match = re.search(r"(\{.*\"probability\".*\})", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Last resort: extract probability with regex from truncated responses
    prob_match = re.search(r"\"probability\"\s*:\s*([\d.]+)", text)
    conf_match = re.search(r"\"confidence\"\s*:\s*\"(\w+)\"", text)
    if prob_match:
        return {
            "probability": float(prob_match.group(1)),
            "confidence": conf_match.group(1) if conf_match else "low",
            "reasoning": "Extracted from truncated response",
        }

    raise ValueError(f"Could not extract JSON from response: {text[:200]}")


def clamp_probability(p: float) -> float:
    """Clamp probability to [0.01, 0.99]."""
    return max(0.01, min(0.99, float(p)))
