"""Search-augmented baseline evaluator.

Each baseline model gets a single Perplexity Sonar search call before predicting.
This gives the model current web context so it can form an independent view
rather than just echoing market price.
"""

from __future__ import annotations

import os
import re
from datetime import date

from openai import AsyncOpenAI

from src.data.normalizer import NormalizedMarket
from src.models.base import BaseEvaluator, clamp_probability, extract_json
from src.models.prompts import FORECASTING_PROMPT, SEARCH_FORECASTING_PROMPT
from src.utils.logger import log


def _build_search_query(question: str) -> str:
    """Convert market question to an effective search query."""
    q = question.rstrip("?").strip()
    for prefix in ["Will ", "Will the ", "Is ", "Does ", "Can "]:
        if q.startswith(prefix):
            q = q[len(prefix) :]
            break
    q = re.sub(r"\b(before|by|on|after)\b \w+ \d{1,2},? \d{4}", "", q).strip()
    return f"{q} latest news March 2026"


class SearchAugmentedEvaluator(BaseEvaluator):
    """Baseline evaluator with a single Perplexity Sonar search per market."""

    def __init__(
        self,
        model: str,
        display_name: str | None = None,
        search_model: str = "perplexity/sonar",
        temperature: float = 0.3,
        max_tokens: int = 16384,
    ):
        super().__init__()
        self.client = AsyncOpenAI(
            api_key=os.environ.get("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
        )
        self.model = model
        self._display_name = display_name or f"search_{model}"
        self.search_model = search_model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def model_name(self) -> str:
        return self._display_name

    async def predict(self, market: NormalizedMarket) -> dict:
        # Step 1: Search for current context
        search_context = await self._search(market.question)

        # Step 2: Build prompt — use search prompt if we got context, fallback otherwise
        if search_context:
            prompt = SEARCH_FORECASTING_PROMPT.format(
                question=market.question,
                description=market.description[:1000] if market.description else "Not provided",
                resolution_date=market.resolution_date.strftime("%Y-%m-%d"),
                market_probability=market.market_probability,
                current_date=date.today().isoformat(),
                search_context=search_context,
            )
        else:
            prompt = FORECASTING_PROMPT.format(
                question=market.question,
                description=market.description[:1000] if market.description else "Not provided",
                resolution_date=market.resolution_date.strftime("%Y-%m-%d"),
                market_probability=market.market_probability,
                current_date=date.today().isoformat(),
            )

        # Step 3: Call reasoning model
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        text = response.choices[0].message.content
        result = extract_json(text)
        result["probability"] = clamp_probability(result["probability"])
        result["search_context"] = bool(search_context)
        return result

    async def _search(self, question: str) -> str:
        """Single Perplexity Sonar search for market context."""
        query = _build_search_query(question)
        try:
            response = await self.client.chat.completions.create(
                model=self.search_model,
                messages=[{"role": "user", "content": query}],
                temperature=0.1,
                max_tokens=2048,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            log.warning(f"Search failed for '{query[:60]}': {e}")
            return ""
