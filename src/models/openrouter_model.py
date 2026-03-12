"""OpenRouter-based evaluator that routes to any model via OpenAI-compatible API."""

from __future__ import annotations

import asyncio
import os
from datetime import date

from openai import AsyncOpenAI

from src.data.normalizer import NormalizedMarket
from src.models.base import BaseEvaluator, clamp_probability, extract_json
from src.models.prompts import FORECASTING_PROMPT


class OpenRouterEvaluator(BaseEvaluator):
    def __init__(
        self,
        model: str,
        display_name: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 16384,
    ):
        super().__init__()
        self.client = AsyncOpenAI(
            api_key=os.environ.get("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
        )
        self.model = model
        self._display_name = display_name or model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def model_name(self) -> str:
        return self._display_name

    async def predict(self, market: NormalizedMarket) -> dict:
        prompt = FORECASTING_PROMPT.format(
            question=market.question,
            description=market.description[:1000] if market.description else "Not provided",
            resolution_date=market.resolution_date.strftime("%Y-%m-%d"),
            market_probability=market.market_probability,
            current_date=date.today().isoformat(),
        )

        for attempt in range(3):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                text = response.choices[0].message.content
                result = extract_json(text)
                result["probability"] = clamp_probability(result["probability"])
                return result
            except Exception as e:
                if "429" in str(e) and attempt < 2:
                    await asyncio.sleep(2 ** attempt)
                    continue
                raise
