"""Critic layer: reviews each constituent prediction before ensembling.

Each agent's prediction + reasoning gets evaluated by a critic that checks
for known failure modes (priced-in news treated as new, trend extrapolation,
hallucinated outcomes, overriding efficient markets without evidence).

The critic adjusts the probability toward market price when reasoning is weak,
and preserves deviations only when the agent found genuinely new information.

Runs as a single LLM call per prediction — no tools, pure reasoning.
"""

import json
import os
from datetime import date

from openai import AsyncOpenAI

from src.data.normalizer import NormalizedMarket
from src.models.base import clamp_probability, extract_json
from src.models.prompts import CRITIC_SYSTEM_PROMPT
from src.utils.logger import log


class Critic:
    """Reviews individual agent predictions for reasoning quality.

    Runs a single LLM call per prediction — no tools, just reasoning
    review. Lightweight and parallelizable.
    """

    def __init__(
        self,
        model: str = "anthropic/claude-sonnet-4-6",
        temperature: float = 0.1,
    ):
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY required for Critic")

        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )
        self.model = model
        self.temperature = temperature

    async def review(
        self,
        market: NormalizedMarket,
        prediction: dict,
    ) -> dict:
        """Review a single prediction and return adjusted probability."""
        agent_prob = prediction.get("probability", market.market_probability)
        agent_reasoning = prediction.get("reasoning", "No reasoning provided")
        agent_model = prediction.get("model", "unknown")
        search_calls = prediction.get("search_calls", 0)

        deviation = abs(agent_prob - market.market_probability)

        # Skip critic for predictions very close to market price — no review needed
        if deviation < 0.03:
            return {
                "original_probability": agent_prob,
                "adjusted_probability": agent_prob,
                "critique": "Near market price, no review needed.",
                "adjustment_reason": "kept",
                "failure_modes": [],
                "skipped": True,
            }

        user_prompt = self._build_prompt(
            market, agent_prob, agent_reasoning, agent_model,
            search_calls,
        )

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                max_tokens=4096,
            )

            text = response.choices[0].message.content or ""
            result = extract_json(text)
            result["adjusted_probability"] = clamp_probability(
                result["adjusted_probability"]
            )
            result["original_probability"] = agent_prob
            result["skipped"] = False

            # Log adjustments
            adj = result["adjusted_probability"]
            if abs(adj - agent_prob) > 0.01:
                log.info(
                    f"  Critic [{market.id}]: {agent_prob:.3f} → {adj:.3f} "
                    f"({result.get('adjustment_reason', '?')}) "
                    f"modes={result.get('failure_modes', [])}"
                )

            return result

        except Exception as e:
            log.warning(f"Critic failed for {market.id}: {e}")
            # On failure, return original prediction unchanged
            return {
                "original_probability": agent_prob,
                "adjusted_probability": agent_prob,
                "critique": f"Critic error: {e}",
                "adjustment_reason": "error",
                "failure_modes": [],
                "skipped": True,
            }

    def _build_prompt(
        self,
        market: NormalizedMarket,
        agent_prob: float,
        agent_reasoning: str,
        agent_model: str,
        search_calls: int,
    ) -> str:
        """Build the critic's review prompt."""
        deviation = agent_prob - market.market_probability
        direction = "ABOVE" if deviation > 0 else "BELOW"

        return (
            f"MARKET: {market.question}\n"
            f"RESOLUTION DATE: {market.resolution_date.strftime('%Y-%m-%d')}\n"
            f"TODAY: {date.today().isoformat()}\n"
            f"PLATFORM: {market.platform.value}\n\n"
            f"MARKET PRICE: {market.market_probability:.1%}\n"
            f"FORECASTER ({agent_model}): {agent_prob:.1%} "
            f"({abs(deviation):.1%} {direction} market)\n"
            f"SEARCH CALLS MADE: {search_calls}\n\n"
            f"FORECASTER'S REASONING:\n{agent_reasoning}\n\n"
            f"Review this prediction. Check for the 6 failure modes. "
            f"Should the probability be kept, pulled toward market, or slightly adjusted?"
        )
