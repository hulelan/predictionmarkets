#!/usr/bin/env python3
"""Run the forecasting agent pipeline.

Pipeline:
  1. Run ToolAgent N times (different reasoning models, each with Perplexity search)
  2. Critic reviews each prediction — catches failure modes, adjusts toward market
  3. Judge reads ALL predictions + reasoning, makes final call (no search, no averaging)
  4. Save final predictions

Each run is independently cached, so re-runs skip completed work.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

from src.agent.critic import Critic
from src.agent.tool_agent import ToolAgent
from src.data.fetcher import load_processed_markets
from src.data.normalizer import NormalizedMarket
from src.models.base import clamp_probability, extract_json
from src.models.prompts import JUDGE_SYSTEM_PROMPT
from src.utils.cache import PredictionCache
from src.utils.logger import log


# Default: 5 runs across 3 models for diversity
DEFAULT_RUNS = [
    {"reasoning_model": "anthropic/claude-opus-4-6", "temperature": 0.3, "label": "claude_t03"},
    {"reasoning_model": "openai/gpt-5.4", "temperature": 0.3, "label": "gpt_t03"},
    {"reasoning_model": "google/gemini-2.5-pro", "temperature": 0.3, "label": "gemini_t03"},
    {"reasoning_model": "anthropic/claude-opus-4-6", "temperature": 0.7, "label": "claude_t07"},
    {"reasoning_model": "openai/gpt-5.4", "temperature": 0.7, "label": "gpt_t07"},
]


async def run_single_agent(
    agent: ToolAgent,
    market: NormalizedMarket,
    run_label: str,
    cache: PredictionCache,
) -> dict | None:
    """Run one ToolAgent prediction for one market, with caching."""
    cache_key = f"tool_agent_{run_label}"
    existing = cache.load_existing(cache_key)

    if market.id in existing:
        return existing[market.id]

    try:
        result = await agent.predict(market)
        pred = {
            "market_id": market.id,
            "platform": market.platform.value,
            "question": market.question,
            "model": cache_key,
            **result,
        }
        cache.append(cache_key, pred)
        return pred
    except Exception as e:
        log.warning(f"[{run_label}] Failed on {market.id}: {e}")
        return None


async def _run_one(
    run_cfg: dict,
    market: NormalizedMarket,
    cache: PredictionCache,
    sem: asyncio.Semaphore,
) -> dict | None:
    """Run a single agent config, respecting the semaphore."""
    async with sem:
        agent = ToolAgent(
            reasoning_model=run_cfg["reasoning_model"],
            temperature=run_cfg["temperature"],
        )
        return await run_single_agent(agent, market, run_cfg["label"], cache)


async def _critique_one(
    critic: Critic,
    market: NormalizedMarket,
    prediction: dict,
    cache: PredictionCache,
    sem: asyncio.Semaphore,
) -> dict:
    """Run critic on a single prediction, with caching."""
    model_label = prediction.get("model", "unknown")
    cache_key = f"critic_{model_label}"
    existing = cache.load_existing(cache_key)

    if market.id in existing:
        cached = existing[market.id]
        prediction["critic"] = cached.get("critic", {})
        prediction["probability"] = cached.get("probability", prediction["probability"])
        return prediction

    async with sem:
        review = await critic.review(market, prediction)

    prediction["critic"] = {
        "original_probability": review["original_probability"],
        "adjusted_probability": review["adjusted_probability"],
        "critique": review.get("critique", ""),
        "adjustment_reason": review.get("adjustment_reason", ""),
        "failure_modes": review.get("failure_modes", []),
    }
    prediction["probability"] = review["adjusted_probability"]
    cache.append(cache_key, prediction)
    return prediction


async def run_all_agents_for_market(
    market: NormalizedMarket,
    runs: list[dict],
    cache: PredictionCache,
    sem: asyncio.Semaphore,
    critic: Critic | None = None,
) -> list[dict]:
    """Run all agent configs for one market in parallel, then critic each."""
    # Step 1: Run all agents in parallel
    tasks = [_run_one(run_cfg, market, cache, sem) for run_cfg in runs]
    raw_results = await asyncio.gather(*tasks)
    results = [r for r in raw_results if r and r.get("confidence") != "failed"]

    if not results or critic is None:
        return results

    # Step 2: Run critic on each prediction in parallel
    critic_tasks = [
        _critique_one(critic, market, pred, cache, sem)
        for pred in results
    ]
    critiqued = await asyncio.gather(*critic_tasks)
    return list(critiqued)


# ---------------------------------------------------------------------------
# Judge — reads all predictions, makes final call. No search.
# ---------------------------------------------------------------------------


async def judge_market(
    market: NormalizedMarket,
    run_results: list[dict],
    judge_model: str,
    cache: PredictionCache,
    sem: asyncio.Semaphore,
) -> dict | None:
    """Judge reads all forecaster outputs and makes the final call. No search."""
    cache_key = "judge"
    existing = cache.load_existing(cache_key)
    if market.id in existing:
        return existing[market.id]

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return None

    client = AsyncOpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")

    # Build forecaster summaries
    forecaster_lines = []
    for i, r in enumerate(run_results, 1):
        model = r.get("model", f"run_{i}")
        prob = r.get("probability", "?")
        reasoning = r.get("reasoning", "No reasoning provided")
        confidence = r.get("confidence", "?")
        search_calls = r.get("search_calls", 0)
        critic_info = ""
        if r.get("critic"):
            c = r["critic"]
            if c.get("failure_modes"):
                critic_info = f"\n  Critic flags: {', '.join(c['failure_modes'])}"
            if c.get("original_probability") != c.get("adjusted_probability"):
                critic_info += (
                    f"\n  Critic adjusted: {c['original_probability']:.3f} → "
                    f"{c['adjusted_probability']:.3f} ({c.get('adjustment_reason', '?')})"
                )
        forecaster_lines.append(
            f"Forecaster {i} ({model}):\n"
            f"  Probability: {prob}\n"
            f"  Confidence: {confidence}\n"
            f"  Searches made: {search_calls}\n"
            f"  Reasoning: {reasoning}"
            f"{critic_info}"
        )
    forecasters_text = "\n\n".join(forecaster_lines)

    description = market.description[:500] if market.description else "Not provided"

    user_prompt = (
        f"MARKET QUESTION: {market.question}\n"
        f"RESOLUTION CRITERIA: {description}\n"
        f"RESOLUTION DATE: {market.resolution_date.strftime('%Y-%m-%d')}\n"
        f"TODAY: {date.today().isoformat()}\n"
        f"MARKET PRICE: {market.market_probability:.1%}\n\n"
        f"FORECASTER PREDICTIONS:\n\n{forecasters_text}\n\n"
        f"Read all predictions carefully. Evaluate reasoning quality. "
        f"Make your final call as JSON."
    )

    try:
        async with sem:
            response = await client.chat.completions.create(
                model=judge_model,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                max_tokens=16384,
                # No tools — judge just reads and decides
            )

        text = response.choices[0].message.content or ""
        result = extract_json(text)
        pred = {
            "market_id": market.id,
            "platform": market.platform.value,
            "question": market.question,
            "model": cache_key,
            "probability": clamp_probability(result["probability"]),
            "confidence": result.get("confidence", "medium"),
            "reasoning": result.get("reasoning", "Judge adjudication"),
            "strategy": "judge",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "n_forecasters": len(run_results),
        }
        cache.append(cache_key, pred)
        return pred

    except Exception as e:
        log.warning(f"Judge failed for {market.id}: {e}")
        return None


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


async def main():
    parser = argparse.ArgumentParser(description="Run forecasting agent pipeline")
    parser.add_argument("--limit", type=int, default=None, help="Limit markets (for testing)")
    parser.add_argument("--concurrency", type=int, default=3, help="Max concurrent runs")
    parser.add_argument("--runs", type=int, default=None,
                        help="Number of agent runs (overrides default 5). "
                             "Uses first N configs from the default list.")
    parser.add_argument("--judge-model", type=str, default="anthropic/claude-opus-4-6",
                        help="Model for the judge (default: claude-opus-4-6)")
    parser.add_argument("--no-critic", action="store_true",
                        help="Skip critic layer")
    parser.add_argument("--critic-model", type=str, default="anthropic/claude-sonnet-4-6",
                        help="Model for critic (default: claude-sonnet-4-6)")
    parser.add_argument("--no-judge", action="store_true",
                        help="Skip judge — just average probabilities instead")
    args = parser.parse_args()

    markets = load_processed_markets()
    if args.limit:
        markets = markets[: args.limit]

    runs = DEFAULT_RUNS
    if args.runs:
        runs = DEFAULT_RUNS[: args.runs]

    critic = None if args.no_critic else Critic(model=args.critic_model)

    log.info(
        f"Pipeline: {len(runs)} agent runs × {len(markets)} markets "
        f"(concurrency={args.concurrency})"
    )
    for r in runs:
        log.info(f"  Run: {r['label']} ({r['reasoning_model']}, temp={r['temperature']})")
    if critic:
        log.info(f"  Critic: {args.critic_model}")
    if not args.no_judge:
        log.info(f"  Judge: {args.judge_model} (final decision, no search)")

    cache = PredictionCache()
    sem = asyncio.Semaphore(args.concurrency)

    # --- Step 1+2: Run all agents + critic ---
    all_predictions: dict[str, dict] = {}

    for market in tqdm(markets, desc="Forecasting"):
        run_results = await run_all_agents_for_market(
            market, runs, cache, sem, critic=critic,
        )

        if not run_results:
            all_predictions[market.id] = {
                "market_id": market.id,
                "platform": market.platform.value,
                "question": market.question,
                "probability": market.market_probability,
                "confidence": "low",
                "strategy": "fallback",
                "reasoning": "All agent runs failed",
                "n_runs": 0,
            }
            continue

        # --- Step 3: Judge makes the final call ---
        if not args.no_judge:
            judged = await judge_market(
                market, run_results, args.judge_model, cache, sem,
            )
            if judged:
                judged["n_runs"] = len(run_results)
                judged["run_probs"] = [r["probability"] for r in run_results]
                all_predictions[market.id] = judged
                continue

        # Fallback if judge is disabled or failed: simple mean
        probs = [r["probability"] for r in run_results]
        all_predictions[market.id] = {
            "market_id": market.id,
            "platform": market.platform.value,
            "question": market.question,
            "probability": statistics.mean(probs),
            "confidence": "medium",
            "strategy": "mean_fallback",
            "reasoning": f"Mean of {len(probs)} runs (judge disabled/failed)",
            "n_runs": len(probs),
            "run_probs": probs,
        }

    # --- Save ---
    cache_key = "ensemble_agent"
    final_preds = list(all_predictions.values())
    cache.save_all(cache_key, final_preds)

    # Summary
    strategies = {}
    for p in final_preds:
        s = p.get("strategy", "unknown")
        strategies[s] = strategies.get(s, 0) + 1

    probs = [p["probability"] for p in final_preds if p.get("strategy") != "fallback"]

    log.info(f"Strategy distribution: {strategies}")
    if probs:
        log.info(f"Mean probability: {statistics.mean(probs):.3f}")
    log.info(f"Saved to data/predictions/{cache_key}.jsonl")


if __name__ == "__main__":
    asyncio.run(main())
