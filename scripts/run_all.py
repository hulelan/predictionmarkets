#!/usr/bin/env python3
"""Unified runner: search baselines + agent pipeline, all concurrent per market.

For each batch of markets, ALL processes run concurrently:
  - 3 search-augmented baselines (GPT-5.4, Claude Opus, Gemini 2.5 Pro)
  - 5 ToolAgent runs (Claude t=0.3, GPT t=0.3, Gemini t=0.3, Claude t=0.7, GPT t=0.7)
  - 5 Critics (one per agent)
  - 1 Judge (reads all agent predictions, makes final call)

Markets are processed in batches (default 10). Within each batch, every market's
full pipeline runs concurrently. When the batch finishes, the next batch starts.

All predictions get timestamps. Logs go to logs/run_YYYYMMDD_HHMMSS.log.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
from datetime import date, datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

from openai import AsyncOpenAI

from src.agent.critic import Critic
from src.agent.tool_agent import ToolAgent
from src.data.fetcher import load_processed_markets
from src.data.normalizer import NormalizedMarket
from src.models.base import clamp_probability, extract_json
from src.models.prompts import JUDGE_SYSTEM_PROMPT
from src.models.search_evaluator import SearchAugmentedEvaluator
from src.utils.cache import PredictionCache
from src.utils.logger import log

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SEARCH_BASELINES = [
    {"model": "openai/gpt-5.4", "display_name": "search_openai/gpt-5.4", "max_tokens": 16384},
    {"model": "anthropic/claude-opus-4.6", "display_name": "search_anthropic/claude-opus-4-6", "max_tokens": 16384},
    {"model": "google/gemini-2.5-pro", "display_name": "search_google/gemini-2.5-pro", "max_tokens": 32768},
]

AGENT_RUNS = [
    {"reasoning_model": "anthropic/claude-opus-4-6", "temperature": 0.3, "label": "claude_t03"},
    {"reasoning_model": "openai/gpt-5.4", "temperature": 0.3, "label": "gpt_t03"},
    {"reasoning_model": "google/gemini-2.5-pro", "temperature": 0.3, "label": "gemini_t03"},
    {"reasoning_model": "anthropic/claude-opus-4-6", "temperature": 0.7, "label": "claude_t07"},
    {"reasoning_model": "openai/gpt-5.4", "temperature": 0.7, "label": "gpt_t07"},
]


# ---------------------------------------------------------------------------
# Per-market pipeline
# ---------------------------------------------------------------------------

async def run_search_baseline_for_market(
    cfg: dict,
    market: NormalizedMarket,
    cache: PredictionCache,
) -> dict | None:
    """Run one search baseline on one market."""
    name = cfg["display_name"]
    existing = cache.load_existing(name)
    if market.id in existing:
        return existing[market.id]

    evaluator = SearchAugmentedEvaluator(
        model=cfg["model"],
        display_name=cfg["display_name"],
        max_tokens=cfg.get("max_tokens", 16384),
    )
    try:
        result = await evaluator.predict(market)
        pred = {
            "market_id": market.id,
            "platform": market.platform.value,
            "question": market.question,
            "model": name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **result,
        }
        cache.append(name, pred)
        return pred
    except Exception as e:
        log.warning(f"[{name}] Failed on {market.id}: {e}")
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
        cache.append(name, pred)
        return pred


async def run_agent_for_market(
    run_cfg: dict,
    market: NormalizedMarket,
    cache: PredictionCache,
) -> dict | None:
    """Run one ToolAgent on one market."""
    label = run_cfg["label"]
    cache_key = f"tool_agent_{label}"
    existing = cache.load_existing(cache_key)
    if market.id in existing:
        return existing[market.id]

    agent = ToolAgent(
        reasoning_model=run_cfg["reasoning_model"],
        temperature=run_cfg["temperature"],
    )
    try:
        result = await agent.predict(market)
        pred = {
            "market_id": market.id,
            "platform": market.platform.value,
            "question": market.question,
            "model": cache_key,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **result,
        }
        cache.append(cache_key, pred)
        return pred
    except Exception as e:
        log.warning(f"[{cache_key}] Failed on {market.id}: {e}")
        return None


async def run_critic_for_prediction(
    critic: Critic,
    market: NormalizedMarket,
    prediction: dict,
    cache: PredictionCache,
) -> dict:
    """Run critic on one agent prediction."""
    model_label = prediction.get("model", "unknown")
    cache_key = f"critic_{model_label}"
    existing = cache.load_existing(cache_key)

    if market.id in existing:
        cached = existing[market.id]
        prediction["critic"] = cached.get("critic", {})
        prediction["probability"] = cached.get("probability", prediction["probability"])
        return prediction

    review = await critic.review(market, prediction)

    prediction["critic"] = {
        "original_probability": review["original_probability"],
        "adjusted_probability": review["adjusted_probability"],
        "critique": review.get("critique", ""),
        "adjustment_reason": review.get("adjustment_reason", ""),
        "failure_modes": review.get("failure_modes", []),
    }
    prediction["probability"] = review["adjusted_probability"]
    prediction["timestamp"] = datetime.now(timezone.utc).isoformat()
    cache.append(cache_key, prediction)
    return prediction


async def run_judge_for_market(
    market: NormalizedMarket,
    run_results: list[dict],
    judge_model: str,
    cache: PredictionCache,
) -> dict | None:
    """Judge reads all predictions and makes final call."""
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
        response = await client.chat.completions.create(
            model=judge_model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=16384,
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
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "n_forecasters": len(run_results),
        }
        cache.append(cache_key, pred)
        return pred

    except Exception as e:
        log.warning(f"Judge failed for {market.id}: {e}")
        return None


async def process_one_market(
    market: NormalizedMarket,
    cache: PredictionCache,
    critic: Critic,
    judge_model: str,
) -> dict:
    """Run the FULL pipeline for one market: search baselines + agents + critics + judge."""
    market_start = datetime.now(timezone.utc)
    q = market.question[:60]

    # Phase 1: Run search baselines + all agents concurrently
    baseline_tasks = [
        run_search_baseline_for_market(cfg, market, cache)
        for cfg in SEARCH_BASELINES
    ]
    agent_tasks = [
        run_agent_for_market(run_cfg, market, cache)
        for run_cfg in AGENT_RUNS
    ]

    all_results = await asyncio.gather(*baseline_tasks, *agent_tasks)

    baseline_results = all_results[:len(SEARCH_BASELINES)]
    agent_results = [r for r in all_results[len(SEARCH_BASELINES):] if r and r.get("confidence") != "failed"]

    # Phase 2: Run critics on agent predictions
    if agent_results:
        critic_tasks = [
            run_critic_for_prediction(critic, market, pred, cache)
            for pred in agent_results
        ]
        critiqued = await asyncio.gather(*critic_tasks)
        agent_results = list(critiqued)

    # Phase 3: Judge makes final call
    judged = None
    if agent_results:
        judged = await run_judge_for_market(market, agent_results, judge_model, cache)

    # Build ensemble result
    if judged:
        ensemble = judged
        ensemble["n_runs"] = len(agent_results)
        ensemble["run_probs"] = [r["probability"] for r in agent_results]
    elif agent_results:
        probs = [r["probability"] for r in agent_results]
        ensemble = {
            "market_id": market.id,
            "platform": market.platform.value,
            "question": market.question,
            "probability": statistics.mean(probs),
            "confidence": "medium",
            "strategy": "mean_fallback",
            "reasoning": f"Mean of {len(probs)} runs (judge failed)",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "n_runs": len(probs),
            "run_probs": probs,
        }
    else:
        ensemble = {
            "market_id": market.id,
            "platform": market.platform.value,
            "question": market.question,
            "probability": market.market_probability,
            "confidence": "low",
            "strategy": "fallback",
            "reasoning": "All agent runs failed",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "n_runs": 0,
        }

    elapsed = (datetime.now(timezone.utc) - market_start).total_seconds()
    baseline_probs = [r["probability"] for r in baseline_results if r and r.get("confidence") != "failed"]
    agent_probs = [r["probability"] for r in agent_results] if agent_results else []

    log.info(
        f"[{q}] done in {elapsed:.0f}s | "
        f"market={market.market_probability:.0%} | "
        f"baselines={[f'{p:.0%}' for p in baseline_probs]} | "
        f"agents={[f'{p:.0%}' for p in agent_probs]} | "
        f"judge={ensemble['probability']:.0%}"
    )

    return ensemble


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    parser = argparse.ArgumentParser(description="Run full pipeline: search baselines + agent ensemble")
    parser.add_argument("--limit", type=int, default=None, help="Limit markets")
    parser.add_argument("--batch-size", type=int, default=10, help="Markets per batch (default: 10)")
    parser.add_argument("--judge-model", type=str, default="anthropic/claude-opus-4-6")
    parser.add_argument("--critic-model", type=str, default="anthropic/claude-sonnet-4-6")
    parser.add_argument("--top-volume", type=int, default=None,
                        help="Sort by volume and take top N markets")
    args = parser.parse_args()

    markets = load_processed_markets()

    # Sort by volume if requested
    if args.top_volume:
        markets.sort(key=lambda m: getattr(m, 'volume', 0) or 0, reverse=True)
        markets = markets[:args.top_volume]
    elif args.limit:
        markets = markets[:args.limit]

    log.info(f"=== STARTING FULL PIPELINE ===")
    log.info(f"Markets: {len(markets)} | Batch size: {args.batch_size}")
    log.info(f"Search baselines: {[c['display_name'] for c in SEARCH_BASELINES]}")
    log.info(f"Agent runs: {[r['label'] for r in AGENT_RUNS]}")
    log.info(f"Critic: {args.critic_model} | Judge: {args.judge_model}")

    cache = PredictionCache()
    critic = Critic(model=args.critic_model)

    all_ensembles = []
    total_start = datetime.now(timezone.utc)

    # Process in batches
    for batch_start in range(0, len(markets), args.batch_size):
        batch_end = min(batch_start + args.batch_size, len(markets))
        batch = markets[batch_start:batch_end]
        batch_num = batch_start // args.batch_size + 1
        total_batches = (len(markets) + args.batch_size - 1) // args.batch_size

        log.info(f"--- Batch {batch_num}/{total_batches} ({len(batch)} markets) ---")

        tasks = [
            process_one_market(m, cache, critic, args.judge_model)
            for m in batch
        ]
        batch_results = await asyncio.gather(*tasks)
        all_ensembles.extend(batch_results)

        log.info(f"--- Batch {batch_num} complete ({len(all_ensembles)}/{len(markets)} done) ---")

    # Save ensemble
    cache_key = "ensemble_agent"
    cache.save_all(cache_key, all_ensembles)

    elapsed = (datetime.now(timezone.utc) - total_start).total_seconds()
    strategies = {}
    for p in all_ensembles:
        s = p.get("strategy", "unknown")
        strategies[s] = strategies.get(s, 0) + 1

    log.info(f"=== PIPELINE COMPLETE ===")
    log.info(f"Total time: {elapsed:.0f}s ({elapsed/60:.1f}m)")
    log.info(f"Markets: {len(all_ensembles)} | Strategies: {strategies}")
    log.info(f"Saved to data/predictions/{cache_key}.jsonl")


if __name__ == "__main__":
    asyncio.run(main())
