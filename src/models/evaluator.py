"""Run all baseline LLM evaluations via OpenRouter.

Parallelization strategy:
- All models (baseline, search-augmented, multi-run) launch concurrently
- Each model runs up to `concurrency` markets in parallel (default 50)
- Total in-flight requests = n_models × concurrency
- OpenRouter 429s are handled by retry logic in base.py
"""

from __future__ import annotations

import asyncio

from src.data.normalizer import NormalizedMarket
from src.models.openrouter_model import OpenRouterEvaluator
from src.models.search_evaluator import SearchAugmentedEvaluator
from src.utils.logger import log

# OpenRouter model IDs mapped to display names matching downstream expectations
BASELINE_MODELS = {
    "openai": {
        "model": "openai/gpt-5.4",
        "display_name": "openai/gpt-5.4",
        "concurrency": 50,
        "max_tokens": 16384,  # max: 128k
    },
    "anthropic": {
        "model": "anthropic/claude-opus-4.6",
        "display_name": "anthropic/claude-opus-4-6",
        "concurrency": 50,
        "max_tokens": 16384,  # max: 128k
    },
    "gemini": {
        "model": "google/gemini-2.5-pro",
        "display_name": "google/gemini-2.5-pro",
        "concurrency": 50,
        "max_tokens": 32768,  # max: 65k — needs extra for thinking tokens
    },
}


async def _run_one_model(
    cfg: dict,
    markets: list[NormalizedMarket],
) -> tuple[str, list[dict]]:
    """Run a single model on all markets. Returns (model_name, predictions)."""
    evaluator = OpenRouterEvaluator(
        model=cfg["model"],
        display_name=cfg["display_name"],
        max_tokens=cfg.get("max_tokens", 500),
    )
    name = evaluator.model_name()
    log.info(f"Running {name} on {len(markets)} markets...")

    predictions = await evaluator.evaluate_batch(markets, concurrency=cfg["concurrency"])

    probs = [p["probability"] for p in predictions if p.get("confidence") != "failed"]
    failed = sum(1 for p in predictions if p.get("confidence") == "failed")
    log.info(
        f"[{name}] Done: {len(probs)} success, {failed} failed. "
        f"Mean prob: {sum(probs)/len(probs):.3f}" if probs else f"[{name}] All failed"
    )

    return name, predictions


async def run_all_baselines(
    markets: list[NormalizedMarket],
    models: list[str] | None = None,
) -> dict[str, list[dict]]:
    """Run all baseline models on all markets in parallel.

    Returns {model_name: [prediction_dicts]}.
    """
    configs = BASELINE_MODELS
    if models:
        configs = {k: v for k, v in configs.items() if k in models}

    # Run all models concurrently
    tasks = [_run_one_model(cfg, markets) for cfg in configs.values()]
    results = await asyncio.gather(*tasks)

    return dict(results)


# Search-augmented baselines: same models, single Sonar search per market
SEARCH_BASELINE_MODELS = {
    "search_openai": {
        "model": "openai/gpt-5.4",
        "display_name": "search_openai/gpt-5.4",
        "concurrency": 50,
        "max_tokens": 16384,
    },
    "search_anthropic": {
        "model": "anthropic/claude-opus-4.6",
        "display_name": "search_anthropic/claude-opus-4-6",
        "concurrency": 50,
        "max_tokens": 16384,
    },
    "search_gemini": {
        "model": "google/gemini-2.5-pro",
        "display_name": "search_google/gemini-2.5-pro",
        "concurrency": 50,
        "max_tokens": 32768,
    },
}


async def _run_one_search_model(
    cfg: dict,
    markets: list[NormalizedMarket],
) -> tuple[str, list[dict]]:
    """Run a single search-augmented model on all markets."""
    evaluator = SearchAugmentedEvaluator(
        model=cfg["model"],
        display_name=cfg["display_name"],
        max_tokens=cfg.get("max_tokens", 500),
    )
    name = evaluator.model_name()
    log.info(f"Running {name} (search-augmented) on {len(markets)} markets...")

    predictions = await evaluator.evaluate_batch(markets, concurrency=cfg["concurrency"])

    probs = [p["probability"] for p in predictions if p.get("confidence") != "failed"]
    failed = sum(1 for p in predictions if p.get("confidence") == "failed")
    log.info(
        f"[{name}] Done: {len(probs)} success, {failed} failed. "
        f"Mean prob: {sum(probs)/len(probs):.3f}" if probs else f"[{name}] All failed"
    )

    return name, predictions


async def run_all_search_baselines(
    markets: list[NormalizedMarket],
    models: list[str] | None = None,
) -> dict[str, list[dict]]:
    """Run all search-augmented baseline models on all markets.

    Returns {model_name: [prediction_dicts]}.
    """
    configs = SEARCH_BASELINE_MODELS
    if models:
        configs = {k: v for k, v in configs.items() if k in models}

    tasks = [_run_one_search_model(cfg, markets) for cfg in configs.values()]
    results = await asyncio.gather(*tasks)

    return dict(results)


async def run_everything(
    markets: list[NormalizedMarket],
) -> dict[str, list[dict] | dict]:
    """Run search-augmented baselines concurrently.

    Every model sees the same markets at roughly the same time.
    Returns {model_name: predictions}.
    """
    tasks = []

    for cfg in SEARCH_BASELINE_MODELS.values():
        tasks.append(_run_one_search_model(cfg, markets))

    log.info(
        f"Launching {len(tasks)} search-augmented baselines "
        f"on {len(markets)} markets"
    )
    results = await asyncio.gather(*tasks)

    return dict(results)
