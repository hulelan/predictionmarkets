#!/usr/bin/env python3
"""Run all baseline LLM evaluations on fetched markets."""

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

from src.data.fetcher import load_processed_markets
from src.models.evaluator import BASELINE_MODELS, run_all_baselines, run_all_search_baselines, run_everything
from src.utils.logger import log


async def main():
    parser = argparse.ArgumentParser(description="Run LLM baselines")
    parser.add_argument("--models", nargs="+", default=None, help="Models to run: openai anthropic gemini")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of markets (for testing)")
    parser.add_argument("--search", action="store_true",
                        help="Run search-augmented baselines (each model gets 1 Perplexity Sonar search)")
    parser.add_argument("--all", action="store_true",
                        help="Run everything in parallel: baselines + search-augmented")
    args = parser.parse_args()

    markets = load_processed_markets()
    if args.limit:
        markets = markets[: args.limit]

    if args.all:
        log.info(f"Running EVERYTHING in parallel on {len(markets)} markets...")
        results = await run_everything(markets)
        for name, preds in results.items():
            if isinstance(preds, list):
                success = sum(1 for p in preds if p.get("confidence") != "failed")
                log.info(f"{name}: {success}/{len(preds)} successful")
            else:
                log.info(f"{name}: {len(preds)} markets aggregated")
        return

    if args.search:
        log.info(f"Running search-augmented baselines on {len(markets)} markets...")
        search_models = None
        if args.models:
            search_models = [f"search_{m}" for m in args.models]
        predictions = await run_all_search_baselines(markets, models=search_models)

        for model_name, preds in predictions.items():
            success = sum(1 for p in preds if p.get("confidence") != "failed")
            log.info(f"{model_name}: {success}/{len(preds)} successful predictions")
        return

    log.info(f"Running baselines on {len(markets)} markets...")
    predictions = await run_all_baselines(markets, models=args.models)

    for model_name, preds in predictions.items():
        success = sum(1 for p in preds if p.get("confidence") != "failed")
        log.info(f"{model_name}: {success}/{len(preds)} successful predictions")


if __name__ == "__main__":
    asyncio.run(main())
