#!/usr/bin/env python3
"""Score all predictions after markets resolve."""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

from src.data.fetcher import load_processed_markets
from src.scoring.brier import brier_score, brier_score_breakdown, log_score
from src.scoring.calibration import plot_calibration
from src.scoring.resolution import check_all_resolutions
from src.utils.cache import PredictionCache
from src.utils.logger import log


async def main():
    markets = load_processed_markets()
    log.info(f"Checking resolutions for {len(markets)} markets...")

    outcomes = await check_all_resolutions(markets)

    # Build market_id -> outcome mapping (only resolved markets)
    resolved = {}
    for market, outcome in zip(markets, outcomes):
        if outcome is not None:
            resolved[market.id] = outcome

    log.info(f"{len(resolved)} / {len(markets)} markets resolved")

    if not resolved:
        log.warning("No markets resolved yet. Try again later.")
        return

    # Score each model
    cache = PredictionCache()
    model_names = [
        # Baselines (no search)
        "openai/gpt-5.4",
        "anthropic/claude-opus-4-6",
        "google/gemini-2.5-pro",
        # Agent runs (with search)
        "tool_agent_claude_t03",
        "tool_agent_gpt_t03",
        "tool_agent_gemini_t03",
        "tool_agent_claude_t07",
        "tool_agent_gpt_t07",
        # Final ensemble
        "ensemble_agent",
    ]

    # Also score "market" as a baseline
    results = {}
    models_plot_data = {}

    # Market baseline
    market_preds = []
    market_outs = []
    market_cats = []
    market_lookup = {m.id: m for m in markets}
    for mid, outcome in resolved.items():
        m = market_lookup.get(mid)
        if m:
            market_preds.append(m.market_probability)
            market_outs.append(outcome)
            market_cats.append(m.category)

    if market_preds:
        bs = brier_score_breakdown(market_preds, market_outs, market_cats)
        ls = log_score(market_preds, market_outs)
        results["market_consensus"] = {"brier": bs, "log_score": ls}
        models_plot_data["Market Consensus"] = (market_preds, market_outs)
        log.info(f"Market Consensus: Brier={bs['overall']:.4f}, LogScore={ls:.4f}")

    # Each model
    for model_name in model_names:
        existing = cache.load_existing(model_name)
        if not existing:
            log.warning(f"No predictions found for {model_name}")
            continue

        preds = []
        outs = []
        cats = []
        for mid, outcome in resolved.items():
            if mid in existing:
                p = existing[mid].get("probability")
                if p is not None:
                    preds.append(float(p))
                    outs.append(outcome)
                    m = market_lookup.get(mid)
                    cats.append(m.category if m else "unknown")

        if not preds:
            log.warning(f"No matched predictions for {model_name}")
            continue

        bs = brier_score_breakdown(preds, outs, cats)
        ls = log_score(preds, outs)
        results[model_name] = {"brier": bs, "log_score": ls}
        models_plot_data[model_name] = (preds, outs)
        log.info(f"{model_name}: Brier={bs['overall']:.4f}, LogScore={ls:.4f}, N={len(preds)}")

    # Save results
    Path("results").mkdir(exist_ok=True)
    with open("results/scores.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Plot calibration
    if models_plot_data:
        plot_calibration(models_plot_data, "results/calibration.png")
        log.info("Calibration plot saved to results/calibration.png")

    log.info("Scoring complete!")


if __name__ == "__main__":
    asyncio.run(main())
