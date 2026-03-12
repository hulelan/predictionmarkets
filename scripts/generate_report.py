#!/usr/bin/env python3
"""Generate a summary report of all predictions and scores."""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.fetcher import load_processed_markets
from src.utils.cache import PredictionCache
from src.utils.logger import log


def main():
    markets = load_processed_markets()
    cache = PredictionCache()

    model_names = [
        "openai/gpt-5.2",
        "anthropic/claude-opus-4-6",
        "google/gemini-2.5-pro",
        "custom_agent",
    ]

    # Load all predictions
    all_preds = {}
    for model_name in model_names:
        existing = cache.load_existing(model_name)
        all_preds[model_name] = existing

    # Generate submission JSON
    submission = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_markets": len(markets),
        "models": model_names,
        "predictions": [],
    }

    for market in markets:
        entry = {
            "market_id": market.id,
            "platform": market.platform.value,
            "question": market.question,
            "market_probability": market.market_probability,
            "resolution_date": market.resolution_date.isoformat(),
            "volume": market.volume,
            "category": market.category,
            "model_predictions": {},
        }

        for model_name in model_names:
            pred = all_preds.get(model_name, {}).get(market.id)
            if pred:
                entry["model_predictions"][model_name] = {
                    "probability": pred.get("probability"),
                    "confidence": pred.get("confidence"),
                    "strategy": pred.get("strategy", ""),
                }

        submission["predictions"].append(entry)

    # Save
    Path("results").mkdir(exist_ok=True)
    with open("results/submission.json", "w") as f:
        json.dump(submission, f, indent=2)

    log.info(f"Submission saved: {len(submission['predictions'])} markets, {len(model_names)} models")

    # Print summary table
    print("\n=== PREDICTION SUMMARY ===")
    print(f"Markets: {len(markets)}")
    print(f"Generated: {submission['generated_at']}")
    print()

    for model_name in model_names:
        preds = all_preds.get(model_name, {})
        success = sum(1 for p in preds.values() if p.get("confidence") != "failed")
        probs = [p["probability"] for p in preds.values() if p.get("confidence") != "failed"]
        if probs:
            avg = sum(probs) / len(probs)
            print(f"{model_name}: {success}/{len(markets)} predictions, avg prob={avg:.3f}")
        else:
            print(f"{model_name}: no predictions")

    # Load scores if available
    scores_path = Path("results/scores.json")
    if scores_path.exists():
        scores = json.loads(scores_path.read_text())
        print("\n=== BRIER SCORES (lower is better) ===")
        for model_name, data in sorted(scores.items(), key=lambda x: x[1].get("brier", {}).get("overall", 999)):
            bs = data.get("brier", {}).get("overall", "N/A")
            ls = data.get("log_score", "N/A")
            n = data.get("brier", {}).get("n_markets", "?")
            print(f"  {model_name}: Brier={bs:.4f}, LogScore={ls:.4f}, N={n}" if isinstance(bs, float) else f"  {model_name}: {bs}")


if __name__ == "__main__":
    main()
