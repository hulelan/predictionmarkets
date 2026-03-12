"""Multi-run ensembling: run each forecast N times and aggregate."""

import statistics

from src.agent.ensemble import trimmed_mean
from src.data.normalizer import NormalizedMarket
from src.models.openrouter_model import OpenRouterEvaluator
from src.utils.logger import log


async def multi_run_predictions(
    model: str,
    display_name: str,
    markets: list[NormalizedMarket],
    n_runs: int = 3,
    temperature: float = 0.7,
    concurrency: int = 5,
) -> dict[str, dict]:
    """Run N independent predictions per market and aggregate.

    Returns {market_id: {"probability": float, "std": float, "runs": list[float], "aggregation": str}}.
    """
    # Run each pass with a distinct cache key: display_name_run_0, display_name_run_1, ...
    all_run_results: list[list[dict]] = []
    for i in range(n_runs):
        run_name = f"{display_name}_run_{i}"
        evaluator = OpenRouterEvaluator(
            model=model,
            display_name=run_name,
            temperature=temperature,
        )
        log.info(f"Starting run {i+1}/{n_runs} for {display_name} (cache key: {run_name})")
        results = await evaluator.evaluate_batch(markets, concurrency=concurrency)
        all_run_results.append(results)

    # Index predictions by market_id across runs
    market_runs: dict[str, list[float]] = {}
    for run_results in all_run_results:
        for pred in run_results:
            mid = pred["market_id"]
            if pred.get("confidence") == "failed":
                continue
            market_runs.setdefault(mid, []).append(pred["probability"])

    # Aggregate
    use_trimmed = n_runs >= 5
    agg_method = "trimmed_mean" if use_trimmed else "mean"

    aggregated: dict[str, dict] = {}
    for mid, runs in market_runs.items():
        if not runs:
            continue
        if use_trimmed and len(runs) >= 5:
            prob = trimmed_mean(runs)
        else:
            prob = statistics.mean(runs)

        std = statistics.stdev(runs) if len(runs) >= 2 else 0.0
        aggregated[mid] = {
            "probability": prob,
            "std": std,
            "runs": runs,
            "aggregation": agg_method,
        }

    log.info(
        f"[{display_name}] Aggregated {len(aggregated)} markets "
        f"across {n_runs} runs (method: {agg_method})"
    )
    return aggregated
