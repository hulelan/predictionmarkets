"""Brier score computation."""

from __future__ import annotations

from collections import defaultdict


def brier_score(predictions: list[float], outcomes: list[int]) -> float:
    """Compute mean Brier score. Lower is better.

    Perfect = 0.0, coin flip = 0.25, always wrong = 1.0.
    """
    assert len(predictions) == len(outcomes), "Predictions and outcomes must have same length"
    if not predictions:
        return float("nan")
    return sum((p - o) ** 2 for p, o in zip(predictions, outcomes)) / len(predictions)


def brier_score_breakdown(
    predictions: list[float],
    outcomes: list[int],
    categories: list[str] | None = None,
) -> dict:
    """Brier score with optional breakdown by category."""
    result = {
        "overall": brier_score(predictions, outcomes),
        "n_markets": len(predictions),
    }

    if categories:
        by_cat: dict[str, dict] = defaultdict(lambda: {"preds": [], "outs": []})
        for p, o, c in zip(predictions, outcomes, categories):
            by_cat[c]["preds"].append(p)
            by_cat[c]["outs"].append(o)

        result["by_category"] = {
            cat: {
                "brier": brier_score(d["preds"], d["outs"]),
                "n": len(d["preds"]),
            }
            for cat, d in sorted(by_cat.items())
        }

    return result


def log_score(predictions: list[float], outcomes: list[int]) -> float:
    """Compute mean log score. Higher (less negative) is better."""
    import math

    if not predictions:
        return float("nan")

    total = 0.0
    for p, o in zip(predictions, outcomes):
        p = max(0.001, min(0.999, p))  # avoid log(0)
        if o == 1:
            total += math.log(p)
        else:
            total += math.log(1 - p)

    return total / len(predictions)
