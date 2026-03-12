"""Weighted ensemble and extremization for combining forecasts."""

from __future__ import annotations

import math
import statistics


def ensemble_prediction(
    baseline_predictions: dict[str, float],
    weights: dict[str, float] | None = None,
) -> float:
    """Weighted average of baseline predictions."""
    if weights is None:
        weights = {k: 1.0 for k in baseline_predictions}

    total_weight = sum(weights.get(k, 1.0) for k in baseline_predictions)
    ensemble_prob = sum(
        baseline_predictions[model] * weights.get(model, 1.0)
        for model in baseline_predictions
    ) / total_weight

    return ensemble_prob


def extremize(prob: float, factor: float = math.sqrt(3)) -> float:
    """Push probabilities away from 50% to improve calibration.

    Research shows ensemble averages tend to be underconfident.
    Extremizing corrects this: transform via logit space with a scaling factor.

    new_p = sigmoid(factor * logit(p))

    factor > 1 extremizes (pushes away from 0.5)
    factor < 1 moderates (pushes toward 0.5)
    """
    if prob <= 0.01 or prob >= 0.99:
        return prob

    logit = math.log(prob / (1 - prob))
    extremized_logit = logit * factor
    result = 1 / (1 + math.exp(-extremized_logit))

    return max(0.01, min(0.99, result))


def platt_scale(prob: float, a: float, b: float) -> float:
    """Full Platt scaling with bias term.

    new_p = sigmoid(a * logit(p) + b)

    The bias term `b` corrects for systematic over/under-prediction,
    while `a` controls extremization strength (equivalent to `factor`
    in the simple extremize function when b=0).
    """
    if prob <= 0.01 or prob >= 0.99:
        return prob

    logit_p = math.log(prob / (1 - prob))
    scaled_logit = a * logit_p + b
    result = 1 / (1 + math.exp(-scaled_logit))

    return max(0.01, min(0.99, result))


def fit_platt_params(
    predictions: list[float], outcomes: list[int]
) -> tuple[float, float]:
    """Fit optimal Platt scaling parameters (a, b) from calibration data.

    Uses logistic regression on logit-transformed predictions vs binary outcomes.
    Requires scipy (optional dependency, imported locally).

    Args:
        predictions: List of predicted probabilities (0-1).
        outcomes: List of binary outcomes (0 or 1).

    Returns:
        (a, b) tuple of fitted Platt parameters.
    """
    import numpy as np
    from scipy.optimize import minimize

    predictions_arr = np.array(predictions, dtype=np.float64)
    outcomes_arr = np.array(outcomes, dtype=np.float64)

    # Clamp to avoid log(0)
    eps = 1e-6
    predictions_arr = np.clip(predictions_arr, eps, 1 - eps)

    logits = np.log(predictions_arr / (1 - predictions_arr))

    def neg_log_likelihood(params: np.ndarray) -> float:
        a, b = params
        scaled = a * logits + b
        # Numerically stable sigmoid
        log_sig = np.where(
            scaled >= 0,
            -np.log1p(np.exp(-scaled)),
            scaled - np.log1p(np.exp(scaled)),
        )
        log_1_minus_sig = np.where(
            scaled >= 0,
            -scaled - np.log1p(np.exp(-scaled)),
            -np.log1p(np.exp(scaled)),
        )
        nll = -np.sum(outcomes_arr * log_sig + (1 - outcomes_arr) * log_1_minus_sig)
        return float(nll)

    result = minimize(neg_log_likelihood, x0=np.array([1.0, 0.0]), method="Nelder-Mead")
    a_fit, b_fit = result.x
    return (float(a_fit), float(b_fit))


def trimmed_mean(values: list[float], trim_fraction: float = 0.2) -> float:
    """Trim top/bottom fraction of values and return mean of the rest.

    With default trim_fraction=0.2 and N=5, drops the min and max (1 each side).
    """
    if len(values) <= 2:
        return statistics.mean(values)

    sorted_vals = sorted(values)
    n_trim = max(1, int(len(sorted_vals) * trim_fraction))
    trimmed = sorted_vals[n_trim : len(sorted_vals) - n_trim]

    if not trimmed:
        # If trimming removed everything, fall back to full mean
        return statistics.mean(values)

    return statistics.mean(trimmed)


def ensemble_with_variance(
    baseline_runs: dict[str, list[float]],
) -> dict:
    """Aggregate multi-run baseline predictions with variance info.

    Args:
        baseline_runs: {model_name: [prob_run_0, prob_run_1, ...]}

    Returns:
        {"probability": float, "std": float, "per_model_std": {model: float}}
    """
    model_means = {}
    per_model_std = {}
    for model, runs in baseline_runs.items():
        if not runs:
            continue
        model_means[model] = statistics.mean(runs)
        per_model_std[model] = statistics.stdev(runs) if len(runs) >= 2 else 0.0

    if not model_means:
        return {"probability": 0.5, "std": 0.0, "per_model_std": {}}

    overall_prob = statistics.mean(model_means.values())
    # Cross-model std: spread of model means
    overall_std = (
        statistics.stdev(model_means.values()) if len(model_means) >= 2 else 0.0
    )

    return {
        "probability": overall_prob,
        "std": overall_std,
        "per_model_std": per_model_std,
    }
