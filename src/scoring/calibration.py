"""Calibration analysis and plotting."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.scoring.brier import brier_score


def calibration_curve(
    predictions: list[float],
    outcomes: list[int],
    n_bins: int = 10,
) -> tuple[list[float], list[float], list[int]]:
    """Compute calibration curve data.

    Returns (bin_centers, bin_accuracies, bin_counts).
    """
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = []
    bin_accuracies = []
    bin_counts = []

    preds = np.array(predictions)
    outs = np.array(outcomes)

    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (preds >= bins[i]) & (preds <= bins[i + 1])
        else:
            mask = (preds >= bins[i]) & (preds < bins[i + 1])

        count = mask.sum()
        if count > 0:
            bin_centers.append(float(preds[mask].mean()))
            bin_accuracies.append(float(outs[mask].mean()))
            bin_counts.append(int(count))

    return bin_centers, bin_accuracies, bin_counts


def plot_calibration(
    models_data: dict[str, tuple[list[float], list[int]]],
    output_path: str = "results/calibration.png",
) -> None:
    """Plot calibration curves for all models.

    models_data: {model_name: (predictions, outcomes)}
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Calibration plot
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")

    colors = ["#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed"]
    for i, (model_name, (preds, outs)) in enumerate(models_data.items()):
        centers, accs, counts = calibration_curve(preds, outs)
        bs = brier_score(preds, outs)
        color = colors[i % len(colors)]
        ax1.plot(
            centers, accs, "o-",
            color=color,
            label=f"{model_name} (Brier={bs:.4f})",
            markersize=6,
        )

    ax1.set_xlabel("Predicted Probability")
    ax1.set_ylabel("Observed Frequency")
    ax1.set_title("Calibration Plot")
    ax1.legend(fontsize=8)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)

    # Brier score bar chart
    names = list(models_data.keys())
    scores = [brier_score(preds, outs) for preds, outs in models_data.values()]
    bar_colors = colors[: len(names)]
    bars = ax2.barh(names, scores, color=bar_colors)
    ax2.set_xlabel("Brier Score (lower is better)")
    ax2.set_title("Model Comparison")
    ax2.set_xlim(0, max(scores) * 1.3 if scores else 0.3)
    for bar, score in zip(bars, scores):
        ax2.text(score + 0.002, bar.get_y() + bar.get_height() / 2, f"{score:.4f}", va="center")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
