import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def make_plots(report, output):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    matrix = np.asarray(report["uncalibrated"]["confusion_matrix"])
    normalized = matrix / np.maximum(matrix.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(10, 8), layout="constrained")
    image = ax.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(len(matrix)), report["classes"], rotation=45, ha="right")
    ax.set_yticks(range(len(matrix)), report["classes"])
    for row in range(len(matrix)):
        for column in range(len(matrix)):
            ax.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color="white" if normalized[row, column] > 0.5 else "#18334a",
                fontsize=8,
            )
    ax.set(
        xlabel="Predicted class",
        ylabel="Reference label",
        title="EuroSAT RGB · held-out test confusion\nCounts annotated; colour is row-normalized",
    )
    fig.colorbar(image, ax=ax, label="Fraction of reference class")
    fig.savefig(output / "confusion.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for ax, key, title in zip(
        axes, ("uncalibrated", "calibrated"), ("Before calibration", "After calibration")
    ):
        bins = [b for b in report[key]["reliability"] if b["count"]]
        ax.plot([0, 1], [0, 1], "--", color="#8393a0")
        ax.plot(
            [b["confidence"] for b in bins], [b["accuracy"] for b in bins], "o-", color="#167d9a"
        )
        ax.set(
            xlim=(0, 1),
            ylim=(0, 1),
            xlabel="Mean confidence",
            ylabel="Observed accuracy",
            title=f"{title}\nECE = {report[key]['ece']:.4f}",
        )
        ax.grid(alpha=0.15)
    fig.savefig(output / "calibration.png", dpi=160)
    plt.close(fig)
