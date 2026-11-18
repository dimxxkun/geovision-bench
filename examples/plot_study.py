"""Plot every paired seed; no confidence interval is inferred from three runs."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def plot(summary, output):
    report = json.loads(Path(summary).read_text(encoding="utf-8"))
    seeds = report["protocol"]["training_seeds"]
    rows = {(row["seed"], row["arm"]): row for row in report["runs"]}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), layout="constrained")
    for ax, metric, scale, label in (
        (axes[0], "accuracy", 100, "Held-out accuracy (%)"),
        (axes[1], "ece", 1, "Validation-calibrated test ECE"),
    ):
        for seed in seeds:
            values = [
                rows[(seed, arm)][metric] * scale for arm in ["no_augmentation", "augmentation"]
            ]
            ax.plot([0, 1], values, "o-", label=f"Seed {seed}", linewidth=1.8)
        ax.set_xticks([0, 1], ["No augmentation", "Orientation augmentation"])
        ax.set_ylabel(label)
        ax.spines[["top", "right"]].set_visible(False)
        ax.margins(x=0.15, y=0.25)
        ax.grid(axis="y", alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(
        f"EuroSAT · {len(seeds)} paired training seeds · one fixed image split\n"
        "Identical architecture and epoch budget; checkpoint selection uses validation NLL"
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot(args.summary, args.output)
