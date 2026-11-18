"""Fixed-split, paired-seed study; all arms are declared before any test evaluation."""

import json
import statistics
from pathlib import Path

from .data import digest, write_json
from .engine import evaluate, train


def summarize(rows):
    """Describe variation across training seeds, not geographic uncertainty."""
    summaries = {}
    for arm in ("augmentation", "no_augmentation"):
        selected = [row for row in rows if row["arm"] == arm]
        summaries[arm] = {
            metric: {
                "mean": statistics.mean(row[metric] for row in selected),
                "sample_std": statistics.stdev(row[metric] for row in selected),
            }
            for metric in ("accuracy", "macro_f1", "nll", "ece")
        }
    by_seed = {(row["seed"], row["arm"]): row for row in rows}
    differences = [
        by_seed[(seed, "augmentation")]["accuracy"] - by_seed[(seed, "no_augmentation")]["accuracy"]
        for seed in sorted({row["seed"] for row in rows})
    ]
    return {
        "arms": summaries,
        "paired_accuracy_deltas": differences,
        "paired_accuracy_delta_mean": statistics.mean(differences),
        "interpretation": "Descriptive seed variability on one fixed image split; not a confidence interval or spatial generalization estimate.",
    }


def run_study(root, manifest, output, seeds, epochs=12, threads=4):
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("Use at least two distinct training seeds")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Study output must be empty; choose a new directory")
    protocol = {
        "schema_version": 1,
        "training_seeds": seeds,
        "arms": ["augmentation", "no_augmentation"],
        "epochs": epochs,
        "threads": threads,
        "manifest_sha256": digest(manifest),
        "selection": "Lowest validation NLL within each run; calibration fits validation logits only.",
        "comparison": "Same split, architecture, optimizer, batch size and epoch budget; only orientation augmentation changes.",
    }
    write_json(output / "protocol.json", protocol)
    rows = []
    for seed in seeds:
        for arm in protocol["arms"]:
            directory = output / f"{arm}-{seed}"
            train(
                root,
                manifest,
                directory,
                seed=seed,
                epochs=epochs,
                threads=threads,
                augmentation=arm == "augmentation",
            )
            report = evaluate(root, manifest, directory / "best.pt", directory, threads=threads)
            row = {"seed": seed, "arm": arm, "selected_epoch": report["selected_epoch"]}
            row.update(
                {key: report["calibrated"][key] for key in ("accuracy", "macro_f1", "nll", "ece")}
            )
            rows.append(row)
            write_json(output / "runs.json", rows)
    result = {"protocol": protocol, "runs": rows, **summarize(rows)}
    write_json(output / "summary.json", result)
    print(json.dumps(result), flush=True)
    return result
