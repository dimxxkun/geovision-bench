import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .data import prepare
from .engine import configure, evaluate, load_model, train
from .model import preprocess


def main():
    parser = argparse.ArgumentParser(description="Reproduce a satellite classification experiment")
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser(
        "prepare", help="Download, verify, extract and split EuroSAT RGB"
    )
    download.add_argument("--data", type=Path, default=Path("data"))
    download.add_argument("--seed", type=int, default=2026)
    training = commands.add_parser("train")
    evaluation = commands.add_parser("evaluate")
    for command in (training, evaluation):
        command.add_argument("--root", type=Path, required=True)
        command.add_argument("--manifest", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--batch-size", type=int, default=128)
        command.add_argument("--threads", type=int, default=4)
    training.add_argument("--epochs", type=int, default=12)
    training.add_argument("--seed", type=int, default=2026)
    training.add_argument("--learning-rate", type=float, default=0.002)
    training.add_argument("--width", type=int, default=24)
    training.add_argument("--limit", type=int, help="Smoke only: max images per class per split")
    training.add_argument("--no-augmentation", dest="augmentation", action="store_false")
    evaluation.add_argument("--checkpoint", type=Path, required=True)
    baseline = commands.add_parser(
        "baseline", help="Fit and evaluate a fixed colour-statistics baseline"
    )
    baseline.add_argument("--root", type=Path, required=True)
    baseline.add_argument("--manifest", type=Path, required=True)
    baseline.add_argument("--output", type=Path, required=True)
    baseline.add_argument("--threads", type=int, default=4)
    study = commands.add_parser("study", help="Paired multi-seed augmentation comparison")
    study.add_argument("--root", type=Path, required=True)
    study.add_argument("--manifest", type=Path, required=True)
    study.add_argument("--output", type=Path, required=True)
    study.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028])
    study.add_argument("--epochs", type=int, default=12)
    study.add_argument("--threads", type=int, default=4)
    prediction = commands.add_parser("predict")
    prediction.add_argument("--checkpoint", type=Path, required=True)
    prediction.add_argument("--image", type=Path, required=True)
    prediction.add_argument("--temperature", type=float, default=1.0)
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "prepare":
        root, manifest = prepare(args["data"], args["seed"])
        print(json.dumps({"root": str(root), "manifest": str(manifest)}))
    elif command in ("train", "evaluate"):
        args["manifest_path"] = args.pop("manifest")
        (train if command == "train" else evaluate)(**args)
    elif command == "study":
        from .study import run_study

        run_study(**args)
    elif command == "baseline":
        from .baseline import evaluate_baseline
        from .data import write_json

        configure(2026, args["threads"])
        manifest = json.loads(args["manifest"].read_text(encoding="utf-8"))
        result = evaluate_baseline(args["root"], manifest)
        write_json(args["output"], result)
        print(json.dumps(result), flush=True)
    else:
        configure(2026, 4)
        if not 0 < args["temperature"] < float("inf"):
            raise ValueError("temperature must be finite and positive")
        model, saved = load_model(args["checkpoint"])
        with Image.open(args["image"]) as image:
            if image.size != (64, 64):
                raise ValueError(
                    "This model expects a 64x64 RGB satellite patch; no automatic resizing"
                )
            array = np.asarray(image.convert("RGB")).copy()
        tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
        with torch.inference_mode():
            probabilities = (model(preprocess(tensor)) / args["temperature"]).softmax(1)[0]
        print(json.dumps(dict(zip(saved["config"]["classes"], probabilities.tolist())), indent=2))


if __name__ == "__main__":
    main()
