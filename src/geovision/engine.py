import json
import os
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import digest, load_split, write_json
from .metrics import classification_metrics, fit_temperature
from .model import LandCoverCNN, augment, preprocess


def configure(seed, threads):
    if threads < 1:
        raise ValueError("threads must be positive")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


def environment():
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "device": "cpu",
    }


@torch.inference_mode()
def collect_logits(model, loader):
    model.eval()
    logits, labels = [], []
    for images, targets in loader:
        logits.append(model(preprocess(images)).cpu())
        labels.append(targets)
    return torch.cat(logits), torch.cat(labels)


def train(
    root,
    manifest_path,
    output,
    epochs=12,
    batch_size=128,
    seed=2026,
    threads=4,
    learning_rate=0.002,
    width=24,
    limit=None,
):
    if epochs < 1 or batch_size < 1 or width < 1 or learning_rate <= 0:
        raise ValueError("epochs, batch size, width and learning rate must be positive")
    output = Path(output)
    if (output / "best.pt").exists():
        raise ValueError("Output already contains a checkpoint; choose a new run directory")
    configure(seed, threads)
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    started = time.perf_counter()
    training, _ = load_split(root, manifest, "train", limit)
    validation, _ = load_split(root, manifest, "val", limit)
    generator = torch.Generator().manual_seed(seed)
    augment_generator = torch.Generator().manual_seed(seed + 1)
    train_loader = DataLoader(training, batch_size=batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(validation, batch_size=batch_size)
    model = LandCoverCNN(len(manifest["classes"]), width)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    config = {
        "epochs": epochs,
        "batch_size": batch_size,
        "seed": seed,
        "learning_rate": learning_rate,
        "width": width,
        "limit_per_class": limit,
        "manifest_sha256": digest(manifest_path),
        "classes": manifest["classes"],
        "normalization": "uint8 / 255, then (x - 0.5) / 0.5",
        "parameters": sum(p.numel() for p in model.parameters()),
    }
    write_json(output / "config.json", config)
    write_json(output / "environment.json", environment())
    history = []
    best_loss = float("inf")
    cpu_started = time.process_time()
    for epoch in range(1, epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        total_loss, total_correct, count = 0.0, 0, 0
        for images, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(augment(preprocess(images), augment_generator))
            loss = nn.functional.cross_entropy(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            optimizer.step()
            total_loss += float(loss.detach()) * len(labels)
            total_correct += int((logits.argmax(1) == labels).sum())
            count += len(labels)
        val_logits, val_labels = collect_logits(model, val_loader)
        metrics = classification_metrics(val_logits, val_labels)
        record = {
            "epoch": epoch,
            "train_loss": total_loss / count,
            "train_accuracy": total_correct / count,
            "val_nll": metrics["nll"],
            "val_accuracy": metrics["accuracy"],
            "val_macro_f1": metrics["macro_f1"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "seconds": time.perf_counter() - epoch_started,
        }
        if metrics["nll"] < best_loss:
            best_loss = metrics["nll"]
            temporary = output / "best.tmp"
            torch.save(
                {"state_dict": model.state_dict(), "config": config, "epoch": epoch}, temporary
            )
            temporary.replace(output / "best.pt")
        scheduler.step()
        history.append(record)
        write_json(output / "history.json", history)
        print(json.dumps(record), flush=True)
    write_json(
        output / "timing.json",
        {
            "wall_seconds_including_data_load": time.perf_counter() - started,
            "training_process_cpu_seconds": time.process_time() - cpu_started,
        },
    )
    return history


def load_model(checkpoint):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = saved["config"]
    model = LandCoverCNN(len(config["classes"]), config["width"])
    model.load_state_dict(saved["state_dict"])
    model.eval()
    return model, saved


def evaluate(root, manifest_path, checkpoint, output, batch_size=256, threads=4):
    configure(2026, threads)
    model, saved = load_model(checkpoint)
    if saved["config"]["manifest_sha256"] != digest(manifest_path):
        raise ValueError("Checkpoint and manifest identities do not match")
    if saved["config"]["limit_per_class"] is not None:
        raise ValueError("Smoke checkpoints cannot generate a full benchmark report")
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    validation, _ = load_split(root, manifest, "val")
    test, test_rows = load_split(root, manifest, "test")
    validation_logits, validation_labels = collect_logits(model, DataLoader(validation, batch_size))
    temperature = fit_temperature(validation_logits, validation_labels)
    logits, labels = collect_logits(model, DataLoader(test, batch_size))
    report = {
        "classes": manifest["classes"],
        "selected_epoch": saved["epoch"],
        "checkpoint_sha256": digest(checkpoint),
        "manifest_sha256": digest(manifest_path),
        "environment": environment(),
        "calibration_source": "validation split only",
        "uncalibrated": classification_metrics(logits, labels),
        "calibrated": classification_metrics(logits, labels, temperature=temperature),
    }
    write_json(Path(output) / "metrics.json", report)
    probabilities = (logits / temperature).softmax(1)
    confidence, prediction = probabilities.max(1)
    errors = sorted(
        (i for i in range(len(labels)) if prediction[i] != labels[i]),
        key=lambda i: float(confidence[i]),
        reverse=True,
    )[:24]
    write_json(
        Path(output) / "confident_errors.json",
        [
            {
                "path": test_rows[i]["path"],
                "actual": manifest["classes"][labels[i]],
                "predicted": manifest["classes"][prediction[i]],
                "confidence": float(confidence[i]),
            }
            for i in errors
        ],
    )
    from .plots import make_plots

    make_plots(report, Path(output))
    print(
        json.dumps(
            {
                key: {
                    metric: report[key][metric] for metric in ("accuracy", "macro_f1", "nll", "ece")
                }
                for key in ("uncalibrated", "calibrated")
            }
        ),
        flush=True,
    )
    return report
