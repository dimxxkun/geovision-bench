"""A deterministic colour-statistics ridge baseline, fit on the training split only."""

import torch
from torch.utils.data import DataLoader

from .data import load_split
from .metrics import classification_metrics


def colour_features(images):
    values = images.double().div(255)
    mean = values.mean((-2, -1))
    deviation = values.std((-2, -1))
    # Include a 2x2 spatial mean: a cheap representation with no learned convolution.
    quadrant = torch.nn.functional.adaptive_avg_pool2d(values, (2, 2)).flatten(1)
    return torch.cat((mean, deviation, quadrant), dim=1)


def evaluate_baseline(root, manifest, ridge=1.0):
    training, _ = load_split(root, manifest, "train")
    test, _ = load_split(root, manifest, "test")

    def features(dataset):
        values, labels = [], []
        for images, targets in DataLoader(dataset, batch_size=256):
            values.append(colour_features(images))
            labels.append(targets)
        return torch.cat(values), torch.cat(labels)

    x_train, y_train = features(training)
    x_test, y_test = features(test)
    mean, scale = x_train.mean(0), x_train.std(0).clamp_min(1e-8)
    x_train = torch.cat(((x_train - mean) / scale, torch.ones((len(x_train), 1))), 1)
    x_test = torch.cat(((x_test - mean) / scale, torch.ones((len(x_test), 1))), 1)
    targets = torch.nn.functional.one_hot(y_train, len(manifest["classes"])).double()
    penalty = torch.eye(x_train.shape[1]) * ridge
    penalty[-1, -1] = 0  # Do not penalize the intercept.
    weights = torch.linalg.solve(x_train.T @ x_train + penalty, x_train.T @ targets)
    metrics = classification_metrics(x_test @ weights, y_test)
    return {
        "method": "ridge classifier on 18 colour/spatial statistics; train-only standardization",
        "ridge": ridge,
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "confusion_matrix": metrics["confusion_matrix"],
        "samples": metrics["samples"],
        "note": "Scores are regression outputs; no probability calibration claim.",
    }
