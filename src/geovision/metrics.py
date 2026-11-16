import torch
import torch.nn.functional as F


def classification_metrics(logits, labels, bins=15, temperature=1.0):
    if logits.ndim != 2 or labels.shape != (len(logits),) or len(logits) == 0:
        raise ValueError("Expected nonempty [N,C] logits and [N] labels")
    if not torch.isfinite(logits).all() or not 0 < temperature < float("inf"):
        raise ValueError("Logits and temperature must be finite; temperature must be positive")
    if bins < 1 or labels.min() < 0 or labels.max() >= logits.shape[1]:
        raise ValueError("Invalid bins or labels")
    probabilities = (logits.double() / temperature).softmax(1)
    confidence, prediction = probabilities.max(1)
    correct = prediction.eq(labels)
    classes = logits.shape[1]
    confusion = torch.bincount(labels * classes + prediction, minlength=classes * classes).reshape(
        classes, classes
    )
    tp = confusion.diag().double()
    support = confusion.sum(1)
    precision = tp / confusion.sum(0).clamp(min=1)
    recall = tp / support.clamp(min=1)
    f1 = 2 * tp / (confusion.sum(0) + support).clamp(min=1)
    reliability = []
    ece = 0.0
    indices = (confidence * bins).long().clamp(max=bins - 1)
    for index in range(bins):
        mask = indices == index
        count = int(mask.sum())
        accuracy = float(correct[mask].double().mean()) if count else None
        mean_confidence = float(confidence[mask].mean()) if count else None
        if count:
            ece += count / len(labels) * abs(accuracy - mean_confidence)
        reliability.append(
            {
                "lower": index / bins,
                "upper": (index + 1) / bins,
                "count": count,
                "accuracy": accuracy,
                "confidence": mean_confidence,
            }
        )
    one_hot = F.one_hot(labels, classes).double()
    return {
        "samples": len(labels),
        "accuracy": float(correct.double().mean()),
        "macro_f1": float(f1.mean()),
        "nll": float(F.cross_entropy(logits.double() / temperature, labels)),
        "ece": ece,
        "brier": float(((probabilities - one_hot) ** 2).sum(1).mean()),
        "temperature": temperature,
        "confusion_matrix": confusion.tolist(),
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1": f1.tolist(),
        "support": support.tolist(),
        "reliability": reliability,
    }


def fit_temperature(validation_logits, validation_labels):
    """Choose a temperature on validation NLL only; include 1.0 as a no-op baseline."""
    if not torch.isfinite(validation_logits).all() or len(validation_labels) == 0:
        raise ValueError("Calibration requires finite, nonempty validation logits")
    candidates = torch.cat((torch.logspace(-1, 1, 201, dtype=torch.float64), torch.ones(1)))
    losses = [
        float(F.cross_entropy(validation_logits.double() / t, validation_labels))
        for t in candidates
    ]
    return float(candidates[min(range(len(losses)), key=losses.__getitem__)])
