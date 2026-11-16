import json
import zipfile

import numpy as np
import pytest
import torch
from PIL import Image

from geovision.baseline import evaluate_baseline
from geovision.data import extract_archive, load_split, make_manifest
from geovision.engine import evaluate, load_model, train
from geovision.metrics import classification_metrics, fit_temperature


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "images"
    rng = np.random.default_rng(19)
    for label, name in enumerate(("crop", "water")):
        (root / name).mkdir(parents=True)
        for index in range(30):
            image = rng.integers(0, 40, (64, 64, 3), dtype=np.uint8)
            image[..., label] += 160
            Image.fromarray(image).save(root / name / f"{index:03}.jpg")
    return root


def test_splits_group_duplicates_are_reproducible_and_detect_drift(dataset, tmp_path):
    original = dataset / "crop/000.jpg"
    (dataset / "crop/copy.jpg").write_bytes(original.read_bytes())
    first = make_manifest(dataset, tmp_path / "first.json", classes=["crop", "water"])
    second = make_manifest(dataset, tmp_path / "second.json", classes=["crop", "water"])
    assert first == second
    assert first["exact_duplicate_copies"] == 1
    assignments = {}
    for row in first["rows"]:
        assert assignments.setdefault(row["sha256"], row["split"]) == row["split"]
    changed = first["rows"][0]
    (dataset / changed["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="Dataset changed"):
        load_split(dataset, first, changed["split"])


def test_conflicting_labels_fail(dataset, tmp_path):
    (dataset / "water/conflict.jpg").write_bytes((dataset / "crop/000.jpg").read_bytes())
    with pytest.raises(ValueError, match="Conflicting labels"):
        make_manifest(dataset, tmp_path / "manifest.json", classes=["crop", "water"])


def test_colour_baseline_learns_separable_colours(dataset, tmp_path):
    torch.set_num_threads(1)
    manifest = make_manifest(dataset, tmp_path / "manifest.json", classes=["crop", "water"])
    result = evaluate_baseline(dataset, manifest)
    assert result["samples"] == 10
    assert result["accuracy"] == 1.0
    assert result["macro_f1"] == 1.0


def test_archive_rejects_traversal_before_extracting(tmp_path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as file:
        file.writestr("normal.txt", "ok")
        file.writestr("../escaped.txt", "bad")
    with pytest.raises(ValueError, match="Unsafe archive"):
        extract_archive(archive, tmp_path / "out")
    assert not (tmp_path / "out/normal.txt").exists()


def test_metrics_against_hand_calculated_confusion_and_calibration():
    logits = torch.tensor([[10.0, 0], [10, 0], [0, 10], [0, 10]])
    labels = torch.tensor([0, 1, 1, 1])
    result = classification_metrics(logits, labels)
    assert result["confusion_matrix"] == [[1, 0], [1, 2]]
    assert result["accuracy"] == 0.75
    assert result["macro_f1"] == pytest.approx((2 / 3 + 4 / 5) / 2)
    temperature = fit_temperature(logits, labels)
    calibrated = classification_metrics(logits, labels, temperature=temperature)
    assert calibrated["nll"] < result["nll"]
    assert calibrated["accuracy"] == result["accuracy"]


def test_complete_train_checkpoint_evaluate_contract(dataset, tmp_path):
    manifest = tmp_path / "manifest.json"
    make_manifest(dataset, manifest, classes=["crop", "water"])
    run = tmp_path / "run"
    history = train(dataset, manifest, run, epochs=2, batch_size=16, width=4, threads=1)
    model, saved = load_model(run / "best.pt")
    assert not model.training
    assert saved["epoch"] in (1, 2)
    assert len(history) == 2
    repeated = tmp_path / "repeat"
    repeat_history = train(dataset, manifest, repeated, epochs=2, batch_size=16, width=4, threads=1)
    _, repeat_saved = load_model(repeated / "best.pt")
    for key, tensor in saved["state_dict"].items():
        torch.testing.assert_close(tensor, repeat_saved["state_dict"][key], rtol=0, atol=0)
    assert [row["train_loss"] for row in history] == [row["train_loss"] for row in repeat_history]
    result = evaluate(dataset, manifest, run / "best.pt", run, threads=1)
    assert result["uncalibrated"]["samples"] == 10
    assert (run / "confusion.png").exists()
    assert (run / "calibration.png").exists()
    parsed = json.loads(manifest.read_text())
    parsed["seed"] = 123
    manifest.write_text(json.dumps(parsed))
    with pytest.raises(ValueError, match="identities"):
        evaluate(dataset, manifest, run / "best.pt", run, threads=1)
