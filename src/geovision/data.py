"""Dataset ownership: download integrity, exact-duplicate groups, and immutable splits."""

import hashlib
import json
import random
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import TensorDataset

URL = "https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip?download=1"
MD5 = "f46e308c4d50d4bf32fedad2d3d62f3b"
CLASSES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]


def digest(path, algorithm="sha256"):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, algorithm).hexdigest()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    # Schema v1 uses explicit CRLF bytes so recorded manifest hashes also match on Unix.
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\r\n"
    )
    temporary.replace(path)


def extract_archive(archive, destination):
    """Reject unsafe names before extracting anything. Never extract archive symlinks."""
    destination = Path(destination).resolve()
    with zipfile.ZipFile(archive) as source:
        for info in source.infolist():
            target = (destination / info.filename).resolve()
            if not target.is_relative_to(destination) or ":" in info.filename:
                raise ValueError(f"Unsafe archive path: {info.filename}")
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Archive symlinks are not supported")
        source.extractall(destination)


def make_manifest(root, output, seed=2026, classes=None):
    root = Path(root).resolve()
    classes = sorted(classes or CLASSES)
    rows = []
    seen_labels = {}
    rng = random.Random(seed)
    duplicate_count = 0
    for label, name in enumerate(classes):
        groups = defaultdict(list)
        for path in sorted((root / name).glob("*.jpg")):
            sha = digest(path)
            if sha in seen_labels and seen_labels[sha] != name:
                raise ValueError(f"Conflicting labels for duplicate {path}")
            seen_labels[sha] = name
            groups[sha].append(path)
        keys = sorted(groups)
        if len(keys) < 10:
            raise ValueError(f"Need at least 10 distinct images in {name}")
        rng.shuffle(keys)
        first, second = int(len(keys) * 0.7), int(len(keys) * 0.85)
        for index, sha in enumerate(keys):
            split = "train" if index < first else "val" if index < second else "test"
            duplicate_count += len(groups[sha]) - 1
            for path in groups[sha]:
                rows.append(
                    {
                        "path": path.relative_to(root).as_posix(),
                        "label": label,
                        "sha256": sha,
                        "split": split,
                    }
                )
    rows.sort(key=lambda row: row["path"])
    manifest = {
        "schema": 1,
        "seed": seed,
        "classes": classes,
        "rows": rows,
        "split_method": "70/15/15 per class, grouped by exact file SHA-256",
        "exact_duplicate_copies": duplicate_count,
        "counts": dict(Counter(row["split"] for row in rows)),
        "limitation": "Not a geographic split. Near duplicates are not detected.",
    }
    write_json(output, manifest)
    return manifest


def prepare(directory, seed=2026):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    archive = directory / "EuroSAT_RGB.zip"
    if not archive.exists():
        temporary = archive.with_suffix(".part")
        urllib.request.urlretrieve(URL, temporary)
        temporary.replace(archive)
    if digest(archive, "md5") != MD5:
        raise ValueError("Archive checksum differs from the official Zenodo MD5")
    extract_archive(archive, directory / "extracted")
    roots = sorted(path.parent for path in (directory / "extracted").rglob("AnnualCrop"))
    if len(roots) != 1:
        raise ValueError(f"Expected one EuroSAT RGB root, found {len(roots)}")
    manifest_path = directory / "manifest.json"
    manifest = make_manifest(roots[0], manifest_path, seed)
    write_json(
        directory / "provenance.json",
        {
            "url": URL,
            "doi": "10.5281/zenodo.7711810",
            "archive_md5": MD5,
            "archive_sha256": digest(archive),
            "manifest_sha256": digest(manifest_path),
            "image_root": roots[0].relative_to(directory).as_posix(),
            "counts": manifest["counts"],
            "classes": manifest["classes"],
            "exact_duplicate_copies": manifest["exact_duplicate_copies"],
        },
    )
    return roots[0], manifest_path


def load_split(root, manifest, split, limit=None, verify=True):
    """Keep uint8 images resident; convert only each batch to float on its device."""
    root = Path(root).resolve()
    rows = [row for row in manifest["rows"] if row["split"] == split]
    if limit is not None:
        # Balanced deterministic smoke subset, never the published full-data evaluation.
        by_label = defaultdict(list)
        for row in rows:
            by_label[row["label"]].append(row)
        rows = [row for label in sorted(by_label) for row in by_label[label][:limit]]
    images, labels = [], []
    for row in rows:
        path = (root / row["path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Manifest path escapes image root")
        if verify and digest(path) != row["sha256"]:
            raise ValueError(f"Dataset changed: {row['path']}")
        with Image.open(path) as image:
            array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        if array.shape != (64, 64, 3):
            raise ValueError(f"Expected 64x64 RGB image: {row['path']}")
        images.append(torch.from_numpy(array).permute(2, 0, 1))
        labels.append(row["label"])
    if not images:
        raise ValueError(f"Empty split: {split}")
    return TensorDataset(torch.stack(images), torch.tensor(labels)), rows
