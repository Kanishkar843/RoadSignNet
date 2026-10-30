"""Official GTSRB/Kaggle adapters with track-disjoint validation and hash audits."""

import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .labels import NUM_CLASSES
from .preprocess import preprocess_image

SUFFIXES = {".ppm", ".png", ".jpg", ".jpeg"}


def _record(path: Path, class_id: int, group: str | None = None) -> dict:
    if not 0 <= class_id < NUM_CLASSES:
        raise ValueError(f"Class ID outside 0..42: {class_id}")
    if not path.is_file():
        raise ValueError(f"Missing image: {path}")
    return {"path": str(path.resolve()), "class_id": class_id, "group": group,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _track(path: Path, class_id: int) -> str:
    # Official: track_frame.ppm; common Kaggle repack: class_track_frame.png.
    match = re.fullmatch(r"(?:(\d{5})_)?(\d{5})_(\d{5})", path.stem)
    if not match:
        raise ValueError(f"Cannot identify GTSRB track from {path.name}; "
                         "preserve original track/frame filenames.")
    if match[1] is not None and int(match[1]) != class_id:
        raise ValueError(f"Filename class disagrees with directory: {path}")
    return f"{class_id}:{match[2]}"


def discover_training(root: str | Path) -> list[dict]:
    root = Path(root)
    candidates = [root / "GTSRB/Final_Training/Images", root / "Final_Training/Images",
                  root / "Train", root / "train", root]
    folder = next((p for p in candidates if (p / "00000").is_dir() or (p / "0").is_dir()), None)
    if folder is None:
        raise ValueError("No GTSRB class folders found. See docs/DATASET.md.")
    records = []
    for class_id in range(NUM_CLASSES):
        class_dir = folder / f"{class_id:05d}"
        if not class_dir.is_dir():
            class_dir = folder / str(class_id)
        paths = sorted(p for p in class_dir.glob("*") if p.suffix.lower() in SUFFIXES)
        if not paths:
            raise ValueError(f"Missing training images for class {class_id}.")
        records.extend(_record(p, class_id, _track(p, class_id)) for p in paths)
    return records


def split_training(records: list[dict], fraction: float = 0.2, seed: int = 42):
    if not 0 < fraction < 1:
        raise ValueError("Validation fraction must be between 0 and 1.")
    rng = np.random.default_rng(seed)
    training, validation = [], []
    for class_id in range(NUM_CLASSES):
        samples = [r for r in records if r["class_id"] == class_id]
        groups = sorted({r["group"] for r in samples})
        if len(groups) < 2:
            raise ValueError(f"Class {class_id} needs at least two distinct sign tracks.")
        n_val = max(1, min(len(groups) - 1, round(len(groups) * fraction)))
        chosen = set(rng.choice(groups, size=n_val, replace=False))
        training.extend(r for r in samples if r["group"] not in chosen)
        validation.extend(r for r in samples if r["group"] in chosen)
    assert_disjoint(training, validation)
    return training, validation


def assert_disjoint(left: list[dict], right: list[dict]):
    if {r["sha256"] for r in left} & {r["sha256"] for r in right}:
        raise ValueError("Identical image bytes occur across dataset partitions.")
    if {r["group"] for r in left if r.get("group")} & {r["group"] for r in right if r.get("group")}:
        raise ValueError("Sign tracks overlap across dataset partitions.")


def discover_test(root: str | Path, csv_path: str | Path) -> list[dict]:
    root, csv_path = Path(root).resolve(), Path(csv_path).resolve()
    table = pd.read_csv(csv_path, sep=None, engine="python")
    if "ClassId" not in table or not ({"Path", "Filename"} & set(table.columns)):
        raise ValueError("Test CSV needs ClassId and Path or Filename columns.")
    records = []
    for _, row in table.iterrows():
        relative = Path(str(row["Path"] if "Path" in table else row["Filename"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Test image paths must be relative without '..'.")
        candidates = [root / relative, csv_path.parent / relative,
                      root / "GTSRB/Final_Test/Images" / relative,
                      root / "Final_Test/Images" / relative, root / "Test" / relative]
        path = next((p for p in candidates if p.is_file()), None)
        if path is None:
            raise ValueError(f"Cannot find test image: {relative}")
        value = float(row["ClassId"])
        if not np.isfinite(value) or not value.is_integer():
            raise ValueError("Class IDs must be finite integers.")
        records.append(_record(path, int(value)))
    if not records:
        raise ValueError("Test CSV is empty.")
    if len({r["path"] for r in records}) != len(records):
        raise ValueError("Test CSV contains repeated image paths.")
    return records


def load_arrays(records: list[dict]):
    images = np.empty((len(records), 32, 32, 3), dtype=np.float32)
    for i, record in enumerate(records):
        images[i] = preprocess_image(record["path"])
    return images, np.asarray([r["class_id"] for r in records], dtype=np.int32)
