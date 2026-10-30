"""Train on official training images; never read the held-out test labels."""

import argparse
import importlib.metadata
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

from .artifacts import digest
from .data import discover_training, load_arrays, split_training
from .labels import CLASS_NAMES
from .model import build_model, configure_runtime, dataset
from .preprocess import PREPROCESSING


def train(data_dir, output, epochs=20, batch_size=64, seed=42, validation_fraction=0.2, augment=True):
    if epochs < 1 or batch_size < 1:
        raise ValueError("Epochs and batch size must be positive.")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose an empty output directory to preserve existing experiments.")
    configure_runtime(seed)
    train_rows, val_rows = split_training(discover_training(data_dir), validation_fraction, seed)
    train_x, train_y = load_arrays(train_rows)
    val_x, val_y = load_arrays(val_rows)
    output.mkdir(parents=True, exist_ok=True)
    # Keep hashes/groups, relative filenames and labels for portable leakage checks.
    data_root = Path(data_dir).resolve()
    manifest = {name: [{**r, "path": str(Path(r["path"]).relative_to(data_root))} for r in rows]
                for name, rows in [("train", train_rows), ("validation", val_rows)]}
    (output / "split_manifest.json").write_text(json.dumps(manifest, indent=2))
    model = build_model(augment)
    model.summary()
    started = time.perf_counter()
    history = model.fit(
        dataset(train_x, train_y, batch_size, True, seed),
        validation_data=dataset(val_x, val_y, batch_size), epochs=epochs, shuffle=False,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
                   tf.keras.callbacks.ModelCheckpoint(str(output / "model.keras"), monitor="val_loss", save_best_only=True),
                   tf.keras.callbacks.CSVLogger(str(output / "history.csv"))], verbose=2,
    )
    elapsed = time.perf_counter() - started
    metadata = {
        "schema_version": 1, "project": "RoadSignNet", "class_names": CLASS_NAMES,
        "preprocessing": PREPROCESSING, "seed": seed, "augmentation": augment,
        "validation_fraction_requested": validation_fraction, "split_method": "per-class sign-track holdout",
        "train_samples": len(train_rows), "validation_samples": len(val_rows),
        "epochs_requested": epochs, "epochs_completed": len(history.epoch), "batch_size": batch_size,
        "training_seconds": elapsed, "created_at": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(), "platform": platform.platform(),
        "devices": [device.name for device in tf.config.list_physical_devices()],
        "versions": {name: importlib.metadata.version(name) for name in
                     ["keras", "numpy", "opencv-python-headless", "scikit-learn", "pandas", "pillow"]},
        "tensorflow_version": tf.__version__, "model_sha256": digest(output / "model.keras"),
        "split_manifest_sha256": digest(output / "split_manifest.json"),
        "evaluation_status": "Not evaluated on held-out GTSRB test data",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    frame = pd.DataFrame(history.history)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric in zip(axes, ["accuracy", "loss"]):
        ax.plot(frame[metric], label="Training")
        ax.plot(frame[f"val_{metric}"], label="Validation")
        ax.set(xlabel="Epoch (zero-based)", ylabel=metric, title=f"RoadSignNet · {metric}")
        ax.legend()
    fig.tight_layout()
    fig.savefig(output / "training_curves.png", dpi=150)
    plt.close(fig)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", default="models/roadsignnet")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--no-augmentation", action="store_true")
    args = parser.parse_args()
    try:
        metadata = train(args.data_dir, args.output, args.epochs, args.batch_size,
                         args.seed, args.validation_fraction, not args.no_augmentation)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"RoadSignNet: {exc}\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
