"""Versioned model bundle validation."""

import hashlib
import json
from pathlib import Path

from .labels import CLASS_NAMES
from .preprocess import PREPROCESSING


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_bundle(folder):
    import tensorflow as tf

    folder = Path(folder)
    if not (folder / "metadata.json").is_file() or not (folder / "model.keras").is_file():
        raise ValueError("Trained model bundle missing. Run the training command in README.md first.")
    metadata = json.loads((folder / "metadata.json").read_text())
    if metadata.get("schema_version") != 1 or metadata.get("class_names") != CLASS_NAMES:
        raise ValueError("Unsupported metadata or incorrect class ordering.")
    if metadata.get("preprocessing") != PREPROCESSING:
        raise ValueError("Model preprocessing does not match this application.")
    if digest(folder / "model.keras") != metadata.get("model_sha256"):
        raise ValueError("Model checksum does not match metadata.")
    model = tf.keras.models.load_model(folder / "model.keras", compile=False, safe_mode=True)
    if model.input_shape != (None, 32, 32, 3) or model.output_shape != (None, 43):
        raise ValueError("Model must accept 32×32 RGB and return 43 class scores.")
    return model, metadata
