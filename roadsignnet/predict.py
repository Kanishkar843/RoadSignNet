"""Saved-model inference, top-three alternatives and optional prediction records."""

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .artifacts import load_bundle
from .labels import CLASS_NAMES
from .preprocess import preprocess_image


def rank_scores(scores, threshold=0.7):
    if not 0 <= threshold <= 1:
        raise ValueError("Confidence threshold must be between 0 and 1.")
    scores = np.asarray(scores, dtype=float)
    if (scores.shape != (43,) or not np.isfinite(scores).all() or (scores < 0).any()
            or (scores > 1).any() or not np.isclose(scores.sum(), 1.0, atol=1e-5)):
        raise ValueError("Model did not return a valid 43-class probability vector.")
    order = np.argsort(-scores, kind="stable")[:3]
    top = [{"class_id": int(i), "label": CLASS_NAMES[i], "score": float(scores[i])} for i in order]
    return {**top[0], "top_three": top,
            "review_status": "Low confidence — review required" if top[0]["score"] < threshold else "Review required",
            "threshold": threshold,
            "notice": "Known-class softmax scores cannot confirm a sign is present or reject unfamiliar objects."}


def predict_image(model, source, threshold=0.7):
    image = preprocess_image(source)
    start = time.perf_counter()
    scores = model(image[None, ...], training=False).numpy()[0]
    result = rank_scores(scores, threshold)
    result["inference_ms"] = (time.perf_counter() - start) * 1000
    result["timestamp"] = datetime.now(timezone.utc).isoformat()
    return result


def append_record(path, filename, result):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["image_name", "timestamp", "class_id", "label", "score", "review_status"]
    filename = Path(filename).name
    if filename.startswith(("=", "+", "-", "@", "\t", "\r")):
        filename = "'" + filename
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if handle.tell() == 0:
            writer.writeheader()
        writer.writerow({"image_name": filename, **{key: result[key] for key in fields[1:]}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument("--model-dir", default="models/roadsignnet")
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--log", help="Optional CSV path; no images are stored")
    args = parser.parse_args()
    try:
        model, _ = load_bundle(args.model_dir)
        result = predict_image(model, args.image, args.threshold)
        if args.log:
            append_record(args.log, args.image, result)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"RoadSignNet: {exc}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
