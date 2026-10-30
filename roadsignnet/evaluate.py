"""Evaluate a frozen model on separately supplied held-out annotations."""

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from .artifacts import digest, load_bundle
from .data import assert_disjoint, discover_test, load_arrays
from .labels import CLASS_NAMES
from .model import configure_runtime


def evaluate(data_dir, test_csv, model_dir, output, batch_size=128):
    if batch_size < 1:
        raise ValueError("Batch size must be positive.")
    output, model_dir = Path(output), Path(model_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose an empty evaluation output directory.")
    configure_runtime()
    model, metadata = load_bundle(model_dir)
    manifest_path = model_dir / "split_manifest.json"
    if not manifest_path.is_file() or digest(manifest_path) != metadata.get("split_manifest_sha256"):
        raise ValueError("Missing or changed split manifest; cannot audit held-out separation.")
    manifest = json.loads(manifest_path.read_text())
    rows = discover_test(data_dir, test_csv)
    assert_disjoint(manifest["train"] + manifest["validation"], rows)
    images, labels = load_arrays(rows)
    scores = np.concatenate([model(images[i:i+batch_size], training=False).numpy()
                             for i in range(0, len(images), batch_size)])
    if (scores.shape != (len(labels), 43) or not np.isfinite(scores).all()
            or (scores < 0).any() or (scores > 1).any()
            or not np.allclose(scores.sum(axis=1), 1, atol=1e-5)):
        raise ValueError("Invalid model output scores.")
    predicted = scores.argmax(axis=1)
    report = classification_report(labels, predicted, labels=list(range(43)),
                                   target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    matrix = confusion_matrix(labels, predicted, labels=list(range(43)))
    model(images[:1], training=False).numpy()
    timings = []
    for image in images[:min(100, len(images))]:
        started = time.perf_counter()
        model(image[None, ...], training=False).numpy()
        timings.append((time.perf_counter() - started) * 1000)
    off_diagonal = matrix.copy()
    np.fill_diagonal(off_diagonal, 0)
    pairs = sorted(np.argwhere(off_diagonal > 0).tolist(), key=lambda p: -off_diagonal[tuple(p)])[:10]
    metrics = {"test_samples": len(rows), "classes_present": sorted(set(labels.tolist())),
               "accuracy": float(accuracy_score(labels, predicted)),
               "macro_precision": report["macro avg"]["precision"],
               "macro_recall": report["macro avg"]["recall"],
               "macro_f1": report["macro avg"]["f1-score"],
               "macro_definition": "All 43 classes; missing-class metrics set to zero",
               "mean_single_image_inference_ms": float(np.mean(timings)),
               "latency_samples": len(timings), "latency_excludes": "preprocessing and model load",
               "evaluation_devices": [device.name for device in tf.config.list_physical_devices()],
               "model_sha256": metadata["model_sha256"], "test_csv_sha256": digest(test_csv),
               "dataset_protocol": "User-supplied held-out annotations; verify official dataset provenance",
               "frequent_confusions": [{"true_id": a, "predicted_id": b, "count": int(matrix[a, b])} for a, b in pairs]}
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (output / "classification_report.json").write_text(json.dumps(report, indent=2))
    pd.DataFrame(report).T.to_csv(output / "classification_report.csv")
    pd.DataFrame(matrix, index=range(43), columns=range(43)).to_csv(output / "confusion_matrix.csv")
    pd.DataFrame({"filename": [Path(r["path"]).name for r in rows], "true_class_id": labels,
                  "predicted_class_id": predicted, "score": scores.max(axis=1)}).to_csv(output / "predictions.csv", index=False)
    portable_test = [{"filename": Path(r["path"]).name, "class_id": r["class_id"], "sha256": r["sha256"]} for r in rows]
    (output / "test_manifest.json").write_text(json.dumps(portable_test, indent=2))
    fig, ax = plt.subplots(figsize=(13, 11))
    plotted = ax.imshow(matrix, cmap="Blues")
    ax.set(xticks=range(43), yticks=range(43), xlabel="Predicted class ID", ylabel="True class ID",
           title="RoadSignNet · held-out confusion matrix")
    ax.tick_params(axis="x", rotation=90, labelsize=7)
    ax.tick_params(axis="y", labelsize=7)
    fig.colorbar(plotted, ax=ax)
    fig.tight_layout()
    fig.savefig(output / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--model-dir", default="models/roadsignnet")
    parser.add_argument("--output", default="results/held-out")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    try:
        result = evaluate(args.data_dir, args.test_csv, args.model_dir, args.output, args.batch_size)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"RoadSignNet: {exc}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
