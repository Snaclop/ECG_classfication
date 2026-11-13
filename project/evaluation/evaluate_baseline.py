"""Evaluate the saved best model on validation or paired final test records."""
import csv
import io
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader

from common.baseline_checkpoint import atomic_text, load_predictor, write_json
from common.baseline_data import ROOT, CLASSES, DEFAULT_TEST, DEFAULT_VALIDATION, WaveformDataset, read_manifest
from common.baseline_metrics import classification_metrics
from training.baseline_training import collect_predictions


def evaluate_checkpoint(checkpoint_path, manifest, split, batch_size, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    model, checkpoint = load_predictor(checkpoint_path)
    rows = read_manifest(manifest)
    loader = DataLoader(WaveformDataset(rows), batch_size=batch_size, shuffle=False, num_workers=0)
    labels, probabilities, indices, loss = collect_predictions(model, loader, torch.device("cuda"))
    ordered_rows = [rows[index] for index in indices]
    thresholds = np.asarray(checkpoint["thresholds"], dtype=np.float64)
    versions = np.array([row["version"] for row in ordered_rows])
    report = {
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "epoch": checkpoint["epoch"], "split": split,
        "classes": CLASSES, "thresholds": thresholds.tolist(),
        "threshold_source": checkpoint["threshold_source"], "loss": loss,
        "conditions": {
            version: classification_metrics(labels[versions == version], probabilities[versions == version], thresholds)
            for version in sorted(set(versions.tolist()))
        },
    }
    predictions = probabilities >= thresholds
    buffer = io.StringIO(newline="")
    fields = ["ecg_id", "patient_id", "version", *[f"label_{name}" for name in CLASSES],
              *[f"probability_{name}" for name in CLASSES], *[f"prediction_{name}" for name in CLASSES]]
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    for index, row in enumerate(ordered_rows):
        item = {key: row[key] for key in ("ecg_id", "patient_id", "version")}
        for column, name in enumerate(CLASSES):
            item.update({
                f"label_{name}": int(labels[index, column]),
                f"probability_{name}": float(probabilities[index, column]),
                f"prediction_{name}": int(predictions[index, column]),
            })
        writer.writerow(item)
    atomic_text(buffer.getvalue(), output / "predictions.csv")
    write_json(report, output / "metrics.json")
    return report


def main():
    checkpoint = ROOT / "outputs/baseline_cnn/best.pt"
    split = "test"
    manifest = {"test": DEFAULT_TEST, "validation": DEFAULT_VALIDATION}[split]
    batch_size = 32
    output = ROOT / "outputs/baseline_cnn" / f"evaluation_{split}"
    torch.set_num_threads(4)
    report = evaluate_checkpoint(checkpoint, manifest, split, batch_size, output)
    for condition, metrics in report["conditions"].items():
        print(f"{condition}: n={metrics['samples']} macro={metrics['macro']}", flush=True)


if __name__ == "__main__":
    main()
