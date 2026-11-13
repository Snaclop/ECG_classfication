"""Evaluate frozen best.pt thresholds on original validation or paired final test records."""
import argparse
import csv
import io
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

if __package__:
    from .baseline_checkpoint import atomic_text, load_predictor, resolve_device, write_json
    from .baseline_data import ROOT, CLASSES, DEFAULT_TEST, DEFAULT_VALIDATION, WaveformDataset, read_manifest
    from .baseline_metrics import classification_metrics
    from .baseline_training import collect_predictions, manifest_hash
else:
    from baseline_checkpoint import atomic_text, load_predictor, resolve_device, write_json
    from baseline_data import ROOT, CLASSES, DEFAULT_TEST, DEFAULT_VALIDATION, WaveformDataset, read_manifest
    from baseline_metrics import classification_metrics
    from baseline_training import collect_predictions, manifest_hash


def evaluate_checkpoint(checkpoint_path, manifest, split="test", data_root=ROOT, device="cpu",
                        batch_size=32, output=None, progress=True):
    if split not in {"validation", "test"} or batch_size < 1:
        raise ValueError("Expected validation/test split and positive batch size")
    model, checkpoint = load_predictor(checkpoint_path, device)
    if split == "test" and checkpoint["config"]["smoke"]:
        raise ValueError("A smoke checkpoint is not eligible for final test evaluation")
    rows = read_manifest(manifest, split)
    patients, ecgs = {r["patient_id"] for r in rows}, {r["ecg_id"] for r in rows}
    for prior_split in (("train", "validation") if split == "test" else ("train",)):
        used = checkpoint["split_ids"][prior_split]
        if patients.intersection(used["patients"]) or ecgs.intersection(used["ecgs"]):
            raise ValueError(f"Evaluation overlaps {prior_split} patients or ECG IDs")
    if split == "validation" and manifest_hash(manifest) != checkpoint["manifest_hashes"]["validation"]:
        raise ValueError("Validation manifest differs from the checkpoint's selection manifest")
    output = Path(output) if output else Path(checkpoint_path).resolve().parent / f"evaluation_{split}"
    if output.exists():
        raise FileExistsError(f"Evaluation directory already exists: {output}; select a new --output")
    loader = DataLoader(WaveformDataset(rows, data_root), batch_size=batch_size, shuffle=False, num_workers=0)
    labels, probabilities, indices, loss = collect_predictions(model, loader, resolve_device(device), progress)
    ordered_rows = [rows[i] for i in indices]
    thresholds = np.asarray(checkpoint["thresholds"])
    versions = np.array([r["version"] for r in ordered_rows])
    report = {"checkpoint": str(Path(checkpoint_path).resolve()), "checkpoint_sha256": manifest_hash(checkpoint_path),
              "epoch": checkpoint["epoch"], "split": split, "smoke": checkpoint["config"]["smoke"],
              "manifest_sha256": manifest_hash(manifest), "classes": CLASSES, "thresholds": thresholds.tolist(),
              "threshold_source": checkpoint["threshold_source"], "loss": loss,
              "conditions": {version: classification_metrics(labels[versions == version], probabilities[versions == version], thresholds)
                             for version in sorted(set(versions.tolist()))}}
    predictions = probabilities >= thresholds
    buffer = io.StringIO(newline="")
    fields = ["ecg_id", "patient_id", "version", *[f"label_{c}" for c in CLASSES],
              *[f"probability_{c}" for c in CLASSES], *[f"prediction_{c}" for c in CLASSES]]
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    for i, row in enumerate(ordered_rows):
        item = {key: row[key] for key in ("ecg_id", "patient_id", "version")}
        for j, name in enumerate(CLASSES):
            item.update({f"label_{name}": int(labels[i, j]), f"probability_{name}": float(probabilities[i, j]),
                         f"prediction_{name}": int(predictions[i, j])})
        writer.writerow(item)
    output.mkdir(parents=True, exist_ok=False)
    atomic_text(buffer.getvalue(), output / "predictions.csv")
    write_json(report, output / "metrics.json")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--data-root", type=Path, default=ROOT)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args(argv)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    manifest = args.manifest or (DEFAULT_TEST if args.split == "test" else DEFAULT_VALIDATION)
    report = evaluate_checkpoint(args.checkpoint, manifest, args.split, args.data_root, args.device,
                                 args.batch_size, args.output, not args.no_progress)
    for condition, metrics in report["conditions"].items():
        print(f"{condition}: n={metrics['samples']} macro={metrics['macro']}", flush=True)


if __name__ == "__main__":
    main()
