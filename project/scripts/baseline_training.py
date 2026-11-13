"""Training and epoch-boundary recovery for the waveform CNN baseline."""
from collections import Counter
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

if __package__:
    from .baseline_data import (ROOT, CLASSES, LEADS, INPUT_SPEC, DEFAULT_TRAIN, DEFAULT_VALIDATION,
                                WaveformDataset, read_manifest, check_disjoint)
    from .baseline_model import MODEL_SPEC, WaveformCNN
    from .baseline_metrics import classification_metrics, select_thresholds, is_better
    from .baseline_checkpoint import (atomic_torch_save, atomic_text, write_json, capture_rng, restore_rng,
                                      model_snapshot, load_checkpoint, resolve_device)
else:
    from baseline_data import (ROOT, CLASSES, LEADS, INPUT_SPEC, DEFAULT_TRAIN, DEFAULT_VALIDATION,
                               WaveformDataset, read_manifest, check_disjoint)
    from baseline_model import MODEL_SPEC, WaveformCNN
    from baseline_metrics import classification_metrics, select_thresholds, is_better
    from baseline_checkpoint import (atomic_torch_save, atomic_text, write_json, capture_rng, restore_rng,
                                     model_snapshot, load_checkpoint, resolve_device)


def make_config(**overrides):
    config = dict(train_manifest=DEFAULT_TRAIN, validation_manifest=DEFAULT_VALIDATION,
                  data_root=ROOT, output=ROOT / "outputs/baseline_cnn", device="cpu",
                  seed=42, epochs=30, batch_size=32, learning_rate=0.001, weight_decay=0.0001,
                  num_workers=0, num_threads=min(4, os.cpu_count() or 1), progress=True,
                  smoke=False, train_limit=None, validation_limit=None)
    unknown = overrides.keys() - config.keys()
    if unknown:
        raise ValueError(f"Unknown config fields: {sorted(unknown)}")
    config.update(overrides)
    for field in ("train_manifest", "validation_manifest", "data_root", "output"):
        config[field] = str(Path(config[field]).resolve())
    for field in ("epochs", "batch_size", "num_threads"):
        if not isinstance(config[field], int) or isinstance(config[field], bool) or config[field] < 1:
            raise ValueError(f"{field} must be a positive integer")
    if not isinstance(config["seed"], int) or not 0 <= config["seed"] < 2 ** 32:
        raise ValueError("Seed must be an integer in [0, 2**32)")
    if config["num_workers"] != 0:
        raise ValueError("This baseline uses num_workers=0 for reproducible epoch-boundary recovery")
    if (not math.isfinite(config["learning_rate"]) or config["learning_rate"] <= 0
            or not math.isfinite(config["weight_decay"]) or config["weight_decay"] < 0):
        raise ValueError("Invalid learning rate or weight decay")
    for field in ("train_limit", "validation_limit"):
        limit = config[field]
        if limit is not None and (not isinstance(limit, int) or limit < 1 or not config["smoke"]):
            raise ValueError("Sample limits require --smoke and positive integer limits")
    return config


def manifest_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def environment_report(device, model):
    return {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
            "packages": {p: importlib.metadata.version(p) for p in ("torch", "numpy", "pandas", "scikit-learn", "wfdb", "tqdm")},
            "device": str(device), "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "torch_cuda_build": torch.version.cuda, "torch_hip_build": torch.version.hip,
            "parameters": sum(p.numel() for p in model.parameters())}


def train_epoch(model, loader, optimizer, device, progress=True):
    model.train()
    total_loss, count = 0.0, 0
    order = hashlib.sha256()
    for waveform, labels, indices in tqdm(loader, desc="train", disable=not progress):
        waveform, labels = waveform.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(waveform)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite training loss")
        loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError("Missing or nonfinite gradient")
        optimizer.step()
        total_loss += float(loss.detach()) * len(labels)
        count += len(labels)
        order.update(indices.numpy().astype("<i8").tobytes())
    return total_loss / count, order.hexdigest()


def collect_predictions(model, loader, device, progress=True):
    model.eval()
    all_labels, probabilities, all_indices = [], [], []
    total_loss, count = 0.0, 0
    with torch.inference_mode():
        for waveform, labels, indices in tqdm(loader, desc="evaluate", disable=not progress):
            logits = model(waveform.to(device))
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels.to(device))
            if not torch.isfinite(logits).all() or not torch.isfinite(loss):
                raise ValueError("Nonfinite evaluation outputs")
            all_labels.append(labels.numpy())
            probabilities.append(logits.sigmoid().cpu().numpy())
            all_indices.extend(indices.tolist())
            total_loss += float(loss) * len(labels)
            count += len(labels)
    return np.concatenate(all_labels), np.concatenate(probabilities), all_indices, total_loss / count


def publish_epoch(checkpoint, output):
    """last.pt is the commit point; its embedded best/history repair interrupted publication."""
    atomic_torch_save(checkpoint, output / "last.pt")
    atomic_torch_save(checkpoint["best_checkpoint"], output / "best.pt")
    atomic_text("".join(json.dumps(e, allow_nan=False) + "\n" for e in checkpoint["history"]), output / "history.jsonl")


def run_training(config, resume=None):
    config = make_config(**config)
    device = resolve_device(config["device"])
    train_rows = read_manifest(config["train_manifest"], "train")
    validation_rows = read_manifest(config["validation_manifest"], "validation")
    check_disjoint(train_rows, validation_rows)
    hashes = {name: manifest_hash(config[name + "_manifest"]) for name in ("train", "validation")}
    train_rows = train_rows[:config["train_limit"]]
    validation_rows = validation_rows[:config["validation_limit"]]
    positives = np.array([[int(r[c]) for c in CLASSES] for r in validation_rows]).sum(axis=0)
    if np.any((positives == 0) | (positives == len(validation_rows))):
        raise ValueError("Validation must contain positives and negatives for every class, including in smoke mode")
    output = Path(config["output"])
    saved = None
    if resume:
        resume = Path(resume).resolve()
        saved = load_checkpoint(resume)
        if saved.get("kind") != "last" or resume.parent != output:
            raise ValueError("Resume requires last.pt in the selected run directory")
        # Device may change, but experiment inputs and numerical settings must match.
        ignored = {"device", "progress"}
        if any(saved["config"].get(k) != v for k, v in config.items() if k not in ignored):
            raise ValueError("Resume configuration differs from the saved experiment")
        if saved["manifest_hashes"] != hashes:
            raise ValueError("Manifest changed since checkpoint")
    elif output.exists():
        raise FileExistsError(f"Run directory already exists: {output}; use --resume or another --output")

    torch.set_num_threads(config["num_threads"])
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    generator = torch.Generator().manual_seed(config["seed"])
    train_loader = DataLoader(WaveformDataset(train_rows, config["data_root"]),
                              batch_size=config["batch_size"], shuffle=True, generator=generator, num_workers=0)
    validation_loader = DataLoader(WaveformDataset(validation_rows, config["data_root"]),
                                   batch_size=config["batch_size"], shuffle=False, num_workers=0)
    model = WaveformCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"], foreach=False)
    start, history, best, best_checkpoint = 0, [], None, None
    if saved:
        model.load_state_dict(saved["model_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        start, history, best = saved["epoch"], saved["history"], saved["best"]
        best_checkpoint = saved["best_checkpoint"]
        restore_rng(saved["rng_state"], generator)
        publish_epoch(saved, output)
        write_json(environment_report(device, model), output / "resume_environment.json")
    else:
        output.mkdir(parents=True, exist_ok=False)
        write_json(config, output / "config.json")
        write_json(environment_report(device, model), output / "environment.json")
        write_json({"sha256": hashes, "rows": {"train": len(train_rows), "validation": len(validation_rows)},
                    "training_versions": dict(Counter(r["version"] for r in train_rows))}, output / "data_summary.json")

    common = {"schema_version": 1, "classes": CLASSES, "leads": LEADS, "input_spec": INPUT_SPEC,
              "model_spec": MODEL_SPEC, "config": config, "manifest_hashes": hashes,
              "split_ids": {"train": {"patients": sorted({r["patient_id"] for r in train_rows}),
                                      "ecgs": sorted({r["ecg_id"] for r in train_rows})},
                            "validation": {"patients": sorted({r["patient_id"] for r in validation_rows}),
                                           "ecgs": sorted({r["ecg_id"] for r in validation_rows})}}}
    for epoch in range(start + 1, config["epochs"] + 1):
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        began = time.perf_counter()
        train_loss, order_hash = train_epoch(model, train_loader, optimizer, device, config["progress"])
        labels, probabilities, _, validation_loss = collect_predictions(model, validation_loader, device, config["progress"])
        metrics = classification_metrics(labels, probabilities)
        improved = is_better(metrics["macro"]["auroc"], validation_loss, best)
        if improved:
            best = {"auroc": metrics["macro"]["auroc"], "loss": validation_loss, "epoch": epoch}
            thresholds = select_thresholds(labels, probabilities)
            best_checkpoint = {**common, "kind": "best", "epoch": epoch, "model_state": model_snapshot(model),
                               "thresholds": thresholds.tolist(), "validation_loss": validation_loss,
                               "validation_metrics": classification_metrics(labels, probabilities, thresholds),
                               "threshold_source": "validation only; per-class F1 grid 0.05..0.95 step 0.01"}
        if device.type == "cuda":
            torch.cuda.synchronize()
        record = {"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss,
                  "validation_metrics_at_0_5": metrics, "train_order_sha256": order_hash,
                  "seconds": time.perf_counter() - began, "selected_best": improved,
                  "gpu_peak_memory_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None}
        history.append(record)
        checkpoint = {**common, "kind": "last", "epoch": epoch, "model_state": model_snapshot(model),
                      "optimizer_state": optimizer.state_dict(), "rng_state": capture_rng(generator),
                      "best": best, "best_checkpoint": best_checkpoint, "history": history}
        publish_epoch(checkpoint, output)
        print(f"Epoch {epoch}/{config['epochs']} train_loss={train_loss:.6f} val_loss={validation_loss:.6f} "
              f"macro_AUROC={metrics['macro']['auroc']:.6f} saved={output / 'last.pt'}", flush=True)
    return output
