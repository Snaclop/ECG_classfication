"""Waveform training and recovery at complete epoch boundaries."""
from collections import Counter
import importlib.metadata
import json
from pathlib import Path
import platform
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from common.baseline_data import CLASSES, LEADS, INPUT_SPEC, WaveformDataset, read_manifest
from common.baseline_metrics import classification_metrics, select_thresholds, is_better
from common.baseline_checkpoint import (
    atomic_torch_save, atomic_text, write_json, capture_rng, restore_rng,
    model_snapshot, load_checkpoint,
)
from models.baseline_model import MODEL_SPEC, WaveformCNN


def environment_report(model):
    return {
        "python": sys.version, "executable": sys.executable, "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name)
                     for name in ("torch", "numpy", "pandas", "scikit-learn", "wfdb", "tqdm")},
        "device": "cuda", "gpu_name": torch.cuda.get_device_name(),
        "torch_hip_build": torch.version.hip,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, count = 0.0, 0
    for waveform, labels, _ in tqdm(loader, desc="train"):
        waveform, labels = waveform.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(waveform), labels)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach()) * len(labels)
        count += len(labels)
    return total_loss / count


def collect_predictions(model, loader, device):
    model.eval()
    all_labels, probabilities, all_indices = [], [], []
    total_loss, count = 0.0, 0
    with torch.inference_mode():
        for waveform, labels, indices in tqdm(loader, desc="evaluate"):
            logits = model(waveform.to(device))
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels.to(device))
            all_labels.append(labels.numpy())
            probabilities.append(logits.sigmoid().cpu().numpy())
            all_indices.extend(indices.tolist())
            total_loss += float(loss) * len(labels)
            count += len(labels)
    return np.concatenate(all_labels), np.concatenate(probabilities), all_indices, total_loss / count


def publish_epoch(checkpoint, output):
    # last.pt carries the matching best/history so an interrupted save can be resumed.
    atomic_torch_save(checkpoint, output / "last.pt")
    if checkpoint["best_checkpoint"] is not None:
        atomic_torch_save(checkpoint["best_checkpoint"], output / "best.pt")
    atomic_text("".join(json.dumps(record, allow_nan=False) + "\n"
                        for record in checkpoint["history"]), output / "history.jsonl")


def run_training(config, resume=None):
    saved = load_checkpoint(resume) if resume else None
    if saved:
        # Reuse training settings; only the requested total epoch count changes.
        total_epochs = config["epochs"]
        config = {key: saved["config"][key] for key in config}
        config.update(epochs=total_epochs, device="cuda", output=str(Path(resume).parent))
    output = Path(config["output"])
    if saved is None:
        output.mkdir(parents=True, exist_ok=False)

    device = torch.device("cuda")
    print(f"Waveform CNN training\nDevice: {device}; output: {output}", flush=True)
    torch.set_num_threads(config["num_threads"])
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    generator = torch.Generator().manual_seed(config["seed"])

    train_rows = read_manifest(config["train_manifest"])
    validation_rows = read_manifest(config["validation_manifest"])
    train_loader = DataLoader(
        WaveformDataset(train_rows, config["data_root"]), batch_size=config["batch_size"],
        shuffle=True, generator=generator, num_workers=config["num_workers"],
    )
    validation_loader = DataLoader(
        WaveformDataset(validation_rows, config["data_root"]), batch_size=config["batch_size"],
        shuffle=False, num_workers=config["num_workers"],
    )
    model = WaveformCNN().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"],
        weight_decay=config["weight_decay"], foreach=False,
    )
    start, history, best, best_checkpoint = 0, [], None, None
    if saved:
        model.load_state_dict(saved["model_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        start, history, best = saved["epoch"], saved["history"], saved["best"]
        best_checkpoint = saved["best_checkpoint"]
        restore_rng(saved["rng_state"], generator)
        publish_epoch(saved, output)
        write_json(environment_report(model), output / "resume_environment.json")
    else:
        write_json(config, output / "config.json")
        write_json(environment_report(model), output / "environment.json")
        write_json({
            "rows": {"train": len(train_rows), "validation": len(validation_rows)},
            "training_versions": dict(Counter(row["version"] for row in train_rows)),
        }, output / "data_summary.json")

    common = {
        "classes": CLASSES, "leads": LEADS, "input_spec": INPUT_SPEC,
        "model_spec": MODEL_SPEC, "config": config,
    }
    for epoch in range(start + 1, config["epochs"] + 1):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        began = time.perf_counter()
        train_loss = train_epoch(model, train_loader, optimizer, device)
        labels, probabilities, _, validation_loss = collect_predictions(model, validation_loader, device)
        metrics = classification_metrics(labels, probabilities)
        improved = is_better(metrics["macro"]["auroc"], validation_loss, best)
        if improved:
            best = {"auroc": metrics["macro"]["auroc"], "loss": validation_loss, "epoch": epoch}
            thresholds = select_thresholds(labels, probabilities)
            best_checkpoint = {
                **common, "kind": "best", "epoch": epoch, "model_state": model_snapshot(model),
                "thresholds": thresholds.tolist(), "validation_loss": validation_loss,
                "validation_metrics": classification_metrics(labels, probabilities, thresholds),
                "threshold_source": "validation only; per-class F1 grid 0.05..0.95 step 0.01",
            }
        torch.cuda.synchronize()
        history.append({
            "epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss,
            "validation_metrics_at_0_5": metrics,
            "seconds": time.perf_counter() - began, "selected_best": improved,
            "gpu_peak_memory_bytes": torch.cuda.max_memory_allocated(),
        })
        checkpoint = {
            **common, "kind": "last", "epoch": epoch, "model_state": model_snapshot(model),
            "optimizer_state": optimizer.state_dict(), "rng_state": capture_rng(generator),
            "best": best, "best_checkpoint": best_checkpoint, "history": history,
        }
        publish_epoch(checkpoint, output)
        print(f"Epoch {epoch}/{config['epochs']} train_loss={train_loss:.6f} "
              f"val_loss={validation_loss:.6f} macro_AUROC={metrics['macro']['auroc']} "
              f"saved={output / 'last.pt'}", flush=True)
    return output
