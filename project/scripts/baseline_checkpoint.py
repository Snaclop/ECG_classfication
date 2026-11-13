"""Portable, atomic checkpoint IO with explicit model/input contracts."""
import json
import os
from pathlib import Path
import random

import numpy as np
import torch

if __package__:
    from .baseline_data import CLASSES, LEADS, INPUT_SPEC
    from .baseline_model import MODEL_SPEC, WaveformCNN
else:
    from baseline_data import CLASSES, LEADS, INPUT_SPEC
    from baseline_model import MODEL_SPEC, WaveformCNN


def atomic_torch_save(payload, path):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_text(text, path):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def write_json(value, path):
    atomic_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", path)


def capture_rng(generator):
    state = np.random.get_state()
    return {"python": random.getstate(),
            "numpy": [state[0], state[1].tolist(), state[2], state[3], state[4]],
            "torch": torch.get_rng_state(), "shuffle": generator.get_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state, generator):
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:]))
    torch.set_rng_state(state["torch"])
    generator.set_state(state["shuffle"])
    if torch.cuda.is_available():
        for i, cuda_state in enumerate(state["cuda"][:torch.cuda.device_count()]):
            torch.cuda.set_rng_state(cuda_state, i)


def model_snapshot(model):
    # A saved best model must not share tensors with subsequent optimizer updates.
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    expected = {"schema_version": 1, "classes": CLASSES, "leads": LEADS,
                "input_spec": INPUT_SPEC, "model_spec": MODEL_SPEC}
    if not isinstance(checkpoint, dict) or any(checkpoint.get(k) != v for k, v in expected.items()):
        raise ValueError("Unsupported checkpoint or mismatched class/lead/model/input contract")
    return checkpoint


def resolve_device(name):
    if name not in {"cpu", "cuda"}:
        raise ValueError("Device must be cpu or cuda")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA/ROCm requested but unavailable; refusing CPU fallback")
    return torch.device(name)


def load_predictor(path, device="cpu"):
    device = resolve_device(device)
    checkpoint = load_checkpoint(path)
    if checkpoint.get("kind") != "best":
        raise ValueError("Prediction/evaluation requires best.pt with validation-fitted thresholds")
    thresholds = np.asarray(checkpoint.get("thresholds"), dtype=float)
    if thresholds.shape != (5,) or not np.isfinite(thresholds).all() or np.any((thresholds < 0.05) | (thresholds > 0.95)):
        raise ValueError("Missing or invalid fitted thresholds")
    model = WaveformCNN().to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return model, checkpoint
