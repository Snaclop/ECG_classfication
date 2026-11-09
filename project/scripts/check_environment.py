"""Functional check using real records; not a trained experimental model."""
import importlib.metadata
import argparse
import json
import math
import os
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/setup"
OUT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".artifacts/matplotlib"))


def main(device="cpu"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy.signal import resample_poly
    import torch
    from torch import nn
    import wfdb

    torch.manual_seed(42)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("GPU requested but unavailable; refusing CPU fallback")
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    frame = pd.read_csv(ROOT / "data/manifests/train.csv")
    classes = ["NORM", "MI", "STTC", "CD", "HYP"]
    rows = frame.head(2)
    signals = [wfdb.rdrecord(str(ROOT / "data/raw/ptb-xl-1.0.3" / row.filename_hr)).p_signal for row in rows.itertuples()]
    x = torch.from_numpy(np.stack(signals).transpose(0, 2, 1).astype("float32"))
    y = torch.from_numpy(rows[classes].to_numpy(dtype="float32"))
    model = nn.Sequential(nn.Conv1d(12, 8, 7, padding=3), nn.ReLU(), nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(8, 5))
    x, y, model = x.to(device), y.to(device), model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, foreach=False)
    before = model[0].weight.detach().clone()
    logits = model(x)
    loss = nn.BCEWithLogitsLoss()(logits, y)
    if not torch.isfinite(loss):
        raise ValueError("Nonfinite loss")
    loss.backward()
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
        raise ValueError("Invalid gradient")
    optimizer.step()
    assert not torch.equal(before, model[0].weight.detach())
    fig, axes = plt.subplots(4, 1, figsize=(11, 8), constrained_layout=True)
    axes[0].plot(np.arange(5000) / 500, signals[0][:, 1], lw=0.8)
    axes[0].set_title("PTB-XL original ECG (lead II; not assumed noise-free)")
    noise_summary = {}
    for ax, name in zip(axes[1:], ["bw", "ma", "em"]):
        record = wfdb.rdrecord(str(ROOT / "data/raw/nstdb-1.0.0" / name), sampfrom=0, sampto=3600)
        divisor = math.gcd(int(record.fs), 500)
        resampled = resample_poly(record.p_signal, 500 // divisor, int(record.fs) // divisor, axis=0)
        assert resampled.shape == (5000, 2) and np.isfinite(resampled).all()
        ax.plot(np.arange(5000) / 500, resampled[:, 0], lw=0.8)
        ax.set_title(f"NSTDB {name}: first training segment, resampled to 500 Hz")
        noise_summary[name] = list(resampled.shape)
    for ax in axes:
        ax.set_ylabel("mV")
        ax.set_xlabel("Time (s)")
        ax.grid(alpha=0.2)
    fig.savefig(OUT / "waveform_preview.png", dpi=150)
    plt.close(fig)
    report = {
        "python": sys.version, "executable": sys.executable, "platform": platform.platform(),
        "packages": {p: importlib.metadata.version(p) for p in ["torch", "numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "wfdb", "ipykernel"]},
        "training_test_device": device, "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "torch_cuda_build": torch.version.cuda, "torch_hip_build": torch.version.hip,
        "ecg_batch_shape": list(x.shape), "output_shape": list(logits.shape),
        "one_step_loss": float(loss.detach()), "parameters_updated": True,
        "noise_resampled_shapes": noise_summary,
        "note": "Functional smoke test only. No accuracy claim, DP implementation, or formal training experiment.",
    }
    filename = "environment_check_gpu.json" if device == "cuda" else "environment_check.json"
    (OUT / filename).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    main(parser.parse_args().device)
