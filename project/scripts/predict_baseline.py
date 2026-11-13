"""Predict five independent diagnostic labels from one 500 Hz, mV, 12-lead waveform."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

if __package__:
    from .baseline_checkpoint import load_predictor, resolve_device, write_json
    from .baseline_data import CLASSES, INPUT_SPEC, LEADS, read_waveform
else:
    from baseline_checkpoint import load_predictor, resolve_device, write_json
    from baseline_data import CLASSES, INPUT_SPEC, LEADS, read_waveform


def predict_record(checkpoint_path, waveform_path, waveform_format=None, device="cpu"):
    waveform_path = Path(waveform_path)
    waveform_format = waveform_format or ("npy" if waveform_path.suffix.lower() == ".npy" else "wfdb")
    waveform = read_waveform(waveform_path, waveform_format)
    model, checkpoint = load_predictor(checkpoint_path, device)
    with torch.inference_mode():
        probabilities = model(torch.from_numpy(waveform)[None].to(resolve_device(device))).sigmoid()[0].cpu().numpy()
    if not np.isfinite(probabilities).all():
        raise ValueError("Nonfinite prediction")
    thresholds = checkpoint["thresholds"]
    return {"waveform": str(waveform_path.resolve()), "epoch": checkpoint["epoch"],
            "smoke": checkpoint["config"]["smoke"], "input_spec": INPUT_SPEC, "leads": LEADS,
            "probabilities": dict(zip(CLASSES, probabilities.astype(float).tolist())),
            "thresholds": dict(zip(CLASSES, thresholds)),
            "predicted_labels": [name for name, p, threshold in zip(CLASSES, probabilities, thresholds)
                                 if float(p) >= threshold]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--waveform", type=Path, required=True, help="WFDB record base path (no extension), or .npy file")
    parser.add_argument("--format", choices=["wfdb", "npy"])
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--output", type=Path, help="Optional JSON output file")
    args = parser.parse_args(argv)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    result = predict_record(args.checkpoint, args.waveform, args.format, args.device)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(result, args.output)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
