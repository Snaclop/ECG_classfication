"""Predict five independent labels from a single 500 Hz, mV, 12-lead waveform."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from common.baseline_checkpoint import load_predictor, write_json
from common.baseline_data import ROOT, CLASSES, INPUT_SPEC, LEADS, DEFAULT_VALIDATION, read_manifest, read_waveform


def predict_record(checkpoint_path, waveform_path):
    waveform_path = Path(waveform_path)
    waveform_format = "npy" if waveform_path.suffix.lower() == ".npy" else "wfdb"
    waveform = read_waveform(waveform_path, waveform_format)
    model, checkpoint = load_predictor(checkpoint_path)
    with torch.inference_mode():
        probabilities = model(torch.from_numpy(waveform)[None].to("cuda")).sigmoid()[0].cpu().numpy()
    thresholds = checkpoint["thresholds"]
    return {
        "waveform": str(waveform_path.resolve()), "epoch": checkpoint["epoch"],
        "input_spec": INPUT_SPEC, "leads": LEADS,
        "probabilities": dict(zip(CLASSES, probabilities.astype(float).tolist())),
        "thresholds": dict(zip(CLASSES, thresholds)),
        "predicted_labels": [name for name, probability, threshold in zip(CLASSES, probabilities, thresholds)
                             if float(probability) >= threshold],
    }


def main():
    checkpoint = ROOT / "outputs/baseline_cnn/best.pt"
    waveform_path = ROOT / read_manifest(DEFAULT_VALIDATION)[0]["waveform_path"]
    output = ROOT / "outputs/baseline_cnn/prediction.json"
    torch.set_num_threads(4)
    result = predict_record(checkpoint, waveform_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(result, output)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
