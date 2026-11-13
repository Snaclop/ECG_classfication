"""Train the waveform CNN on the local AMD GPU."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.baseline_data import ROOT, DEFAULT_TRAIN, DEFAULT_VALIDATION
from training.baseline_training import run_training


def main():
    config = {
        "train_manifest": str(DEFAULT_TRAIN),
        "validation_manifest": str(DEFAULT_VALIDATION),
        "data_root": str(ROOT),
        "output": str(ROOT / "outputs/baseline_cnn"),
        "device": "cuda",
        "seed": 42,
        "epochs": 30,
        "batch_size": 32,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "num_workers": 0,
        "num_threads": 4,
    }
    resume = False
    checkpoint = Path(config["output"]) / "last.pt" if resume else None
    output = run_training(config, checkpoint)
    print(f"Run complete: {output}", flush=True)


if __name__ == "__main__":
    main()
