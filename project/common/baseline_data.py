"""Shared waveform and manifest loading for the five-label ECG task."""
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
import wfdb

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
LEADS = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
INPUT_SPEC = {"shape": [12, 5000], "fs_hz": 500, "units": "mV", "dtype": "float32", "preprocessing": "none"}
DEFAULT_TRAIN = ROOT / "data/derived/train_6db/train.csv"
DEFAULT_VALIDATION = ROOT / "data/manifests/validation.csv"
DEFAULT_TEST = ROOT / "data/derived/test_50hz_6db/test.csv"


def read_manifest(path):
    """Read either a derived waveform manifest or an original PTB-XL manifest."""
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row["ecg_id"] = str(int(float(row["ecg_id"])))
        row["patient_id"] = str(int(float(row["patient_id"])))
        if "waveform_path" not in row:
            row.update(waveform_path="data/raw/ptb-xl-1.0.3/" + row["filename_hr"],
                       format="wfdb", version="original")
    return rows


def read_waveform(path, waveform_format):
    """Load physical mV values as contiguous float32 [12,5000]."""
    if waveform_format == "wfdb":
        waveform = wfdb.rdrecord(str(path)).p_signal.T
    else:
        waveform = np.load(path, allow_pickle=False)
    return np.ascontiguousarray(waveform, dtype=np.float32)


class WaveformDataset(Dataset):
    def __init__(self, rows, root=ROOT):
        self.rows = rows
        self.root = Path(root)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        waveform = read_waveform(self.root / row["waveform_path"], row["format"])
        labels = np.array([int(row[c]) for c in CLASSES], dtype=np.float32)
        return torch.from_numpy(waveform), torch.from_numpy(labels), index
