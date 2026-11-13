"""Waveform-only PTB-XL input contract; no DP files or noise parameters are read."""
import csv
from decimal import Decimal, InvalidOperation
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


def identifier(value):
    """Canonicalize CSV IDs, including patient IDs written as 15709.0."""
    try:
        number = Decimal(str(value))
        if not number.is_finite() or number <= 0 or number != number.to_integral_value():
            raise ValueError(f"Invalid identifier: {value}")
        return str(int(number))
    except (InvalidOperation, TypeError) as exc:
        raise ValueError(f"Invalid identifier: {value}") from exc


def read_manifest(path, split):
    """Read and validate one split; paths in its rows remain relative to project/."""
    conditions = {"train": {"original", "bw", "ma", "em", "mix"},
                  "validation": {"original"}, "test": {"original", "50hz"}}
    if split not in conditions:
        raise ValueError(f"Unknown split: {split}")
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Empty manifest: {path}")
    seen, records = set(), {}
    for row in rows:
        try:
            row["ecg_id"] = identifier(row["ecg_id"])
            row["patient_id"] = identifier(row["patient_id"])
            fold = int(row["strat_fold"])
            allowed = range(1, 9) if split == "train" else (9 if split == "validation" else 10,)
            if row["split"] != split or fold not in allowed:
                raise ValueError("Incorrect split or official fold")
            labels = tuple(row[c] for c in CLASSES)
            if not set(labels) <= {"0", "1"} or "1" not in labels:
                raise ValueError("Expected nonempty binary diagnostic labels")
            if not row.get("waveform_path") and row.get("filename_hr"):
                row.update(waveform_path="data/raw/ptb-xl-1.0.3/" + row["filename_hr"],
                           format="wfdb", version="original")
            if (row["version"] not in conditions[split] or row["format"] not in {"wfdb", "npy"}
                    or not row["waveform_path"]):
                raise ValueError("Invalid waveform format, path or version")
            key = (row["ecg_id"], row["version"])
            if key in seen:
                raise ValueError("Duplicate ECG/version")
            seen.add(key)
            identity = (row["patient_id"], fold, labels)
            if row["ecg_id"] in records and records[row["ecg_id"]] != identity:
                raise ValueError("Paired versions disagree on patient, fold or labels")
            records[row["ecg_id"]] = identity
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid {split} row {row.get('ecg_id', '?')} in {path}: {exc}") from exc
    return rows


def check_disjoint(*groups):
    """Check both patient and ECG identity across any supplied split groups."""
    patients, ecgs = set(), set()
    for rows in groups:
        current_patients = {identifier(r["patient_id"]) for r in rows}
        current_ecgs = {identifier(r["ecg_id"]) for r in rows}
        if patients & current_patients or ecgs & current_ecgs:
            raise ValueError("Overlapping patients or ECG IDs across splits")
        patients.update(current_patients)
        ecgs.update(current_ecgs)


def read_waveform(path, waveform_format):
    """Return the stored physical waveform unchanged as contiguous float32 [12,5000]."""
    if waveform_format == "wfdb":
        record = wfdb.rdrecord(str(path))
        if record.fs != 500 or record.sig_name != LEADS or record.units != ["mV"] * 12:
            raise ValueError(f"Expected 500 Hz, fixed lead order and mV: {path}")
        waveform = record.p_signal.T
    elif waveform_format == "npy":
        waveform = np.load(path, allow_pickle=False)
        if waveform.dtype != np.float32:
            raise ValueError(f"Expected stored float32 waveform: {path}")
    else:
        raise ValueError(f"Unsupported waveform format: {waveform_format}")
    waveform = np.ascontiguousarray(waveform, dtype=np.float32)
    if waveform.shape != (12, 5000) or not np.isfinite(waveform).all():
        raise ValueError(f"Expected finite [12,5000] waveform: {path}")
    return waveform


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
