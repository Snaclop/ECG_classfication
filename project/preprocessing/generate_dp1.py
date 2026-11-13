"""Encode the actual waveform of each training, validation and test sample."""
import csv
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from tqdm import tqdm

from common.baseline_data import CLASSES, ROOT, DEFAULT_TRAIN, DEFAULT_VALIDATION, DEFAULT_TEST, read_manifest, read_waveform
from common.dp_encoding import encode_dp1

WAVEFORM_FIELDS = [
    "ecg_id", "patient_id", "strat_fold", "split", *CLASSES,
    "filename_hr", "filename_lr", "version", "format", "waveform_path", "snr_db", "flat_leads",
    "bw_start_360", "ma_start_360", "em_start_360", "noise_frequency_hz", "noise_phase_rad",
]


def encoded_path(split, row):
    return Path(split) / f"{int(row['ecg_id']):05d}_{row['version']}.npy"


def generate(folder, sources, encoder, path_column):
    """Write one encoding and its path per waveform; reused by the DP2 generator."""
    relative_folder = folder.relative_to(ROOT)
    for split, rows in sources.items():
        (folder / split).mkdir()
        with (folder / f"{split}.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=[*rows[0], path_column])
            writer.writeheader()
            for row in tqdm(rows, desc=f"{path_column} {split}"):
                waveform = read_waveform(ROOT / row["waveform_path"], row["format"])
                relative = encoded_path(split, row)
                np.save(folder / relative, encoder(waveform), allow_pickle=False)
                writer.writerow({**row, path_column: (relative_folder / relative).as_posix()})


def main():
    target = ROOT / "data/derived/dp1"
    inputs = {"train": DEFAULT_TRAIN, "validation": DEFAULT_VALIDATION, "test": DEFAULT_TEST}
    target.mkdir(parents=True, exist_ok=False)
    sources = {split: read_manifest(path) for split, path in inputs.items()}
    generate(target, sources, encode_dp1, "dp1_path")
    print(f"DP1 data: {target}", flush=True)


if __name__ == "__main__":
    main()
