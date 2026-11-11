"""Cache DP1 for existing training, original validation, and 50 Hz test waveforms."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import wfdb

if __package__:
    from .dp_encoding import encode_dp1
    from .generate_train_6db import CLASSES, LEADS, PTB, ROOT, SAMPLES
else:
    from dp_encoding import encode_dp1
    from generate_train_6db import CLASSES, LEADS, PTB, ROOT, SAMPLES


DEST = Path("data/derived/dp1")
INPUTS = {
    "train": ROOT / "data/derived/train_6db/train.csv",
    "validation": ROOT / "data/manifests/validation.csv",
    "test": ROOT / "data/derived/test_50hz_6db/test.csv",
}
COUNTS = {
    "train": {"original": 17084, "bw": 4271, "ma": 4271, "em": 4271, "mix": 4271},
    "validation": {"original": 2146},
    "test": {"original": 2158, "50hz": 2158},
}
EXTRA_FIELDS = ["dp1_path", "waveform_sha256"]


def input_hashes():
    return {split: hashlib.sha256(path.read_bytes()).hexdigest() for split, path in INPUTS.items()}


def read_sources():
    sources, patients, ids = {}, {}, {}
    for split, path in INPUTS.items():
        with path.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        keys = set()
        for row in rows:
            ecg_id = int(row["ecg_id"])
            if split == "validation":
                row.update(version="original", format="wfdb",
                           waveform_path=(PTB / row["filename_hr"]).as_posix())
            key = (ecg_id, row["version"])
            fold = int(row["strat_fold"])
            allowed = range(1, 9) if split == "train" else (9 if split == "validation" else 10,)
            labels = [row[label] for label in CLASSES]
            if (key in keys or ecg_id <= 0 or row["split"] != split or fold not in allowed
                    or "1" not in labels or not set(labels) <= {"0", "1"} or not row["patient_id"]):
                raise ValueError(f"Invalid or duplicate {split} sample {key}")
            keys.add(key)
        if Counter(row["version"] for row in rows) != COUNTS[split]:
            raise ValueError(f"Unexpected source conditions in {split}")
        patients[split] = {row["patient_id"] for row in rows}
        ids[split] = {row["ecg_id"] for row in rows}
        sources[split] = rows
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if patients[left] & patients[right] or ids[left] & ids[right]:
            raise ValueError(f"Overlapping records or patients in {left} and {right}")
    return sources


def read_waveform(row):
    path = ROOT / row["waveform_path"]
    if row["format"] == "wfdb":
        record = wfdb.rdrecord(str(path))
        if record.fs != 500 or record.sig_name != LEADS or record.units != ["mV"] * 12:
            raise ValueError(f"Unexpected ECG metadata: {path}")
        x = record.p_signal.T
    elif row["format"] == "npy":
        x = np.load(path, allow_pickle=False)
        if x.dtype != np.float32:
            raise ValueError(f"Unexpected stored waveform dtype: {path}")
    else:
        raise ValueError(f"Unsupported waveform format: {row['format']}")
    # Encode exactly the float32 amplitudes that the model's data loader will read.
    x = np.ascontiguousarray(x, dtype="<f4")
    if x.shape != (12, SAMPLES) or not np.isfinite(x).all():
        raise ValueError(f"Invalid waveform: {path}")
    return x


def encoded_path(split, row):
    return Path(split) / f"{int(row['ecg_id']):05d}_{row['version']}.npy"


def output_row(split, row, waveform):
    return {**row, "dp1_path": (DEST / encoded_path(split, row)).as_posix(),
            "waveform_sha256": hashlib.sha256(waveform.tobytes()).hexdigest()}


def progress(stage, split, count, total):
    if count % 5000 == 0 or count == total:
        print(f"{stage} {split}: {count}/{total}", flush=True)


def generate(folder, sources):
    for split, rows in sources.items():
        (folder / split).mkdir(parents=True)
        with (folder / f"{split}.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=[*rows[0], *EXTRA_FIELDS])
            writer.writeheader()
            for count, row in enumerate(rows, 1):
                waveform = read_waveform(row)
                encoded = encode_dp1(waveform)
                np.save(folder / encoded_path(split, row), encoded, allow_pickle=False)
                writer.writerow(output_row(split, row, waveform))
                progress("Generated", split, count, len(rows))


def check_encoding(waveform, encoded):
    if encoded.shape != waveform.shape or encoded.dtype != np.int8:
        raise ValueError("Invalid DP1 shape or dtype")
    if np.any(encoded[:, :2] != 0):
        raise ValueError("DP1 boundary samples must be zero")
    # Independent verification through neighboring amplitudes, without taking differences.
    up = waveform[:, 1:] > waveform[:, :-1]
    down = waveform[:, 1:] < waveform[:, :-1]
    expected = (up[:, 1:] & ~up[:, :-1]).astype(np.int8)
    expected -= (down[:, 1:] & ~down[:, :-1]).astype(np.int8)
    if not np.array_equal(encoded[:, 2:], expected):
        raise ValueError("DP1 events do not match the actual input waveform")


def verify(folder, sources, hashes):
    report = {
        "order": 1, "fs_hz": 500, "shape": [12, SAMPLES], "dtype": "int8",
        "values": [-1, 0, 1], "smoothing": "none",
        "difference": "d[t] = x[t] - x[t-1]; computed in float64 from float32 input",
        "boundary": "first two samples zero",
        "source": "https://arxiv.org/html/2405.00724v1#S2.E1",
        "source_manifest_sha256": hashes, "splits": {},
    }
    for split, rows in sources.items():
        with (folder / f"{split}.csv").open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != [*rows[0], *EXTRA_FIELDS]:
                raise ValueError(f"Unexpected {split} output columns")
            saved_rows = list(reader)
        if len(saved_rows) != len(rows) or sum(1 for _ in (folder / split).glob("*.npy")) != len(rows):
            raise ValueError(f"Unexpected {split} record/file count")
        positives, negatives, flat_leads = 0, 0, 0
        for count, (row, saved) in enumerate(zip(rows, saved_rows), 1):
            waveform = read_waveform(row)
            if saved != output_row(split, row, waveform):
                raise ValueError(f"Changed source waveform, metadata or pairing: {split} {row['ecg_id']}")
            encoded = np.load(folder / encoded_path(split, row), allow_pickle=False)
            check_encoding(waveform, encoded)
            positives += int(np.count_nonzero(encoded == 1))
            negatives += int(np.count_nonzero(encoded == -1))
            flat_leads += int(np.count_nonzero(np.ptp(waveform, axis=1) == 0))
            progress("Verified", split, count, len(rows))
        report["splits"][split] = {
            "records": len(rows), "condition_counts": dict(Counter(row["version"] for row in rows)),
            "positive_events": positives, "negative_events": negatives,
            "zero_values": len(rows) * 12 * SAMPLES - positives - negatives,
            "constant_lead_instances": flat_leads,
        }
    if input_hashes() != hashes:
        raise ValueError("Source manifests changed during generation/verification")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true", help="Recheck cached DP1 against its waveforms")
    args = parser.parse_args()
    hashes = input_hashes()
    sources = read_sources()
    target = ROOT / DEST
    if args.verify_only or target.exists():
        saved = json.loads((target / "audit.json").read_text(encoding="utf-8"))
        report = verify(target, sources, hashes)
        if saved != report:
            raise ValueError("Stored DP1 audit does not match the current data")
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=".dp1-", dir=target.parent) as temp:
            staging = Path(temp) / target.name
            generate(staging, sources)
            report = verify(staging, sources, hashes)
            (staging / "audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            staging.rename(target)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
