"""Cache DP2 from actual waveforms and pair it with the existing DP1 records."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

if __package__:
    from . import generate_dp1 as dp1
    from .dp_encoding import encode_dp2
else:
    import generate_dp1 as dp1
    from dp_encoding import encode_dp2


ROOT = dp1.ROOT
DEST = Path("data/derived/dp2")
INPUTS = {split: ROOT / dp1.DEST / f"{split}.csv" for split in dp1.INPUTS}


def input_hashes():
    return {
        "waveform_manifests": dp1.input_hashes(),
        "dp1_manifests": {split: hashlib.sha256(path.read_bytes()).hexdigest()
                          for split, path in INPUTS.items()},
    }


def read_sources():
    original = dp1.read_sources()
    sources = {}
    for split, source_rows in original.items():
        with INPUTS[split].open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != [*source_rows[0], *dp1.EXTRA_FIELDS]:
                raise ValueError(f"Unexpected {split} DP1 manifest columns")
            rows = list(reader)
        if len(rows) != len(source_rows):
            raise ValueError(f"Unexpected {split} DP1 manifest length")
        for source, row in zip(source_rows, rows):
            metadata = {key: value for key, value in row.items() if key not in dp1.EXTRA_FIELDS}
            expected_path = (dp1.DEST / dp1.encoded_path(split, source)).as_posix()
            if metadata != source or row["dp1_path"] != expected_path:
                raise ValueError(f"Mismatched DP1 pairing: {split} ECG {source['ecg_id']}")
        sources[split] = rows
    return sources


def read_waveform(row):
    waveform = dp1.read_waveform(row)
    if hashlib.sha256(waveform.tobytes()).hexdigest() != row["waveform_sha256"]:
        raise ValueError(f"Waveform changed since DP1 generation: ECG {row['ecg_id']}")
    return waveform


def output_row(split, row):
    return {**row, "dp2_path": (DEST / dp1.encoded_path(split, row)).as_posix()}


def generate(folder, sources):
    for split, rows in sources.items():
        (folder / split).mkdir(parents=True)
        with (folder / f"{split}.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=[*rows[0], "dp2_path"])
            writer.writeheader()
            for count, row in enumerate(rows, 1):
                encoded = encode_dp2(read_waveform(row))
                np.save(folder / dp1.encoded_path(split, row), encoded, allow_pickle=False)
                writer.writerow(output_row(split, row))
                dp1.progress("Generated DP2", split, count, len(rows))


def check_encoding(waveform, encoded):
    if encoded.shape != waveform.shape or encoded.dtype != np.int8:
        raise ValueError("Invalid DP2 shape or dtype")
    if np.any(encoded[:, :3] != 0):
        raise ValueError("DP2 boundary samples must be zero")
    # Check crossings independently by comparing neighboring first differences.
    x = waveform.astype(np.float64)
    slope = x[:, 1:] - x[:, :-1]
    up = slope[:, 1:] > slope[:, :-1]
    down = slope[:, 1:] < slope[:, :-1]
    expected = (up[:, 1:] & ~up[:, :-1]).astype(np.int8)
    expected -= (down[:, 1:] & ~down[:, :-1]).astype(np.int8)
    if not np.array_equal(encoded[:, 3:], expected):
        raise ValueError("DP2 events do not match the actual waveform curvature")


def verify(folder, sources, hashes):
    report = {
        "order": 2, "fs_hz": 500, "shape": [12, dp1.SAMPLES], "dtype": "int8",
        "values": [-1, 0, 1], "smoothing": "none",
        "difference": "d2[t] = (x[t]-x[t-1]) - (x[t-1]-x[t-2]); float64 from float32 input",
        "boundary": "first three samples zero",
        "fusion_channel_order": "DP1 leads I..V6 followed by DP2 leads I..V6",
        "source": "https://arxiv.org/html/2405.00724v1#S2.E1",
        "source_manifest_sha256": hashes, "splits": {},
    }
    for split, rows in sources.items():
        with (folder / f"{split}.csv").open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != [*rows[0], "dp2_path"]:
                raise ValueError(f"Unexpected {split} DP2 manifest columns")
            saved_rows = list(reader)
        if len(saved_rows) != len(rows) or sum(1 for _ in (folder / split).glob("*.npy")) != len(rows):
            raise ValueError(f"Unexpected {split} DP2 record/file count")
        positives, negatives, flat_leads = 0, 0, 0
        for count, (row, saved) in enumerate(zip(rows, saved_rows), 1):
            if saved != output_row(split, row):
                raise ValueError(f"Changed waveform/DP1/DP2 pairing: {split} ECG {row['ecg_id']}")
            waveform = read_waveform(row)
            first = np.load(ROOT / row["dp1_path"], allow_pickle=False)
            dp1.check_encoding(waveform, first)
            second = np.load(folder / dp1.encoded_path(split, row), allow_pickle=False)
            check_encoding(waveform, second)
            positives += int(np.count_nonzero(second == 1))
            negatives += int(np.count_nonzero(second == -1))
            flat_leads += int(np.count_nonzero(np.ptp(waveform, axis=1) == 0))
            dp1.progress("Verified DP1/DP2", split, count, len(rows))
        report["splits"][split] = {
            "records": len(rows), "condition_counts": dict(Counter(row["version"] for row in rows)),
            "positive_events": positives, "negative_events": negatives,
            "zero_values": len(rows) * 12 * dp1.SAMPLES - positives - negatives,
            "constant_lead_instances": flat_leads,
        }
    if input_hashes() != hashes:
        raise ValueError("Source manifests changed during DP2 generation/verification")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true", help="Recheck DP2 and its paired waveform/DP1")
    args = parser.parse_args()
    hashes = input_hashes()
    sources = read_sources()
    target = ROOT / DEST
    if args.verify_only or target.exists():
        saved = json.loads((target / "audit.json").read_text(encoding="utf-8"))
        report = verify(target, sources, hashes)
        if saved != report:
            raise ValueError("Stored DP2 audit does not match the current data")
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=".dp2-", dir=target.parent) as temp:
            staging = Path(temp) / target.name
            generate(staging, sources)
            report = verify(staging, sources, hashes)
            (staging / "audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            staging.rename(target)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
