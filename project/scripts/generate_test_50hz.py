"""Generate paired original / 50 Hz, 6 dB waveforms for the held-out test fold."""
import argparse
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

if __package__:
    from .generate_train_6db import (
        CLASSES, LEADS, META, PTB, ROOT, SAMPLES, SNR_DB, TOLERANCE_DB,
        augment, read_ecg, validate_audit, wave_name,
    )
    from .prepare_data import verify_hashes
else:
    from generate_train_6db import (
        CLASSES, LEADS, META, PTB, ROOT, SAMPLES, SNR_DB, TOLERANCE_DB,
        augment, read_ecg, validate_audit, wave_name,
    )
    from prepare_data import verify_hashes


DEST = Path("data/derived/test_50hz_6db")
TEST_SIZE = 2158
FS = 500
FREQUENCY_HZ = 50
PHASE_RAD = 0.0
SINE = np.sin(2 * np.pi * FREQUENCY_HZ * np.arange(SAMPLES) / FS + PHASE_RAD)
FIELDS = [*META, "version", "format", "waveform_path", "snr_db", "flat_leads",
          "noise_frequency_hz", "noise_phase_rad"]


def read_sources():
    manifests = ROOT / "data/manifests"
    with (manifests / "test.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    records = {int(row["ecg_id"]): row for row in rows}
    if len(records) != TEST_SIZE or len(rows) != TEST_SIZE:
        raise ValueError(f"Expected {TEST_SIZE} distinct test ECGs")
    for row in rows:
        labels = [row[c] for c in CLASSES]
        if (row["split"] != "test" or row["strat_fold"] != "10" or not row["patient_id"]
                or "1" not in labels or not set(labels) <= {"0", "1"}):
            raise ValueError(f"Invalid test metadata for ECG {row['ecg_id']}")
    patients = {row["patient_id"] for row in rows}
    for split in ("train", "validation"):
        with (manifests / f"{split}.csv").open(encoding="utf-8", newline="") as stream:
            other = list(csv.DictReader(stream))
        if patients.intersection(row["patient_id"] for row in other):
            raise ValueError(f"Test patients overlap with {split}")
        if records.keys() & {int(row["ecg_id"]) for row in other}:
            raise ValueError(f"Test ECGs overlap with {split}")
    files = [row["filename_hr"] + suffix for row in rows for suffix in (".hea", ".dat")]
    checked = verify_hashes(ROOT / PTB, files)
    print(f"Verified {checked} source ECG files against official checksums", flush=True)
    return records


def add_powerline(clean):
    # Identical copies allow reuse of the training generator's per-lead RMS scaling.
    return augment(clean, [np.stack([SINE, SINE])])


def check_waveform(clean, noisy):
    """Check actual saved noise using its SNR, frequency and phase."""
    if noisy.shape != clean.shape or noisy.dtype != np.float32 or not np.isfinite(noisy).all():
        raise ValueError("Invalid saved test waveform")
    active = np.ptp(clean, axis=1) != 0
    if not np.array_equal(noisy[~active], clean[~active].astype(np.float32)):
        raise ValueError("A constant lead changed")
    added = noisy[active].astype(np.float64) - clean[active]
    if not active.any():
        return np.array([], dtype=float)
    db = 20 * np.log10(clean[active].std(axis=1) / added.std(axis=1))
    if not np.isfinite(db).all() or np.any(np.abs(db - SNR_DB) > TOLERANCE_DB):
        raise ValueError(f"Measured test SNR outside tolerance: {db}")
    peak_bins = np.abs(np.fft.rfft(added, axis=1)).argmax(axis=1)
    if not np.all(peak_bins == FREQUENCY_HZ * SAMPLES / FS):
        raise ValueError("Added noise has the wrong peak frequency")
    expected = (SINE - SINE.mean()) / SINE.std()
    if not np.allclose(added / added.std(axis=1, keepdims=True), expected, rtol=0, atol=1e-5):
        raise ValueError("Added noise does not match the declared 50 Hz sine and phase")
    return db


def manifest_row(source, version, flat):
    original = version == "original"
    path = PTB / source["filename_hr"] if original else DEST / wave_name(int(source["ecg_id"]))
    row = {field: source[field] for field in META}
    row.update(
        version=version, format="wfdb" if original else "npy", waveform_path=path.as_posix(),
        snr_db="" if original else str(SNR_DB), flat_leads=",".join(flat),
        noise_frequency_hz="" if original else str(FREQUENCY_HZ),
        noise_phase_rad="" if original else str(PHASE_RAD),
    )
    return row


def progress(stage, count, total):
    if count % 500 == 0 or count == total:
        print(f"{stage}: {count}/{total} records", flush=True)


def generate(folder, records):
    (folder / "waveforms").mkdir(parents=True)
    with (folder / "test.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for count, ecg_id in enumerate(sorted(records), 1):
            source = records[ecg_id]
            noisy, flat = add_powerline(read_ecg(source))
            np.save(folder / wave_name(ecg_id), noisy, allow_pickle=False)
            writer.writerow(manifest_row(source, "original", flat))
            writer.writerow(manifest_row(source, "50hz", flat))
            progress("Generated", count, len(records))


def verify(folder, records):
    with (folder / "test.csv").open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != FIELDS:
            raise ValueError("Unexpected test manifest columns")
        rows = list(reader)
    index = {(int(row["ecg_id"]), row["version"]): row for row in rows}
    if len(index) != 2 * len(records) or len(index) != len(rows):
        raise ValueError("Unexpected test count or duplicate ECG/version pairs")
    if sum(1 for _ in (folder / "waveforms").glob("*.npy")) != len(records):
        raise ValueError("Unexpected number of test waveform files")
    exceptions = {}
    lowest, highest, measured_leads = float("inf"), -float("inf"), 0
    for count, ecg_id in enumerate(sorted(records), 1):
        source = records[ecg_id]
        clean = read_ecg(source)
        flat = [LEADS[i] for i in np.flatnonzero(np.ptp(clean, axis=1) == 0)]
        for version in ("original", "50hz"):
            if index.get((ecg_id, version)) != manifest_row(source, version, flat):
                raise ValueError(f"ECG {ecg_id}: wrong labels, paths or noise parameters")
        noisy = np.load(folder / wave_name(ecg_id), allow_pickle=False)
        db = check_waveform(clean, noisy)
        if db.size:
            lowest, highest = min(lowest, float(db.min())), max(highest, float(db.max()))
            measured_leads += db.size
        if flat:
            exceptions[str(ecg_id)] = flat
        progress("Verified", count, len(records))
    if not measured_leads:
        raise ValueError("Test set has no measurable ECG leads")
    return {
        "split": "test", "strat_fold": 10, "fs_hz": FS, "shape": [12, SAMPLES],
        "dtype": "float32", "units": "mV", "noise_frequency_hz": FREQUENCY_HZ,
        "noise_phase_rad": PHASE_RAD, "original_records": len(records),
        "augmented_records": len(records), "manifest_rows": len(rows),
        "condition_counts": {"original": len(records), "50hz": len(records)},
        "target_snr_db": SNR_DB, "tolerance_db": TOLERANCE_DB,
        "measured_leads": measured_leads, "min_snr_db": lowest, "max_snr_db": highest,
        "frequency_and_phase_verified": True, "flat_leads": exceptions,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true", help="Recheck the published test dataset")
    args = parser.parse_args()
    records = read_sources()
    target = ROOT / DEST
    if args.verify_only or target.exists():
        saved = json.loads((target / "audit.json").read_text(encoding="utf-8"))
        report = verify(target, records)
        validate_audit(saved, report)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=".test_50hz_6db-", dir=target.parent) as temp:
            staging = Path(temp) / target.name
            generate(staging, records)
            report = verify(staging, records)
            (staging / "audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            staging.rename(target)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
