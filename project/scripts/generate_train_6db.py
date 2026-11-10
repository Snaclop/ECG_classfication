"""Generate and verify the paired PTB-XL / NSTDB 6 dB training set."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.signal import resample_poly
import wfdb

if __package__:
    from .prepare_data import CLASSES, ROOT
else:
    from prepare_data import CLASSES, ROOT


MANIFESTS = ROOT / "data/manifests"
PTB = Path("data/raw/ptb-xl-1.0.3")
NSTDB = ROOT / "data/raw/nstdb-1.0.0"
DEST = Path("data/derived/train_6db")
LEADS = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
NOISES = ("bw", "ma", "em")
CONDITIONS = (*NOISES, "mix")
TRAIN_SIZE = 17_084
SEED = 42
SNR_DB = 6.0
TOLERANCE_DB = 0.01
WINDOW_360 = 3600
SAMPLES = 5000
META = ["ecg_id", "patient_id", "strat_fold", "split", *CLASSES]
FIELDS = [*META, "version", "format", "waveform_path", "snr_db", "flat_leads",
          *(f"{name}_start_360" for name in NOISES)]


def read_sources():
    with (MANIFESTS / "train.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    records = {int(row["ecg_id"]): row for row in rows}
    if len(records) != TRAIN_SIZE or len(rows) != TRAIN_SIZE:
        raise ValueError(f"Expected {TRAIN_SIZE} distinct training ECGs, got {len(records)}")
    for ecg_id, row in records.items():
        labels = [row[c] for c in CLASSES]
        if (row["split"] != "train" or int(row["strat_fold"]) not in range(1, 9)
                or "1" not in labels or not set(labels) <= {"0", "1"}):
            raise ValueError(f"Invalid training metadata for ECG {ecg_id}")

    splits = json.loads((MANIFESTS / "noise_splits.json").read_text(encoding="utf-8"))
    bounds, noises = {}, {}
    for name in NOISES:
        split = splits[name]
        begin, end = split["train"]
        if (split["fs"] != 360 or not 0 <= begin <= end - WINDOW_360
                or not end <= split["validation"][0] < split["validation"][1] <= split["test"][0]):
            raise ValueError(f"Invalid training bounds for {name}: {split}")
        record = wfdb.rdrecord(str(ROOT / NSTDB / name), sampfrom=begin, sampto=end)
        if (record.fs != 360 or record.sig_name != ["noise1", "noise2"]
                or record.p_signal.shape != (end - begin, 2) or not np.isfinite(record.p_signal).all()):
            raise ValueError(f"Unexpected NSTDB record {name}")
        bounds[name] = (begin, end)
        noises[name] = record.p_signal
    return records, bounds, noises


def make_recipes(ids, bounds):
    """Choose balanced conditions and absolute 360 Hz start samples, independent of CSV order."""
    rng = np.random.default_rng(SEED)
    shuffled = rng.permutation(sorted(ids))
    if len(shuffled) % len(CONDITIONS):
        raise ValueError("Training records must divide equally across conditions")
    recipes = {}
    for condition, group in zip(CONDITIONS, np.split(shuffled, len(CONDITIONS))):
        names = NOISES if condition == "mix" else (condition,)
        for ecg_id in group:
            starts = {}
            for name in names:
                begin, end = bounds[name]
                starts[name] = int(rng.integers(begin, end - WINDOW_360 + 1))
            recipes[int(ecg_id)] = (condition, starts)
    return recipes


def read_ecg(source):
    record = wfdb.rdrecord(str(ROOT / PTB / source["filename_hr"]))
    clean = np.ascontiguousarray(record.p_signal.T)
    if (record.fs != 500 or record.sig_name != LEADS or record.units != ["mV"] * 12
            or clean.shape != (12, SAMPLES) or not np.isfinite(clean).all()):
        raise ValueError(f"Unexpected ECG waveform: {source['ecg_id']}")
    return clean


def noise_pieces(starts, bounds, noises):
    pieces = []
    for name, start in starts.items():
        begin, end = bounds[name]
        if not begin <= start <= end - WINDOW_360:
            raise ValueError(f"{name} window at {start} falls outside [{begin}, {end})")
        offset = start - begin
        window = noises[name][offset:offset + WINDOW_360]
        # Reflect only the selected ten seconds at its boundaries when filtering.
        pieces.append(resample_poly(window.T, 25, 18, axis=1, padtype="reflect"))
    return pieces


def augment(clean, pieces):
    mixture = np.zeros((2, SAMPLES), dtype=np.float64)
    for piece in pieces:
        centered = piece - piece.mean(axis=1, keepdims=True)
        rms = np.sqrt(np.mean(centered ** 2, axis=1, keepdims=True))
        if np.any(rms == 0):
            raise ValueError("A noise component has zero AC power")
        mixture += centered / rms
    mixture_rms = mixture.std(axis=1, keepdims=True)
    if np.any(mixture_rms == 0):
        raise ValueError("Mixed noise has zero AC power")
    normalized = np.repeat(mixture / mixture_rms, 6, axis=0)
    signal_rms = clean.std(axis=1, keepdims=True)
    flat = np.ptp(clean, axis=1) == 0
    signal_rms[flat] = 0
    noisy = clean + normalized * signal_rms * 10 ** (-SNR_DB / 20)
    return noisy.astype(np.float32), [LEADS[i] for i in np.flatnonzero(flat)]


def wave_name(ecg_id):
    return Path("waveforms") / f"{ecg_id:05d}.npy"


def manifest_row(source, version, starts, flat):
    original = version == "original"
    path = PTB / source["filename_hr"] if original else DEST / wave_name(int(source["ecg_id"]))
    row = {field: source[field] for field in META}
    row.update(version=version, format="wfdb" if original else "npy", waveform_path=path.as_posix(),
               snr_db="" if original else str(SNR_DB), flat_leads=",".join(flat))
    for name in NOISES:
        row[f"{name}_start_360"] = str(starts.get(name, ""))
    return row


def progress(stage, count):
    if count % 4000 == 0 or count == TRAIN_SIZE:
        print(f"{stage}: {count}/{TRAIN_SIZE} records", flush=True)


def generate(folder, records, bounds, noises, recipes):
    (folder / "waveforms").mkdir(parents=True)
    with (folder / "train.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for count, ecg_id in enumerate(sorted(records), 1):
            source = records[ecg_id]
            condition, starts = recipes[ecg_id]
            noisy, flat = augment(read_ecg(source), noise_pieces(starts, bounds, noises))
            np.save(folder / wave_name(ecg_id), noisy, allow_pickle=False)
            writer.writerow(manifest_row(source, "original", {}, flat))
            writer.writerow(manifest_row(source, condition, starts, flat))
            progress("Generated", count)


def verify(folder, records, bounds, noises, recipes):
    """Reopen every file, check pairing and provenance, and measure added power from the difference."""
    with (folder / "train.csv").open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != FIELDS:
            raise ValueError("Unexpected training manifest columns")
        rows = list(reader)
    index = {(int(row["ecg_id"]), row["version"]): row for row in rows}
    if len(index) != 2 * len(records) or len(index) != len(rows):
        raise ValueError("Unexpected manifest count or duplicate ECG/version pairs")
    if sum(1 for _ in (folder / "waveforms").glob("*.npy")) != len(records):
        raise ValueError("Unexpected number of augmented waveform files")

    counts, exceptions = Counter(), {}
    lowest, highest, valid_leads = float("inf"), -float("inf"), 0
    for count, ecg_id in enumerate(sorted(records), 1):
        source = records[ecg_id]
        condition, starts = recipes[ecg_id]
        clean = read_ecg(source)
        active = np.ptp(clean, axis=1) != 0
        flat = [LEADS[i] for i in np.flatnonzero(~active)]
        for version, offsets in (("original", {}), (condition, starts)):
            if index.get((ecg_id, version)) != manifest_row(source, version, offsets, flat):
                raise ValueError(f"ECG {ecg_id}: wrong {version} metadata, path, labels or offsets")
        noisy = np.load(folder / wave_name(ecg_id), allow_pickle=False)
        if noisy.shape != clean.shape or noisy.dtype != np.float32 or not np.isfinite(noisy).all():
            raise ValueError(f"ECG {ecg_id}: invalid stored waveform")
        if flat:
            exceptions[str(ecg_id)] = flat
            if not np.array_equal(noisy[~active], clean[~active].astype(np.float32)):
                raise ValueError(f"ECG {ecg_id}: flat lead changed")
        added = noisy[active].astype(np.float64) - clean[active]
        db = 20 * np.log10(clean[active].std(axis=1) / added.std(axis=1))
        if not np.isfinite(db).all() or np.any(np.abs(db - SNR_DB) > TOLERANCE_DB):
            raise ValueError(f"ECG {ecg_id}: measured SNR outside tolerance: {db}")
        expected, _ = augment(clean, noise_pieces(starts, bounds, noises))
        if not np.allclose(noisy, expected, rtol=1e-6, atol=1e-7):
            raise ValueError(f"ECG {ecg_id}: waveform does not match its recorded noise windows")
        lowest, highest = min(lowest, float(db.min())), max(highest, float(db.max()))
        valid_leads += db.size
        counts[condition] += 1
        progress("Verified", count)

    if counts != {name: TRAIN_SIZE // 4 for name in CONDITIONS} or exceptions != {"12722": ["V5"]}:
        raise ValueError(f"Unexpected condition counts or flat leads: {counts}, {exceptions}")
    return {"seed": SEED, "fs_hz": 500, "shape": [12, SAMPLES], "dtype": "float32", "units": "mV",
            "original_records": len(records), "augmented_records": sum(counts.values()),
            "manifest_rows": len(rows), "condition_counts": dict(counts),
            "target_snr_db": SNR_DB, "tolerance_db": TOLERANCE_DB, "measured_leads": valid_leads,
            "min_snr_db": lowest, "max_snr_db": highest, "flat_leads": exceptions,
            "train_noise_bounds_360hz_end_exclusive": {k: list(v) for k, v in bounds.items()}}


def validate_audit(saved, measured):
    comparable = saved.copy()
    # Allow sub-microdecibel differences when recomputing on another platform.
    for key in ("min_snr_db", "max_snr_db"):
        if key in saved and np.isclose(saved[key], measured[key], rtol=0, atol=1e-6):
            comparable[key] = measured[key]
    if comparable != measured:
        raise ValueError("Published audit.json does not match the measured training set")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true", help="Recheck the published dataset")
    args = parser.parse_args()
    records, bounds, noises = read_sources()
    recipes = make_recipes(records, bounds)
    target = ROOT / DEST
    if args.verify_only or target.exists():
        saved = json.loads((target / "audit.json").read_text(encoding="utf-8"))
        report = verify(target, records, bounds, noises, recipes)
        validate_audit(saved, report)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=".train_6db-", dir=target.parent) as temp:
            staging = Path(temp) / target.name
            generate(staging, records, bounds, noises, recipes)
            report = verify(staging, records, bounds, noises, recipes)
            (staging / "audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            staging.rename(target)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
