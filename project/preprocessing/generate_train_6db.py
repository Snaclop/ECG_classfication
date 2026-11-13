"""Generate paired PTB-XL / NSTDB training waveforms at 6 dB."""
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy.signal import resample_poly
import wfdb

from common.baseline_data import CLASSES, LEADS, ROOT, read_manifest

MANIFESTS = ROOT / "data/manifests"
PTB = Path("data/raw/ptb-xl-1.0.3")
NSTDB = ROOT / "data/raw/nstdb-1.0.0"
DEST = Path("data/derived/train_6db")
NOISES = ("bw", "ma", "em")
CONDITIONS = (*NOISES, "mix")
TRAIN_SIZE = 17_084
SEED = 42
SNR_DB = 6.0
WINDOW_360 = 3600
SAMPLES = 5000
META = ["ecg_id", "patient_id", "strat_fold", "split", *CLASSES]
FIELDS = [*META, "version", "format", "waveform_path", "snr_db", "flat_leads",
          *(f"{name}_start_360" for name in NOISES)]


def read_sources():
    records = {int(row["ecg_id"]): row for row in read_manifest(MANIFESTS / "train.csv")}
    splits = json.loads((MANIFESTS / "noise_splits.json").read_text(encoding="utf-8"))
    bounds, noises = {}, {}
    for name in NOISES:
        begin, end = splits[name]["train"]
        bounds[name] = (begin, end)
        noises[name] = wfdb.rdrecord(str(NSTDB / name), sampfrom=begin, sampto=end).p_signal
    return records, bounds, noises


def make_recipes(ids, bounds):
    """Choose balanced conditions and absolute 360 Hz start samples, independent of CSV order."""
    rng = np.random.default_rng(SEED)
    shuffled = rng.permutation(sorted(ids))
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
    return clean


def noise_pieces(starts, bounds, noises):
    pieces = []
    for name, start in starts.items():
        begin, end = bounds[name]
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
        mixture += centered / rms
    mixture_rms = mixture.std(axis=1, keepdims=True)
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


def main():
    target = ROOT / DEST
    target.mkdir(parents=True, exist_ok=False)
    records, bounds, noises = read_sources()
    recipes = make_recipes(records, bounds)
    generate(target, records, bounds, noises, recipes)
    print(f"Training data: {target}", flush=True)


if __name__ == "__main__":
    main()
