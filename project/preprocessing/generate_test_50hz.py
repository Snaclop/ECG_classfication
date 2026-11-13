"""Generate paired original / 50 Hz, 6 dB waveforms for the held-out test fold."""
import csv
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from common.baseline_data import ROOT, read_manifest
from preprocessing.generate_train_6db import META, PTB, SAMPLES, SNR_DB, augment, read_ecg, wave_name

DEST = Path("data/derived/test_50hz_6db")
FS = 500
FREQUENCY_HZ = 50
PHASE_RAD = 0.0
SINE = np.sin(2 * np.pi * FREQUENCY_HZ * np.arange(SAMPLES) / FS + PHASE_RAD)
FIELDS = [*META, "version", "format", "waveform_path", "snr_db", "flat_leads",
          "noise_frequency_hz", "noise_phase_rad"]


def add_powerline(clean):
    # Identical copies allow reuse of the training generator's per-lead RMS scaling.
    return augment(clean, [np.stack([SINE, SINE])])


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


def main():
    target = ROOT / DEST
    target.mkdir(parents=True, exist_ok=False)
    records = {int(row["ecg_id"]): row for row in read_manifest(ROOT / "data/manifests/test.csv")}
    generate(target, records)
    print(f"50 Hz data: {target}", flush=True)


if __name__ == "__main__":
    main()
