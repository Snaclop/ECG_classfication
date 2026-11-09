"""Verify official files and create patient-independent five-label manifests."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]


def diagnostic_labels(codes, mapping):
    codes = ast.literal_eval(codes) if isinstance(codes, str) else codes
    if not isinstance(codes, dict):
        raise ValueError("scp_codes must be a dictionary")
    # Zero means unknown likelihood, not absence of the statement.
    found = {mapping[code] for code in codes if code in mapping}
    return [label for label in CLASSES if label in found]


def split_for_fold(fold):
    if fold in range(1, 9):
        return "train"
    if fold == 9:
        return "validation"
    if fold == 10:
        return "test"
    raise ValueError(f"Unexpected fold {fold}")


def validate_patient_folds(pairs):
    assignments = {}
    for patient, fold in pairs:
        split = split_for_fold(fold)
        if patient in assignments and assignments[patient] != split:
            raise ValueError(f"Patient {patient} occurs in multiple splits")
        assignments[patient] = split


def verify_hashes(folder, selected=None):
    manifest = folder / "SHA256SUMS.txt"
    checked = 0
    pending = set(selected) if selected is not None else None
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split(maxsplit=1)
        relative = relative.lstrip("*")
        if pending is not None and relative not in pending:
            continue
        path = (folder / relative).resolve()
        if not path.is_relative_to(folder.resolve()):
            raise ValueError(f"Unsafe checksum path {relative}")
        # Checksum lists normally exclude themselves, handle that explicitly.
        if path == manifest.resolve():
            continue
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"SHA256 mismatch: {path}")
        checked += 1
        if pending is not None:
            pending.remove(relative)
        if checked % 10000 == 0:
            print(f"Verified {checked} files in {folder.name}", flush=True)
    if pending:
        raise ValueError(f"Required files absent from checksum manifest: {sorted(pending)}")
    if not checked:
        raise ValueError(f"Empty checksum manifest: {manifest}")
    return checked


def main():
    import numpy as np
    import pandas as pd
    import wfdb

    ptb = ROOT / "data/raw/ptb-xl-1.0.3"
    noise = ROOT / "data/raw/nstdb-1.0.0"
    report = {"ptb_version": "1.0.3", "nstdb_version": "1.0.0", "classes": CLASSES}
    noise_files = [f"{name}.{ext}" for name in ("bw", "ma", "em") for ext in ("hea", "dat")]
    report["sha256_verified_files"] = {str(ptb.relative_to(ROOT)): verify_hashes(ptb), str(noise.relative_to(ROOT)): verify_hashes(noise, noise_files)}
    report["nstdb_scope"] = "Three raw noise records only: bw/ma/em, each .hea and .dat; other NSTDB records are not required."
    frame = pd.read_csv(ptb / "ptbxl_database.csv", index_col="ecg_id")
    statements = pd.read_csv(ptb / "scp_statements.csv", index_col=0)
    mapping = statements.loc[statements.diagnostic == 1, "diagnostic_class"].dropna().to_dict()
    if frame.patient_id.isna().any() or not frame.index.is_unique:
        raise ValueError("Missing patient IDs or duplicate ECG IDs")
    validate_patient_folds(zip(frame.patient_id, frame.strat_fold))
    labels = frame.scp_codes.map(lambda codes: diagnostic_labels(codes, mapping))
    labeled = labels.map(bool)
    output = ROOT / "data/manifests"
    output.mkdir(parents=True, exist_ok=True)
    frame.loc[~labeled, ["patient_id", "strat_fold", "scp_codes"]].to_csv(output / "excluded_no_target.csv")
    manifest = frame.loc[labeled, ["patient_id", "strat_fold", "filename_hr", "filename_lr"]].copy()
    manifest["split"] = manifest.strat_fold.map(split_for_fold)
    for label in CLASSES:
        manifest[label] = labels.loc[labeled].map(lambda values: int(label in values))
    for column in ("filename_hr", "filename_lr"):
        for relative in manifest[column]:
            for suffix in (".hea", ".dat"):
                if not (ptb / (relative + suffix)).is_file():
                    raise FileNotFoundError(relative + suffix)
    manifest.to_csv(output / "all.csv")
    report.update(total_records=len(frame), labeled_records=len(manifest), excluded_no_target=int((~labeled).sum()))
    report["splits"] = {}
    for split in ("train", "validation", "test"):
        subset = manifest.loc[manifest.split == split]
        subset.to_csv(output / f"{split}.csv")
        # Read several real records per split; all files were hash-verified above.
        for relative in subset.filename_hr.head(3):
            record = wfdb.rdrecord(str(ptb / relative))
            if record.fs != 500 or record.p_signal.shape != (5000, 12) or not np.isfinite(record.p_signal).all():
                raise ValueError(f"Unexpected ECG shape/rate/values: {relative}")
        report["splits"][split] = {
            "records": len(subset), "patients": subset.patient_id.nunique(),
            "positive_counts": {c: int(subset[c].sum()) for c in CLASSES},
        }
    report["noise"] = {}
    noise_splits = {}
    for name in ("bw", "ma", "em"):
        record = wfdb.rdrecord(str(noise / name))
        if not np.isfinite(record.p_signal).all():
            raise ValueError(f"Nonfinite noise: {name}")
        report["noise"][name] = {"fs": record.fs, "shape": list(record.p_signal.shape), "units": record.units}
        # Split before resampling/filtering, with 10-second unused boundary gaps.
        gap = int(10 * record.fs)
        n = record.sig_len
        a, b = int(n * 0.6), int(n * 0.8)
        noise_splits[name] = {"fs": record.fs, "units": "original sample indices; end exclusive", "train": [0, a], "validation": [a + gap, b], "test": [b + gap, n]}
    (output / "noise_splits.json").write_text(json.dumps(noise_splits, indent=2), encoding="utf-8")
    (output / "audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
