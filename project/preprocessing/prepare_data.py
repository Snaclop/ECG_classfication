"""Create the official-fold, five-label manifests and noise intervals."""
import ast
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import wfdb

from common.baseline_data import CLASSES, ROOT


def diagnostic_labels(codes, mapping):
    codes = ast.literal_eval(codes) if isinstance(codes, str) else codes
    # Zero means unknown likelihood, not absence of the statement.
    found = {mapping[code] for code in codes if code in mapping}
    return [label for label in CLASSES if label in found]


def split_for_fold(fold):
    return {**dict.fromkeys(range(1, 9), "train"), 9: "validation", 10: "test"}[fold]


def main():
    ptb = ROOT / "data/raw/ptb-xl-1.0.3"
    noise = ROOT / "data/raw/nstdb-1.0.0"
    output = ROOT / "data/manifests"
    output.mkdir(parents=True, exist_ok=False)

    frame = pd.read_csv(ptb / "ptbxl_database.csv", index_col="ecg_id")
    statements = pd.read_csv(ptb / "scp_statements.csv", index_col=0)
    mapping = statements.loc[statements.diagnostic == 1, "diagnostic_class"].dropna().to_dict()
    labels = frame.scp_codes.map(lambda codes: diagnostic_labels(codes, mapping))
    labeled = labels.map(bool)
    frame.loc[~labeled, ["patient_id", "strat_fold", "scp_codes"]].to_csv(output / "excluded_no_target.csv")
    manifest = frame.loc[labeled, ["patient_id", "strat_fold", "filename_hr", "filename_lr"]].copy()
    manifest["split"] = manifest.strat_fold.map(split_for_fold)
    for label in CLASSES:
        manifest[label] = labels.loc[labeled].map(lambda values: int(label in values))
    manifest.to_csv(output / "all.csv")
    for split in ("train", "validation", "test"):
        subset = manifest.loc[manifest.split == split]
        subset.to_csv(output / f"{split}.csv")
        print(f"{split}: {len(subset)} records", flush=True)

    noise_splits = {}
    for name in ("bw", "ma", "em"):
        record = wfdb.rdheader(str(noise / name))
        gap = int(10 * record.fs)
        n = record.sig_len
        a, b = int(n * 0.6), int(n * 0.8)
        noise_splits[name] = {
            "fs": record.fs, "units": "original sample indices; end exclusive",
            "train": [0, a], "validation": [a + gap, b], "test": [b + gap, n],
        }
    (output / "noise_splits.json").write_text(json.dumps(noise_splits, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
