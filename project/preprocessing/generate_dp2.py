"""Add second-order DP paths to the paired waveform and DP1 manifests."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.baseline_data import ROOT, read_manifest
from common.dp_encoding import encode_dp2
from preprocessing.generate_dp1 import WAVEFORM_FIELDS, generate


def main():
    target = ROOT / "data/derived/dp2"
    inputs = {split: ROOT / "data/derived/dp1" / f"{split}.csv"
              for split in ("train", "validation", "test")}
    target.mkdir(parents=True, exist_ok=False)
    fields = [*WAVEFORM_FIELDS, "dp1_path"]
    sources = {
        split: [{key: value for key, value in row.items() if key in fields}
                for row in read_manifest(path)]
        for split, path in inputs.items()
    }
    generate(target, sources, encode_dp2, "dp2_path")
    print(f"DP2 data: {target}", flush=True)


if __name__ == "__main__":
    main()
