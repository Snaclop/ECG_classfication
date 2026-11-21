"""Single-branch CNN receiving only first-order Derived Peak encoding."""
from models.baseline_cnn import MODEL_SPEC as CNN_SPEC, WaveformCNN

MODEL_SPEC = {
    **CNN_SPEC,
    "name": "dp1_cnn_v1",
    "uses_dp1": True,
    "dp1_only": True,
    "input_spec": {
        "shape": [12, 5000], "fs_hz": 500, "units": "signed events",
        "dtype": "float32", "preprocessing": "dp1", "smoothing": "none",
    },
}


class DP1CNN(WaveformCNN):
    def forward(self, dp1):
        return self.classifier(self.encode(dp1))
