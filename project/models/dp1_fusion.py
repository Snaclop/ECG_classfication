"""Gated fusion of waveform and first-order Derived Peak features."""
import torch
from torch import nn

from models.baseline_cnn import MODEL_SPEC as CNN_SPEC, build_cnn_features

MODEL_SPEC = {
    **CNN_SPEC,
    "name": "waveform_dp1_fusion_v1",
    "uses_dp1": True,
    "gate": "linear_softmax_per_record",
}


class DP1FusionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.waveform_encoder = build_cnn_features()
        self.dp1_encoder = build_cnn_features()
        self.gate = nn.Linear(256, 2)
        self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(128, 5))

    def forward(self, waveform, dp1):
        waveform_features = self.waveform_encoder(waveform)
        dp1_features = self.dp1_encoder(dp1)
        weights = self.gate(torch.cat((waveform_features, dp1_features), dim=1)).softmax(dim=1)
        fused = weights[:, :1] * waveform_features + weights[:, 1:] * dp1_features
        return self.classifier(fused)
