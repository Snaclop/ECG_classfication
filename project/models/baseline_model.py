"""Ordinary 1D CNN with a reusable 128-dimensional waveform encoder."""
from torch import nn

MODEL_SPEC = {"name": "waveform_cnn_v1", "channels": [32, 64, 128, 128],
              "kernel_size": 7, "stride": 1, "padding": 3, "conv_bias": False,
              "max_pool_first_three": 4, "dropout": 0.2, "outputs": 5}


class WaveformCNN(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        incoming = 12
        for index, outgoing in enumerate(MODEL_SPEC["channels"]):
            layers.extend([nn.Conv1d(incoming, outgoing, 7, padding=3, bias=False),
                           nn.BatchNorm1d(outgoing), nn.ReLU()])
            if index < 3:
                layers.append(nn.MaxPool1d(4))
            incoming = outgoing
        self.features = nn.Sequential(*layers, nn.AdaptiveAvgPool1d(1), nn.Flatten(1))
        self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(128, 5))

    def encode(self, waveform):
        return self.features(waveform)

    def forward(self, waveform):
        return self.classifier(self.encode(waveform))
