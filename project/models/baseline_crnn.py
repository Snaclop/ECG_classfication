"""Waveform CNN followed by a bidirectional LSTM for the backbone comparison."""
from torch import nn

from models.baseline_cnn import MODEL_SPEC as CNN_SPEC, WaveformCNN

MODEL_SPEC = {
    **CNN_SPEC,
    "name": "waveform_crnn_v1",
    "recurrent": "LSTM",
    "hidden_size": 64,
    "num_layers": 1,
    "bidirectional": True,
    "sequence_pooling": "mean",
}


class WaveformCRNN(WaveformCNN):
    def __init__(self):
        super().__init__()
        # Retain the same convolutions; pool only after sequence modeling.
        self.features = self.features[:-2]
        self.recurrent = nn.LSTM(
            input_size=128, hidden_size=64, num_layers=1,
            batch_first=True, bidirectional=True,
        )

    def encode(self, waveform):
        sequence = self.features(waveform).transpose(1, 2)  # [B,78,128]
        sequence, _ = self.recurrent(sequence)
        return sequence.mean(dim=1)
