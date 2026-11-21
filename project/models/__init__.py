"""Model construction shared by training, evaluation and prediction."""
from models.baseline_cnn import MODEL_SPEC as CNN_SPEC, WaveformCNN
from models.baseline_crnn import MODEL_SPEC as CRNN_SPEC, WaveformCRNN
from models.dp1_fusion import MODEL_SPEC as DP1_SPEC, DP1FusionCNN

MODELS = {
    CNN_SPEC["name"]: (WaveformCNN, CNN_SPEC),
    CRNN_SPEC["name"]: (WaveformCRNN, CRNN_SPEC),
    DP1_SPEC["name"]: (DP1FusionCNN, DP1_SPEC),
}


def build_model(name):
    model_class, specification = MODELS[name]
    return model_class(), specification
