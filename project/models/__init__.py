"""Model construction shared by training, evaluation and prediction."""
from models.baseline_cnn import MODEL_SPEC as CNN_SPEC, WaveformCNN
from models.baseline_crnn import MODEL_SPEC as CRNN_SPEC, WaveformCRNN

MODELS = {
    CNN_SPEC["name"]: (WaveformCNN, CNN_SPEC),
    CRNN_SPEC["name"]: (WaveformCRNN, CRNN_SPEC),
}


def build_model(name):
    model_class, specification = MODELS[name]
    return model_class(), specification
