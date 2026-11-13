"""Multilabel metrics and validation-only threshold fitting."""
import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from common.baseline_data import CLASSES


def classification_metrics(labels, probabilities, thresholds=0.5):
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    thresholds = np.broadcast_to(np.asarray(thresholds, dtype=np.float64), (5,))
    predictions = probabilities >= thresholds
    per_class = {}
    for i, name in enumerate(CLASSES):
        positive = int(labels[:, i].sum())
        valid = 0 < positive < len(labels)
        per_class[name] = {
            "positives": positive,
            "auroc": float(roc_auc_score(labels[:, i], probabilities[:, i])) if valid else None,
            "ap": float(average_precision_score(labels[:, i], probabilities[:, i])) if valid else None,
            "f1": float(f1_score(labels[:, i], predictions[:, i], zero_division=0)),
            "undefined_reason": None if valid else "Class lacks positive or negative examples; ranking metrics unavailable",
        }
    # Never silently change the classes contributing to a macro metric.
    macro = {}
    for metric in ("auroc", "ap", "f1"):
        values = [per_class[c][metric] for c in CLASSES]
        macro[metric] = float(np.mean(values)) if all(v is not None for v in values) else None
    return {"samples": len(labels), "per_class": per_class, "macro": macro,
            "valid_auroc_classes": sum(v["auroc"] is not None for v in per_class.values()),
            "valid_ap_classes": sum(v["ap"] is not None for v in per_class.values()),
            "ap_definition": "sklearn average_precision_score (not trapezoidal integration)"}


def select_thresholds(labels, probabilities):
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    thresholds = []
    for column in range(5):
        scored = []
        for percent in range(5, 96):
            prediction = probabilities[:, column] >= percent / 100
            tp = int(np.sum(prediction & (labels[:, column] == 1)))
            denominator = int(prediction.sum() + labels[:, column].sum())
            score = 2 * tp / denominator if denominator else 0.0
            scored.append((score, -abs(percent - 50), -percent))
        thresholds.append(-max(scored)[2] / 100)
    return np.array(thresholds, dtype=np.float64)


def is_better(auroc, loss, best):
    return auroc is not None and (best is None or auroc > best["auroc"]
                                 or (auroc == best["auroc"] and loss < best["loss"]))
