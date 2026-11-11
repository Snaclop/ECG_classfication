"""Derived Peak encoding"""
import numpy as np


def encode_dp1(waveform):
    """Encode [lead, time] without smoothing; retain the original sample positions.

    Use d[t] = x[t] - x[t-1]. A nonpositive-to-positive transition emits +1;
    a nonnegative-to-negative transition emits -1. Zero derivatives participate
    in these comparisons.
    """
    return _encode(waveform, order=1)


def encode_dp2(waveform):
    """Encode crossings of the waveform's second difference without smoothing.

    Use d2[t] = (x[t]-x[t-1]) - (x[t-1]-x[t-2]). The first three samples
    lack two valid second differences and are set to zero.
    """
    return _encode(waveform, order=2)


def _encode(waveform, order):
    x = np.asarray(waveform, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] < order + 2 or not np.isfinite(x).all():
        raise ValueError(f"DP{order} requires a finite [lead, time] array with at least {order + 2} samples")
    delta = np.diff(x, n=order, axis=1)
    positive = (delta[:, 1:] > 0) & (delta[:, :-1] <= 0)
    negative = (delta[:, 1:] < 0) & (delta[:, :-1] >= 0)
    encoded = np.zeros(x.shape, dtype=np.int8)
    encoded[:, order + 1:] = positive.astype(np.int8) - negative.astype(np.int8)
    return encoded
