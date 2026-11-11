"""First-order Derived Peak encoding using equation (1) of arXiv:2405.00724."""
import numpy as np


def encode_dp1(waveform):
    """Encode [lead, time] without smoothing; retain the original sample positions.

    Use d[t] = x[t] - x[t-1]. A nonpositive-to-positive transition emits +1;
    a nonnegative-to-negative transition emits -1. Zero derivatives participate
    in these comparisons. The first two samples have insufficient history
    for two real differences and are set to zero.
    """
    x = np.asarray(waveform, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] < 3 or not np.isfinite(x).all():
        raise ValueError("DP1 requires a finite [lead, time] array with at least three samples")
    delta = np.diff(x, axis=1)
    positive = (delta[:, 1:] > 0) & (delta[:, :-1] <= 0)
    negative = (delta[:, 1:] < 0) & (delta[:, :-1] >= 0)
    encoded = np.zeros(x.shape, dtype=np.int8)
    encoded[:, 2:] = positive.astype(np.int8) - negative.astype(np.int8)
    return encoded
