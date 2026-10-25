import numpy as np


def conformalize(y_val, val_lower, val_upper, test_lower, test_upper, coverage=0.8):
    """Split-conformal calibration of intervals (CQR), per horizon.

    Native interval models pass their quantile bounds. Point forecasters pass
    lower = upper = forecast, which gives the symmetric forecast +/- quantile(|residual|) interval.
    """
    scores = np.maximum(val_lower - y_val, y_val - val_upper)
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * coverage) / n)
    q = np.quantile(scores, level, axis=0)
    return test_lower - q, test_upper + q