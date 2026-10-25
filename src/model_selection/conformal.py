import numpy as np


def level(history, window=96):
    """Recent mean occupancy per sample (last `window` steps = 1 day at 15 min)."""
    return history[:, -window:].mean(axis=1)


def conformalize(y_val, val_lower, val_upper, test_lower, test_upper, val_history, test_history, coverage=0.8):
    """Locally scaled split-conformal calibration (CQR), per horizon.

    Typical error size is modelled as a linear function of the recent occupancy level (fitted on
    validation). Misses are divided by that scale before taking the conformal quantile, and the
    quantile is multiplied back by each test sample's own scale, so intervals widen when occupancy
    is higher than during calibration. Native interval models pass their quantile bounds; point
    forecasters pass lower = upper = forecast.
    """
    err = np.abs(y_val - (val_lower + val_upper) / 2).mean(axis=1)
    slope, intercept = np.polyfit(level(val_history), err, 1)
    if slope < 0:  # error does not grow with level: fall back to a constant scale
        slope, intercept = 0.0, err.mean()
    floor = 0.05 * err.mean()

    def scale(history):
        return np.maximum(intercept + slope * level(history), floor)[:, None]

    scores = np.maximum(val_lower - y_val, y_val - val_upper) / scale(val_history)
    n = len(scores)
    q = np.quantile(scores, min(1.0, np.ceil((n + 1) * coverage) / n), axis=0)
    return test_lower - q * scale(test_history), test_upper + q * scale(test_history)