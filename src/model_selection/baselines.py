import os

import numpy as np


def persistence(history, horizon):
    """Repeat the most recent observation across the forecast horizon."""
    if history.ndim != 2 or history.shape[1] == 0:
        raise ValueError("history must be a 2D array with at least one timestep")
    if horizon < 1:
        raise ValueError("horizon must be positive")
    return np.repeat(history[:, -1:], horizon, axis=1)


def seasonal_naive(history, horizon):
    """Use the first horizon values in the input history as the forecast."""
    if history.ndim != 2:
        raise ValueError("history must be a 2D array")
    if horizon < 1 or horizon > history.shape[1]:
        raise ValueError("horizon must be positive and no longer than history")
    return history[:, :horizon].copy()


def make_baseline_predictions(val_history, test_history, horizon):
    return {
        "persistence": (
            persistence(val_history, horizon),
            persistence(test_history, horizon),
        ),
        "seasonal_naive": (
            seasonal_naive(val_history, horizon),
            seasonal_naive(test_history, horizon),
        ),
    }


def save_baseline_predictions(predictions, y_val, y_test, outdir):
    """Write baseline forecasts using the same archive schema as trained models."""
    os.makedirs(outdir, exist_ok=True)
    for name, (val_pred, test_pred) in predictions.items():
        path = os.path.join(outdir, name + "_predictions.npz")
        np.savez_compressed(
            path,
            val_pred=val_pred.astype(np.float32),
            test_pred=test_pred.astype(np.float32),
            y_val=y_val,
            y_test=y_test,
        )
        print("Saved:", path)
