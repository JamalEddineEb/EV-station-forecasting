import numpy as np


def conformalize(
    y_val,
    val_lower,
    val_upper,
    test_lower,
    test_upper,
    val_history=None,
    test_history=None,
    coverage=0.80,
):
    """
    Horizon-wise conformalized quantile regression.

    Validation data calibrates the intervals.
    Test data is used only to construct the final intervals.

    Arrays have shape (n_samples, horizon).
    """
    if not 0 < coverage < 1:
        raise ValueError("coverage must be between 0 and 1")

    y_val = np.asarray(y_val)
    val_lower = np.asarray(val_lower)
    val_upper = np.asarray(val_upper)

    # CQR nonconformity scores, computed separately per horizon.
    scores = np.maximum(
        val_lower - y_val,
        y_val - val_upper,
    )

    n = scores.shape[0]
    k = int(np.ceil((n + 1) * coverage))

    if k > n:
        # Not enough calibration samples for a finite correction
        # at the requested finite-sample coverage level.
        q = np.full(scores.shape[1], np.inf)
    else:
        # Exact order statistic; avoids quantile interpolation.
        q = np.partition(scores, k - 1, axis=0)[k - 1, :]

    lower = test_lower - q[None, :]
    upper = test_upper + q[None, :]

    return lower, upper