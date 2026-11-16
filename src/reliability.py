"""Reliability and safety methods for EV charging station occupancy forecasting.

Covers:
1. Data-level: Jittering (Additive Gaussian noise augmentation for time series)
2. Inference-level: Input & Output Sanitization (physical bounds, missing value imputation, rate-of-change filtering)
3. Inference-level: Confidence Thresholds & Abstention (selective forecasting based on interval uncertainty)
4. Model-level: Ensembling (multi-model predictive aggregation)
"""
import numpy as np
import torch
from scipy.optimize import minimize


# =====================================================================
# 1. DATA-LEVEL: TIME-SERIES JITTERING AUGMENTATION
# =====================================================================

class JitterAugmentation:
    """Additive Gaussian noise augmentation tailored for time-series forecasting.

    Adds zero-mean Gaussian noise to historical occupancy values during training,
    simulating sensor noise, minor driver arrival/departure variance, and telemetry jitter.
    Maintains physical reality by clipping negative occupancy to 0.
    """
    def __init__(self, sigma=0.03, clip_min=0.0, relative=False):
        """
        Args:
            sigma: Standard deviation of Gaussian noise.
            clip_min: Minimum allowable occupancy value (physical constraint >= 0).
            relative: If True, noise scales with signal magnitude; if False, absolute noise.
        """
        self.sigma = sigma
        self.clip_min = clip_min
        self.relative = relative

    def __call__(self, x):
        """Apply jitter to a NumPy array or PyTorch tensor."""
        if isinstance(x, torch.Tensor):
            return self.jitter_tensor(x)
        return self.jitter_numpy(x)

    def jitter_numpy(self, hist, rng=None):
        """Augment a NumPy history array of shape (..., seq_len)."""
        if self.sigma <= 0.0:
            return hist
        rng = rng or np.random.default_rng()
        if self.relative:
            std = np.maximum(np.abs(hist) * self.sigma, 1e-4)
            noise = rng.normal(0.0, std, size=hist.shape).astype(hist.dtype)
        else:
            noise = rng.normal(0.0, self.sigma, size=hist.shape).astype(hist.dtype)
        return np.maximum(hist + noise, self.clip_min)

    def jitter_tensor(self, hist_tensor):
        """Augment a PyTorch tensor batch of shape (batch, seq_len, 1)."""
        if self.sigma <= 0.0:
            return hist_tensor
        if self.relative:
            std = torch.clamp(torch.abs(hist_tensor) * self.sigma, min=1e-4)
            noise = torch.randn_like(hist_tensor) * std
        else:
            noise = torch.randn_like(hist_tensor) * self.sigma
        return torch.clamp(hist_tensor + noise, min=self.clip_min)


# =====================================================================
# 2. INFERENCE-LEVEL: INPUT & OUTPUT SANITIZATION
# =====================================================================

class InputSanitizer:
    """Inference-time defensive validation and sanitization for EV station telemetry.

    Protects downstream models against:
    - Missing packets / NaNs / Infs (forward fills or interpolates).
    - Impossible negative occupancy readings from sensor reset glitches (clips to >= 0).
    - Readings exceeding physical station charging capacity (e.g. > 54 ports for Caltech).
    - Severe single-step transient sensor glitches / spikes (rate-of-change clamping).
    - Out-of-bound calendar cyclic features (clips sin/cos features to [-1, 1]).
    """
    def __init__(self, max_capacity=54.0, max_rate_per_step=15.0):
        """
        Args:
            max_capacity: Physical maximum number of charging ports at the station.
            max_rate_per_step: Maximum physically plausible change in 1 15-minute interval.
        """
        self.max_capacity = float(max_capacity)
        self.max_rate_per_step = float(max_rate_per_step)

    def sanitize_history(self, hist):
        """Sanitizes raw historical occupancy sequence.

        Args:
            hist: Array of shape (N, seq_len) or (seq_len,).

        Returns:
            sanitized: Cleaned float32 array satisfying physical invariants.
            audit: Dict tracking detected issues (nan_count, negative_count, overflow_count, spike_count).
        """
        arr = np.array(hist, dtype=np.float32, copy=True)
        is_1d = (arr.ndim == 1)
        if is_1d:
            arr = arr[None, :]

        audit = {
            "nan_or_inf_count": int(np.isnan(arr).sum() + np.isinf(arr).sum()),
            "negative_count": int((arr < 0.0).sum()),
            "capacity_overflow_count": int((arr > self.max_capacity).sum()),
            "spike_count": 0,
        }

        # 1. Impute NaNs and Infs (vectorized forward fill, then backward fill for initial values)
        invalid = np.isnan(arr) | np.isinf(arr)
        if np.any(invalid):
            # Fast vectorized forward fill using index accumulation
            idx = np.where(~invalid, np.arange(arr.shape[1]), 0)
            np.maximum.accumulate(idx, axis=1, out=idx)
            arr = np.take_along_axis(arr, idx, axis=1)
            # Backward fill any remaining leading invalid values per row
            still_invalid = np.isnan(arr) | np.isinf(arr)
            if np.any(still_invalid):
                rev_arr = arr[:, ::-1]
                rev_inv = np.isnan(rev_arr) | np.isinf(rev_arr)
                rev_idx = np.where(~rev_inv, np.arange(arr.shape[1]), 0)
                np.maximum.accumulate(rev_idx, axis=1, out=rev_idx)
                rev_arr = np.take_along_axis(rev_arr, rev_idx, axis=1)
                arr = rev_arr[:, ::-1]
                arr = np.nan_to_num(arr, nan=0.0, posinf=self.max_capacity, neginf=0.0)

        # 2. Physical bound clamping [0, max_capacity]
        np.clip(arr, 0.0, self.max_capacity, out=arr)

        # 3. Rate-of-change smoothing for anomalous transient single-step spikes (vectorized)
        if arr.shape[1] > 2:
            diff_left = np.abs(arr[:, 1:-1] - arr[:, :-2]) > self.max_rate_per_step
            diff_right = np.abs(arr[:, 1:-1] - arr[:, 2:]) > self.max_rate_per_step
            spike_mask = diff_left & diff_right
            if np.any(spike_mask):
                audit["spike_count"] = int(np.sum(spike_mask))
                smoothed = 0.5 * (arr[:, :-2] + arr[:, 2:])
                arr[:, 1:-1] = np.where(spike_mask, smoothed, arr[:, 1:-1])

        return arr[0] if is_1d else arr, audit

    def sanitize_calendar(self, cal):
        """Ensures cyclic trigonometric calendar features stay within [-1.0, 1.0]."""
        arr = np.array(cal, dtype=np.float32, copy=True)
        # Features 0..3 are typically sin/cos pairs for time-of-day and day-of-week
        arr[..., :4] = np.clip(arr[..., :4], -1.0, 1.0)
        # Any binary indicator (e.g. holiday flag in index 4) clipped to [0, 1]
        if arr.shape[-1] > 4:
            arr[..., 4:] = np.clip(arr[..., 4:], 0.0, 1.0)
        return arr

    def sanitize_output(self, pred, lower=None, upper=None):
        """Enforces physical bounds [0, max_capacity] and quantile monotonicity on predictions."""
        clean_pred = np.clip(np.array(pred, dtype=np.float32), 0.0, self.max_capacity)
        if lower is None or upper is None:
            return clean_pred

        clean_lower = np.clip(np.array(lower, dtype=np.float32), 0.0, self.max_capacity)
        clean_upper = np.clip(np.array(upper, dtype=np.float32), 0.0, self.max_capacity)

        # Enforce quantile ordering: lower <= pred <= upper
        clean_lower = np.minimum(clean_lower, clean_pred)
        clean_upper = np.maximum(clean_upper, clean_pred)

        return clean_lower, clean_pred, clean_upper


# =====================================================================
# 3. INFERENCE-LEVEL: CONFIDENCE THRESHOLDS & ABSTENTION
# =====================================================================

class SelectiveForecaster:
    """Confidence-based abstention policy for safety-critical EV energy management.

    Uses prediction interval width (W = upper - lower) as an uncertainty metric.
    When uncertainty exceeds threshold tau, the forecaster abstains from automated
    action, alerting the operator or falling back to a robust baseline (Seasonal Naive).
    """
    def __init__(self, uncertainty_threshold=None, max_abstention_rate=0.10):
        """
        Args:
            uncertainty_threshold: Max allowed interval width before abstaining.
        from scipy.optimize import minimize
                                   If None, dynamically calibrated to reject at most max_abstention_rate.
            max_abstention_rate: Maximum fraction of predictions to abstain from.
        """
        self.uncertainty_threshold = uncertainty_threshold
        self.max_abstention_rate = max_abstention_rate

    def calibrate_threshold(self, val_lower, val_upper, quantile=0.90):
        """Calibrates uncertainty threshold on validation set (e.g. 90th percentile width)."""
        val_widths = np.mean(val_upper - val_lower, axis=-1)
        self.uncertainty_threshold = float(np.quantile(val_widths, quantile))
        return self.uncertainty_threshold

    def evaluate_abstention(self, pred, lower, upper, fallback_pred=None, threshold=None):
        """
        Decides which forecasts to accept and which to abstain.

        Returns:
            final_pred: Array of forecasts (model predictions where accepted, fallback where abstained).
            abstained_mask: Boolean array of shape (N,) where True indicates abstention.
            uncertainty: Average interval width per sample, shape (N,).
        """
        uncertainty = np.mean(upper - lower, axis=-1)  # average across horizon steps
        tau = threshold if threshold is not None else self.uncertainty_threshold
        if tau is None:
            tau = float(np.quantile(uncertainty, 1.0 - self.max_abstention_rate))

        abstained_mask = uncertainty > tau
        final_pred = np.array(pred, copy=True)
        if fallback_pred is not None:
            final_pred[abstained_mask] = fallback_pred[abstained_mask]

        return final_pred, abstained_mask, uncertainty

    @staticmethod
    def compute_risk_coverage_curve(errors, uncertainties, n_points=20):
        """Computes Risk (MAE) as a function of Coverage (fraction of retained predictions)."""
        coverage_levels = np.linspace(0.1, 1.0, n_points)
        curve = []
        for cov in coverage_levels:
            cutoff = np.quantile(uncertainties, cov)
            retained = uncertainties <= cutoff
            mae = float(np.mean(errors[retained]))
            curve.append({"coverage": float(cov), "cutoff_uncertainty": float(cutoff), "mae": mae})
        return curve


# =====================================================================
# 4. MODEL-LEVEL: ENSEMBLE FORECASTER
# =====================================================================

class EnsembleForecaster:
    """Ensemble aggregator combining complementary forecasting architectures.

    Supports:
    - 'equal': Equal weighting (1/M simple average).
    - 'inverse_error': Weighted by inverse validation MAE (1/MAE_m).
    - 'stacking': Constrained optimization (Super Learner) minimizing MSE on validation data.
    - 'horizon_wise': Horizon-specific optimal weights accounting for short- vs long-term model strengths.
    """
    def __init__(self, weights=None, strategy="equal"):
        """
        Args:
            weights: Optional explicit weight array/dict.
            strategy: 'equal', 'inverse_error', 'stacking', or 'horizon_wise'.
        """
        self.weights = weights
        self.strategy = strategy

    def fit_weights(self, y_val, model_val_preds, strategy="horizon_wise"):
        """Learns optimal ensemble weights from validation data.

        Args:
            y_val: Ground truth validation targets, shape (N, horizon).
            model_val_preds: List of prediction arrays [pred_1, ..., pred_M], each (N, horizon).
            strategy: 'equal', 'inverse_error', 'stacking', or 'horizon_wise'.
        """
        self.strategy = strategy
        M = len(model_val_preds)
        horizon = y_val.shape[1]

        if strategy == "equal":
            self.weights = np.ones(M, dtype=np.float32) / M

        elif strategy == "inverse_error":
            # Global inverse-MAE weights
            maes = [np.mean(np.abs(y_val - p)) for p in model_val_preds]
            inv = 1.0 / np.maximum(np.array(maes), 1e-6)
            self.weights = inv / inv.sum()

        elif strategy == "stacking":
            # Global constrained quadratic optimization: min MSE s.t. w >= 0, sum(w) = 1
            X = np.column_stack([p.ravel() for p in model_val_preds])
            y_flat = y_val.ravel()
            def loss(w):
                return np.mean((y_flat - X @ w) ** 2)
            res = minimize(loss, np.ones(M)/M, bounds=[(0, 1)] * M,
                           constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
            self.weights = res.x

        elif strategy == "horizon_wise":
            # Optimal weights per horizon step h in {0..H-1}
            weights_per_h = []
            for h in range(horizon):
                X_h = np.column_stack([p[:, h] for p in model_val_preds])
                y_h = y_val[:, h]
                def loss_h(w):
                    return np.mean((y_h - X_h @ w) ** 2)
                res = minimize(loss_h, np.ones(M)/M, bounds=[(0, 1)] * M,
                               constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
                weights_per_h.append(res.x)
            self.weights = np.array(weights_per_h)  # shape (horizon, M)

        return self.weights

    def predict(self, model_predictions):
        """Generates ensemble forecasts.

        Args:
            model_predictions: Dict of {name: pred_array} or list of pred_arrays.

        Returns:
            ensemble_pred: Weighted average forecast, shape (N, horizon).
        """
        if isinstance(model_predictions, dict):
            names = list(model_predictions.keys())
            preds = [model_predictions[k] for k in names]
        else:
            preds = model_predictions

        M = len(preds)
        if self.weights is None:
            w = np.ones(M, dtype=np.float32) / M
            return sum(weight * pred for weight, pred in zip(w, preds))

        w = np.asarray(self.weights, dtype=np.float32)

        # Check if horizon-specific weights: shape (horizon, M)
        if w.ndim == 2:
            horizon = preds[0].shape[1]
            out = np.zeros_like(preds[0])
            for h in range(horizon):
                out[:, h] = sum(w[h, m] * preds[m][:, h] for m in range(M))
            return out
        else:
            w = w / w.sum()
            return sum(weight * pred for weight, pred in zip(w, preds))
