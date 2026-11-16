"""Simulation and Adaptation Benchmark for Concept and Data Drift (Time-of-Use Shift).

Simulates a real-world campus pricing policy change that pushes EV arrival surges
later in the day (by 8 to 12 timesteps, i.e., 2 to 3 hours).

Stages evaluated:
1. Baseline Metric: Evaluate original model on clean test set (MAE, RMSE, 80% coverage, inference latency).
2. Inject Drift: Shift occupancy curves forward by delta steps using numpy.roll() / pandas shift().
3. Degradation (Stale Model): Pass shifted test set through stale model; log spiked MAE and collapsed coverage.
4. Adaptation (Retrain): Feed shifted training block into training loop to re-fit DLinear weights,
   Ridge meta-learner, and recalculate Split-CQR quantiles on shifted calibration split.
5. Recovery (Updated Model): Evaluate recovered MAE, 80% coverage, latency, and model sizes.

Preserves all original models, datasets, and codebase without modification.

Usage:
    python src/simulate_data_drift.py --site all
    python src/simulate_data_drift.py --site Caltech --shift-steps 10
    python src/simulate_data_drift.py --site JPL --shift-steps 10
"""
import argparse
import io
import os
import sys
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

# Ensure src is in python search path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from model_selection.conformal import conformalize
from models.train_neural import (
    QUANTILES,
    DLinear,
    PinballLoss,
    fit,
    make_tensors,
    window_scale,
)

SITE_CONFIGS = {
    "Caltech": {
        "data_path": "outputs/caltech_occupancy_data.npz",
        "dlinear_path": "outputs/dlinear_caltech.pt",
        "ridge_path": "outputs/benchmark_caltech/ridge.joblib",
        "max_capacity": 54.0,
    },
    "JPL": {
        "data_path": "outputs/jpl_occupancy_data.npz",
        "dlinear_path": "outputs/dlinear_jpl.pt",
        "ridge_path": "outputs/benchmark_jpl/ridge.joblib",
        "max_capacity": 56.0,
    },
}


def bootstrap_ci(metric_fn, y_true, y_pred, n_boot=500, ci=0.95):
    """Compute bootstrapped 95% confidence interval for a metric."""
    n = len(y_true)
    rng = np.random.default_rng(42)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        scores.append(metric_fn(y_true[idx], y_pred[idx]))
    alpha = (1.0 - ci) / 2.0
    return float(np.percentile(scores, alpha * 100)), float(np.percentile(scores, (1.0 - alpha) * 100))


def run_latency_benchmark(fn, hist, cal, warmup=15, iters=100, bs=64):
    """Measures single-sample and batched inference latency and throughput."""
    span = len(hist) - bs + 1
    # Warmup
    for i in range(warmup):
        s = (i * bs) % span
        fn(hist[s:s + bs], cal[s:s + bs])
    # Batch timing
    latencies = []
    t0 = time.perf_counter()
    for i in range(iters):
        s = (i * bs) % span
        t = time.perf_counter()
        fn(hist[s:s + bs], cal[s:s + bs])
        latencies.append(time.perf_counter() - t)
    total_time = time.perf_counter() - t0
    batch_p50 = float(np.percentile(latencies, 50) * 1000)
    batch_p90 = float(np.percentile(latencies, 90) * 1000)
    throughput = (iters * bs) / total_time

    # Single-sample warmup & timing
    for i in range(warmup):
        s = i % len(hist)
        fn(hist[s:s + 1], cal[s:s + 1])
    single_latencies = []
    for i in range(iters):
        s = i % len(hist)
        t = time.perf_counter()
        fn(hist[s:s + 1], cal[s:s + 1])
        single_latencies.append(time.perf_counter() - t)
    single_p50 = float(np.percentile(single_latencies, 50) * 1000)
    single_p90 = float(np.percentile(single_latencies, 90) * 1000)

    return {
        "single_p50_ms": single_p50,
        "single_p90_ms": single_p90,
        "batch_p50_ms": batch_p50,
        "batch_p90_ms": batch_p90,
        "throughput_sps": throughput,
    }


def get_model_size_and_params(model):
    """Computes total parameter count and serialized in-memory checkpoint size in bytes."""
    buf = io.BytesIO()
    if isinstance(model, torch.nn.Module):
        params = sum(p.numel() for p in model.parameters())
        torch.save(model.state_dict(), buf)
    elif hasattr(model, "models"):
        # StackingMetaLearner
        params = int(sum(m.coef_.size + m.intercept_.size for m in model.models))
        joblib.dump(model, buf)
    else:
        # Standard sklearn model
        params = int(model.coef_.size + model.intercept_.size)
        joblib.dump(model, buf)
    return params, buf.tell()


class StackingRidgeMetaLearner:
    """Horizon-wise Ridge regression meta-learner combining multiple base forecasts."""
    def __init__(self, alpha=10.0):
        self.alpha = float(alpha)
        self.models = []

    def fit(self, preds_list, y):
        self.models = []
        H = y.shape[1]
        for h in range(H):
            X = np.column_stack([p[:, h] for p in preds_list])
            m = Ridge(alpha=self.alpha, positive=True)
            m.fit(X, y[:, h])
            self.models.append(m)
        return self

    def predict(self, preds_list):
        H = len(self.models)
        out = np.zeros_like(preds_list[0])
        for h in range(H):
            X = np.column_stack([p[:, h] for p in preds_list])
            out[:, h] = self.models[h].predict(X)
        return out


def inject_drift_time_of_use(arr, shift_steps=10, method="roll"):
    """
    Simulates a new campus pricing policy that pushes arrivals later in the day.
    Uses numpy.roll() or pandas shift() to shift occupancy curves forward in time.
    """
    if method == "roll":
        return np.roll(arr, shift_steps, axis=0)
    elif method == "shift":
        df = pd.DataFrame(arr)
        return df.shift(shift_steps).bfill().to_numpy(dtype=arr.dtype)
    else:
        raise ValueError(f"Unknown shift method: {method}")


def run_experiment_for_site(
    site_name,
    shift_steps=10,
    shift_method="roll",
    block_size=15000,
    shift_weight=2.0,
    epochs=15,
    patience=4,
    lr=1e-3,
    batch_size=256,
    outdir="outputs",
):
    print("\n" + "=" * 100)
    print(f" SIMULATING TIME-OF-USE DATA DRIFT EXPERIMENT FOR {site_name.upper()} HUB")
    print(f" Shift: {shift_steps} steps ({shift_steps * 15 / 60:.2f} hours) forward | Method: {shift_method}")
    print(f" Adaptation: Full training set with shifted block (last {block_size:,} samples weighted {shift_weight}x) | Max Epochs: {epochs}")
    print("=" * 100)

    cfg = SITE_CONFIGS[site_name]
    data_path = cfg["data_path"]
    dlinear_path = cfg["dlinear_path"]
    ridge_path = cfg["ridge_path"]
    max_capacity = cfg["max_capacity"]

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    if not os.path.exists(dlinear_path):
        raise FileNotFoundError(f"DLinear checkpoint not found: {dlinear_path}")
    if not os.path.exists(ridge_path):
        raise FileNotFoundError(f"Ridge checkpoint not found: {ridge_path}")

    # Load preprocessed arrays (Read-only, without modifying originals)
    data = np.load(data_path, allow_pickle=False)
    th, tc, ty = data["train_X_hist"], data["train_X_cal"], data["train_y"]
    vh, vc, vy = data["val_X_hist"], data["val_X_cal"], data["val_y"]
    sh, sc, sy = data["test_X_hist"], data["test_X_cal"], data["test_y"]
    test_origins = data["test_origins"] if "test_origins" in data else None
    timestamps = pd.to_datetime(data["timestamps"][test_origins]) if "timestamps" in data and test_origins is not None else None

    # Morning surge window (8:00 AM - 12:00 PM local = 14:00 - 19:00 UTC on weekdays)
    if timestamps is not None:
        morning_mask = (timestamps.hour >= 14) & (timestamps.hour <= 19) & (timestamps.dayofweek < 5)
    else:
        morning_mask = np.ones(len(sy), dtype=bool)

    L, H, C = th.shape[1], ty.shape[1], tc.shape[2]
    device = torch.device("cpu")

    # 1. Load Original Pre-Trained Models
    dlinear_orig = DLinear(L, H, C)
    dl_state = torch.load(dlinear_path, map_location=device)
    dlinear_orig.load_state_dict(dl_state if "seasonal.weight" in dl_state else dl_state["state_dict"])
    dlinear_orig.eval()

    rd_data = joblib.load(ridge_path)
    ridge_scaler_orig = rd_data["scaler"]
    ridge_model_orig = rd_data["model"]

    # Inference wrappers
    def dl_predict(model, h_arr, c_arr):
        s = window_scale(h_arr)
        h_t = torch.from_numpy((h_arr / s).astype(np.float32)).unsqueeze(-1)
        c_t = torch.from_numpy(c_arr.astype(np.float32))
        with torch.no_grad():
            out = model(h_t, c_t).numpy()
        return out * s[:, :, None]

    def ridge_predict(scaler, model, h_arr, c_arr):
        feats = np.concatenate([h_arr, c_arr.reshape(len(c_arr), -1)], axis=1)
        return model.predict(scaler.transform(feats))

    # Clean Validation Predictions for Original Models
    v_dl_clean = dl_predict(dlinear_orig, vh, vc)
    v_rd_clean = ridge_predict(ridge_scaler_orig, ridge_model_orig, vh, vc)

    # Clean Meta-Learner Fit
    meta_clean = StackingRidgeMetaLearner(alpha=10.0)
    meta_clean.fit([v_dl_clean[..., 1], v_rd_clean], vy)
    v_ens_clean_pt = meta_clean.predict([v_dl_clean[..., 1], v_rd_clean])
    v_ens_hw = 0.5 * (v_dl_clean[..., 2] - v_dl_clean[..., 0])
    v_ens_lo = np.maximum(v_ens_clean_pt - v_ens_hw, 0.0)
    v_ens_hi = np.minimum(v_ens_clean_pt + v_ens_hw, max_capacity)

    # -------------------------------------------------------------------------
    # STAGE 1: BASELINE METRICS ON CLEAN TEST SET
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("STAGE 1: BASELINE EVALUATION (CLEAN TEST SET)")
    print("-" * 70)

    s_dl_clean = dl_predict(dlinear_orig, sh, sc)
    s_rd_clean = ridge_predict(ridge_scaler_orig, ridge_model_orig, sh, sc)
    s_ens_clean_pt = meta_clean.predict([s_dl_clean[..., 1], s_rd_clean])
    s_ens_hw = 0.5 * (s_dl_clean[..., 2] - s_dl_clean[..., 0])
    s_ens_lo = np.maximum(s_ens_clean_pt - s_ens_hw, 0.0)
    s_ens_hi = np.minimum(s_ens_clean_pt + s_ens_hw, max_capacity)

    # Baseline Conformal Calibration (Split-CQR) on Clean Validation Data
    lo_dl_clean_cal, hi_dl_clean_cal = conformalize(vy, v_dl_clean[..., 0], v_dl_clean[..., 2], s_dl_clean[..., 0], s_dl_clean[..., 2])
    lo_ens_clean_cal, hi_ens_clean_cal = conformalize(vy, v_ens_lo, v_ens_hi, s_ens_lo, s_ens_hi)

    # Ridge baseline interval: point +/- mean DLinear interval width
    rd_v_hw = np.abs(vy - v_rd_clean)
    rd_lo_raw = np.maximum(s_rd_clean - np.mean(rd_v_hw), 0.0)
    rd_hi_raw = np.minimum(s_rd_clean + np.mean(rd_v_hw), max_capacity)
    v_rd_lo_raw = np.maximum(v_rd_clean - np.mean(rd_v_hw), 0.0)
    v_rd_hi_raw = np.minimum(v_rd_clean + np.mean(rd_v_hw), max_capacity)
    lo_rd_clean_cal, hi_rd_clean_cal = conformalize(vy, v_rd_lo_raw, v_rd_hi_raw, rd_lo_raw, rd_hi_raw)

    mae_fn = lambda yt, yp: float(np.mean(np.abs(yt - yp)))
    rmse_fn = lambda yt, yp: float(np.sqrt(np.mean((yt - yp) ** 2)))
    cov_fn = lambda yt, lo, hi: float(np.mean((yt >= lo) & (yt <= hi)))
    width_fn = lambda lo, hi: float(np.mean(hi - lo))

    # Measure clean latency
    lat_dl_clean = run_latency_benchmark(lambda h, c: dl_predict(dlinear_orig, h, c), sh, sc)
    lat_rd_clean = run_latency_benchmark(lambda h, c: ridge_predict(ridge_scaler_orig, ridge_model_orig, h, c), sh, sc)
    lat_ens_clean = run_latency_benchmark(
        lambda h, c: meta_clean.predict([dl_predict(dlinear_orig, h, c)[..., 1], ridge_predict(ridge_scaler_orig, ridge_model_orig, h, c)]),
        sh, sc
    )

    p_dl, sz_dl = get_model_size_and_params(dlinear_orig)
    p_rd, sz_rd = get_model_size_and_params(ridge_model_orig)
    p_ens, sz_ens = get_model_size_and_params(meta_clean)

    results = []

    # Record Clean Metrics
    results.append({
        "Site": site_name,
        "Stage": "1. Baseline (Clean)",
        "Model": "DLinear (Neural)",
        "Overall MAE": mae_fn(sy, s_dl_clean[..., 1]),
        "Overall MAE 95% CI": str([round(x, 4) for x in bootstrap_ci(mae_fn, sy, s_dl_clean[..., 1])]),
        "Overall RMSE": rmse_fn(sy, s_dl_clean[..., 1]),
        "Morning MAE": mae_fn(sy[morning_mask], s_dl_clean[morning_mask, :, 1]),
        "Raw 80% Cov": cov_fn(sy, s_dl_clean[..., 0], s_dl_clean[..., 2]),
        "Calibrated 80% Cov": cov_fn(sy, lo_dl_clean_cal, hi_dl_clean_cal),
        "Morning 80% Cov": cov_fn(sy[morning_mask], lo_dl_clean_cal[morning_mask], hi_dl_clean_cal[morning_mask]),
        "Mean Interval Width": width_fn(lo_dl_clean_cal, hi_dl_clean_cal),
        "Single Latency p50 (ms)": lat_dl_clean["single_p50_ms"],
        "Batch Latency p50 (ms)": lat_dl_clean["batch_p50_ms"],
        "Throughput (sps)": lat_dl_clean["throughput_sps"],
        "Parameters": p_dl,
        "Model Bytes": sz_dl,
    })

    results.append({
        "Site": site_name,
        "Stage": "1. Baseline (Clean)",
        "Model": "Ridge (Linear)",
        "Overall MAE": mae_fn(sy, s_rd_clean),
        "Overall MAE 95% CI": str([round(x, 4) for x in bootstrap_ci(mae_fn, sy, s_rd_clean)]),
        "Overall RMSE": rmse_fn(sy, s_rd_clean),
        "Morning MAE": mae_fn(sy[morning_mask], s_rd_clean[morning_mask]),
        "Raw 80% Cov": cov_fn(sy, rd_lo_raw, rd_hi_raw),
        "Calibrated 80% Cov": cov_fn(sy, lo_rd_clean_cal, hi_rd_clean_cal),
        "Morning 80% Cov": cov_fn(sy[morning_mask], lo_rd_clean_cal[morning_mask], hi_rd_clean_cal[morning_mask]),
        "Mean Interval Width": width_fn(lo_rd_clean_cal, hi_rd_clean_cal),
        "Single Latency p50 (ms)": lat_rd_clean["single_p50_ms"],
        "Batch Latency p50 (ms)": lat_rd_clean["batch_p50_ms"],
        "Throughput (sps)": lat_rd_clean["throughput_sps"],
        "Parameters": p_rd,
        "Model Bytes": sz_rd,
    })

    results.append({
        "Site": site_name,
        "Stage": "1. Baseline (Clean)",
        "Model": "Stacking Meta-Learner (DLinear + Ridge)",
        "Overall MAE": mae_fn(sy, s_ens_clean_pt),
        "Overall MAE 95% CI": str([round(x, 4) for x in bootstrap_ci(mae_fn, sy, s_ens_clean_pt)]),
        "Overall RMSE": rmse_fn(sy, s_ens_clean_pt),
        "Morning MAE": mae_fn(sy[morning_mask], s_ens_clean_pt[morning_mask]),
        "Raw 80% Cov": cov_fn(sy, s_ens_lo, s_ens_hi),
        "Calibrated 80% Cov": cov_fn(sy, lo_ens_clean_cal, hi_ens_clean_cal),
        "Morning 80% Cov": cov_fn(sy[morning_mask], lo_ens_clean_cal[morning_mask], hi_ens_clean_cal[morning_mask]),
        "Mean Interval Width": width_fn(lo_ens_clean_cal, hi_ens_clean_cal),
        "Single Latency p50 (ms)": lat_ens_clean["single_p50_ms"],
        "Batch Latency p50 (ms)": lat_ens_clean["batch_p50_ms"],
        "Throughput (sps)": lat_ens_clean["throughput_sps"],
        "Parameters": p_dl + p_rd + p_ens,
        "Model Bytes": sz_dl + sz_rd + sz_ens,
    })

    print(f"Clean Baseline DLinear MAE:   {results[0]['Overall MAE']:.4f} | Calibrated 80% Cov: {results[0]['Calibrated 80% Cov']:.3f}")
    print(f"Clean Baseline Ridge MAE:     {results[1]['Overall MAE']:.4f} | Calibrated 80% Cov: {results[1]['Calibrated 80% Cov']:.3f}")
    print(f"Clean Baseline Stacking MAE:  {results[2]['Overall MAE']:.4f} | Calibrated 80% Cov: {results[2]['Calibrated 80% Cov']:.3f}")

    # -------------------------------------------------------------------------
    # STAGE 2 & 3: INJECT DRIFT AND EVALUATE DEGRADATION (STALE MODEL)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print(f"STAGE 2 & 3: INJECT TIME-OF-USE DRIFT & EVALUATE STALE MODEL ({shift_steps} STEPS SHIFT)")
    print("-" * 70)

    # Fair shift: The entire continuous occupancy curve shifts forward in time.
    # Both history telemetry (sh) and future targets (sy) shift together by shift_steps.
    # Calendar features (sc) stay locked to true clock time.
    sh_drift = inject_drift_time_of_use(sh, shift_steps=shift_steps, method=shift_method)
    sy_drift = inject_drift_time_of_use(sy, shift_steps=shift_steps, method=shift_method)

    # Pass shifted test data through stale original models
    s_dl_stale = dl_predict(dlinear_orig, sh_drift, sc)
    s_rd_stale = ridge_predict(ridge_scaler_orig, ridge_model_orig, sh_drift, sc)
    s_ens_stale_pt = meta_clean.predict([s_dl_stale[..., 1], s_rd_stale])
    s_ens_stale_hw = 0.5 * (s_dl_stale[..., 2] - s_dl_stale[..., 0])
    s_ens_stale_lo = np.maximum(s_ens_stale_pt - s_ens_stale_hw, 0.0)
    s_ens_stale_hi = np.minimum(s_ens_stale_pt + s_ens_stale_hw, max_capacity)

    # Stale Split-CQR calibration quantiles (computed from pre-drift validation set)
    scores_dl_clean = np.maximum(v_dl_clean[..., 0] - vy, vy - v_dl_clean[..., 2])
    k_cov = int(np.ceil((len(scores_dl_clean) + 1) * 0.80))
    q_dl_clean = np.partition(scores_dl_clean, k_cov - 1, axis=0)[k_cov - 1, :]
    lo_dl_stale_cal = s_dl_stale[..., 0] - q_dl_clean[None, :]
    hi_dl_stale_cal = s_dl_stale[..., 2] + q_dl_clean[None, :]

    scores_ens_clean = np.maximum(v_ens_lo - vy, vy - v_ens_hi)
    q_ens_clean = np.partition(scores_ens_clean, k_cov - 1, axis=0)[k_cov - 1, :]
    lo_ens_stale_cal = s_ens_stale_lo - q_ens_clean[None, :]
    hi_ens_stale_cal = s_ens_stale_hi + q_ens_clean[None, :]

    rd_stale_lo_raw = np.maximum(s_rd_stale - np.mean(rd_v_hw), 0.0)
    rd_stale_hi_raw = np.minimum(s_rd_stale + np.mean(rd_v_hw), max_capacity)
    scores_rd_clean = np.maximum(v_rd_lo_raw - vy, vy - v_rd_hi_raw)
    q_rd_clean = np.partition(scores_rd_clean, k_cov - 1, axis=0)[k_cov - 1, :]
    lo_rd_stale_cal = rd_stale_lo_raw - q_rd_clean[None, :]
    hi_rd_stale_cal = rd_stale_hi_raw + q_rd_clean[None, :]

    # Record Stale Degraded Metrics
    results.append({
        "Site": site_name,
        "Stage": "2. Degraded (Stale Model)",
        "Model": "DLinear (Neural)",
        "Overall MAE": mae_fn(sy_drift, s_dl_stale[..., 1]),
        "Overall MAE 95% CI": str([round(x, 4) for x in bootstrap_ci(mae_fn, sy_drift, s_dl_stale[..., 1])]),
        "Overall RMSE": rmse_fn(sy_drift, s_dl_stale[..., 1]),
        "Morning MAE": mae_fn(sy_drift[morning_mask], s_dl_stale[morning_mask, :, 1]),
        "Raw 80% Cov": cov_fn(sy_drift, s_dl_stale[..., 0], s_dl_stale[..., 2]),
        "Calibrated 80% Cov": cov_fn(sy_drift, lo_dl_stale_cal, hi_dl_stale_cal),
        "Morning 80% Cov": cov_fn(sy_drift[morning_mask], lo_dl_stale_cal[morning_mask], hi_dl_stale_cal[morning_mask]),
        "Mean Interval Width": width_fn(lo_dl_stale_cal, hi_dl_stale_cal),
        "Single Latency p50 (ms)": lat_dl_clean["single_p50_ms"],
        "Batch Latency p50 (ms)": lat_dl_clean["batch_p50_ms"],
        "Throughput (sps)": lat_dl_clean["throughput_sps"],
        "Parameters": p_dl,
        "Model Bytes": sz_dl,
    })

    results.append({
        "Site": site_name,
        "Stage": "2. Degraded (Stale Model)",
        "Model": "Ridge (Linear)",
        "Overall MAE": mae_fn(sy_drift, s_rd_stale),
        "Overall MAE 95% CI": str([round(x, 4) for x in bootstrap_ci(mae_fn, sy_drift, s_rd_stale)]),
        "Overall RMSE": rmse_fn(sy_drift, s_rd_stale),
        "Morning MAE": mae_fn(sy_drift[morning_mask], s_rd_stale[morning_mask]),
        "Raw 80% Cov": cov_fn(sy_drift, rd_stale_lo_raw, rd_stale_hi_raw),
        "Calibrated 80% Cov": cov_fn(sy_drift, lo_rd_stale_cal, hi_rd_stale_cal),
        "Morning 80% Cov": cov_fn(sy_drift[morning_mask], lo_rd_stale_cal[morning_mask], hi_rd_stale_cal[morning_mask]),
        "Mean Interval Width": width_fn(lo_rd_stale_cal, hi_rd_stale_cal),
        "Single Latency p50 (ms)": lat_rd_clean["single_p50_ms"],
        "Batch Latency p50 (ms)": lat_rd_clean["batch_p50_ms"],
        "Throughput (sps)": lat_rd_clean["throughput_sps"],
        "Parameters": p_rd,
        "Model Bytes": sz_rd,
    })

    results.append({
        "Site": site_name,
        "Stage": "2. Degraded (Stale Model)",
        "Model": "Stacking Meta-Learner (DLinear + Ridge)",
        "Overall MAE": mae_fn(sy_drift, s_ens_stale_pt),
        "Overall MAE 95% CI": str([round(x, 4) for x in bootstrap_ci(mae_fn, sy_drift, s_ens_stale_pt)]),
        "Overall RMSE": rmse_fn(sy_drift, s_ens_stale_pt),
        "Morning MAE": mae_fn(sy_drift[morning_mask], s_ens_stale_pt[morning_mask]),
        "Raw 80% Cov": cov_fn(sy_drift, s_ens_stale_lo, s_ens_stale_hi),
        "Calibrated 80% Cov": cov_fn(sy_drift, lo_ens_stale_cal, hi_ens_stale_cal),
        "Morning 80% Cov": cov_fn(sy_drift[morning_mask], lo_ens_stale_cal[morning_mask], hi_ens_stale_cal[morning_mask]),
        "Mean Interval Width": width_fn(lo_ens_stale_cal, hi_ens_stale_cal),
        "Single Latency p50 (ms)": lat_ens_clean["single_p50_ms"],
        "Batch Latency p50 (ms)": lat_ens_clean["batch_p50_ms"],
        "Throughput (sps)": lat_ens_clean["throughput_sps"],
        "Parameters": p_dl + p_rd + p_ens,
        "Model Bytes": sz_dl + sz_rd + sz_ens,
    })

    print(f"Stale DLinear MAE:   {results[3]['Overall MAE']:.4f} (Morning MAE: {results[3]['Morning MAE']:.4f}, Cov: {results[3]['Calibrated 80% Cov']:.3f})")
    print(f"Stale Ridge MAE:     {results[4]['Overall MAE']:.4f} (Morning MAE: {results[4]['Morning MAE']:.4f}, Cov: {results[4]['Calibrated 80% Cov']:.3f})")
    print(f"Stale Stacking MAE:  {results[5]['Overall MAE']:.4f} (Morning MAE: {results[5]['Morning MAE']:.4f}, Cov: {results[5]['Calibrated 80% Cov']:.3f})")

    # -------------------------------------------------------------------------
    # STAGE 4: ADAPTATION (RETRAIN ON ALL DATA WITH SHIFTED DATA WEIGHTED HIGHER)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print(f"STAGE 4: ADAPTATION (RETRAINING ON ALL DATA, SHIFTED BLOCK WEIGHTED {shift_weight}x)")
    print("-" * 70)

    # Prepare combined training dataset: all historical data, with the recent block shifted and given higher weight
    actual_block_size = min(block_size, len(th))
    n_old = len(th) - actual_block_size

    # Older unshifted historical data (weight = 1.0)
    th_old, tc_old, ty_old = th[:n_old], tc[:n_old], ty[:n_old]

    # Recent operational block under new tariff (shifted, weight = shift_weight)
    th_rec = inject_drift_time_of_use(th[-actual_block_size:], shift_steps=shift_steps, method=shift_method)
    tc_rec = tc[-actual_block_size:]
    ty_rec = inject_drift_time_of_use(ty[-actual_block_size:], shift_steps=shift_steps, method=shift_method)

    # Combine all training data
    th_all = np.concatenate([th_old, th_rec], axis=0)
    tc_all = np.concatenate([tc_old, tc_rec], axis=0)
    ty_all = np.concatenate([ty_old, ty_rec], axis=0)

    # Sample weights
    sample_weights_np = np.concatenate([np.ones(n_old, dtype=np.float64), shift_weight * np.ones(actual_block_size, dtype=np.float64)])

    # Shifted calibration split (fair continuous shift)
    vh_drift = inject_drift_time_of_use(vh, shift_steps=shift_steps, method=shift_method)
    vc_drift = vc
    vy_drift = inject_drift_time_of_use(vy, shift_steps=shift_steps, method=shift_method)

    # 4A. Retrain DLinear weights from scratch with WeightedRandomSampler
    print(f"Retraining DLinear neural forecaster from scratch on all {len(th_all):,} samples (shifted weighted {shift_weight}x)...")
    dlinear_adapted = DLinear(L, H, C)  # Fresh uninitialized weights
    tr_tensors = make_tensors(th_all, tc_all, ty_all)
    va_tensors = make_tensors(vh_drift, vc_drift, vy_drift)
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights_np),
        num_samples=len(sample_weights_np),
        replacement=True
    )
    loader = DataLoader(TensorDataset(*tr_tensors), batch_size=batch_size, sampler=sampler)
    dlinear_adapted, dl_hist = fit(
        dlinear_adapted, loader, va_tensors, device,
        epochs=epochs, patience=patience, lr=lr, name=f"DLinear_{site_name}_Retrained"
    )

    # Free memory from DLinear training
    del tr_tensors, loader, va_tensors
    import gc
    gc.collect()

    # 4B. Retrain Ridge baseline on all data with sample weights
    print(f"Retraining Ridge regression baseline on all data with {shift_weight}x weight on shifted block...")
    feats_all = np.concatenate([th_all, tc_all.reshape(len(tc_all), -1)], axis=1).astype(np.float32)
    del th_all, tc_all, th_old, tc_old, ty_old, th_rec, tc_rec
    gc.collect()

    ridge_scaler_adapted = StandardScaler(copy=False)
    xt_all = ridge_scaler_adapted.fit_transform(feats_all)
    ridge_weights = (sample_weights_np / np.mean(sample_weights_np)).astype(np.float32)
    ridge_model_adapted = Ridge(alpha=10.0, solver="auto")
    ridge_model_adapted.fit(xt_all, ty_all, sample_weight=ridge_weights)
    del feats_all, xt_all, ty_all, sample_weights_np, ridge_weights
    gc.collect()

    # 4C. Retrain Stacking Ridge Meta-Learner on shifted calibration predictions
    print("Retraining Stacking Ridge Meta-Learner and recalculating Split-CQR quantiles...")
    v_dl_adapted = dl_predict(dlinear_adapted, vh_drift, vc_drift)
    v_rd_adapted = ridge_predict(ridge_scaler_adapted, ridge_model_adapted, vh_drift, vc_drift)

    meta_adapted = StackingRidgeMetaLearner(alpha=10.0)
    meta_adapted.fit([v_dl_adapted[..., 1], v_rd_adapted], vy_drift)
    v_ens_adapted_pt = meta_adapted.predict([v_dl_adapted[..., 1], v_rd_adapted])
    v_ens_adapted_hw = 0.5 * (v_dl_adapted[..., 2] - v_dl_adapted[..., 0])
    v_ens_adapted_lo = np.maximum(v_ens_adapted_pt - v_ens_adapted_hw, 0.0)
    v_ens_adapted_hi = np.minimum(v_ens_adapted_pt + v_ens_adapted_hw, max_capacity)

    # 4D. Recalculate Split-CQR quantiles on the shifted calibration split
    # For DLinear
    s_dl_adapted = dl_predict(dlinear_adapted, sh_drift, sc)
    lo_dl_rec_cal, hi_dl_rec_cal = conformalize(
        vy_drift, v_dl_adapted[..., 0], v_dl_adapted[..., 2],
        s_dl_adapted[..., 0], s_dl_adapted[..., 2]
    )

    # For Ridge
    rd_v_hw_adapt = np.abs(vy_drift - v_rd_adapted)
    v_rd_lo_adapt = np.maximum(v_rd_adapted - np.mean(rd_v_hw_adapt), 0.0)
    v_rd_hi_adapt = np.minimum(v_rd_adapted + np.mean(rd_v_hw_adapt), max_capacity)

    s_rd_adapted = ridge_predict(ridge_scaler_adapted, ridge_model_adapted, sh_drift, sc)
    s_rd_lo_adapt = np.maximum(s_rd_adapted - np.mean(rd_v_hw_adapt), 0.0)
    s_rd_hi_adapt = np.minimum(s_rd_adapted + np.mean(rd_v_hw_adapt), max_capacity)
    lo_rd_rec_cal, hi_rd_rec_cal = conformalize(vy_drift, v_rd_lo_adapt, v_rd_hi_adapt, s_rd_lo_adapt, s_rd_hi_adapt)

    # For Stacking Meta-Learner
    s_ens_adapted_pt = meta_adapted.predict([s_dl_adapted[..., 1], s_rd_adapted])
    s_ens_adapted_hw = 0.5 * (s_dl_adapted[..., 2] - s_dl_adapted[..., 0])
    s_ens_adapted_lo = np.maximum(s_ens_adapted_pt - s_ens_adapted_hw, 0.0)
    s_ens_adapted_hi = np.minimum(s_ens_adapted_pt + s_ens_adapted_hw, max_capacity)
    lo_ens_rec_cal, hi_ens_rec_cal = conformalize(vy_drift, v_ens_adapted_lo, v_ens_adapted_hi, s_ens_adapted_lo, s_ens_adapted_hi)

    # -------------------------------------------------------------------------
    # STAGE 5: RECOVERY (UPDATED MODEL ON SHIFTED TEST DATA)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("STAGE 5: RECOVERY EVALUATION (ADAPTED MODEL ON SHIFTED TEST SET)")
    print("-" * 70)

    # Latency benchmarking for adapted models
    lat_dl_adapt = run_latency_benchmark(lambda h, c: dl_predict(dlinear_adapted, h, c), sh_drift, sc)
    lat_rd_adapt = run_latency_benchmark(lambda h, c: ridge_predict(ridge_scaler_adapted, ridge_model_adapted, h, c), sh_drift, sc)
    lat_ens_adapt = run_latency_benchmark(
        lambda h, c: meta_adapted.predict([dl_predict(dlinear_adapted, h, c)[..., 1], ridge_predict(ridge_scaler_adapted, ridge_model_adapted, h, c)]),
        sh_drift, sc
    )

    p_dl_ad, sz_dl_ad = get_model_size_and_params(dlinear_adapted)
    p_rd_ad, sz_rd_ad = get_model_size_and_params(ridge_model_adapted)
    p_ens_ad, sz_ens_ad = get_model_size_and_params(meta_adapted)

    # Record Recovered Metrics
    results.append({
        "Site": site_name,
        "Stage": "3. Recovered (Adapted Model)",
        "Model": "DLinear (Neural)",
        "Overall MAE": mae_fn(sy_drift, s_dl_adapted[..., 1]),
        "Overall MAE 95% CI": str([round(x, 4) for x in bootstrap_ci(mae_fn, sy_drift, s_dl_adapted[..., 1])]),
        "Overall RMSE": rmse_fn(sy_drift, s_dl_adapted[..., 1]),
        "Morning MAE": mae_fn(sy_drift[morning_mask], s_dl_adapted[morning_mask, :, 1]),
        "Raw 80% Cov": cov_fn(sy_drift, s_dl_adapted[..., 0], s_dl_adapted[..., 2]),
        "Calibrated 80% Cov": cov_fn(sy_drift, lo_dl_rec_cal, hi_dl_rec_cal),
        "Morning 80% Cov": cov_fn(sy_drift[morning_mask], lo_dl_rec_cal[morning_mask], hi_dl_rec_cal[morning_mask]),
        "Mean Interval Width": width_fn(lo_dl_rec_cal, hi_dl_rec_cal),
        "Single Latency p50 (ms)": lat_dl_adapt["single_p50_ms"],
        "Batch Latency p50 (ms)": lat_dl_adapt["batch_p50_ms"],
        "Throughput (sps)": lat_dl_adapt["throughput_sps"],
        "Parameters": p_dl_ad,
        "Model Bytes": sz_dl_ad,
    })

    results.append({
        "Site": site_name,
        "Stage": "3. Recovered (Adapted Model)",
        "Model": "Ridge (Linear)",
        "Overall MAE": mae_fn(sy_drift, s_rd_adapted),
        "Overall MAE 95% CI": str([round(x, 4) for x in bootstrap_ci(mae_fn, sy_drift, s_rd_adapted)]),
        "Overall RMSE": rmse_fn(sy_drift, s_rd_adapted),
        "Morning MAE": mae_fn(sy_drift[morning_mask], s_rd_adapted[morning_mask]),
        "Raw 80% Cov": cov_fn(sy_drift, s_rd_lo_adapt, s_rd_hi_adapt),
        "Calibrated 80% Cov": cov_fn(sy_drift, lo_rd_rec_cal, hi_rd_rec_cal),
        "Morning 80% Cov": cov_fn(sy_drift[morning_mask], lo_rd_rec_cal[morning_mask], hi_rd_rec_cal[morning_mask]),
        "Mean Interval Width": width_fn(lo_rd_rec_cal, hi_rd_rec_cal),
        "Single Latency p50 (ms)": lat_rd_adapt["single_p50_ms"],
        "Batch Latency p50 (ms)": lat_rd_adapt["batch_p50_ms"],
        "Throughput (sps)": lat_rd_adapt["throughput_sps"],
        "Parameters": p_rd_ad,
        "Model Bytes": sz_rd_ad,
    })

    results.append({
        "Site": site_name,
        "Stage": "3. Recovered (Adapted Model)",
        "Model": "Stacking Meta-Learner (DLinear + Ridge)",
        "Overall MAE": mae_fn(sy_drift, s_ens_adapted_pt),
        "Overall MAE 95% CI": str([round(x, 4) for x in bootstrap_ci(mae_fn, sy_drift, s_ens_adapted_pt)]),
        "Overall RMSE": rmse_fn(sy_drift, s_ens_adapted_pt),
        "Morning MAE": mae_fn(sy_drift[morning_mask], s_ens_adapted_pt[morning_mask]),
        "Raw 80% Cov": cov_fn(sy_drift, s_ens_adapted_lo, s_ens_adapted_hi),
        "Calibrated 80% Cov": cov_fn(sy_drift, lo_ens_rec_cal, hi_ens_rec_cal),
        "Morning 80% Cov": cov_fn(sy_drift[morning_mask], lo_ens_rec_cal[morning_mask], hi_ens_rec_cal[morning_mask]),
        "Mean Interval Width": width_fn(lo_ens_rec_cal, hi_ens_rec_cal),
        "Single Latency p50 (ms)": lat_ens_adapt["single_p50_ms"],
        "Batch Latency p50 (ms)": lat_ens_adapt["batch_p50_ms"],
        "Throughput (sps)": lat_ens_adapt["throughput_sps"],
        "Parameters": p_dl_ad + p_rd_ad + p_ens_ad,
        "Model Bytes": sz_dl_ad + sz_rd_ad + sz_ens_ad,
    })

    print(f"Recovered DLinear MAE:   {results[6]['Overall MAE']:.4f} (Morning MAE: {results[6]['Morning MAE']:.4f}, Cov: {results[6]['Calibrated 80% Cov']:.3f})")
    print(f"Recovered Ridge MAE:     {results[7]['Overall MAE']:.4f} (Morning MAE: {results[7]['Morning MAE']:.4f}, Cov: {results[7]['Calibrated 80% Cov']:.3f})")
    print(f"Recovered Stacking MAE:  {results[8]['Overall MAE']:.4f} (Morning MAE: {results[8]['Morning MAE']:.4f}, Cov: {results[8]['Calibrated 80% Cov']:.3f})")

    df_site = pd.DataFrame(results)
    payload = {
        "sy_clean": sy, "s_dl_clean": s_dl_clean, "lo_dl_clean_cal": lo_dl_clean_cal, "hi_dl_clean_cal": hi_dl_clean_cal,
        "sy_drift": sy_drift, "s_dl_stale": s_dl_stale, "lo_dl_stale_cal": lo_dl_stale_cal, "hi_dl_stale_cal": hi_dl_stale_cal,
        "s_dl_adapted": s_dl_adapted, "lo_dl_rec_cal": lo_dl_rec_cal, "hi_dl_rec_cal": hi_dl_rec_cal,
        "timestamps": timestamps, "morning_mask": morning_mask,
        "meta_clean": meta_clean, "meta_adapt": meta_adapted,
    }
    return df_site, payload


def generate_drift_visualizations(site_data, outdir="outputs/figures"):
    """Generates aggregate visual diagnostics for the data drift experiment."""
    os.makedirs(outdir, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # 1. Diurnal Forecast Error Profile (MAE across 24 Hours)
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
    for i, (site, d) in enumerate(site_data.items()):
        ax = axes[i]
        ts = d["timestamps"]
        if ts is not None:
            local_hours = (ts.hour - 7) % 24
            clean_err = np.mean(np.abs(d["sy_clean"] - d["s_dl_clean"][..., 1]), axis=1)
            stale_err = np.mean(np.abs(d["sy_drift"] - d["s_dl_stale"][..., 1]), axis=1)
            rec_err = np.mean(np.abs(d["sy_drift"] - d["s_dl_adapted"][..., 1]), axis=1)
            df_err = pd.DataFrame({"hour": local_hours, "clean": clean_err, "stale": stale_err, "recovered": rec_err})
            hourly_err = df_err.groupby("hour").mean()

            ax.plot(hourly_err.index.to_numpy(), hourly_err["clean"].to_numpy(), "o-", color="#2ca02c", lw=2.2, label="Clean Baseline")
            ax.plot(hourly_err.index.to_numpy(), hourly_err["stale"].to_numpy(), "s--", color="#d62728", lw=2.5, label="Stale Model (Degraded)")
            ax.plot(hourly_err.index.to_numpy(), hourly_err["recovered"].to_numpy(), "^-", color="#1f77b4", lw=2.2, label="Adapted Model (Retrained)")
            ax.axvspan(7, 13, color="orange", alpha=0.15, label="Arrival Transition Window (7am-1pm PT)")
            ax.set_title(f"{site}: Diurnal Forecast Error (MAE Across 24 Hours)", fontsize=12, fontweight="bold")
            ax.set_xlabel("Hour of Day (Local Pacific Time)", fontsize=11)
            ax.set_ylabel("MAE (Cars)", fontsize=11)
            ax.set_xticks(range(0, 24, 2))
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right", frameon=True, fontsize=9.5)
    plt.tight_layout()
    fpath1 = os.path.join(outdir, "data_drift_hourly_mae_profile.png")
    fig.savefig(fpath1, dpi=200)
    plt.close(fig)
    print(f"Saved diurnal error profile plot to: {fpath1}")

    # 2. Conformal Coverage by Hour
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
    for i, (site, d) in enumerate(site_data.items()):
        ax = axes[i]
        ts = d["timestamps"]
        if ts is not None:
            local_hours = (ts.hour - 7) % 24
            stale_cov = np.mean((d["sy_drift"] >= d["lo_dl_stale_cal"]) & (d["sy_drift"] <= d["hi_dl_stale_cal"]), axis=1) * 100
            rec_cov = np.mean((d["sy_drift"] >= d["lo_dl_rec_cal"]) & (d["sy_drift"] <= d["hi_dl_rec_cal"]), axis=1) * 100
            df_cov = pd.DataFrame({"hour": local_hours, "stale": stale_cov, "recovered": rec_cov})
            hourly_cov = df_cov.groupby("hour").mean()

            ax.plot(hourly_cov.index.to_numpy(), hourly_cov["stale"].to_numpy(), "s--", color="#d62728", lw=2.5, label="Stale Model")
            ax.plot(hourly_cov.index.to_numpy(), hourly_cov["recovered"].to_numpy(), "o-", color="#1f77b4", lw=2.5, label="Adapted Model (Recalibrated)")
            ax.axhline(80.0, color="black", linestyle="--", lw=2.0, label="Nominal Target (80%)")
            ax.axvspan(7, 13, color="orange", alpha=0.15, label="Arrival Transition (7am-1pm PT)")
            ax.set_title(f"{site}: Empirical 80% Conformal Coverage Across Time of Day", fontsize=12, fontweight="bold")
            ax.set_xlabel("Hour of Day (Local Pacific Time)", fontsize=11)
            ax.set_ylabel("Empirical 80% Coverage (%)", fontsize=11)
            ax.set_xticks(range(0, 24, 2))
            ax.set_ylim(min(45, hourly_cov["stale"].min() - 5), 95)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="lower right", frameon=True, fontsize=9.5)
    plt.tight_layout()
    fpath2 = os.path.join(outdir, "data_drift_conformal_coverage_by_hour.png")
    fig.savefig(fpath2, dpi=200)
    plt.close(fig)
    print(f"Saved hourly conformal coverage plot to: {fpath2}")

    # 3. Average Morning Surge Arrival Trajectory
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
    for i, (site, d) in enumerate(site_data.items()):
        ax = axes[i]
        ts = d["timestamps"]
        if ts is not None:
            local_hours = (ts.hour - 7) % 24
            rush_mask = (local_hours >= 7) & (local_hours <= 9) & (ts.dayofweek < 5)
            horizon_minutes = np.arange(1, 17) * 15
            mean_gt = np.mean(d["sy_drift"][rush_mask], axis=0)
            mean_stale = np.mean(d["s_dl_stale"][rush_mask, :, 1], axis=0)
            mean_rec = np.mean(d["s_dl_adapted"][rush_mask, :, 1], axis=0)
            mean_lo_rec = np.mean(d["lo_dl_rec_cal"][rush_mask], axis=0)
            mean_hi_rec = np.mean(d["hi_dl_rec_cal"][rush_mask], axis=0)

            ax.plot(horizon_minutes, mean_gt, "k*-", lw=2.5, markersize=8, label="Ground Truth (Shifted Arrivals)")
            ax.plot(horizon_minutes, mean_stale, "r--o", lw=2.2, label="Stale Model (Pre-Drift Bias)")
            ax.plot(horizon_minutes, mean_rec, "b-s", lw=2.2, label="Adapted Model (Retrained)")
            ax.fill_between(horizon_minutes, mean_lo_rec, mean_hi_rec, color="blue", alpha=0.15, label="Adapted 80% Split-CQR Interval")
            n_samples = np.sum(rush_mask)
            ax.set_title(f"{site}: Average Morning Surge Forecast ({n_samples:,} Weekday 7am–9am PT Origins)", fontsize=12, fontweight="bold")
            ax.set_xlabel("Forecast Horizon (Minutes Ahead)", fontsize=11)
            ax.set_ylabel("Occupancy (Cars)", fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper left", frameon=True, fontsize=9.5)
    plt.tight_layout()
    fpath3 = os.path.join(outdir, "data_drift_morning_surge_bias.png")
    fig.savefig(fpath3, dpi=200)
    plt.close(fig)
    print(f"Saved average morning surge trajectory plot to: {fpath3}")


def main():
    parser = argparse.ArgumentParser(description="Simulate Time-of-Use Data Drift Experiment")
    parser.add_argument("--site", choices=["Caltech", "JPL", "all", "caltech", "jpl"], default="all",
                        help="EV charging hub to benchmark ('Caltech', 'JPL', or 'all')")
    parser.add_argument("--shift-steps", type=int, default=10, choices=range(8, 13),
                        help="Number of 15-minute steps to shift forward (8 to 12 steps, i.e. 2 to 3 hours, default: 10)")
    parser.add_argument("--shift-method", choices=["roll", "shift"], default="roll",
                        help="Array shift mechanism ('roll' using np.roll, 'shift' using pandas.shift)")
    parser.add_argument("--block-size", type=int, default=15000,
                        help="Number of recent training samples to use for adaptation block (default: 15000)")
    parser.add_argument("--shift-weight", type=float, default=2.0,
                        help="Sample weight multiplier for shifted data in adaptation (default: 2.0)")
    parser.add_argument("--epochs", type=int, default=15,
                        help="Maximum training epochs for adaptation (default: 15)")
    parser.add_argument("--patience", type=int, default=4,
                        help="Early stopping patience for adaptation (default: 4)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate for neural model adaptation (default: 1e-3)")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Training batch size (default: 256)")
    parser.add_argument("--outdir", default="outputs",
                        help="Root directory for saving output tables and figures")
    parser.add_argument("--no-plots", action="store_true", default=False,
                        help="Disable saving diagnostic figures to outputs/figures")
    args = parser.parse_args()

    sites = ["Caltech", "JPL"] if args.site.lower() == "all" else [args.site.capitalize()]
    all_dfs = []
    site_payloads = {}

    for site in sites:
        df_site, payload = run_experiment_for_site(
            site_name=site,
            shift_steps=args.shift_steps,
            shift_method=args.shift_method,
            block_size=args.block_size,
            shift_weight=args.shift_weight,
            epochs=args.epochs,
            patience=args.patience,
            lr=args.lr,
            batch_size=args.batch_size,
            outdir=args.outdir,
        )
        all_dfs.append(df_site)
        site_payloads[site] = payload

        # Save individual site CSV
        site_tag = site.lower()
        tbl_dir = os.path.join(args.outdir, "tables")
        os.makedirs(tbl_dir, exist_ok=True)
        csv_path = os.path.join(tbl_dir, f"data_drift_{site_tag}.csv")
        df_site.to_csv(csv_path, index=False)
        print(f"\nSaved {site} data drift benchmark table to: {csv_path}")

    # Combine and save summary
    df_combined = pd.concat(all_dfs, ignore_index=True)
    summary_path = os.path.join(args.outdir, "tables", "data_drift_summary.csv")
    df_combined.to_csv(summary_path, index=False)
    print(f"Saved combined data drift summary table to: {summary_path}")

    # Save diagnostic visualizations if requested
    if not args.no_plots:
        fig_dir = os.path.join(args.outdir, "figures")
        generate_drift_visualizations(site_payloads, outdir=fig_dir)

    print("\n" + "=" * 100)
    print("DATA DRIFT SIMULATION BENCHMARK COMPLETED SUCCESSFULLY!")
    print("=" * 100)
    print(df_combined[["Site", "Stage", "Model", "Overall MAE", "Morning MAE", "Calibrated 80% Cov", "Mean Interval Width", "Single Latency p50 (ms)"]].to_string(index=False))


if __name__ == "__main__":
    main()
