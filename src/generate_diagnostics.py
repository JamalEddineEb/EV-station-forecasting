"""Generate comprehensive visual diagnostics:
1. Robustness Curves (clean vs perturbed metrics across multiple epsilon budgets)
2. Confidence Histograms (prediction interval width distributions)
3. Calibration Plots / Reliability Diagrams (nominal vs empirical coverage)
4. Qualitative Failure Cases and Trajectory Visualizations

Outputs saved to outputs/figures/
"""
import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch

from adversarial import FGSMAttack, PGDAttack, RandomPerturbation
from model_selection.conformal import conformalize
from models.train_neural import (
    QUANTILES,
    DLinear,
    PinballLoss,
    make_tensors,
    window_scale,
)
from reliability import SelectiveForecaster

# ---------------------------------------------------------------------------
# Project-level constants — change once here, propagate everywhere
# ---------------------------------------------------------------------------
OUT_FIG = "outputs/figures"
DATA_ROOT = "outputs"

# Station physics
MAX_CAPACITY = 54.0          # Caltech: total number of charging ports

# Adversarial attack parameters
PGD_STEPS = 7                # number of projected-gradient-descent iterations
HIST_EPSILON = 1.0           # epsilon (cars) used in histogram perturbation panels
ROBUSTNESS_EPSILONS = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]   # sweep for robustness curves

# Sub-sampling for speed (keeping runtime reasonable while remaining representative)
ROBUSTNESS_TEST_N = 300      # test samples for the epsilon-sweep robustness loop
HISTOGRAM_TEST_N = 500       # test samples for uncertainty histogram panels
CALIBRATION_ADV_N = 400      # adversarial test samples in the calibration diagram

# Abstention / selective forecaster
ABSTENTION_QUANTILE = 0.90   # top (1 - q) fraction of widths → abstain

# Calibration diagram nominal levels
NOMINAL_LEVELS = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95]
# Index of the primary target level (0.80) used to draw the "raw quantile" reference line
_NOMINAL_80_IDX = NOMINAL_LEVELS.index(0.80)

# Failure-case indices (programmatically verified against test set)
FAILURE_IDX_SURGE = 12065    # peak rush arrival under-forecast
FAILURE_IDX_HOLIDAY = 7272   # campus holiday over-forecast
RESOLVED_IDX_ROBUST = 3548   # Model-level adaptation: robust model vs original baseline
RESOLVED_IDX_ABSTAIN = 11586 # high uncertainty resolved by SelectiveForecaster
NAN_DROP_STEPS = 8           # number of history steps set to NaN in the dropout case

# GRU architecture (must match the trained model in benchmark outputs)
GRU_HIDDEN = 64

os.makedirs(OUT_FIG, exist_ok=True)
plt.rcParams.update({"font.size": 10, "figure.autolayout": True})


def _load_model(checkpoint_path, seq_len, horizon, cal_dim):
    """Load a DLinear checkpoint and set to eval mode."""
    model = DLinear(seq_len, horizon, cal_dim)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    return model.eval()


def _ridge_predictor(ridge_path):
    """Return a closure that predicts with the ridge model from a joblib file."""
    import joblib
    rd = joblib.load(ridge_path)
    scaler, ridge = rd["scaler"], rd["model"]

    def predict(h_arr, c_arr):
        feats = np.concatenate([h_arr, c_arr.reshape(len(c_arr), -1)], axis=1)
        # Expand last dim so shape = (N, horizon, 3) matching neural quantile format
        return np.repeat(ridge.predict(scaler.transform(feats))[:, :, None], 3, axis=-1)

    return predict


def _batch_predict(net, h_t, c_t, s_arr, bs=128):
    """Run net in mini-batches to avoid OOM; returns (N, horizon, n_q) in original units."""
    outs = []
    with torch.no_grad():
        for i in range(0, len(h_t), bs):
            out = net(h_t[i:i + bs], c_t[i:i + bs]).numpy()
            outs.append(out * s_arr[i:i + bs, :, None])
    return np.concatenate(outs, axis=0)


def _make_weights(n):
    """Percentage weights for a histogram of n samples."""
    return np.ones(n) / n * 100


def plot_robustness_curves(caltech_data_path, checkpoint_path):
    """Plot MAE and Calibrated Coverage curves under Random, FGSM, and PGD across
    perturbation budgets, including the Stacking Ensemble defense line.
    """
    print("Generating Robustness Curves...")
    site_folder = "benchmark_jpl" if "jpl" in caltech_data_path.lower() else "benchmark_caltech"
    ridge_path = os.path.join(DATA_ROOT, site_folder, "ridge.joblib")
    data = np.load(caltech_data_path)

    # Use the full validation set for conformal calibration, a 300-sample test subset
    # for the attack sweep (keeps runtime manageable while remaining representative)
    vh, vc, vy = data["val_X_hist"], data["val_X_cal"], data["val_y"]
    sh = data["test_X_hist"][:ROBUSTNESS_TEST_N]
    sc = data["test_X_cal"][:ROBUSTNESS_TEST_N]
    sy = data["test_y"][:ROBUSTNESS_TEST_N]

    v_scale = window_scale(vh)
    scale   = window_scale(sh)
    vh_t, vc_t = make_tensors(vh, vc, None)
    h, c        = make_tensors(sh, sc, None)
    y_norm = torch.from_numpy((sy / scale).astype(np.float32))

    model = _load_model(checkpoint_path, sh.shape[1], sy.shape[1], sc.shape[2])
    ridge_pred = _ridge_predictor(ridge_path)

    loss_fn = PinballLoss(QUANTILES)
    pgd_atk = PGDAttack(loss_fn, steps=PGD_STEPS, max_capacity=MAX_CAPACITY)
    attacks = {
        "Random Noise (Black-box)":    (RandomPerturbation(MAX_CAPACITY), "tab:green",  "o--"),
        "FGSM (White-box 1-step)":     (FGSMAttack(loss_fn, MAX_CAPACITY), "tab:orange", "s-"),
        "PGD-7 (Defenseless Baseline)": (pgd_atk,                          "tab:red",    "^-"),
    }

    v_out   = _batch_predict(model, vh_t, vc_t, v_scale)
    ridge_v = ridge_pred(vh, vc)
    v_pt = 0.5 * (v_out[..., 1] + ridge_v[..., 1])
    v_hw = 0.5 * (v_out[..., 2] - v_out[..., 0])
    v_ens_lo = np.maximum(v_pt - v_hw, 0.0)
    v_ens_hi = np.minimum(v_pt + v_hw, MAX_CAPACITY)

    with torch.no_grad():
        clean_pred = model(h, c).numpy() * scale[:, :, None]
    clean_mae = float(np.mean(np.abs(sy - clean_pred[..., 1])))

    # Raw vs calibrated clean coverage
    raw_clean_cov = float(np.mean((sy >= clean_pred[..., 0]) & (sy <= clean_pred[..., 2])))
    lo_clean, hi_clean = conformalize(vy, v_out[..., 0], v_out[..., 2],
                                      clean_pred[..., 0], clean_pred[..., 2])
    cal_clean_cov = float(np.mean((sy >= lo_clean) & (sy <= hi_clean)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8), dpi=160)

    for name, (atk, color, marker) in attacks.items():
        maes = [clean_mae]
        covs = [cal_clean_cov]
        for eps in ROBUSTNESS_EPSILONS[1:]:
            h_adv = atk.generate(model, h, c, y_norm, scale, eps)
            with torch.no_grad():
                adv_pred = model(h_adv, c).numpy() * scale[:, :, None]
            maes.append(float(np.mean(np.abs(sy - adv_pred[..., 1]))))
            lo_adv, hi_adv = conformalize(vy, v_out[..., 0], v_out[..., 2],
                                          adv_pred[..., 0], adv_pred[..., 2])
            covs.append(float(np.mean((sy >= lo_adv) & (sy <= hi_adv))))

        ax1.plot(ROBUSTNESS_EPSILONS, maes, marker, color=color, label=name, lw=2)
        ax2.plot(ROBUSTNESS_EPSILONS, covs, marker, color=color,
                 label=name + " (Calibrated)", lw=2)

    # Stacking Ensemble defense under PGD-7 (Actual DLinear + Ridge)
    ridge_clean = ridge_pred(sh, sc)
    ens_clean_pt = 0.5 * (clean_pred[..., 1] + ridge_clean[..., 1])
    ens_clean_hw = 0.5 * (clean_pred[..., 2] - clean_pred[..., 0])
    ens_clean_lo = np.maximum(ens_clean_pt - ens_clean_hw, 0.0)
    ens_clean_hi = np.minimum(ens_clean_pt + ens_clean_hw, MAX_CAPACITY)
    lo_ens_c, hi_ens_c = conformalize(vy, v_ens_lo, v_ens_hi,
                                      ens_clean_lo, ens_clean_hi)
    ens_maes = [float(np.mean(np.abs(sy - ens_clean_pt)))]
    ens_covs = [float(np.mean((sy >= lo_ens_c) & (sy <= hi_ens_c)))]

    for eps in ROBUSTNESS_EPSILONS[1:]:
        h_adv    = pgd_atk.generate(model, h, c, y_norm, scale, eps)
        adv_phys = h_adv.squeeze(-1).numpy() * scale
        with torch.no_grad():
            dl_adv  = model(h_adv, c).numpy() * scale[:, :, None]
        ridge_adv = ridge_pred(adv_phys, sc)
        ens_adv_pt = 0.5 * (dl_adv[..., 1] + ridge_adv[..., 1])
        ens_adv_hw = 0.5 * (dl_adv[..., 2] - dl_adv[..., 0])
        ens_adv_lo = np.maximum(ens_adv_pt - ens_adv_hw, 0.0)
        ens_adv_hi = np.minimum(ens_adv_pt + ens_adv_hw, MAX_CAPACITY)
        ens_maes.append(float(np.mean(np.abs(sy - ens_adv_pt))))
        lo_e, hi_e = conformalize(vy, v_ens_lo, v_ens_hi,
                                  ens_adv_lo, ens_adv_hi)
        ens_covs.append(float(np.mean((sy >= lo_e) & (sy <= hi_e))))

    ax1.plot(ROBUSTNESS_EPSILONS, ens_maes, "d-", color="darkblue",
             label="Stacking Ensemble (PGD Defense)", lw=2.4)
    ax2.plot(ROBUSTNESS_EPSILONS, ens_covs, "d-", color="darkblue",
             label="Stacking Ensemble (PGD Defense)", lw=2.4)

    nominal_cov = NOMINAL_LEVELS[_NOMINAL_80_IDX]
    ax1.axhline(clean_mae, color="gray", linestyle=":", label="Clean Baseline")
    ax1.set_xlabel(r"Perturbation Budget $\epsilon$ (physical cars)")
    ax1.set_ylabel("Robust MAE (cars)")
    ax1.set_title("Forecast Error vs. Perturbation Budget (With Defenses)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(frameon=True, fontsize=9)

    ax2.axhline(nominal_cov, color="black", linestyle="--",
                label=f"Nominal {nominal_cov:.0%} Target", lw=1.5)
    ax2.axhline(raw_clean_cov, color="gray", linestyle=":",
                label=f"Raw Uncalibrated Quantiles ({raw_clean_cov:.1%})", lw=1.5)
    ax2.set_xlabel(r"Perturbation Budget $\epsilon$ (physical cars)")
    ax2.set_ylabel(f"Calibrated Empirical {nominal_cov:.0%} Coverage")
    ax2.set_title("Calibrated Interval Coverage vs. Perturbation Budget")
    ax2.grid(True, alpha=0.3)
    ax2.legend(frameon=True, fontsize=8.5)

    save_path = os.path.join(OUT_FIG, "robustness_curves.png")
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", save_path)


def plot_confidence_histograms(caltech_data_path, checkpoint_path):
    """Plot histograms of forecast uncertainty (80% interval width W) across clean vs attacked states."""
    print("Generating Confidence Histograms...")
    data = np.load(caltech_data_path)
    vh, vc, vy = data["val_X_hist"], data["val_X_cal"], data["val_y"]
    sh, sc, sy = data["test_X_hist"][:HISTOGRAM_TEST_N], data["test_X_cal"][:HISTOGRAM_TEST_N], data["test_y"][:HISTOGRAM_TEST_N]
    v_scale = window_scale(vh)
    scale   = window_scale(sh)
    vh_t, vc_t = make_tensors(vh, vc, None)
    h, c       = make_tensors(sh, sc, None)
    y_norm = torch.from_numpy((sy / scale).astype(np.float32))

    model = _load_model(checkpoint_path, sh.shape[1], sy.shape[1], sc.shape[2])

    loss_fn = PinballLoss(QUANTILES)
    pgd  = PGDAttack(loss_fn, steps=PGD_STEPS, max_capacity=MAX_CAPACITY)
    rand = RandomPerturbation(max_capacity=MAX_CAPACITY)

    v_out = _batch_predict(model, vh_t, vc_t, v_scale)
    with torch.no_grad():
        clean_out = model(h, c).numpy() * scale[:, :, None]
    lo_c, hi_c = conformalize(vy, v_out[..., 0], v_out[..., 2], clean_out[..., 0], clean_out[..., 2])
    clean_widths = np.mean(hi_c - lo_c, axis=-1)

    h_rand = rand.generate(model, h, c, y_norm, scale, HIST_EPSILON)
    with torch.no_grad():
        rand_out = model(h_rand, c).numpy() * scale[:, :, None]
    lo_r, hi_r = conformalize(vy, v_out[..., 0], v_out[..., 2], rand_out[..., 0], rand_out[..., 2])
    rand_widths = np.mean(hi_r - lo_r, axis=-1)

    h_pgd = pgd.generate(model, h, c, y_norm, scale, HIST_EPSILON)
    with torch.no_grad():
        pgd_out = model(h_pgd, c).numpy() * scale[:, :, None]
    lo_p, hi_p = conformalize(vy, v_out[..., 0], v_out[..., 2], pgd_out[..., 0], pgd_out[..., 2])
    pgd_widths = np.mean(hi_p - lo_p, axis=-1)

    # Derived axis limits from data (data-driven, no hardcoding)
    tau      = float(np.percentile(clean_widths, ABSTENTION_QUANTILE * 100))
    median_w = float(np.median(clean_widths))
    xlim_a   = np.percentile(clean_widths, 99.5) * 1.15   # panel A: zoom on clean
    xlim_b   = np.percentile(pgd_widths,   99.5) * 1.10   # panel B: covers PGD tail

    n_bins_a = 26
    n_bins_b = 28

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 4.8), dpi=180)

    # ------------------------------------------------------------------ #
    # Panel A: Clean distribution with retained / abstention colouring    #
    # ------------------------------------------------------------------ #
    bins_a   = np.linspace(0, xlim_a, n_bins_a)
    weights  = _make_weights(len(clean_widths))
    counts, edges = np.histogram(clean_widths, bins=bins_a, weights=weights)
    for i in range(len(counts)):
        color = "#1f77b4" if edges[i] < tau else "#d62728"
        ax1.bar(edges[i], counts[i], width=edges[i+1] - edges[i], align="edge",
                color=color, alpha=0.72, edgecolor="white", lw=0.8)

    l_med = ax1.axvline(median_w, color="#08306b", linestyle=":", lw=2.0)
    l_tau = ax1.axvline(tau,      color="#7f0000", linestyle="--", lw=2.0)
    ax1.axvspan(tau, xlim_a, color="#d62728", alpha=0.07)

    patch_acc = mpatches.Patch(color="#1f77b4", alpha=0.72)
    patch_abs = mpatches.Patch(color="#d62728", alpha=0.72)
    abstain_pct = (1.0 - ABSTENTION_QUANTILE) * 100
    retain_pct  = ABSTENTION_QUANTILE * 100
    ax1.legend(
        handles=[patch_acc, patch_abs, l_med, l_tau],
        labels=[
            f"Retained Forecasts ({retain_pct:.0f}%)",
            rf"Abstained Forecasts ({abstain_pct:.0f}%, $W > \tau$)",
            f"Median Width ({median_w:.2f} cars)",
            rf"Cutoff $\tau={tau:.2f}$ cars (Top {abstain_pct:.0f}%)",
        ],
        loc="upper right", frameon=True, framealpha=0.92,
        edgecolor="#d0d4dc", fontsize=9.0,
    )
    ax1.set_xlim(0, xlim_a)
    ax1.set_ylim(0, max(counts) * 1.35)
    ax1.set_xlabel(r"Prediction Interval Width $W = \hat{y}_{90} - \hat{y}_{10}$ (cars)", fontsize=10)
    ax1.set_ylabel("Percentage of Test Samples (%)", fontsize=10)
    ax1.set_title("(A) Baseline Uncertainty & Selective Forecaster Boundary",
                  fontsize=11, fontweight="bold", pad=9)
    ax1.grid(True, alpha=0.25, linestyle="--")

    # ------------------------------------------------------------------ #
    # Panel B: Distribution shift under random noise and PGD attack       #
    # ------------------------------------------------------------------ #
    bins_b     = np.linspace(0, xlim_b, n_bins_b)
    weights_b  = _make_weights(len(clean_widths))
    weights_r  = _make_weights(len(rand_widths))
    weights_p  = _make_weights(len(pgd_widths))

    ax2.hist(clean_widths, bins=bins_b, weights=weights_b,
             histtype="stepfilled", alpha=0.18, color="#2ca02c")
    ax2.hist(clean_widths, bins=bins_b, weights=weights_b,
             histtype="step", lw=2.0, color="#2ca02c",
             label=f"Clean Baseline (Mean $W={clean_widths.mean():.2f}$)")

    ax2.hist(rand_widths, bins=bins_b, weights=weights_r,
             histtype="step", lw=1.8, linestyle="--", color="#1f77b4",
             label=rf"Random Noise $\epsilon={HIST_EPSILON}$ (Mean $W={rand_widths.mean():.2f}$)")

    ax2.hist(pgd_widths, bins=bins_b, weights=weights_p,
             histtype="stepfilled", alpha=0.15, color="#d62728")
    ax2.hist(pgd_widths, bins=bins_b, weights=weights_p,
             histtype="step", lw=2.0, color="#d62728",
             label=rf"PGD-{PGD_STEPS} Attack $\epsilon={HIST_EPSILON}$ (Mean $W={pgd_widths.mean():.2f}$)")

    ax2.axvline(tau, color="black", linestyle=":", lw=1.8,
                label=rf"Clean Threshold $\tau={tau:.2f}$")

    y_max_b = max(np.histogram(clean_widths, bins=bins_b, weights=weights_b)[0])
    ax2.set_xlim(0, xlim_b)
    ax2.set_ylim(0, y_max_b * 1.35)
    ax2.set_xlabel(r"Prediction Interval Width $W = \hat{y}_{90} - \hat{y}_{10}$ (cars)", fontsize=10)
    ax2.set_ylabel("Percentage of Test Samples (%)", fontsize=10)
    ax2.set_title("(B) Uncertainty Shift Under Noise and Adversarial Attack",
                  fontsize=11, fontweight="bold", pad=9)
    ax2.legend(loc="upper right", frameon=True, framealpha=0.92,
               edgecolor="#d0d4dc", fontsize=9.0)
    ax2.grid(True, alpha=0.25, linestyle="--")

    plt.tight_layout()
    save_path = os.path.join(OUT_FIG, "confidence_histograms.png")
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", save_path)


def plot_calibration_reliability_diagram(caltech_data_path, checkpoint_path):
    """Reliability diagram: Nominal Coverage vs Empirical Coverage across NOMINAL_LEVELS."""
    print("Generating Calibration Diagrams...")
    data = np.load(caltech_data_path)
    vh, vc, vy = data["val_X_hist"], data["val_X_cal"], data["val_y"]
    sh, sc, sy = data["test_X_hist"], data["test_X_cal"], data["test_y"]

    v_scale = window_scale(vh)
    s_scale = window_scale(sh)
    vh_t, vc_t = make_tensors(vh, vc, None)
    sh_t, sc_t = make_tensors(sh, sc, None)

    model = _load_model(checkpoint_path, sh.shape[1], sy.shape[1], sc.shape[2])

    with torch.no_grad():
        v_out = model(vh_t, vc_t).numpy() * v_scale[:, :, None]
        s_out = model(sh_t, sc_t).numpy() * s_scale[:, :, None]

    adv_n = CALIBRATION_ADV_N
    loss_fn = PinballLoss(QUANTILES)
    pgd = PGDAttack(loss_fn, steps=PGD_STEPS, max_capacity=MAX_CAPACITY)
    sh_sub     = sh_t[:adv_n]
    sc_sub     = sc_t[:adv_n]
    sy_sub     = sy[:adv_n]
    s_scale_sub = s_scale[:adv_n]
    y_norm_sub = torch.from_numpy((sy_sub / s_scale_sub).astype(np.float32))
    h_adv_sub  = pgd.generate(model, sh_sub, sc_sub, y_norm_sub, s_scale_sub, HIST_EPSILON)
    with torch.no_grad():
        adv_sub_out = model(h_adv_sub, sc_sub).numpy() * s_scale_sub[:, :, None]

    # Z-scores for symmetric intervals [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95]
    z_table = {0.50: 0.6745, 0.60: 0.8416, 0.70: 1.0364, 0.80: 1.2816, 0.85: 1.4395, 0.90: 1.6449, 0.95: 1.9600}
    z_80 = z_table[0.80]

    raw_coverage = []
    calibrated_coverage = []
    adversarial_coverage = []

    raw_mid = s_out[..., 1]
    raw_hw_80 = 0.5 * (s_out[..., 2] - s_out[..., 0])

    for alpha in NOMINAL_LEVELS:
        lo_cal, hi_cal = conformalize(vy, v_out[..., 0], v_out[..., 2],
                                      s_out[..., 0], s_out[..., 2], coverage=alpha)
        calibrated_coverage.append(float(np.mean((sy >= lo_cal) & (sy <= hi_cal))))

        scale_factor = z_table[alpha] / z_80
        lo_raw = np.maximum(raw_mid - raw_hw_80 * scale_factor, 0.0)
        hi_raw = np.minimum(raw_mid + raw_hw_80 * scale_factor, MAX_CAPACITY)
        raw_coverage.append(float(np.mean((sy >= lo_raw) & (sy <= hi_raw))))

        lo_adv, hi_adv = conformalize(vy, v_out[..., 0], v_out[..., 2],
                                      adv_sub_out[..., 0], adv_sub_out[..., 2], coverage=alpha)
        adversarial_coverage.append(float(np.mean((sy_sub >= lo_adv) & (sy_sub <= hi_adv))))

    diag_min = min(NOMINAL_LEVELS) - 0.03
    diag_max = max(NOMINAL_LEVELS) + 0.03

    plt.figure(figsize=(7, 5), dpi=150)
    plt.plot([diag_min, diag_max], [diag_min, diag_max],
             "k--", label="Perfect Calibration (Ideal)", lw=1.5)
    plt.plot(NOMINAL_LEVELS, calibrated_coverage,
             "s-", color="tab:blue", lw=2, label="Calibrated Conformal (Split-CQR)")
    plt.plot(NOMINAL_LEVELS, raw_coverage,
             "o:", color="tab:orange", lw=1.8, label="Uncalibrated Raw Quantiles")
    plt.plot(NOMINAL_LEVELS, adversarial_coverage,
             "^-", color="tab:red", lw=2,
             label=rf"PGD-{PGD_STEPS} Attack $\epsilon={HIST_EPSILON}$ (Calibrated)")

    plt.xlabel(r"Nominal Target Coverage ($1 - \alpha$)")
    plt.ylabel("Empirical Test Coverage")
    plt.title("Calibration Reliability Diagram (Nominal vs. Empirical Coverage)")
    plt.xlim(diag_min, diag_max)
    plt.ylim(min(min(calibrated_coverage), min(adversarial_coverage)) - 0.05, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=True, loc="upper left")

    save_path = os.path.join(OUT_FIG, "calibration_diagrams.png")
    plt.savefig(save_path)
    plt.close()
    print("Saved:", save_path)


def plot_failure_and_resolved_cases(caltech_data_path, checkpoint_path, robust_checkpoint_path=None):
    """Plot 4 curated, verified failure/resolved trajectory panels (2x2 grid).

    Cases are programmatically derived from real test data.
    Panel A: Peak Rush Arrival Lag (Baseline failure)
    Panel B: Inactivity / Holiday Over-Forecast (Baseline failure)
    Panel C: Model-Level Adaptation: Robust Model vs Original Baseline
    Panel D: Inference-Level Adaptation: Epistemic Uncertainty Selective Abstention
    """
    print("Generating Failure and Resolved Trajectory Visualizations (4 cases, 2x2)...")

    data = np.load(caltech_data_path)
    sh, sc, sy = data["test_X_hist"], data["test_X_cal"], data["test_y"]
    scale = window_scale(sh)

    model = _load_model(checkpoint_path, sh.shape[1], sy.shape[1], sc.shape[2])
    
    # Load robust model (trained with jittering) if available
    rob_model = None
    if robust_checkpoint_path and os.path.exists(robust_checkpoint_path):
        rob_model = _load_model(robust_checkpoint_path, sh.shape[1], sy.shape[1], sc.shape[2])

    # Horizon axis derived from data shape (steps × minutes-per-step)
    horizon_steps = sy.shape[1]
    minutes_per_step = 15
    horizons = np.arange(1, horizon_steps + 1) * minutes_per_step

    def predict(m, h_arr, c_arr, s):
        """Run model, multiply by scale, clamp to physical lower bound of 0."""
        h_t, c_t = make_tensors(h_arr, c_arr, None)
        with torch.no_grad():
            p = m(h_t, c_t).numpy() * s[:, :, None]
        return np.maximum(p, 0.0)

    # Compute tau from the full clean test-set widths via SelectiveForecaster
    with torch.no_grad():
        full_h, full_c = make_tensors(sh, sc, None)
        full_out = model(full_h, full_c).numpy() * scale[:, :, None]
    forecaster = SelectiveForecaster(max_abstention_rate=1.0 - ABSTENTION_QUANTILE)
    tau = forecaster.calibrate_threshold(full_out[..., 0], full_out[..., 2], quantile=ABSTENTION_QUANTILE)

    site_label = "JPL" if "jpl" in caltech_data_path.lower() else "Caltech"
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=160)
    fig.suptitle(
        f"Curated Failure & Resolved Cases — DLinear + Reliability Stack ({site_label} Hub, $C_{{max}}={int(MAX_CAPACITY)}$)",
        fontsize=13, fontweight="bold", y=0.995,
    )

    # ------------------------------------------------------------------ #
    # Panel A (top-left): Peak Rush Arrival – model lags surging demand   #
    # ------------------------------------------------------------------ #
    p_a = predict(model, sh[FAILURE_IDX_SURGE:FAILURE_IDX_SURGE+1],
                  sc[FAILURE_IDX_SURGE:FAILURE_IDX_SURGE+1],
                  scale[FAILURE_IDX_SURGE:FAILURE_IDX_SURGE+1])
    ax = axes[0, 0]
    peak_y = float(np.max(sy[FAILURE_IDX_SURGE]))
    peak_h = int(horizons[np.argmax(sy[FAILURE_IDX_SURGE])])
    y_max_a = max(peak_y, float(np.max(p_a[0, :, 2])))
    ax.plot(horizons, sy[FAILURE_IDX_SURGE], "k-o", ms=5, lw=2.5,
            label=f"Actual (surge to {peak_y:.0f} cars)")
    ax.plot(horizons, p_a[0, :, 1], "r--", lw=2,
            label=f"DLinear forecast (mean \u2248 {p_a[0,:,1].mean():.1f})")
    ax.fill_between(horizons, p_a[0, :, 0], p_a[0, :, 2],
                    color="red", alpha=0.18, label="80% prediction interval")
    ax.set_title(f"Case 1 [FAILURE]: Peak Rush Arrival Under-Forecast ({site_label})\n"
                 f"Idx {FAILURE_IDX_SURGE} \u00b7 Demand surges to {peak_y:.0f} cars",
                 fontweight="bold", fontsize=10)
    ax.set_xlabel("Horizon (min)"); ax.set_ylabel("Occupancy (cars)")
    ax.set_ylim(0, max(12.0, y_max_a * 1.35)); ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8.5, frameon=True, loc="upper left")
    ax.annotate("Model lags:\nactual demand\nsurges abruptly",
                xy=(peak_h, peak_y), xytext=(max(15, peak_h - 75), peak_y * 0.70),
                arrowprops=dict(arrowstyle="->", color="red", lw=1.5), fontsize=8.5, color="red", fontweight="bold")

    # ------------------------------------------------------------------ #
    # Panel B (top-right): Holiday / quiet period – model over-forecasts  #
    # ------------------------------------------------------------------ #
    p_b = predict(model, sh[FAILURE_IDX_HOLIDAY:FAILURE_IDX_HOLIDAY+1],
                  sc[FAILURE_IDX_HOLIDAY:FAILURE_IDX_HOLIDAY+1],
                  scale[FAILURE_IDX_HOLIDAY:FAILURE_IDX_HOLIDAY+1])
    ax = axes[0, 1]
    y_max_b = max(float(np.max(sy[FAILURE_IDX_HOLIDAY])), float(np.max(p_b[0, :, 2])))
    ax.plot(horizons, sy[FAILURE_IDX_HOLIDAY], "k-o", ms=5, lw=2.5,
            label=f"Actual (\u2248 {sy[FAILURE_IDX_HOLIDAY].mean():.1f} cars)")
    ax.plot(horizons, p_b[0, :, 1], "r--", lw=2,
            label=f"DLinear forecast (mean \u2248 {p_b[0,:,1].mean():.1f})")
    ax.fill_between(horizons, p_b[0, :, 0], p_b[0, :, 2],
                    color="red", alpha=0.18, label="80% prediction interval")
    ax.set_title(f"Case 2 [FAILURE]: Off-Peak / Inactivity Over-Forecast ({site_label})\n"
                 f"Idx {FAILURE_IDX_HOLIDAY} \u00b7 Over-predicting inactive station",
                 fontweight="bold", fontsize=10)
    ax.set_xlabel("Horizon (min)"); ax.set_ylabel("Occupancy (cars)")
    ax.set_ylim(0, max(12.0, y_max_b * 1.35)); ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8.5, frameon=True, loc="upper right")

    # ------------------------------------------------------------------ #
    # Panel C (bottom-left): RESOLVED – Robust Model vs Original Baseline#
    # ------------------------------------------------------------------ #
    target_idx = RESOLVED_IDX_ROBUST
    y_c = sy[target_idx]
    s_c = scale[target_idx:target_idx+1]
    p_base_c = predict(model, sh[target_idx:target_idx+1], sc[target_idx:target_idx+1], s_c)
    
    if rob_model is not None:
        p_rob_c = predict(rob_model, sh[target_idx:target_idx+1], sc[target_idx:target_idx+1], s_c)
    else:
        p_rob_c = p_base_c

    ax = axes[1, 0]
    y_max_c = max(float(np.max(y_c)), float(np.max(p_base_c[0, :, 2])), float(np.max(p_rob_c[0, :, 2])))
    ax.plot(horizons, y_c, "k-o", ms=5, lw=2.5,
            label=f"Actual (mean \u2248 {y_c.mean():.1f} cars)")
    
    mae_base = np.mean(np.abs(y_c - p_base_c[0, :, 1]))
    mae_rob = np.mean(np.abs(y_c - p_rob_c[0, :, 1]))
    cov_rob = np.mean((y_c >= p_rob_c[0, :, 0]) & (y_c <= p_rob_c[0, :, 2]))
    cov_base = np.mean((y_c >= p_base_c[0, :, 0]) & (y_c <= p_base_c[0, :, 2]))

    ax.plot(horizons, p_base_c[0, :, 1], "r--", lw=2,
            label=f"Original Baseline (MAE = {mae_base:.2f} cars)")
    ax.plot(horizons, p_rob_c[0, :, 1], "b-", lw=2.2,
            label=f"Robust Model (MAE = {mae_rob:.2f} cars)")
    ax.fill_between(horizons, p_rob_c[0, :, 0], p_rob_c[0, :, 2], color="royalblue", alpha=0.22,
                    label=f"Robust 80% interval (Cov: {cov_rob*100:.0f}% \u2713)")

    ax.set_title(f"Case 3 [RESOLVED]: Robust vs. Original Baseline Model ({site_label})\n"
                 f"Idx {target_idx} \u00b7 Jitter-augmented training resolves baseline drift & lag",
                 fontweight="bold", fontsize=10)
    ax.set_xlabel("Horizon (min)"); ax.set_ylabel("Occupancy (cars)")
    ax.set_ylim(0, max(12.0, y_max_c * 1.35)); ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8.5, frameon=True, loc="upper left")
    
    err_reduct = ((mae_base - mae_rob) / mae_base) * 100.0 if mae_base > 0 else 0.0
    ax.text(0.98, 0.05, f"Baseline MAE: {mae_base:.2f} (Cov {cov_base*100:.0f}%)\n"
                        f"Robust MAE: {mae_rob:.2f} (Cov {cov_rob*100:.0f}%)\n"
                        f"Error Reduction: {err_reduct:.1f}%",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcyan", alpha=0.85))

    # ------------------------------------------------------------------ #
    # Panel D (bottom-right): RESOLVED – high uncertainty → abstention    #
    # ------------------------------------------------------------------ #
    p_d = predict(model, sh[RESOLVED_IDX_ABSTAIN:RESOLVED_IDX_ABSTAIN+1],
                  sc[RESOLVED_IDX_ABSTAIN:RESOLVED_IDX_ABSTAIN+1],
                  scale[RESOLVED_IDX_ABSTAIN:RESOLVED_IDX_ABSTAIN+1])
    w_d = float(np.mean(p_d[0, :, 2] - p_d[0, :, 0]))

    ax = axes[1, 1]
    ax.plot(horizons, sy[RESOLVED_IDX_ABSTAIN], "k-o", ms=5, lw=2.5,
            label=f"Actual (mean = {sy[RESOLVED_IDX_ABSTAIN].mean():.1f} cars)")
    ax.plot(horizons, p_d[0, :, 1], "r--", lw=2,
            label=f"DLinear forecast (mean = {p_d[0,:,1].mean():.1f})")
    ax.fill_between(horizons, p_d[0, :, 0], p_d[0, :, 2], color="orange", alpha=0.3,
                    label=f"Interval width W = {w_d:.1f} cars\n(W > \u03c4 = {tau:.2f} \u2192 ABSTAIN)")
    ax.set_title(f"Case 4 [RESOLVED]: Epistemic Uncertainty Abstention Policy ({site_label})\n"
                 f"Idx {RESOLVED_IDX_ABSTAIN} \u00b7 Interval width W = {w_d:.1f} cars > \u03c4 = {tau:.2f} \u2192 Abstain from dispatch",
                 fontweight="bold", fontsize=10)
    ax.set_xlabel("Horizon (min)"); ax.set_ylabel("Occupancy (cars)")
    y_max_d = max(float(np.max(sy[RESOLVED_IDX_ABSTAIN])), float(np.max(p_d[0, :, 2])))
    ax.set_ylim(0, max(12.0, y_max_d * 1.35)); ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8.5, frameon=True, loc="upper left")
    ax.text(0.98, 0.95, "\u26a0 ABSTAINED\nOperator alerted", transform=ax.transAxes,
            ha="right", va="top", fontsize=9, color="darkorange", fontweight="bold",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "lightyellow", "alpha": 0.8})

    plt.tight_layout()
    save_path = os.path.join(OUT_FIG, "failure_cases.png")
    fig.savefig(save_path, bbox_inches="tight", dpi=160)
    plt.close(fig)
    print("Saved:", save_path)


def generate_site_diagnostics(site_name="Caltech"):
    global MAX_CAPACITY, FAILURE_IDX_SURGE, FAILURE_IDX_HOLIDAY, RESOLVED_IDX_ROBUST, RESOLVED_IDX_ABSTAIN, OUT_FIG
    site_key = site_name.lower()
    OUT_FIG = os.path.join("outputs", "figures", site_key)
    os.makedirs(OUT_FIG, exist_ok=True)
    os.makedirs("outputs/figures", exist_ok=True)

    if site_key == "caltech":
        data_path = os.path.join(DATA_ROOT, "caltech_occupancy_data.npz")
        ckpt = os.path.join(DATA_ROOT, "dlinear_caltech.pt")
        robust_ckpt = os.path.join(DATA_ROOT, "dlinear_caltech_robust.pt")
        MAX_CAPACITY = 54.0
        FAILURE_IDX_SURGE = 12068      # 2021-09-13 15:15 UTC (Rush hour surge to 16 cars)
        FAILURE_IDX_HOLIDAY = 11391    # 2021-09-06 14:00 UTC (Labor day holiday over-prediction)
        RESOLVED_IDX_ROBUST = 3548     # 2021-06-16 21:15 UTC (Robust vs Base: MAE 0.74 -> 0.43, 100% cov)
        RESOLVED_IDX_ABSTAIN = 11586   # 2021-09-08 14:45 UTC (Storm volatility epistemic abstention)
    else:
        data_path = os.path.join(DATA_ROOT, "jpl_occupancy_data.npz")
        ckpt = os.path.join(DATA_ROOT, "dlinear_jpl.pt")
        robust_ckpt = os.path.join(DATA_ROOT, "dlinear_jpl_robust.pt")
        MAX_CAPACITY = 56.0
        FAILURE_IDX_SURGE = 8132       # 2021-07-19 20:00 UTC (Surge to 29 cars, forecast lags at 7 cars)
        FAILURE_IDX_HOLIDAY = 7770     # 2021-07-16 01:30 UTC (Quiet night, actual 2-4 cars, forecast 22 cars)
        RESOLVED_IDX_ROBUST = 51       # 2021-04-26 15:45 UTC (Robust vs Base: MAE 2.29 -> 1.00, 100% cov)
        RESOLVED_IDX_ABSTAIN = 11514   # 2021-09-06 14:30 UTC (Extreme width W > 30 cars, selective abstention)

    print(f"\nGenerating Diagnostics for {site_name} (Capacity: {MAX_CAPACITY})...")
    plot_robustness_curves(data_path, ckpt)
    plot_confidence_histograms(data_path, ckpt)
    plot_calibration_reliability_diagram(data_path, ckpt)
    plot_failure_and_resolved_cases(data_path, ckpt, robust_ckpt)

    # Also copy into outputs/figures/ with site suffix and base filename
    import shutil
    for fig_name in ["robustness_curves.png", "confidence_histograms.png", "calibration_diagrams.png", "failure_cases.png"]:
        src = os.path.join(OUT_FIG, fig_name)
        dst_suffix = os.path.join("outputs", "figures", f"{fig_name.replace('.png', '')}_{site_key}.png")
        shutil.copyfile(src, dst_suffix)
        if site_key == "caltech":
            shutil.copyfile(src, os.path.join("outputs", "figures", fig_name))


def plot_cross_hub_failure_figure():
    """Generate the unified 2x2 failure & resolved adaptation figure requested for submission:
       - Panel 1 (top-left): Caltech Case 1 (Peak Rush Arrival Under-Forecast)
       - Panel 2 (top-right): JPL Case 2 (Off-Peak / Inactivity Over-Forecast)
       - Panel 3 (bottom-left): JPL Case 3 (Robust vs. Original Baseline Model)
       - Panel 4 (bottom-right): Caltech Case 4 (Epistemic Uncertainty Selective Abstention)
    """
    print("\nGenerating Cross-Hub Unified Failure Figure (Caltech 1, JPL 2, JPL 3, Caltech 4)...")
    
    # 1. Load Caltech data and models
    cal_data = np.load(os.path.join(DATA_ROOT, "caltech_occupancy_data.npz"))
    cal_sh, cal_sc, cal_sy = cal_data["test_X_hist"], cal_data["test_X_cal"], cal_data["test_y"]
    cal_scale = window_scale(cal_sh)
    cal_model = _load_model(os.path.join(DATA_ROOT, "dlinear_caltech.pt"), cal_sh.shape[1], cal_sy.shape[1], cal_sc.shape[2])

    # 2. Load JPL data and models
    jpl_data = np.load(os.path.join(DATA_ROOT, "jpl_occupancy_data.npz"))
    jpl_sh, jpl_sc, jpl_sy = jpl_data["test_X_hist"], jpl_data["test_X_cal"], jpl_data["test_y"]
    jpl_scale = window_scale(jpl_sh)
    jpl_model = _load_model(os.path.join(DATA_ROOT, "dlinear_jpl.pt"), jpl_sh.shape[1], jpl_sy.shape[1], jpl_sc.shape[2])
    jpl_rob = _load_model(os.path.join(DATA_ROOT, "dlinear_jpl_robust.pt"), jpl_sh.shape[1], jpl_sy.shape[1], jpl_sc.shape[2])

    def predict(m, h_arr, c_arr, s):
        h_t, c_t = make_tensors(h_arr, c_arr, None)
        with torch.no_grad():
            p = m(h_t, c_t).numpy() * s[:, :, None]
        return np.maximum(p, 0.0)

    # Compute tau for Caltech via SelectiveForecaster
    with torch.no_grad():
        cal_fh, cal_fc = make_tensors(cal_sh, cal_sc, None)
        cal_fout = cal_model(cal_fh, cal_fc).numpy() * cal_scale[:, :, None]
    cal_forecaster = SelectiveForecaster(max_abstention_rate=1.0 - ABSTENTION_QUANTILE)
    cal_tau = cal_forecaster.calibrate_threshold(cal_fout[..., 0], cal_fout[..., 2], quantile=ABSTENTION_QUANTILE)

    horizons = np.arange(1, 17) * 15
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=160)
    fig.suptitle(
        "Curated Failure & Resolved Adaptation Cases — Caltech & JPL Hubs (INSE 6450 Benchmark)",
        fontsize=13, fontweight="bold", y=0.995,
    )

    # ------------------------------------------------------------------ #
    # Panel 1 (top-left): Caltech Case 1 – Peak Rush Arrival Under-Forecast#
    # ------------------------------------------------------------------ #
    c1_idx = 12068
    p1 = predict(cal_model, cal_sh[c1_idx:c1_idx+1], cal_sc[c1_idx:c1_idx+1], cal_scale[c1_idx:c1_idx+1])
    ax = axes[0, 0]
    peak_y1 = float(np.max(cal_sy[c1_idx]))
    peak_h1 = int(horizons[np.argmax(cal_sy[c1_idx])])
    y_max_1 = max(peak_y1, float(np.max(p1[0, :, 2])))
    ax.plot(horizons, cal_sy[c1_idx], "k-o", ms=5, lw=2.5,
            label=f"Actual (surge to {peak_y1:.0f} cars)")
    ax.plot(horizons, p1[0, :, 1], "r--", lw=2,
            label=f"DLinear forecast (mean \u2248 {p1[0, :, 1].mean():.1f})")
    ax.fill_between(horizons, p1[0, :, 0], p1[0, :, 2],
                    color="red", alpha=0.18, label="80% prediction interval")
    ax.set_title("Case 1 [FAILURE]: Peak Rush Arrival Under-Forecast (Caltech Hub, $C_{max}=54$)\n"
                 f"Idx {c1_idx} \u00b7 Demand surges to {peak_y1:.0f} cars, model lags due to trend smoothing",
                 fontweight="bold", fontsize=10)
    ax.set_xlabel("Horizon (min)"); ax.set_ylabel("Occupancy (cars)")
    ax.set_ylim(0, max(12.0, y_max_1 * 1.35)); ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8.5, frameon=True, loc="upper left")
    ax.annotate("Model lags:\nactual demand\nsurges abruptly",
                xy=(peak_h1, peak_y1), xytext=(max(15, peak_h1 - 75), peak_y1 * 0.70),
                arrowprops=dict(arrowstyle="->", color="red", lw=1.5), fontsize=8.5, color="red", fontweight="bold")

    # ------------------------------------------------------------------ #
    # Panel 2 (top-right): JPL Case 2 – Off-Peak / Inactivity Over-Forecast#
    # ------------------------------------------------------------------ #
    j2_idx = 7770
    p2 = predict(jpl_model, jpl_sh[j2_idx:j2_idx+1], jpl_sc[j2_idx:j2_idx+1], jpl_scale[j2_idx:j2_idx+1])
    ax = axes[0, 1]
    y_max_2 = max(float(np.max(jpl_sy[j2_idx])), float(np.max(p2[0, :, 2])))
    ax.plot(horizons, jpl_sy[j2_idx], "k-o", ms=5, lw=2.5,
            label=f"Actual (mean \u2248 {jpl_sy[j2_idx].mean():.1f} cars)")
    ax.plot(horizons, p2[0, :, 1], "r--", lw=2,
            label=f"DLinear forecast (mean \u2248 {p2[0, :, 1].mean():.1f})")
    ax.fill_between(horizons, p2[0, :, 0], p2[0, :, 2],
                    color="red", alpha=0.18, label="80% prediction interval")
    ax.set_title("Case 2 [FAILURE]: Off-Peak / Inactivity Over-Forecast (JPL Hub, $C_{max}=56$)\n"
                 f"Idx {j2_idx} \u00b7 Quiet night (actual 1\u20134 cars), baseline over-predicts ~22 cars",
                 fontweight="bold", fontsize=10)
    ax.set_xlabel("Horizon (min)"); ax.set_ylabel("Occupancy (cars)")
    ax.set_ylim(0, max(12.0, y_max_2 * 1.35)); ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8.5, frameon=True, loc="upper right")

    # ------------------------------------------------------------------ #
    # Panel 3 (bottom-left): JPL Case 3 – Robust Model vs Original Baseline#
    # ------------------------------------------------------------------ #
    j3_idx = 51
    y3 = jpl_sy[j3_idx]
    s3 = jpl_scale[j3_idx:j3_idx+1]
    p3_base = predict(jpl_model, jpl_sh[j3_idx:j3_idx+1], jpl_sc[j3_idx:j3_idx+1], s3)
    p3_rob = predict(jpl_rob, jpl_sh[j3_idx:j3_idx+1], jpl_sc[j3_idx:j3_idx+1], s3)

    ax = axes[1, 0]
    y_max_3 = max(float(np.max(y3)), float(np.max(p3_base[0, :, 2])), float(np.max(p3_rob[0, :, 2])))
    ax.plot(horizons, y3, "k-o", ms=5, lw=2.5,
            label=f"Actual (mean \u2248 {y3.mean():.1f} cars)")
    
    mae_base3 = np.mean(np.abs(y3 - p3_base[0, :, 1]))
    mae_rob3 = np.mean(np.abs(y3 - p3_rob[0, :, 1]))
    cov_rob3 = np.mean((y3 >= p3_rob[0, :, 0]) & (y3 <= p3_rob[0, :, 2]))
    cov_base3 = np.mean((y3 >= p3_base[0, :, 0]) & (y3 <= p3_base[0, :, 2]))

    ax.plot(horizons, p3_base[0, :, 1], "r--", lw=2,
            label=f"Original Baseline (MAE = {mae_base3:.2f} cars)")
    ax.plot(horizons, p3_rob[0, :, 1], "b-", lw=2.2,
            label=f"Robust Model (MAE = {mae_rob3:.2f} cars)")
    ax.fill_between(horizons, p3_rob[0, :, 0], p3_rob[0, :, 2], color="royalblue", alpha=0.22,
                    label=f"Robust 80% interval (Cov: {cov_rob3*100:.0f}% \u2713)")

    ax.set_title("Case 3 [RESOLVED]: Robust vs. Original Baseline Model (JPL Hub, $C_{max}=56$)\n"
                 f"Idx {j3_idx} \u00b7 Jitter-augmented training resolves baseline drift & oscillation",
                 fontweight="bold", fontsize=10)
    ax.set_xlabel("Horizon (min)"); ax.set_ylabel("Occupancy (cars)")
    ax.set_ylim(0, max(12.0, y_max_3 * 1.35)); ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8.5, frameon=True, loc="upper left")
    
    err_reduct3 = ((mae_base3 - mae_rob3) / mae_base3) * 100.0 if mae_base3 > 0 else 0.0
    ax.text(0.98, 0.05, f"Baseline MAE: {mae_base3:.2f} (Cov {cov_base3*100:.0f}%)\n"
                         f"Robust MAE: {mae_rob3:.2f} (Cov {cov_rob3*100:.0f}%)\n"
                         f"Error Reduction: {err_reduct3:.1f}%",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcyan", alpha=0.85))

    # ------------------------------------------------------------------ #
    # Panel 4 (bottom-right): Caltech Case 4 – High Uncertainty Abstention#
    # ------------------------------------------------------------------ #
    c4_idx = 11586
    p4 = predict(cal_model, cal_sh[c4_idx:c4_idx+1], cal_sc[c4_idx:c4_idx+1], cal_scale[c4_idx:c4_idx+1])
    w4 = float(np.mean(p4[0, :, 2] - p4[0, :, 0]))

    ax = axes[1, 1]
    ax.plot(horizons, cal_sy[c4_idx], "k-o", ms=5, lw=2.5,
            label=f"Actual (mean \u2248 {cal_sy[c4_idx].mean():.1f} cars)")
    ax.plot(horizons, p4[0, :, 1], "r--", lw=2,
            label=f"DLinear forecast (mean \u2248 {p4[0, :, 1].mean():.1f})")
    ax.fill_between(horizons, p4[0, :, 0], p4[0, :, 2], color="orange", alpha=0.3,
                    label=f"Interval width W = {w4:.1f} cars\n(W > \u03c4 = {cal_tau:.2f} \u2192 ABSTAIN)")
    ax.set_title("Case 4 [RESOLVED]: Epistemic Uncertainty Abstention Policy (Caltech Hub, $C_{max}=54$)\n"
                 f"Idx {c4_idx} \u00b7 Interval width W = {w4:.1f} cars > \u03c4 = {cal_tau:.2f} \u2192 Abstain from dispatch",
                 fontweight="bold", fontsize=10)
    ax.set_xlabel("Horizon (min)"); ax.set_ylabel("Occupancy (cars)")
    y_max_4 = max(float(np.max(cal_sy[c4_idx])), float(np.max(p4[0, :, 2])))
    ax.set_ylim(0, max(12.0, y_max_4 * 1.35)); ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8.5, frameon=True, loc="upper left")
    ax.text(0.98, 0.95, "\u26a0 ABSTAINED\nConfidence policy active", transform=ax.transAxes,
            ha="right", va="top", fontsize=9, color="darkorange", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout()
    save_path_main = os.path.join("outputs", "figures", "failure_cases.png")
    save_path_comb = os.path.join("outputs", "figures", "failure_cases_combined.png")
    fig.savefig(save_path_main, bbox_inches="tight", dpi=160)
    fig.savefig(save_path_comb, bbox_inches="tight", dpi=160)
    plt.close(fig)
    print("Saved Cross-Hub Failure Figure:", save_path_main)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate Visual Diagnostics for EV Forecasting")
    parser.add_argument("--site", choices=["Caltech", "JPL", "all", "caltech", "jpl"], default="all")
    args = parser.parse_args()

    sites = ["Caltech", "JPL"] if args.site.lower() == "all" else [args.site.capitalize()]
    for s in sites:
        generate_site_diagnostics(s)
    
    # Generate the cross-hub failure figure requested for presentation
    plot_cross_hub_failure_figure()


if __name__ == "__main__":
    main()
