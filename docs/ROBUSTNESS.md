# Safety, Robustness, and Reliability Benchmark (Milestone 3)

Comprehensive documentation of the reliability, defensive engineering, and adversarial benchmarking framework developed for multi-horizon EV charging station occupancy forecasting ($H=16$, 4-hour horizon at 15-minute resolution) across the **Caltech Hub** ($C_{\max}=54$ ports) and **JPL Hub** ($C_{\max}=56$ ports).

---

## 1. Architectural Overview & Reliability Tiers

The framework implements defense-in-depth across four distinct operational tiers:

```
+-------------------------------------------------------------------------------+
|                             OPERATIONAL TIERS                                 |
+-------------------------------------------------------------------------------+
|  TIER 1: DATA-LEVEL DEFENSE                                                   |
|  - Time-series jittering augmentation (Gaussian noise sigma = 0.03)            |
|  - Prevents linear trend overfitting & stabilizes moving-average weights      |
+-------------------------------------------------------------------------------+
                                      |
                                      v
+-------------------------------------------------------------------------------+
|  TIER 2: INFERENCE-LEVEL DEFENSIVE SANITIZATION                               |
|  - InputSanitizer: Causal forward/backward-fill for NaN packet blackouts      |
|  - Physical boundary clamping: x in [0, C_max]                                |
|  - Rate-of-change smoothing: suppresses single-step voltage spike glitches    |
+-------------------------------------------------------------------------------+
                                      |
                                      v
+-------------------------------------------------------------------------------+
|  TIER 3: MODEL-LEVEL DEFENSE & ENSEMBLE                                       |
|  - Split Conformal Quantile Regression (Split-CQR): Coverage guarantees       |
|  - Stacking Ensemble: DLinear + Ridge regression lag weighting                |
|  - Dynamic INT8 Quantization: 3.78x model size reduction                      |
+-------------------------------------------------------------------------------+
                                      |
                                      v
+-------------------------------------------------------------------------------+
|  TIER 4: SYSTEM-LEVEL SELECTIVE FORECASTING & RISK CONTROL                    |
|  - SelectiveForecaster: Real-time interval width monitoring (W = y_90 - y_10) |
|  - Uncertainty gating: If W > tau, abstain and alert microgrid operator       |
+-------------------------------------------------------------------------------+
```

---

## 2. Implemented Defense Modules

### A. Data-Level Augmentation (`src/reliability.py` -> `JitterAugmentation`)
- **Problem:** EV telemetry exhibits sensor resolution limits, minor arrival jitter, and clock drift. DLinear's moving-average trend kernel ($k=25$, 6.25h) tends to overfit to static integer occupancy plateaus.
- **Method:** Injects zero-mean Gaussian noise $\mathcal{N}(0, \sigma^2)$ with $\sigma=0.03$ during mini-batch training while strictly enforcing non-negativity: $x_{\text{aug}} = \max(x + \epsilon, 0)$.
- **Impact:** Regularizes the model weights, achieving 100% empirical coverage on steady daytime plateaus (Case 7).

### B. Defensive Input Sanitizer (`src/reliability.py` -> `InputSanitizer`)
- **Problem:** Real-world cellular IoT modems experience packet loss. Ingesting `NaN` readings causes the per-window scale factor $\text{scale} = \max(x)$ to evaluate to `NaN` or divide by zero, crashing downstream PyTorch linear layers (100% pipeline failure).
- **Method:**
  1. *Vectorized Forward Fill:* Propagates the last valid reading across missing intervals.
  2. *Physical Bounding:* Clamps readings to $[0, C_{\max}]$ (e.g. $[0, 54]$ for Caltech, $[0, 56]$ for JPL).
  3. *Rate-of-Change Filter:* Detects single-step sensor voltage glitches ($\Delta x > 15$ cars in 15 min) and smooths with adjacent readings.
- **Impact:** **100% crash recovery** (50 out of 50 simulated blackouts recovered), restoring valid calibrated forecasts without operator intervention.

### C. Conformal Quantile Calibration (`src/model_selection/conformal.py` -> `conformalize`)
- **Problem:** Uncalibrated neural quantile regression (pinball loss) suffers from severe undercoverage under distribution shifts (empirical 80% coverage drops to 51–61%).
- **Method:** Computes horizon-specific nonconformity scores on held-out validation sequences:
  $$\hat{E}_{i,h} = \max(\hat{y}_{10,h}(x_i) - y_{i,h},\; y_{i,h} - \hat{y}_{90,h}(x_i))$$
  Dynamically expands test prediction intervals using the $(1 - \alpha)(1 + 1/n)$-th quantile of calibration residuals.
- **Impact:** Mathematically guarantees finite-sample, distribution-free coverage, restoring empirical coverage to **76.7%–80.0%** across all 16 forecast horizons.

### D. Selective Forecaster & Abstention Policy (`src/reliability.py` -> `SelectiveForecaster`)
- **Problem:** During severe demand volatility or grid anomalies, point predictions can exhibit disastrous errors (MAE > 10–15 cars). Automated smart-charging systems blindly trusting these forecasts risk transformer overload.
- **Method:** Uses conformal prediction interval width $W = \hat{y}_{90} - \hat{y}_{10}$ as a measure of epistemic uncertainty. When $W > \tau$ (calibrated 90th percentile threshold), the forecaster abstains from automated dispatch, alerts human operators, and reverts to a safe baseline policy.
- **Impact:** Reduces retained forecast error by **10%–15%** (Caltech: 0.90 $\to$ 0.82 cars; JPL: 1.84 $\to$ 1.56 cars) and eliminates catastrophic dispatches.

### E. Multi-Model Stacking Ensemble (`src/reliability.py` -> `EnsembleForecaster`)
- **Problem:** DLinear's moving-average kernel smooths near-term transitions, creating inertia lag at the immediate next step ($H=1$, 15 min ahead).
- **Method:** Blends DLinear's seasonal decomposition with Ridge regression lag weights:
  $$\hat{y}_{\text{ens}} = 0.5 \cdot \hat{y}_{\text{DLinear}} + 0.5 \cdot \hat{y}_{\text{Ridge}}$$
- **Impact:** Ridge's direct weight on the current observation ($x_t$) reduces immediate Horizon-1 lag by **18.2% on JPL** and **5.0% on Caltech**.

### F. Dynamic INT8 Quantization (`torch.ao.quantization.quantize_dynamic`)
- **Problem:** Microcontrollers and embedded edge gateways have limited RAM and flash storage.
- **Method:** Post-training dynamic quantization converts linear layer weights from 32-bit floating point (`FP32`) to 8-bit signed integers (`qint8`).
- **Impact:** Shrinks weight footprint by **3.78x** (254.3 KiB $\to$ 67.2 KiB) with zero loss in forecasting accuracy or coverage.

---

## 3. Adversarial Threat Models (`src/adversarial.py`)

The framework formally tests models against three distinct attack vectors constrained by physical station capacity:

| Threat Model | Class Name | Perturbation Formulation | Physical Constraints |
|:---|:---|:---|:---|
| **Black-Box Uniform Noise** | `RandomPerturbation` | $x_{\text{adv}} = x + \delta,\; \delta \sim \mathcal{U}(-\epsilon, \epsilon)$ | Clamped to $[0, C_{\max}]$ |
| **White-Box 1-Step (Analytical)** | `FGSMAttack` | $x_{\text{adv}} = x + \epsilon \cdot \text{sign}(\nabla_x \mathcal{L}(f(x), y))$ | Projected onto physical box |
| **White-Box Multi-Step (Worst-Case)** | `PGDAttack` | $x_{t+1} = \Pi_{x + \mathcal{S}} \left( x_t + \alpha \cdot \text{sign}(\nabla_x \mathcal{L}) \right)$ | $L_\infty$ ball + station limits $[0, C_{\max}]$ |

---

## 4. Execution & CLI Usage Guide

All commands run directly from the repository root:

### 1. Train Robust Models (with Jittering Augmentation)
```bash
# Train on both Caltech and JPL stations (~1 min on CPU)
python src/train_robust.py --site all

# Or train on a single hub
python src/train_robust.py --site Caltech
python src/train_robust.py --site JPL
```

### 2. Run Comprehensive Quantitative Robustness Benchmark
```bash
# Benchmark all reliability tiers and export primary CSV tables
python src/benchmark_robustness.py --site all
```
*Outputs generated:*
- `outputs/tables/benchmark_task_metrics_caltech.csv` & `benchmark_task_metrics_jpl.csv`
- `outputs/tables/benchmark_latency_caltech.csv` & `benchmark_latency_jpl.csv`
- `outputs/tables/benchmark_model_sizes_caltech.csv` & `benchmark_model_sizes_jpl.csv`
- `outputs/tables/failure_cases_extracted_caltech.csv` & `failure_cases_extracted_jpl.csv`

### 3. Generate Visual Diagnostic Figures
```bash
# Generate all 4 diagnostic figures for both hubs
python src/generate_diagnostics.py --site all
```
*Figures generated in `outputs/figures/`:*
- `robustness_curves.png`: MAE and coverage decay across $\epsilon \in [0.0, 5.0]$ cars.
- `confidence_histograms.png`: Interval width distributions under clean vs attacked states.
- `calibration_diagrams.png`: Reliability diagrams comparing uncalibrated vs Split-CQR coverage.
- `failure_cases.png` & `failure_cases_combined.png`: 2x2 multi-panel failure and adaptation trajectories.

### 4. Run Standalone Adversarial & Reliability Sweeps
```bash
# Run multi-budget adversarial sweep (eps = 0.5 to 5.0 cars)
python src/evaluate_adversarial.py --site all

# Run standalone telemetry sanitization and selective risk-coverage trade-off
python src/evaluate_reliability.py --site all
```

---

## 5. Quantitative Benchmark Summary

### Primary Task Metrics Across Robustness Tiers (95% Bootstrapped CIs)

| Condition / Pipeline | Caltech MAE [95% CI] | Caltech 80% Cov | JPL MAE [95% CI] | JPL 80% Cov | Key Reliability Finding |
|:---|:---:|:---:|:---:|:---:|:---|
| **Clean: Original Model (Raw)** | 0.9040 [0.893, 0.916] | 0.611 | 1.8404 [1.815, 1.867] | 0.510 | Dangerously overconfident |
| **Clean: Conformal Split-CQR** | **0.9040** [0.893, 0.916] | **0.767** | **1.8404** [1.815, 1.867] | **0.796** | **Valid statistical guarantee** |
| **Clean: Updated Robust Model** | 0.9241 [0.912, 0.937] | 0.769 | 1.8420 [1.816, 1.868] | 0.812 | Stabilizes daytime plateaus |
| **Clean: Stacking Ensemble** | 0.8997 [0.888, 0.911] | 0.777 | 1.7858 [1.761, 1.811] | 0.818 | Cuts near-term lag by 5–18% |
| **Clean: Confidence Abstention** | **0.8157** [0.805, 0.826] | 0.768 | **1.5645** [1.542, 1.587] | 0.805 | **10–15% error reduction on retained** |
| **Corrupted (Without Sanitizer)** | *FAILED (NaN Crash)* | 0.000 | *FAILED (NaN Crash)* | 0.000 | Zero-scale division crash |
| **Corrupted (With InputSanitizer)**| **0.7654** [0.724, 0.812] | **0.760** | **1.7483** [1.688, 1.808] | **0.781** | **100% crash recovery** |
| **White-Box PGD-7 ($\epsilon=1.0$)** | 2.8234 [2.746, 2.899] | 0.249 | 4.5422 [4.416, 4.672] | 0.329 | Adversary drives false troughs |
| **PGD-7 + Stacking Ensemble** | 2.7970 [2.715, 2.874] | 0.166 | 4.3512 [4.240, 4.464] | 0.274 | Non-differentiable anchor |
| **PGD-7 + Confidence Abstention** | 2.7767 [2.709, 2.863] | 0.211 | 4.1950 [4.077, 4.314] | 0.287 | Flags high-risk attacks |

---

## 6. Failure Modes Analysis: What is Resolved vs Unresolved

| Scenario / Case ID | Failure Description | Root Cause | Status | Resolution / Limitation |
|:---:|:---|:---|:---:|:---|
| **Case 1** | Monday Morning Surge | Moving average ($k=25$) smooths recent history | **UNMITIGATED** | Fundamental information barrier ("ghost demand"); unobserved future arrivals require external reservation API. |
| **Case 2** | Post-Holiday Rebound Dip | 7-day cyclical history contains preceding holiday | **UNMITIGATED** | Requires custom post-holiday transition weights. |
| **Case 3** | Campus Event Surge | Surge invisible in sensor telemetry | **UNMITIGATED** | Requires calendar booking API. |
| **Case 4** | Statutory Holiday False Commute | Sinusoidal day-of-week embeddings inject $+6$ cars | **RESOLVED** | Multiplicative holiday gating cuts holiday MAE by **42.6% on JPL**. |
| **Case 6** | White-Box PGD-7 Attack | Gradients steer predictions into false troughs | **ADAPTATION** | Median filtering + non-negative clamping $[0, C_{\max}]$. |
| **Case 7** | Telemetry Packet Loss | Missing readings produce NaN scale factors | **RESOLVED** | `InputSanitizer` forward-fill achieves **100% crash recovery**. |
| **Case 8** | Epistemic Volatility Spike | Severe prediction error under stormy weather | **RESOLVED** | `SelectiveForecaster` ($W > \tau$) **abstains from automated dispatch**. |
| **Case 9** | Near-Term Step-1 Lag | DLinear moving-average trend delay at $t+1$ | **PARTIAL** | Stacking Ensemble cuts Horizon-1 error by **18.2% on JPL**, but multi-horizon surge persists. |
