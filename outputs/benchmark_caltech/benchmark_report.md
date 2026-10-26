# Forecast model benchmark

Generated: 2026-10-26 11:34:08 EDT

## Setup

- CPU: AMD EPYC 7763 64-Core Processor (256 logical CPUs available, 1 thread(s) used).
- Training device: cuda:0 (NVIDIA A100-SXM4-80GB MIG 7g.80gb, 79.25 GiB).
- Data: 56461 train / 12087 val / 12102 test; input length 672, horizon 16.
- Batch size: training 256, inference 64. Latency = 200 timed calls after 20 warmup calls, including NumPy preprocessing (and H2D/D2H for CUDA, with synchronization).
- Versions: Python 3.8.20, PyTorch 1.12.1+cu113, scikit-learn 1.3.2.

## Training efficiency

| Model | Total fit (s) | Epochs | s/epoch | Samples/s | Peak GPU (MiB) | RSS growth (MiB) | Params | Size (MiB) | GFLOPs/sample (analytic) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dlinear | 21.79 | 13 | 1.393 | 40531 | 53.1 | 2373.6 | 64,626 | 0.2483 | 0.000146 |
| gru | 75.03 | 16 | 4.302 | 13124 | 5763.9 | 2859.1 | 17,539 | 0.0691 | 0.016921 |
| ridge | 28.39 | n/a | n/a | 1988 | n/a | 167.7 | 12,048 | 0.0643 | 0.000024 |
| persistence | 0.00 | n/a | n/a | n/a | n/a | n/a | 0 | 0.0000 | 0.000000 |
| seasonal_naive | 0.00 | n/a | n/a | n/a | n/a | n/a | 0 | 0.0000 | 0.000000 |

Neural fit time includes per-epoch validation and early stopping (first CUDA epoch includes warmup). Ridge time is the solver only; feature preparation is in the JSON. RSS growth is the process peak minus RSS after data loading (each model runs in its own process). FLOPs are hand-derived forward-pass estimates (2 per multiply-add, activations ignored).

## Inference

| Model / runtime | Single p50 (ms) | Single p90 (ms) | Batch | Batch p50 (ms) | Batch p90 (ms) | Samples/s (batch) | RSS (MiB) | GPU peak (+MiB) | Weights (KiB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dlinear/cpu_fp32 | 0.1376 | 0.1451 | 64 | 1.2860 | 1.2925 | 49756 | 3230 | n/a | 254.3 |
| dlinear/cpu_int8 | 0.2656 | 0.2757 | 64 | 1.1452 | 1.2923 | 54573 | 3291 | n/a | 67.2 |
| dlinear/cuda_fp32 | 7.3926 | 7.5398 | 64 | 5.0202 | 5.0891 | 14464 | 3291 | 0.7 | n/a |
| gru/cpu_fp32 | 6.5586 | 6.7472 | 64 | 59.6708 | 65.7338 | 1052 | 3775 | n/a | 70.8 |
| gru/cpu_int8 | 5.9484 | 6.7587 | 64 | 41.1499 | 43.0016 | 1578 | 3725 | n/a | 22.5 |
| gru/cuda_fp32 | 7.8380 | 7.9735 | 64 | 5.4710 | 5.6260 | 12658 | 3794 | 187.0 | n/a |
| ridge/cpu_fp32 | 0.1124 | 0.1207 | 64 | 0.2712 | 0.2801 | 233843 | 849 | n/a | n/a |
| persistence/cpu_numpy | 0.0023 | 0.0024 | 64 | 0.0058 | 0.0060 | 10531140 | 650 | n/a | n/a |
| seasonal_naive/cpu_numpy | 0.0030 | 0.0031 | 64 | 0.0033 | 0.0042 | 12458451 | 665 | n/a | n/a |

## Test accuracy and effect of optimizations

| Model / runtime | MAE (median) | RMSE | dMAE vs first runtime | dRMSE vs first runtime | Raw 80% coverage | Calibrated 80% coverage | Mean width |
|---|---:|---:|---:|---:|---:|---:|---:|
| dlinear/cpu_fp32 | 0.92064 | 1.36472 | +0.00000 | +0.00000 | 0.621 | 0.789 | 3.0406 |
| dlinear/cpu_int8 | 0.92247 | 1.36508 | +0.00182 | +0.00036 | 0.622 | 0.789 | 3.0477 |
| dlinear/cuda_fp32 | 0.92064 | 1.36472 | +0.00000 | +0.00000 | 0.621 | 0.789 | 3.0406 |
| gru/cpu_fp32 | 0.87996 | 1.36738 | +0.00000 | +0.00000 | 0.662 | 0.714 | 2.3724 |
| gru/cpu_int8 | 0.88292 | 1.37090 | +0.00297 | +0.00352 | 0.677 | 0.718 | 2.3867 |
| gru/cuda_fp32 | 0.87996 | 1.36738 | +0.00000 | +0.00000 | 0.662 | 0.714 | 2.3724 |
| ridge/cpu_fp32 | 0.95671 | 1.36285 | +0.00000 | +0.00000 | n/a | 0.724 | 2.5281 |
| persistence/cpu_numpy | 1.21277 | 2.09061 | +0.00000 | +0.00000 | n/a | 0.749 | 3.4628 |
| seasonal_naive/cpu_numpy | 1.21691 | 1.97715 | +0.00000 | +0.00000 | n/a | 0.716 | 2.8231 |

Every system outputs an 80% interval. Neural models predict the 10th and 90th percentiles directly (raw coverage); point forecasters have none. All intervals are then calibrated on the validation set with split-conformal (same as the model comparison). MAE/RMSE use the median (neural) or point forecast; coverage and width average all test samples and horizons.

## Optimizations attempted

- dlinear dynamic INT8: measured
- gru dynamic INT8: measured
- ONNX export and distillation: not attempted. Energy/cost: not measured (no power telemetry).

Numbers are specific to this machine and software stack; rerun on the deployment target.
