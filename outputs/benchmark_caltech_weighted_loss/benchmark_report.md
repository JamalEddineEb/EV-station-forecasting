# Forecast model benchmark

Generated: 2026-10-26 11:12:10 EDT

## Setup

- CPU: AMD EPYC 7763 64-Core Processor (256 logical CPUs available, 1 thread(s) used).
- Training device: cuda:0 (NVIDIA A100-SXM4-80GB MIG 7g.80gb, 79.25 GiB).
- Data: 56461 train / 12087 val / 12102 test; input length 672, horizon 16.
- Batch size: training 256, inference 64. Latency = 200 timed calls after 20 warmup calls, including NumPy preprocessing (and H2D/D2H for CUDA, with synchronization).
- Versions: Python 3.8.20, PyTorch 1.12.1+cu113, scikit-learn 1.3.2.

## Training efficiency

| Model | Total fit (s) | Epochs | s/epoch | Samples/s | Peak GPU (MiB) | RSS growth (MiB) | Params | Size (MiB) | GFLOPs/sample (analytic) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dlinear | 24.31 | 15 | 1.383 | 40814 | 53.1 | 2384.7 | 64,626 | 0.2483 | 0.000146 |
| gru | 100.80 | 22 | 4.291 | 13158 | 5763.9 | 2852.1 | 17,539 | 0.0691 | 0.016921 |
| ridge | 27.13 | n/a | n/a | 2081 | n/a | 167.6 | 12,048 | 0.0643 | 0.000024 |
| persistence | 0.00 | n/a | n/a | n/a | n/a | n/a | 0 | 0.0000 | 0.000000 |
| seasonal_naive | 0.00 | n/a | n/a | n/a | n/a | n/a | 0 | 0.0000 | 0.000000 |

Neural fit time includes per-epoch validation and early stopping (first CUDA epoch includes warmup). Ridge time is the solver only; feature preparation is in the JSON. RSS growth is the process peak minus RSS after data loading (each model runs in its own process). FLOPs are hand-derived forward-pass estimates (2 per multiply-add, activations ignored).

## Inference

| Model / runtime | Single p50 (ms) | Single p90 (ms) | Batch | Batch p50 (ms) | Batch p90 (ms) | Samples/s (batch) | RSS (MiB) | GPU peak (+MiB) | Weights (KiB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dlinear/cpu_fp32 | 0.1365 | 0.1542 | 64 | 1.2743 | 1.2867 | 49876 | 3236 | n/a | 254.3 |
| dlinear/cpu_int8 | 0.2663 | 0.2757 | 64 | 1.1381 | 1.2337 | 55307 | 3309 | n/a | 67.2 |
| dlinear/cuda_fp32 | 7.4161 | 7.5176 | 64 | 5.0020 | 5.1094 | 15556 | 3309 | 0.7 | n/a |
| gru/cpu_fp32 | 6.5791 | 6.9499 | 64 | 59.6328 | 61.7081 | 1065 | 3769 | n/a | 70.8 |
| gru/cpu_int8 | 5.9155 | 6.0458 | 64 | 38.1943 | 42.3836 | 1622 | 3767 | n/a | 22.5 |
| gru/cuda_fp32 | 7.8463 | 7.9746 | 64 | 5.4736 | 5.6132 | 12634 | 3771 | 187.0 | n/a |
| ridge/cpu_fp32 | 0.1138 | 0.1197 | 64 | 0.2693 | 0.2813 | 233540 | 849 | n/a | n/a |
| persistence/cpu_numpy | 0.0022 | 0.0024 | 64 | 0.0057 | 0.0059 | 10574029 | 647 | n/a | n/a |
| seasonal_naive/cpu_numpy | 0.0030 | 0.0033 | 64 | 0.0037 | 0.0050 | 12058349 | 663 | n/a | n/a |

## Test accuracy and effect of optimizations

| Model / runtime | MAE (median) | RMSE | dMAE vs first runtime | dRMSE vs first runtime | Raw 80% coverage | Calibrated 80% coverage | Mean width |
|---|---:|---:|---:|---:|---:|---:|---:|
| dlinear/cpu_fp32 | 0.92897 | 1.37938 | +0.00000 | +0.00000 | 0.624 | 0.785 | 3.0247 |
| dlinear/cpu_int8 | 0.93119 | 1.38087 | +0.00222 | +0.00150 | 0.621 | 0.785 | 3.0281 |
| dlinear/cuda_fp32 | 0.92897 | 1.37938 | +0.00000 | +0.00000 | 0.624 | 0.785 | 3.0247 |
| gru/cpu_fp32 | 0.91004 | 1.40475 | +0.00000 | +0.00000 | 0.695 | 0.716 | 2.3016 |
| gru/cpu_int8 | 0.91456 | 1.40777 | +0.00452 | +0.00302 | 0.704 | 0.731 | 2.3722 |
| gru/cuda_fp32 | 0.91004 | 1.40475 | -0.00000 | -0.00000 | 0.695 | 0.716 | 2.3016 |
| ridge/cpu_fp32 | 0.95671 | 1.36285 | +0.00000 | +0.00000 | n/a | 0.724 | 2.5281 |
| persistence/cpu_numpy | 1.21277 | 2.09061 | +0.00000 | +0.00000 | n/a | 0.749 | 3.4628 |
| seasonal_naive/cpu_numpy | 1.21691 | 1.97715 | +0.00000 | +0.00000 | n/a | 0.716 | 2.8231 |

Every system outputs an 80% interval. Neural models predict the 10th and 90th percentiles directly (raw coverage); point forecasters have none. All intervals are then calibrated on the validation set with split-conformal (same as the model comparison). MAE/RMSE use the median (neural) or point forecast; coverage and width average all test samples and horizons.

## Optimizations attempted

- dlinear dynamic INT8: measured
- gru dynamic INT8: measured
- ONNX export and distillation: not attempted. Energy/cost: not measured (no power telemetry).

Numbers are specific to this machine and software stack; rerun on the deployment target.
