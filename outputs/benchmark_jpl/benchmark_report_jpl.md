# Forecast model benchmark

Generated: 2026-10-26 12:40:43 EDT

## Setup

- CPU: AMD EPYC 7763 64-Core Processor (256 logical CPUs available, 1 thread(s) used).
- Training device: cuda:0 (NVIDIA A100-SXM4-80GB MIG 7g.80gb, 79.25 GiB).
- Data: 57242 train / 12255 val / 12270 test; input length 672, horizon 16.
- Batch size: training 256, inference 64. Latency = 200 timed calls after 20 warmup calls, including NumPy preprocessing (and H2D/D2H for CUDA, with synchronization).
- Versions: Python 3.8.20, PyTorch 1.12.1+cu113, scikit-learn 1.3.2.

## Training efficiency

| Model | Total fit (s) | Epochs | s/epoch | Samples/s | Peak GPU (MiB) | RSS growth (MiB) | Params | Size (MiB) | GFLOPs/sample (analytic) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dlinear | 22.33 | 12 | 1.529 | 37442 | 54.1 | 2382.2 | 64,626 | 0.2483 | 0.000146 |
| gru | 60.61 | 8 | 6.303 | 9081 | 5765.0 | 2854.7 | 17,539 | 0.0691 | 0.016921 |
| ridge | 40.84 | n/a | n/a | 1402 | n/a | 169.9 | 12,048 | 0.0643 | 0.000024 |
| persistence | 0.00 | n/a | n/a | n/a | n/a | n/a | 0 | 0.0000 | 0.000000 |
| seasonal_naive | 0.00 | n/a | n/a | n/a | n/a | n/a | 0 | 0.0000 | 0.000000 |

Neural fit time includes per-epoch validation and early stopping (first CUDA epoch includes warmup). Ridge time is the solver only; feature preparation is in the JSON. RSS growth is the process peak minus RSS after data loading (each model runs in its own process). FLOPs are hand-derived forward-pass estimates (2 per multiply-add, activations ignored).

## Inference

| Model / runtime | Single p50 (ms) | Single p90 (ms) | Batch | Batch p50 (ms) | Batch p90 (ms) | Samples/s (batch) | RSS (MiB) | GPU peak (+MiB) | Weights (KiB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dlinear/cpu_fp32 | 0.1338 | 0.1528 | 64 | 1.2763 | 1.2940 | 49863 | 3245 | n/a | 254.3 |
| dlinear/cpu_int8 | 0.2682 | 0.3045 | 64 | 1.1360 | 1.2826 | 55317 | 3304 | n/a | 67.2 |
| dlinear/cuda_fp32 | 7.9782 | 8.5802 | 64 | 5.3496 | 5.5545 | 13004 | 3304 | 0.7 | n/a |
| gru/cpu_fp32 | 7.5489 | 7.5700 | 64 | 66.7701 | 67.1883 | 959 | 3781 | n/a | 70.8 |
| gru/cpu_int8 | 6.6375 | 6.6960 | 64 | 40.6178 | 43.5003 | 1570 | 3779 | n/a | 22.5 |
| gru/cuda_fp32 | 7.8409 | 8.0045 | 64 | 5.4709 | 5.6021 | 12339 | 3774 | 187.0 | n/a |
| ridge/cpu_fp32 | 0.1271 | 0.1348 | 64 | 0.3009 | 0.3111 | 212303 | 853 | n/a | n/a |
| persistence/cpu_numpy | 0.0042 | 0.0044 | 64 | 0.0058 | 0.0073 | 7305085 | 651 | n/a | n/a |
| seasonal_naive/cpu_numpy | 0.0015 | 0.0015 | 64 | 0.0037 | 0.0040 | 17149049 | 662 | n/a | n/a |

## Test accuracy and effect of optimizations

| Model / runtime | MAE (median) | RMSE | dMAE vs first runtime | dRMSE vs first runtime | Raw 80% coverage | Calibrated 80% coverage | Mean width |
|---|---:|---:|---:|---:|---:|---:|---:|
| dlinear/cpu_fp32 | 1.80643 | 3.10826 | +0.00000 | +0.00000 | 0.701 | 0.798 | 6.3837 |
| dlinear/cpu_int8 | 1.87480 | 3.17968 | +0.06837 | +0.07142 | 0.694 | 0.792 | 6.5140 |
| dlinear/cuda_fp32 | 1.80643 | 3.10826 | +0.00000 | +0.00000 | 0.701 | 0.798 | 6.3837 |
| gru/cpu_fp32 | 1.90251 | 3.71436 | +0.00000 | +0.00000 | 0.744 | 0.812 | 6.3749 |
| gru/cpu_int8 | 2.01469 | 3.88585 | +0.11218 | +0.17150 | 0.722 | 0.795 | 6.5139 |
| gru/cuda_fp32 | 1.90251 | 3.71436 | +0.00000 | +0.00000 | 0.744 | 0.812 | 6.3749 |
| ridge/cpu_fp32 | 1.99510 | 3.04615 | +0.00000 | +0.00000 | n/a | 0.696 | 4.4882 |
| persistence/cpu_numpy | 3.24855 | 6.23421 | +0.00000 | +0.00000 | n/a | 0.750 | 7.6250 |
| seasonal_naive/cpu_numpy | 3.17414 | 6.28547 | +0.00000 | +0.00000 | n/a | 0.741 | 6.0000 |

Every system outputs an 80% interval. Neural models predict the 10th and 90th percentiles directly (raw coverage); point forecasters have none. All intervals are then calibrated on the validation set with split-conformal (same as the model comparison). MAE/RMSE use the median (neural) or point forecast; coverage and width average all test samples and horizons.

## Optimizations attempted

- dlinear dynamic INT8: measured
- gru dynamic INT8: measured
- ONNX export and distillation: not attempted. Energy/cost: not measured (no power telemetry).

Numbers are specific to this machine and software stack; rerun on the deployment target.
