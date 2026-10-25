# Forecast model benchmark

Generated: 2026-10-25 17:25:58 EDT

## Setup

- CPU: AMD EPYC 7763 64-Core Processor (256 logical CPUs available, 1 thread(s) used).
- Training device: cuda:0 (NVIDIA A100-SXM4-80GB MIG 7g.80gb, 79.25 GiB).
- Data: 56461 train / 12087 val / 12102 test; input length 672, horizon 16.
- Batch size: training 256, inference 64. Latency = 200 timed calls after 20 warmup calls, including NumPy preprocessing (and H2D/D2H for CUDA, with synchronization).
- Versions: Python 3.8.20, PyTorch 1.12.1+cu113, scikit-learn 1.3.2.

## Training efficiency

| Model | Total fit (s) | Epochs | s/epoch | Samples/s | Peak GPU (MiB) | RSS growth (MiB) | Params | Size (MiB) | GFLOPs/sample (analytic) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dlinear | 20.03 | 12 | 1.390 | 40620 | 53.1 | 2375.3 | 64,626 | 0.2483 | 0.000146 |
| gru | 49.97 | 10 | 4.341 | 13007 | 5763.9 | 2869.6 | 17,539 | 0.0691 | 0.016921 |
| ridge | 31.35 | n/a | n/a | 1801 | n/a | 167.6 | 12,048 | 0.0643 | 0.000024 |
| persistence | 0.00 | n/a | n/a | n/a | n/a | n/a | 0 | 0.0000 | 0.000000 |
| seasonal_naive | 0.00 | n/a | n/a | n/a | n/a | n/a | 0 | 0.0000 | 0.000000 |

Neural fit time includes per-epoch validation and early stopping (first CUDA epoch includes warmup). Ridge time is the solver only; feature preparation is in the JSON. RSS growth is the process peak minus RSS after data loading (each model runs in its own process). FLOPs are hand-derived forward-pass estimates (2 per multiply-add, activations ignored).

## Inference

| Model / runtime | Single p50 (ms) | Single p90 (ms) | Batch | Batch p50 (ms) | Batch p90 (ms) | Samples/s (batch) | RSS (MiB) | GPU peak (+MiB) | Weights (KiB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dlinear/cpu_fp32 | 0.1158 | 0.1259 | 64 | 1.2393 | 1.4011 | 50089 | 3232 | n/a | 254.3 |
| dlinear/cpu_int8 | 0.2587 | 0.2702 | 64 | 1.1323 | 1.2479 | 55049 | 3267 | n/a | 67.2 |
| dlinear/cuda_fp32 | 7.4078 | 7.5445 | 64 | 5.0168 | 5.1243 | 14658 | 3267 | 0.7 | n/a |
| gru/cpu_fp32 | 6.8006 | 7.4849 | 64 | 60.3984 | 66.5880 | 1034 | 3819 | n/a | 70.8 |
| gru/cpu_int8 | 5.9114 | 6.4988 | 64 | 39.2369 | 40.3982 | 1629 | 3786 | n/a | 22.5 |
| gru/cuda_fp32 | 7.8301 | 7.9622 | 64 | 5.4691 | 5.5717 | 12498 | 3747 | 187.0 | n/a |
| ridge/cpu_fp32 | 0.1360 | 0.1454 | 64 | 0.2978 | 0.3372 | 210062 | 847 | n/a | n/a |
| persistence/cpu_numpy | 0.0025 | 0.0028 | 64 | 0.0090 | 0.0098 | 7497811 | 648 | n/a | n/a |
| seasonal_naive/cpu_numpy | 0.0015 | 0.0015 | 64 | 0.0076 | 0.0087 | 9042065 | 649 | n/a | n/a |

## Test accuracy and effect of optimizations

| Model / runtime | MAE (median) | RMSE | dMAE vs first runtime | dRMSE vs first runtime | Raw 80% coverage | Calibrated 80% coverage | Mean width |
|---|---:|---:|---:|---:|---:|---:|---:|
| dlinear/cpu_fp32 | 0.93962 | 1.36819 | +0.00000 | +0.00000 | 0.611 | 0.713 | 2.4842 |
| dlinear/cpu_int8 | 0.94031 | 1.36911 | +0.00069 | +0.00091 | 0.614 | 0.711 | 2.4781 |
| dlinear/cuda_fp32 | 0.93962 | 1.36819 | +0.00000 | +0.00000 | 0.611 | 0.713 | 2.4842 |
| gru/cpu_fp32 | 1.04685 | 1.70861 | +0.00000 | +0.00000 | 0.650 | 0.716 | 2.4988 |
| gru/cpu_int8 | 1.05886 | 1.72119 | +0.01201 | +0.01258 | 0.630 | 0.725 | 2.5343 |
| gru/cuda_fp32 | 1.04685 | 1.70861 | +0.00000 | +0.00000 | 0.650 | 0.716 | 2.4988 |
| ridge/cpu_fp32 | 0.95671 | 1.36285 | +0.00000 | +0.00000 | n/a | 0.658 | 2.0747 |
| persistence/cpu_numpy | 1.21277 | 2.09061 | +0.00000 | +0.00000 | n/a | 0.749 | 2.3750 |
| seasonal_naive/cpu_numpy | 1.21691 | 1.97715 | +0.00000 | +0.00000 | n/a | 0.715 | 2.0000 |

Every system outputs an 80% interval. Neural models predict the 10th and 90th percentiles directly (raw coverage); point forecasters have none. All intervals are then calibrated on the validation set with split-conformal (same as the model comparison). MAE/RMSE use the median (neural) or point forecast; coverage and width average all test samples and horizons.

## Optimizations attempted

- dlinear dynamic INT8: measured
- gru dynamic INT8: measured
- ONNX export and distillation: not attempted. Energy/cost: not measured (no power telemetry).

Numbers are specific to this machine and software stack; rerun on the deployment target.
