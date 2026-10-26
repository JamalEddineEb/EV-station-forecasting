# Forecast model benchmark

Generated: 2026-10-26 12:14:58 EDT

## Setup

- CPU: AMD EPYC 7763 64-Core Processor (256 logical CPUs available, 1 thread(s) used).
- Training device: cuda:0 (NVIDIA A100-SXM4-80GB MIG 7g.80gb, 79.25 GiB).
- Data: 57242 train / 12255 val / 12270 test; input length 672, horizon 16.
- Batch size: training 256, inference 64. Latency = 200 timed calls after 20 warmup calls, including NumPy preprocessing (and H2D/D2H for CUDA, with synchronization).
- Versions: Python 3.8.20, PyTorch 1.12.1+cu113, scikit-learn 1.3.2.

## Training efficiency

| Model | Total fit (s) | Epochs | s/epoch | Samples/s | Peak GPU (MiB) | RSS growth (MiB) | Params | Size (MiB) | GFLOPs/sample (analytic) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dlinear | 20.26 | 12 | 1.408 | 40649 | 54.1 | 2384.9 | 64,626 | 0.2483 | 0.000146 |
| gru | 41.47 | 8 | 4.404 | 12997 | 5765.0 | 2866.0 | 17,539 | 0.0691 | 0.016921 |
| ridge | 39.40 | n/a | n/a | 1453 | n/a | 169.9 | 12,048 | 0.0643 | 0.000024 |
| persistence | 0.00 | n/a | n/a | n/a | n/a | n/a | 0 | 0.0000 | 0.000000 |
| seasonal_naive | 0.00 | n/a | n/a | n/a | n/a | n/a | 0 | 0.0000 | 0.000000 |

Neural fit time includes per-epoch validation and early stopping (first CUDA epoch includes warmup). Ridge time is the solver only; feature preparation is in the JSON. RSS growth is the process peak minus RSS after data loading (each model runs in its own process). FLOPs are hand-derived forward-pass estimates (2 per multiply-add, activations ignored).

## Inference

| Model / runtime | Single p50 (ms) | Single p90 (ms) | Batch | Batch p50 (ms) | Batch p90 (ms) | Samples/s (batch) | RSS (MiB) | GPU peak (+MiB) | Weights (KiB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dlinear/cpu_fp32 | 0.1361 | 0.1431 | 64 | 1.2713 | 1.3181 | 49860 | 3247 | n/a | 254.3 |
| dlinear/cpu_int8 | 0.2613 | 0.2730 | 64 | 1.1310 | 1.1490 | 56051 | 3319 | n/a | 67.2 |
| dlinear/cuda_fp32 | 7.4091 | 7.5680 | 64 | 5.0182 | 5.1424 | 15186 | 3320 | 0.7 | n/a |
| gru/cpu_fp32 | 6.7160 | 6.7374 | 64 | 59.3717 | 60.3518 | 1073 | 3791 | n/a | 70.8 |
| gru/cpu_int8 | 5.9764 | 6.0142 | 64 | 38.6768 | 42.3147 | 1610 | 3791 | n/a | 22.5 |
| gru/cuda_fp32 | 7.8076 | 7.9365 | 64 | 5.4672 | 5.5940 | 12510 | 3796 | 187.0 | n/a |
| ridge/cpu_fp32 | 0.1141 | 0.1245 | 64 | 0.2644 | 0.3039 | 235666 | 857 | n/a | n/a |
| persistence/cpu_numpy | 0.0025 | 0.0026 | 64 | 0.0065 | 0.0067 | 9393447 | 650 | n/a | n/a |
| seasonal_naive/cpu_numpy | 0.0015 | 0.0030 | 64 | 0.0035 | 0.0049 | 11638238 | 666 | n/a | n/a |

## Test accuracy and effect of optimizations

| Model / runtime | MAE (median) | RMSE | dMAE vs first runtime | dRMSE vs first runtime | Raw 80% coverage | Calibrated 80% coverage | Mean width |
|---|---:|---:|---:|---:|---:|---:|---:|
| dlinear/cpu_fp32 | 1.80643 | 3.10826 | +0.00000 | +0.00000 | 0.701 | 0.811 | 6.5426 |
| dlinear/cpu_int8 | 1.87480 | 3.17968 | +0.06837 | +0.07142 | 0.694 | 0.805 | 6.6933 |
| dlinear/cuda_fp32 | 1.80643 | 3.10826 | +0.00000 | +0.00000 | 0.701 | 0.811 | 6.5426 |
| gru/cpu_fp32 | 1.90251 | 3.71436 | +0.00000 | +0.00000 | 0.744 | 0.820 | 6.4808 |
| gru/cpu_int8 | 2.01469 | 3.88585 | +0.11218 | +0.17150 | 0.722 | 0.805 | 6.6380 |
| gru/cuda_fp32 | 1.90251 | 3.71436 | +0.00000 | +0.00000 | 0.744 | 0.820 | 6.4808 |
| ridge/cpu_fp32 | 1.99510 | 3.04615 | +0.00000 | +0.00000 | n/a | 0.736 | 5.0498 |
| persistence/cpu_numpy | 3.24855 | 6.23421 | +0.00000 | +0.00000 | n/a | 0.771 | 9.8618 |
| seasonal_naive/cpu_numpy | 3.17414 | 6.28547 | +0.00000 | +0.00000 | n/a | 0.746 | 6.9774 |

Every system outputs an 80% interval. Neural models predict the 10th and 90th percentiles directly (raw coverage); point forecasters have none. All intervals are then calibrated on the validation set with split-conformal (same as the model comparison). MAE/RMSE use the median (neural) or point forecast; coverage and width average all test samples and horizons.

## Optimizations attempted

- dlinear dynamic INT8: measured
- gru dynamic INT8: measured
- ONNX export and distillation: not attempted. Energy/cost: not measured (no power telemetry).

Numbers are specific to this machine and software stack; rerun on the deployment target.
