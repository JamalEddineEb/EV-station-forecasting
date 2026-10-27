# Forecast model benchmark

Generated: 2026-10-26 12:40:42 EDT

## Setup

- CPU: AMD EPYC 7763 64-Core Processor (256 logical CPUs available, 1 thread(s) used).
- Training device: cuda:0 (NVIDIA A100-SXM4-80GB MIG 7g.80gb, 79.25 GiB).
- Data: 56461 train / 12087 val / 12102 test; input length 672, horizon 16.
- Batch size: training 256, inference 64. Latency = 200 timed calls after 20 warmup calls, including NumPy preprocessing (and H2D/D2H for CUDA, with synchronization).
- Versions: Python 3.8.20, PyTorch 1.12.1+cu113, scikit-learn 1.3.2.

## Training efficiency

| Model | Total fit (s) | Epochs | s/epoch | Samples/s | Peak GPU (MiB) | RSS growth (MiB) | Params | Size (MiB) | GFLOPs/sample (analytic) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dlinear | 22.40 | 13 | 1.461 | 38658 | 53.1 | 2391.5 | 64,626 | 0.2483 | 0.000146 |
| gru | 90.41 | 16 | 5.232 | 10791 | 5763.9 | 2858.9 | 17,539 | 0.0691 | 0.016921 |
| ridge | 27.24 | n/a | n/a | 2073 | n/a | 167.8 | 12,048 | 0.0643 | 0.000024 |
| persistence | 0.00 | n/a | n/a | n/a | n/a | n/a | 0 | 0.0000 | 0.000000 |
| seasonal_naive | 0.00 | n/a | n/a | n/a | n/a | n/a | 0 | 0.0000 | 0.000000 |

Neural fit time includes per-epoch validation and early stopping (first CUDA epoch includes warmup). Ridge time is the solver only; feature preparation is in the JSON. RSS growth is the process peak minus RSS after data loading (each model runs in its own process). FLOPs are hand-derived forward-pass estimates (2 per multiply-add, activations ignored).

## Inference

| Model / runtime | Single p50 (ms) | Single p90 (ms) | Batch | Batch p50 (ms) | Batch p90 (ms) | Samples/s (batch) | RSS (MiB) | GPU peak (+MiB) | Weights (KiB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dlinear/cpu_fp32 | 0.1548 | 0.1628 | 64 | 1.4376 | 1.4554 | 45636 | 3247 | n/a | 254.3 |
| dlinear/cpu_int8 | 0.2980 | 0.3113 | 64 | 1.2894 | 1.3035 | 49937 | 3301 | n/a | 67.2 |
| dlinear/cuda_fp32 | 8.3519 | 8.7341 | 64 | 5.7290 | 5.8085 | 12256 | 3314 | 0.7 | n/a |
| gru/cpu_fp32 | 6.9424 | 7.5108 | 64 | 59.1992 | 61.4909 | 1072 | 3781 | n/a | 70.8 |
| gru/cpu_int8 | 5.8519 | 5.9789 | 64 | 39.0194 | 39.7735 | 1635 | 3770 | n/a | 22.5 |
| gru/cuda_fp32 | 7.8238 | 7.9542 | 64 | 5.4693 | 5.5764 | 12906 | 3776 | 187.0 | n/a |
| ridge/cpu_fp32 | 0.1253 | 0.1326 | 64 | 0.2665 | 0.3010 | 233204 | 849 | n/a | n/a |
| persistence/cpu_numpy | 0.0041 | 0.0043 | 64 | 0.0064 | 0.0067 | 9351093 | 649 | n/a | n/a |
| seasonal_naive/cpu_numpy | 0.0014 | 0.0015 | 64 | 0.0035 | 0.0038 | 18100137 | 659 | n/a | n/a |

## Test accuracy and effect of optimizations

| Model / runtime | MAE (median) | RMSE | dMAE vs first runtime | dRMSE vs first runtime | Raw 80% coverage | Calibrated 80% coverage | Mean width |
|---|---:|---:|---:|---:|---:|---:|---:|
| dlinear/cpu_fp32 | 0.92064 | 1.36472 | +0.00000 | +0.00000 | 0.621 | 0.766 | 2.8622 |
| dlinear/cpu_int8 | 0.92247 | 1.36508 | +0.00182 | +0.00036 | 0.622 | 0.766 | 2.8684 |
| dlinear/cuda_fp32 | 0.92064 | 1.36472 | +0.00000 | +0.00000 | 0.621 | 0.766 | 2.8622 |
| gru/cpu_fp32 | 0.87996 | 1.36738 | +0.00000 | +0.00000 | 0.662 | 0.708 | 2.3524 |
| gru/cpu_int8 | 0.88292 | 1.37090 | +0.00297 | +0.00352 | 0.677 | 0.713 | 2.3694 |
| gru/cuda_fp32 | 0.87996 | 1.36738 | +0.00000 | +0.00000 | 0.662 | 0.708 | 2.3524 |
| ridge/cpu_fp32 | 0.95671 | 1.36285 | +0.00000 | +0.00000 | n/a | 0.658 | 2.0746 |
| persistence/cpu_numpy | 1.21277 | 2.09061 | +0.00000 | +0.00000 | n/a | 0.749 | 2.3750 |
| seasonal_naive/cpu_numpy | 1.21691 | 1.97715 | +0.00000 | +0.00000 | n/a | 0.715 | 2.0000 |

Every system outputs an 80% interval. Neural models predict the 10th and 90th percentiles directly (raw coverage); point forecasters have none. All intervals are then calibrated on the validation set with split-conformal (same as the model comparison). MAE/RMSE use the median (neural) or point forecast; coverage and width average all test samples and horizons.

## Optimizations attempted

- dlinear dynamic INT8: measured
- gru dynamic INT8: measured
- ONNX export and distillation: not attempted. Energy/cost: not measured (no power telemetry).

Numbers are specific to this machine and software stack; rerun on the deployment target.
