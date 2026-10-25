# Forecast model benchmark

Generated: 2026-10-25 16:34:29 EDT

## Setup

- CPU: AMD EPYC 7763 64-Core Processor (256 logical CPUs available, 1 thread(s) used).
- Training device: cuda:0 (NVIDIA A100-SXM4-80GB MIG 7g.80gb, 79.25 GiB).
- Data: 56461 train / 12087 val / 12102 test; input length 672, horizon 16.
- Batch size: training 256, inference 64. Latency = 200 timed calls after 20 warmup calls, including NumPy preprocessing (and H2D/D2H for CUDA, with synchronization).
- Versions: Python 3.8.20, PyTorch 1.12.1+cu113, scikit-learn 1.3.2.

## Training efficiency

| Model | Total fit (s) | Epochs | s/epoch | Samples/s | Peak GPU (MiB) | RSS growth (MiB) | Params | Size (MiB) | GFLOPs/sample (analytic) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dlinear | 17.43 | 10 | 1.419 | 39781 | 52.2 | 2363.1 | 21,542 | 0.0840 | 0.000060 |
| gru | 61.24 | 15 | 3.664 | 15411 | 5763.9 | 2853.2 | 17,409 | 0.0686 | 0.016916 |
| ridge | 28.05 | n/a | n/a | 2013 | n/a | 167.6 | 12,048 | 0.0643 | 0.000024 |
| persistence | 0.00 | n/a | n/a | n/a | n/a | n/a | 0 | 0.0000 | 0.000000 |
| seasonal_naive | 0.00 | n/a | n/a | n/a | n/a | n/a | 0 | 0.0000 | 0.000000 |

Neural fit time includes per-epoch validation and early stopping (first CUDA epoch includes warmup). Ridge time is the solver only; feature preparation is in the JSON. RSS growth is the process peak minus RSS after data loading (each model runs in its own process). FLOPs are hand-derived forward-pass estimates (2 per multiply-add, activations ignored).

## Inference

| Model / runtime | Single p50 (ms) | Single p90 (ms) | Batch | Batch p50 (ms) | Batch p90 (ms) | Samples/s (batch) | RSS (MiB) | GPU peak (+MiB) | Weights (KiB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dlinear/cpu_fp32 | 0.0959 | 0.0994 | 64 | 0.9951 | 1.1219 | 61864 | 3219 | n/a | 86.0 |
| dlinear/cpu_int8 | 0.2357 | 0.2477 | 64 | 1.0214 | 1.0301 | 62158 | 3225 | n/a | 24.9 |
| dlinear/cuda_fp32 | 7.3663 | 7.4991 | 64 | 4.9870 | 5.0861 | 15754 | 3225 | 0.7 | n/a |
| gru/cpu_fp32 | 7.4348 | 7.4612 | 64 | 65.7914 | 66.7009 | 970 | 3763 | n/a | 70.3 |
| gru/cpu_int8 | 6.6078 | 6.6490 | 64 | 42.3450 | 47.0776 | 1468 | 3762 | n/a | 22.4 |
| gru/cuda_fp32 | 7.7974 | 7.9002 | 64 | 5.4416 | 5.5436 | 12865 | 3769 | 187.0 | n/a |
| ridge/cpu_fp32 | 0.1307 | 0.1377 | 64 | 0.3050 | 0.3196 | 212211 | 837 | n/a | n/a |
| persistence/cpu_numpy | 0.0023 | 0.0023 | 64 | 0.0060 | 0.0062 | 10343586 | 651 | n/a | n/a |
| seasonal_naive/cpu_numpy | 0.0013 | 0.0014 | 64 | 0.0050 | 0.0054 | 13486261 | 653 | n/a | n/a |

## Test accuracy and effect of optimizations

| Model / runtime | MAE | RMSE | dMAE vs first runtime | dRMSE vs first runtime |
|---|---:|---:|---:|---:|
| dlinear/cpu_fp32 | 0.95944 | 1.36058 | +0.00000 | +0.00000 |
| dlinear/cpu_int8 | 0.96110 | 1.36190 | +0.00165 | +0.00132 |
| dlinear/cuda_fp32 | 0.95944 | 1.36058 | +0.00000 | +0.00000 |
| gru/cpu_fp32 | 1.06875 | 1.54411 | +0.00000 | +0.00000 |
| gru/cpu_int8 | 1.06464 | 1.53770 | -0.00411 | -0.00641 |
| gru/cuda_fp32 | 1.06875 | 1.54411 | +0.00000 | +0.00000 |
| ridge/cpu_fp32 | 0.95671 | 1.36285 | +0.00000 | +0.00000 |
| persistence/cpu_numpy | 1.21277 | 2.09061 | +0.00000 | +0.00000 |
| seasonal_naive/cpu_numpy | 1.21691 | 1.97715 | +0.00000 | +0.00000 |

## Optimizations attempted

- dlinear dynamic INT8: measured
- gru dynamic INT8: measured
- ONNX export and distillation: not attempted. Energy/cost: not measured (no power telemetry).

Numbers are specific to this machine and software stack; rerun on the deployment target.
