# Forecast model benchmark

Generated: 2026-10-24 21:11:10 EDT

## Setup

- CPU: AMD EPYC 7763 64-Core Processor (256 logical CPUs available, 1 thread(s) used).
- Training device: cuda:0 (NVIDIA A100-SXM4-80GB MIG 7g.80gb, 79.25 GiB).
- Data: 56461 train / 12087 val / 12102 test; input length 672, horizon 16.
- Batch size: training 256, inference 64. Latency = 200 timed calls after 20 warmup calls, including NumPy preprocessing (and H2D/D2H for CUDA, with synchronization).
- Versions: Python 3.8.20, PyTorch 1.12.1+cu113, scikit-learn 1.3.2.

## Training efficiency

| Model | Total fit (s) | Epochs | s/epoch | Samples/s | Peak GPU (MiB) | RSS growth (MiB) | Params | Size (MiB) | GFLOPs/sample (counter / analytic) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| dlinear | 17.53 | 10 | 1.413 | 39955 | 52.2 | 2383.5 | 21,542 | 0.0840 | n/a / 0.000060 |
| gru | 66.82 | 15 | 4.044 | 13961 | 5763.9 | 2850.0 | 17,409 | 0.0686 | n/a / 0.016916 |
| ridge | 27.61 | n/a | n/a | 2045 | n/a | 167.6 | 12,048 | 0.0643 | n/a / 0.000024 |
| persistence | 0.00 | n/a | n/a | n/a | n/a | n/a | 0 | 0.0000 | n/a / 0.000000 |
| seasonal_naive | 0.00 | n/a | n/a | n/a | n/a | n/a | 0 | 0.0000 | n/a / 0.000000 |

Neural fit time includes per-epoch validation and early stopping (first CUDA epoch includes warmup). Ridge time is the solver only; feature preparation is in the JSON. RSS growth is the process peak minus RSS after data loading (each model runs in its own process). FLOPs count 2 per multiply-add; the counter column is n/a where torch's FlopCounterMode cannot cover the model, and the analytic column ignores activations.

## Inference

| Model / runtime | Single p50 (ms) | Single p90 (ms) | Batch | Batch p50 (ms) | Batch p90 (ms) | Samples/s (batch) | RSS (MiB) | GPU peak (+MiB) | Weights (KiB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dlinear/cpu_fp32 | 0.0922 | 0.0978 | 64 | 1.0023 | 1.0071 | 63673 | 3238 | n/a | 86.0 |
| dlinear/cpu_int8 | 0.2125 | 0.2243 | 64 | 1.0243 | 1.1637 | 60523 | 3256 | n/a | 24.9 |
| dlinear/cuda_fp32 | 7.3521 | 7.5098 | 64 | 4.9731 | 5.1222 | 18241 | 3256 | 0.7 | n/a |
| gru/cpu_fp32 | 7.4575 | 7.4878 | 64 | 65.8749 | 66.2273 | 981 | 3770 | n/a | 70.3 |
| gru/cpu_int8 | 6.0501 | 6.6092 | 64 | 42.9447 | 47.3628 | 1439 | 3767 | n/a | 22.4 |
| gru/cuda_fp32 | 7.7969 | 8.0066 | 64 | 5.4628 | 5.6158 | 13117 | 3777 | 187.0 | n/a |
| ridge/cpu_fp32 | 0.1312 | 0.1401 | 64 | 0.3023 | 0.3167 | 210261 | 849 | n/a | n/a |
| persistence/cpu_numpy | 0.0025 | 0.0026 | 64 | 0.0064 | 0.0066 | 9594634 | 648 | n/a | n/a |
| seasonal_naive/cpu_numpy | 0.0014 | 0.0015 | 64 | 0.0035 | 0.0038 | 17693489 | 650 | n/a | n/a |

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

- dlinear cpu_int8: measured
- dlinear onnx_cpu: failed/skipped: onnx / onnxruntime not installed
- gru cpu_int8: measured
- gru onnx_cpu: failed/skipped: onnx / onnxruntime not installed
- Distillation: not attempted. Energy/cost: not measured (no power telemetry).

Numbers are specific to this machine and software stack; rerun on the deployment target.
