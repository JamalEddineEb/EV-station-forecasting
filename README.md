# EV Station Occupancy Forecasting

`data_audit.ipynb` cleans Caltech and JPL charging-session data, creates occupancy plots and features, and prepares training, validation, and test datasets.

## Dependencies

Use Python 3.11. Install the required packages:

```bash
python -m pip install \
  "numpy>=1.20,<2" "pandas>=1.5,<2" matplotlib holidays \
  jupyterlab ipykernel torch scikit-learn threadpoolctl joblib psutil
```

## How to run

1. Download or clone this repository and open a terminal in its root folder.
2. Start JupyterLab:

   ```bash
   python -m jupyterlab
   ```

3. Open `data_audit.ipynb` and select the Python environment with the dependencies installed.
4. Choose **Restart Kernel and Run All Cells** to prepare the occupancy datasets.

The notebook runs from the raw JSON files included in `acn_dataset/` and uses relative paths. Output folders are created automatically.

## Train and compare models

Run the commands below from the repository root after preparing the datasets:

```bash
python src/train_neural.py --data outputs/caltech_occupancy_data.npz
python src/train_ridge.py --data outputs/caltech_occupancy_data.npz
python src/compare_results.py --data outputs/caltech_occupancy_data.npz
```

The neural training command trains both DLinear and GRU. Model code lives in `src/models/`, while baselines and evaluation live in `src/model_selection/`. The comparison step writes persistence and seasonal-naive validation/test forecasts to `outputs/persistence_predictions.npz` and `outputs/seasonal_naive_predictions.npz`, alongside the comparison CSV and intervals for trained models. All prediction archives use the same keys: `val_pred`, `test_pred`, `y_val`, and `y_test`.

## Benchmark training and inference

Use the following command to train DLinear, GRU, and Ridge on the prepared Caltech dataset and generate a hardware-specific report:

```bash
python -m src.benchmark_models --data outputs/caltech_occupancy_data.npz
```

The report and JSON measurements are written to `outputs/benchmark/`. The script measures total fit time, epoch time, peak PyTorch GPU memory, process RSS, parameter count, checkpoint size, estimated forward-pass FLOPs, single-sample latency, batched latency/throughput, test-set accuracy, and CPU dynamic-INT8 quantization impact. It measures the available GPU and CPU inference paths, if CUDA is available. ONNX, distillation, and energy results are identified as not measured unless explicitly implemented. Run the command in the intended Python environment and on the target device; hardware contention and software versions affect results.

## Saved outputs

- `outputs/figures/`: plots.
- `outputs/tables/`: data-quality and summary tables.
- `outputs/caltech_occupancy_data.npz` and `outputs/jpl_occupancy_data.npz`: training, validation, and test datasets.
- `outputs/*_predictions.npz`: trained-model and baseline validation/test forecasts.
- `outputs/model_comparison.csv`: horizon-by-horizon model metrics.
- `outputs/benchmark/benchmark_report.md` and `benchmark_results.json`: hardware-specific performance results.

Saved outputs are included in the repository, for example [the occupancy plot](outputs/figures/occupancy_profiles.png).
