# EV Station Occupancy Forecasting

`data_audit.ipynb` cleans Caltech and JPL charging-session data, creates occupancy plots and features, and prepares training, validation, and test datasets.

## Dependencies

Use Python 3.11. Install the required packages:

```bash
python -m pip install "numpy>=1.20,<2" "pandas>=1.5,<2" matplotlib holidays jupyterlab ipykernel
```

## How to run

1. Download or clone this repository and open a terminal in its root folder.
2. Start JupyterLab:

   ```bash
   python -m jupyterlab
   ```

3. Open `data_audit.ipynb` and select the Python environment with the dependencies installed.
4. Choose **Restart Kernel and Run All Cells**.

The notebook runs from the raw JSON files included in `acn_dataset/` and uses relative paths. Output folders are created automatically.

## Saved outputs

- `outputs/figures/`: plots.
- `outputs/tables/`: data-quality and summary tables.
- `outputs/caltech_occupancy_data.npz` and `outputs/jpl_occupancy_data.npz`: training, validation, and test datasets.

Saved outputs are included in the repository, for example [the occupancy plot](outputs/figures/occupancy_profiles.png).
