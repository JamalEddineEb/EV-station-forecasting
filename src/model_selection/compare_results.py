#!/usr/bin/env python3
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
import argparse
import numpy as np
import pandas as pd

from .baselines import make_baseline_predictions, save_baseline_predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="outputs/caltech_occupancy_data.npz")
    parser.add_argument("--indir", default="outputs")
    parser.add_argument("--out", default="outputs/model_comparison.csv")
    args = parser.parse_args()
    with np.load(args.data, allow_pickle=False) as data:
        val_history, y_val = data["val_X_hist"], data["val_y"]
        test_history, y_test = data["test_X_hist"], data["test_y"]

    horizon = y_test.shape[1]
    baseline_preds = make_baseline_predictions(
        val_history, test_history, horizon
    )
    save_baseline_predictions(baseline_preds, y_val, y_test, args.indir)

    predictions = {}
    for model_key, model_name in [
        ("ridge", "Ridge"),
        ("dlinear", "DLinear"),
        ("gru", "GRU"),
    ]:
        path = os.path.join(args.indir, model_key + "_predictions.npz")
        if os.path.exists(path):
            with np.load(path, allow_pickle=False) as data:
                predictions[model_name] = (data["val_pred"], data["test_pred"])
        else:
            print("Missing {}, skipping".format(path))
    predictions["Persistence"] = baseline_preds["persistence"]
    predictions["Seasonal naive"] = baseline_preds["seasonal_naive"]

    rows = []
    for name, (val_pred, test_pred) in predictions.items():
        if name in ("Persistence", "Seasonal naive"):
            lower = upper = None
        else:
            residuals = np.abs(y_val - val_pred)
            sample_count = len(residuals)
            level = min(1.0, np.ceil((sample_count + 1) * 0.8) / sample_count)
            quantile = np.quantile(residuals, level, axis=0)
            lower, upper = test_pred - quantile, test_pred + quantile
            interval_path = os.path.join(
                args.indir, name.lower() + "_intervals.npz"
            )
            np.savez_compressed(
                interval_path,
                lower=lower.astype(np.float32),
                upper=upper.astype(np.float32),
            )

        error = y_test - test_pred
        for h in range(horizon):
            row = {
                "model": name,
                "horizon_minutes": 15 * (h + 1),
                "MAE": float(np.mean(np.abs(error[:, h]))),
                "RMSE": float(np.sqrt(np.mean(error[:, h] ** 2))),
            }
            if lower is None:
                row.update(coverage_80=np.nan, mean_interval_width=np.nan)
            else:
                row["coverage_80"] = float(
                    np.mean((y_test[:, h] >= lower[:, h]) & (y_test[:, h] <= upper[:, h]))
                )
                row["mean_interval_width"] = float(
                    np.mean(upper[:, h] - lower[:, h])
                )
            rows.append(row)

    results = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    results.to_csv(args.out, index=False)
    print("Saved:", args.out)
    print(
        results[results.horizon_minutes.isin([60, 240])]
        .sort_values(["horizon_minutes", "MAE"])
        .round(3)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
