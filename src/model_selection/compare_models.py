#!/usr/bin/env python3
import os

import argparse
import numpy as np
import pandas as pd

from .baselines import make_baseline_predictions, save_baseline_predictions
from .conformal import conformalize

MODELS = [
    ("ridge", "Ridge"),
    ("dlinear", "DLinear"),
    ("gru", "GRU"),
    ("persistence", "Persistence"),
    ("seasonal_naive", "Seasonal naive"),
]
PENALTY = 2 / 0.2  # interval-score penalty per unit of miss for an 80% interval


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
    save_baseline_predictions(
        make_baseline_predictions(val_history, test_history, horizon), y_val, y_test, args.indir
    )

    rows = []
    for key, name in MODELS:
        path = os.path.join(args.indir, key + "_predictions.npz")
        if not os.path.exists(path):
            print("Missing {}, skipping".format(path))
            continue
        with np.load(path, allow_pickle=False) as data:
            native = "test_lower" in data.files  # model predicts its own interval
            point = data["test_pred"]
            # point forecasters use lower = upper = forecast
            bounds = {
                s: (data[s + "_lower"], data[s + "_upper"]) if native else (data[s + "_pred"],) * 2
                for s in ("val", "test")
            }
        lower, upper = conformalize(y_val, *bounds["val"], *bounds["test"])
        np.savez_compressed(
            os.path.join(args.indir, key + "_intervals.npz"),
            lower=lower.astype(np.float32),
            upper=upper.astype(np.float32),
        )

        error = y_test - point
        for h in range(horizon):
            y, lo, hi = y_test[:, h], lower[:, h], upper[:, h]
            score = (hi - lo) + PENALTY * np.maximum(lo - y, 0) + PENALTY * np.maximum(y - hi, 0)
            raw = (
                float(np.mean((y >= bounds["test"][0][:, h]) & (y <= bounds["test"][1][:, h])))
                if native else np.nan
            )
            rows.append({
                "model": name,
                "horizon_minutes": 15 * (h + 1),
                "MAE": float(np.mean(np.abs(error[:, h]))),
                "RMSE": float(np.sqrt(np.mean(error[:, h] ** 2))),
                "coverage_80": float(np.mean((y >= lo) & (y <= hi))),
                "mean_interval_width": float(np.mean(hi - lo)),
                "interval_score": float(np.mean(score)),
                "raw_coverage_80": raw,
            })

    results = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    results.to_csv(args.out, index=False)
    print("Saved:", args.out)
    print(
        results[results.horizon_minutes.isin([60, 240])]
        .sort_values(["horizon_minutes", "interval_score"])
        .round(3)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()