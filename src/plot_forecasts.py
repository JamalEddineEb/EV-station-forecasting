
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="Saved forecast .npz")
    parser.add_argument("--sample", type=int, default=100)
    parser.add_argument("--outdir", default="outputs/forecast_plots")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    with np.load(args.file, allow_pickle=False) as data:
        actual = data["actual"]
        pred = data["prediction"]
        lower = data["lower"]
        upper = data["upper"]
        steps = data["horizon_steps"]
        timestamps = data["forecast_timestamps"]
        site = str(data["site_name"].item())
        origin_times = data["origin_timestamps"]

    n = len(actual)
    if n == 0:
        raise ValueError("The forecast file contains no test samples.")

    i = min(max(args.sample, 0), n - 1)
    model = os.path.basename(args.file).replace(".npz", "")

    # Plot 1: one forecast origin, all 16 future steps.
    if timestamps.shape == actual.shape:
        x = timestamps[i].astype("datetime64[s]")
    else:
        x = steps

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(x, actual[i], marker="o", label="Actual")
    ax.plot(x, pred[i], marker="o", label="Forecast")
    ax.fill_between(
        x, lower[i], upper[i],
        alpha=0.2, label="Calibrated 80% interval"
    )

    origin = str(origin_times[i]) if len(origin_times) == len(actual) else "unknown"

    ax.set_title(
        f"{model} | Site: {site}\n"
        f"Forecast issued: {origin} | Sample index: {i}"
    )
    ax.set_xlabel("Forecast time (or horizon step)")
    ax.set_ylabel("EV charging occupancy (cars)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()

    path = os.path.join(args.outdir, f"{model}_sample_{i}.png")
    fig.savefig(path, dpi=150)
    plt.show()
    plt.close(fig)

    print("Saved:", path)



if __name__ == "__main__":
    main()
