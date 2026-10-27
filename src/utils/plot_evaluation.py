
import argparse
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv", nargs="+", required=True,
        help="CSV files containing epoch, train_pinball, val_pinball"
    )
    parser.add_argument(
        "--labels", nargs="+",
        help="Labels corresponding to the CSV files"
    )
    parser.add_argument(
        "--out", default="outputs/learning_curves.png"
    )
    args = parser.parse_args()

    if args.labels and len(args.labels) != len(args.csv):
        parser.error("--labels must have one label per CSV file")

    labels = args.labels or [
        Path(f).stem.replace("_learning_curves", "").replace("_", " ").title()
        for f in args.csv
    ]

    fig, ax = plt.subplots(figsize=(8, 5))

    for csv_file, label in zip(args.csv, labels):
        df = pd.read_csv(csv_file)
        required = {"epoch", "train_pinball", "val_pinball"}
        if not required.issubset(df.columns):
            raise ValueError(
                f"{csv_file} must contain columns: {sorted(required)}"
            )

        ax.plot(
            df["epoch"], df["train_pinball"],
            marker="o", markersize=3, linewidth=1.8,
            label=f"{label} — train"
        )
        ax.plot(
            df["epoch"], df["val_pinball"],
            marker="o", markersize=3, linewidth=1.8,
            linestyle="--", label=f"{label} — validation"
        )

        best = df.loc[df["val_pinball"].idxmin()]
        ax.axvline(
            best["epoch"], color="gray", linestyle=":", alpha=0.5
        )
        ax.scatter(
            [best["epoch"]], [best["val_pinball"]],
            marker="*", s=100, zorder=5
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Pinball loss (normalized)")
    ax.set_title("Training and Validation Learning Curves")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {out}")


if __name__ == "__main__":
    main()
