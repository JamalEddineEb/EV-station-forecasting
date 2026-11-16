"""Train robust DLinear forecasting models with time-series jittering augmentation.

Supports training across both Caltech (C_max = 54) and JPL (C_max = 56) EV charging hubs.

Usage:
    python src/train_robust.py --site Caltech
    python src/train_robust.py --site JPL
    python src/train_robust.py --site all
"""

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# Ensure src is on sys.path for direct script execution
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from models.train_neural import DLinear, fit, make_tensors


def train_site(
    site: str,
    data_dir: str = "outputs",
    out_dir: str = "outputs",
    epochs: int = 25,
    patience: int = 5,
    lr: float = 1e-3,
    batch_size: int = 256,
    jitter_sigma: float = 0.03,
    seed: int = 42,
) -> str:
    """Train robust DLinear model for a specific EV charging hub using jittering.

    Args:
        site: Station name ('Caltech' or 'JPL').
        data_dir: Directory containing preprocessed .npz files.
        out_dir: Directory to save the trained model checkpoint.
        epochs: Maximum training epochs.
        patience: Early stopping patience epochs.
        lr: Adam optimizer learning rate.
        batch_size: Mini-batch size.
        jitter_sigma: Standard deviation of Gaussian noise augmentation.
        seed: Random seed for reproducibility.

    Returns:
        Path to saved checkpoint file.
    """
    site_key = site.lower()
    data_path = os.path.join(data_dir, f"{site_key}_occupancy_data.npz")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset for {site} not found at '{data_path}'.")

    out_ckpt = os.path.join(out_dir, f"dlinear_{site_key}_robust.pt")
    print("\n" + "=" * 80)
    print(f" TRAINING ROBUST DLINEAR FOR {site.upper()} (Jittering sigma = {jitter_sigma})")
    print("=" * 80)
    print(f"Data source:       {data_path}")
    print(f"Target checkpoint: {out_ckpt}")

    # Set deterministic seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Compute device:    {device}")

    # Load preprocessed station arrays
    data = np.load(data_path, allow_pickle=False)
    th, tc, ty = data["train_X_hist"], data["train_X_cal"], data["train_y"]
    vh, vc, vy = data["val_X_hist"], data["val_X_cal"], data["val_y"]

    seq_len = th.shape[1]
    horizon = ty.shape[1]
    cal_dim = tc.shape[2]
    print(f"Dimensions:        History={seq_len}, Horizon={horizon}, Calendar Dim={cal_dim}")
    print(f"Dataset sizes:     Train={len(th):,}, Val={len(vh):,}")

    model = DLinear(seq_len, horizon, cal_dim)
    params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters:  {params:,}")

    tr = make_tensors(th, tc, ty)
    va = make_tensors(vh, vc, vy)
    loader = DataLoader(TensorDataset(*tr), batch_size=batch_size, shuffle=True)

    print(f"Beginning training (max {epochs} epochs, patience {patience})...")
    m_rob, history = fit(
        model,
        loader,
        va,
        device,
        epochs=epochs,
        patience=patience,
        lr=lr,
        name=f"DLinear_{site}_Robust",
        jitter_sigma=jitter_sigma,
    )

    os.makedirs(out_dir, exist_ok=True)
    torch.save(m_rob.state_dict(), out_ckpt)
    best_val = min(r["val_loss"] for r in history)
    epochs_run = len(history)
    print(f"--> [SUCCESS] Saved robust checkpoint to '{out_ckpt}'")
    print(f"--> Best validation pinball loss: {best_val:.6f} (Epochs run: {epochs_run})")
    return out_ckpt


def main() -> None:
    """CLI entry point for robust training."""
    parser = argparse.ArgumentParser(
        description="Train robust DLinear models with time-series jittering augmentation."
    )
    parser.add_argument(
        "--site",
        choices=["Caltech", "JPL", "all", "caltech", "jpl"],
        default="all",
        help="Target station hub to train ('Caltech', 'JPL', or 'all')",
    )
    parser.add_argument(
        "--data-dir", default="outputs", help="Directory containing preprocessed data .npz files"
    )
    parser.add_argument(
        "--outdir", default="outputs", help="Directory where trained checkpoints will be saved"
    )
    parser.add_argument(
        "--jitter-sigma",
        type=float,
        default=0.03,
        help="Additive Gaussian noise std for normalized history",
    )
    parser.add_argument("--epochs", type=int, default=25, help="Maximum training epochs")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (Adam)")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    sites = ["Caltech", "JPL"] if args.site.lower() == "all" else [args.site.capitalize()]
    for s in sites:
        train_site(
            site=s,
            data_dir=args.data_dir,
            out_dir=args.outdir,
            epochs=args.epochs,
            patience=args.patience,
            lr=args.lr,
            batch_size=args.batch_size,
            jitter_sigma=args.jitter_sigma,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
