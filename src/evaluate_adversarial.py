"""Adversarial robustness evaluation across Caltech and JPL forecasting models.

Evaluates:
- Plausible adversaries: White-box (FGSM, PGD-7), Grey-box / Black-box (Random bounded noise).
- Multiple perturbation strengths: eps = 0.5, 1.0, 2.0, 3.0, 5.0 cars.
- Metrics: Robust MAE, Robust RMSE, Robust Accuracy within 1 car and 2 cars, and 80% Coverage.

Usage:
    python src/evaluate_adversarial.py --site all
    python src/evaluate_adversarial.py --site Caltech
    python src/evaluate_adversarial.py --site JPL
"""
import argparse
import os
import numpy as np
import pandas as pd
import torch

from adversarial import (
    FGSMAttack,
    PGDAttack,
    RandomPerturbation,
    evaluate_adversarial_robustness,
)
from models.train_neural import (
    DLinear,
    PinballLoss,
    QUANTILES,
    make_tensors,
    window_scale,
)

CONFIGS = {
    "Caltech": {
        "data_path": "outputs/caltech_occupancy_data.npz",
        "checkpoint": "outputs/dlinear_caltech.pt",
        "max_capacity": 54.0,
        "tolerance": (1.0, 2.0),
    },
    "JPL": {
        "data_path": "outputs/jpl_occupancy_data.npz",
        "checkpoint": "outputs/dlinear_jpl.pt",
        "max_capacity": 56.0,
        "tolerance": (2.0, 4.0),
    },
}


def run_adversarial_benchmark(site_name, cfg, n_samples=300):
    print("\n" + "=" * 85)
    print(f"ADVERSARIAL EVALUATION FOR SITE: {site_name}")
    print("=" * 85)

    data = np.load(cfg["data_path"])
    sh = data["test_X_hist"][:n_samples]
    sc = data["test_X_cal"][:n_samples]
    sy = data["test_y"][:n_samples]

    L, H, C = sh.shape[1], sy.shape[1], sc.shape[2]
    device = torch.device("cpu")

    model = DLinear(L, H, C)
    if not os.path.exists(cfg["checkpoint"]):
        raise FileNotFoundError(f"Missing checkpoint: {cfg['checkpoint']}")
    model.load_state_dict(torch.load(cfg["checkpoint"], map_location=device))
    model.eval()

    scale = window_scale(sh)
    h_tensor, c_tensor = make_tensors(sh, sc, None)
    loss_fn = PinballLoss(QUANTILES)

    attacks = {
        "Random Noise (Black-box)": RandomPerturbation(max_capacity=cfg["max_capacity"]),
        "FGSM (White-box 1-step)": FGSMAttack(loss_fn, max_capacity=cfg["max_capacity"]),
        "PGD-7 (White-box Multi-step)": PGDAttack(loss_fn, steps=7, max_capacity=cfg["max_capacity"]),
    }

    epsilons = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]
    t1, t2 = cfg["tolerance"]

    results = evaluate_adversarial_robustness(
        model, h_tensor, c_tensor, sy, scale, attacks, epsilons, tolerance_cars=(t1, t2)
    )

    df = pd.DataFrame(results)

    # Format and display
    print(f"\n--- {site_name} Adversarial Robustness Benchmark (N={n_samples} test sequences) ---")
    print(f"Physical Capacity: {cfg['max_capacity']} cars | Tolerance thresholds: <= {t1:.0f} cars, <= {t2:.0f} cars\n")

    cols = ["attack", "epsilon", "MAE", "RMSE", f"Acc_within_{int(t1)}_car", f"Acc_within_{int(t2)}_car", "Coverage_80"]
    rename_map = {
        "attack": "Adversary / Attack",
        "epsilon": "Budget eps (cars)",
        "MAE": "Robust MAE",
        "RMSE": "Robust RMSE",
        f"Acc_within_{int(t1)}_car": f"Acc (<= {int(t1)} car)",
        f"Acc_within_{int(t2)}_car": f"Acc (<= {int(t2)} cars)",
        "Coverage_80": "80% Coverage",
    }
    table = df[cols].rename(columns=rename_map)

    # Format table without requiring external tabulate dependency
    print(table.to_string(index=False))

    return table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", choices=["Caltech", "JPL", "all"], default="all")
    parser.add_argument("--n-samples", type=int, default=300)
    args = parser.parse_args()

    sites = ["Caltech", "JPL"] if args.site == "all" else [args.site]
    for s in sites:
        run_adversarial_benchmark(s, CONFIGS[s], n_samples=args.n_samples)


if __name__ == "__main__":
    main()
