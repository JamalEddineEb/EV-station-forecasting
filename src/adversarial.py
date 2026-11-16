"""Adversarial evaluation methods for time-series EV occupancy forecasting models.

Implements standard white-box and black-box threat models adapted for multi-horizon
quantile time-series neural networks with physical station box constraints [0, C_max]:
1. FGSMAttack: 1-step Fast Gradient Sign Method (Goodfellow et al., 2014).
2. PGDAttack: Multi-step Projected Gradient Descent with random start (Madry et al., 2018).
3. RandomPerturbation: Uniform random bounded noise baseline (black-box reference).
4. evaluate_adversarial_robustness: Sweeps perturbation budgets epsilon in [0, E_max].
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn


class AdversarialAttack:
    """Base class for adversarial time-series perturbations with capacity constraints."""

    def __init__(self, max_capacity: float = 54.0) -> None:
        """Initialize base attack.

        Args:
            max_capacity: Physical maximum station capacity (charging ports).
        """
        self.max_capacity = float(max_capacity)

    def generate(
        self,
        model: nn.Module,
        h: torch.Tensor,
        c: torch.Tensor,
        y_norm: torch.Tensor,
        scale: np.ndarray,
        epsilon_cars: float,
    ) -> torch.Tensor:
        """Generate adversarial input perturbation.

        Args:
            model: PyTorch forecasting model in eval mode.
            h: Normalized input tensor of shape (batch, seq_len, 1).
            c: Calendar feature tensor of shape (batch, horizon, cal_dim).
            y_norm: Normalized target tensor of shape (batch, horizon).
            scale: NumPy scale array of shape (batch, 1).
            epsilon_cars: Perturbation budget in physical EV counts (e.g. 1.0 car).

        Returns:
            Perturbed normalized history tensor of shape (batch, seq_len, 1).
        """
        raise NotImplementedError


class FGSMAttack(AdversarialAttack):
    """Fast Gradient Sign Method (FGSM) for differentiable time-series forecasters.

    Computes:
        x_adv = clip(x + eps * sign(grad_x Loss(f(x), y)), 0, C_max)
    """

    def __init__(self, loss_fn: nn.Module, max_capacity: float = 54.0) -> None:
        """Initialize FGSM attack.

        Args:
            loss_fn: Differentiable loss function (e.g. PinballLoss).
            max_capacity: Physical maximum station capacity.
        """
        super().__init__(max_capacity=max_capacity)
        self.loss_fn = loss_fn

    def generate(
        self,
        model: nn.Module,
        h: torch.Tensor,
        c: torch.Tensor,
        y_norm: torch.Tensor,
        scale: np.ndarray,
        epsilon_cars: float,
    ) -> torch.Tensor:
        """Generate 1-step analytical gradient-directed perturbation."""
        if epsilon_cars <= 0.0:
            return h.clone().detach()

        s_t = torch.from_numpy(scale).unsqueeze(-1).to(h.device).float()
        eps_norm = epsilon_cars / s_t

        h_in = h.clone().detach().requires_grad_(True)
        pred = model(h_in, c)
        loss = self.loss_fn(pred, y_norm)
        loss.backward()

        with torch.no_grad():
            grad_sign = h_in.grad.sign()
            h_adv = h + eps_norm * grad_sign
            # Project onto physical bounds [0, max_capacity]
            h_adv = torch.clamp(h_adv, min=0.0)
            h_adv = torch.minimum(h_adv, self.max_capacity / s_t)

        return h_adv.detach()


class PGDAttack(AdversarialAttack):
    """Projected Gradient Descent (PGD) multi-step L_inf attack.

    Iteratively steps in the direction of the loss gradient sign:
        x_{t+1} = Proj_{x + S}( x_t + alpha * sign(grad_x Loss) )
    where S is the L_inf ball of radius epsilon and physical box [0, C_max].
    """

    def __init__(
        self,
        loss_fn: nn.Module,
        steps: int = 7,
        alpha: Optional[float] = None,
        random_start: bool = True,
        max_capacity: float = 54.0,
    ) -> None:
        """Initialize PGD attack parameters.

        Args:
            loss_fn: Differentiable loss function.
            steps: Number of ascent steps (default 7).
            alpha: Step size. If None, defaults to epsilon / 4.
            random_start: If True, initialize with random point in L_inf ball.
            max_capacity: Physical maximum station capacity.
        """
        super().__init__(max_capacity=max_capacity)
        self.loss_fn = loss_fn
        self.steps = int(steps)
        self.alpha = alpha
        self.random_start = bool(random_start)

    def generate(
        self,
        model: nn.Module,
        h: torch.Tensor,
        c: torch.Tensor,
        y_norm: torch.Tensor,
        scale: np.ndarray,
        epsilon_cars: float,
    ) -> torch.Tensor:
        """Generate multi-step PGD adversarial perturbation."""
        if epsilon_cars <= 0.0:
            return h.clone().detach()

        s_t = torch.from_numpy(scale).unsqueeze(-1).to(h.device).float()
        eps_norm = epsilon_cars / s_t
        step_alpha = (eps_norm / 4.0) if self.alpha is None else (self.alpha / s_t)

        # Initialize: random start within L_inf ball or exact original point
        if self.random_start:
            delta = (torch.rand_like(h) * 2.0 - 1.0) * eps_norm
            h_adv = torch.clamp(h + delta, min=0.0)
            h_adv = torch.minimum(h_adv, self.max_capacity / s_t)
        else:
            h_adv = h.clone().detach()

        for _ in range(self.steps):
            h_adv = h_adv.clone().detach().requires_grad_(True)
            pred = model(h_adv, c)
            loss = self.loss_fn(pred, y_norm)
            loss.backward()

            with torch.no_grad():
                grad_sign = h_adv.grad.sign()
                h_adv = h_adv + step_alpha * grad_sign
                # 1. Project back onto L_inf epsilon ball around original h
                h_adv = torch.max(torch.min(h_adv, h + eps_norm), h - eps_norm)
                # 2. Project onto physical station boundaries [0, max_capacity]
                h_adv = torch.clamp(h_adv, min=0.0)
                h_adv = torch.minimum(h_adv, self.max_capacity / s_t)

        return h_adv.detach()


class RandomPerturbation(AdversarialAttack):
    """Uniform random bounded noise baseline (black-box reference)."""

    def generate(
        self,
        model: nn.Module,
        h: torch.Tensor,
        c: torch.Tensor,
        y_norm: torch.Tensor,
        scale: np.ndarray,
        epsilon_cars: float,
    ) -> torch.Tensor:
        """Generate uniform random noise within [-eps, +eps] clipped to physical bounds."""
        if epsilon_cars <= 0.0:
            return h.clone().detach()

        s_t = torch.from_numpy(scale).unsqueeze(-1).to(h.device).float()
        eps_norm = epsilon_cars / s_t
        delta = (torch.rand_like(h) * 2.0 - 1.0) * eps_norm
        h_rand = torch.clamp(h + delta, min=0.0)
        h_rand = torch.minimum(h_rand, self.max_capacity / s_t)
        return h_rand.detach()


def evaluate_adversarial_robustness(
    model: nn.Module,
    h: torch.Tensor,
    c: torch.Tensor,
    y: np.ndarray,
    scale: np.ndarray,
    attacks_dict: Dict[str, AdversarialAttack],
    epsilons: List[float],
    tolerance_cars: Tuple[float, ...] = (1.0, 2.0),
) -> List[Dict[str, Any]]:
    """Evaluates robust metrics across multiple perturbation budgets.

    Args:
        model: PyTorch model to evaluate.
        h: Normalized history tensor.
        c: Calendar feature tensor.
        y: Ground truth unnormalized occupancy array.
        scale: Per-sample scale array.
        attacks_dict: Dict mapping attack name to AdversarialAttack instance.
        epsilons: List of perturbation budgets in physical EV cars.
        tolerance_cars: Tuple of physical car thresholds for tolerance accuracy.

    Returns:
        List of result dictionaries containing MAE, RMSE, Acc@tau, Coverage.
    """
    model.eval()
    y_norm = torch.from_numpy((y / scale).astype(np.float32)).to(h.device)
    results: List[Dict[str, Any]] = []

    # Baseline clean evaluation (epsilon = 0.0)
    with torch.no_grad():
        clean_out = model(h, c).cpu().numpy() * scale[:, :, None]
        clean_point = clean_out[..., 1]
        err = np.abs(y - clean_point)

        row: Dict[str, Any] = {
            "attack": "Clean",
            "epsilon": 0.0,
            "MAE": float(np.mean(err)),
            "RMSE": float(np.sqrt(np.mean((y - clean_point) ** 2))),
            "Coverage_80": float(np.mean((y >= clean_out[..., 0]) & (y <= clean_out[..., 2]))),
        }
        for tau in tolerance_cars:
            row[f"Acc_within_{int(tau)}_car"] = float(np.mean(err <= tau) * 100.0)
        results.append(row)

    # Perturbation sweeps
    for eps in epsilons:
        if eps <= 0.0:
            continue
        for attack_name, attack in attacks_dict.items():
            h_adv = attack.generate(model, h, c, y_norm, scale, eps)
            with torch.no_grad():
                adv_out = model(h_adv, c).cpu().numpy() * scale[:, :, None]
                adv_point = adv_out[..., 1]
                err = np.abs(y - adv_point)

                row = {
                    "attack": attack_name,
                    "epsilon": float(eps),
                    "MAE": float(np.mean(err)),
                    "RMSE": float(np.sqrt(np.mean((y - adv_point) ** 2))),
                    "Coverage_80": float(np.mean((y >= adv_out[..., 0]) & (y <= adv_out[..., 2]))),
                }
                for tau in tolerance_cars:
                    row[f"Acc_within_{int(tau)}_car"] = float(np.mean(err <= tau) * 100.0)
                results.append(row)

    return results
