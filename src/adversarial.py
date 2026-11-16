"""Adversarial evaluation methods for time-series EV occupancy forecasting models.

Implements:
1. FGSM (Fast Gradient Sign Method) tailored for multi-horizon quantile time-series models.
2. PGD (Projected Gradient Descent) with L_inf ball projection and physical box constraints [0, C_max].
3. Random uniform baseline for black-box noise comparison.
4. Robust accuracy metrics (MAE, RMSE, Tolerance Accuracy within tau cars, and 80% Coverage).
"""
import numpy as np
import torch
import torch.nn as nn


class AdversarialAttack:
    """Base class for adversarial time-series perturbations."""
    def __init__(self, max_capacity=54.0):
        self.max_capacity = float(max_capacity)

    def generate(self, model, hist_tensor, cal_tensor, y_norm_tensor, scale, epsilon):
        raise NotImplementedError


class FGSMAttack(AdversarialAttack):
    """Fast Gradient Sign Method (FGSM) for differentiable time-series forecasters.

    Computes:
        x_adv = clip(x + eps * sign(grad_x Loss), 0, C_max)
    """
    def __init__(self, loss_fn, max_capacity=54.0):
        super().__init__(max_capacity=max_capacity)
        self.loss_fn = loss_fn

    def generate(self, model, h, c, y_norm, scale, epsilon_cars):
        """
        Args:
            model: PyTorch forecasting model (eval mode).
            h: Normalized input tensor of shape (batch, seq_len, 1).
            c: Calendar tensor of shape (batch, horizon, cal_dim).
            y_norm: Normalized target tensor of shape (batch, horizon).
            scale: NumPy per-window scale factors, shape (batch, 1).
            epsilon_cars: Perturbation budget in physical EV count units (e.g. 1.0 car).

        Returns:
            h_adv: Adversarially perturbed normalized history tensor.
        """
        if epsilon_cars <= 0.0:
            return h.clone().detach()

        s_t = torch.from_numpy(scale).unsqueeze(-1).to(h.device).float()
        eps_norm = (epsilon_cars / s_t)  # convert physical car budget to normalized scale

        h_in = h.clone().detach().requires_grad_(True)
        pred = model(h_in, c)
        loss = self.loss_fn(pred, y_norm)
        loss.backward()

        with torch.no_grad():
            grad_sign = h_in.grad.sign()
            h_adv = h + eps_norm * grad_sign
            # Project onto physical non-negative bound and physical station capacity
            h_adv = torch.clamp(h_adv, min=0.0)
            h_adv = torch.minimum(h_adv, self.max_capacity / s_t)

        return h_adv.detach()


class PGDAttack(AdversarialAttack):
    """Projected Gradient Descent (PGD) multi-step L_inf attack for time-series forecasting.

    Iteratively steps in the direction of the gradient sign, projecting back onto
    the L_inf epsilon-ball around the original sequence and enforcing physical domain bounds [0, C_max].
    """
    def __init__(self, loss_fn, steps=7, alpha=None, random_start=True, max_capacity=54.0):
        super().__init__(max_capacity=max_capacity)
        self.loss_fn = loss_fn
        self.steps = steps
        self.alpha = alpha
        self.random_start = random_start

    def generate(self, model, h, c, y_norm, scale, epsilon_cars):
        """
        Args:
            model: PyTorch forecasting model (eval mode).
            h: Normalized input tensor (batch, seq_len, 1).
            c: Calendar tensor (batch, horizon, cal_dim).
            y_norm: Normalized target tensor (batch, horizon).
            scale: NumPy scale array (batch, 1).
            epsilon_cars: Perturbation budget in physical EV counts.

        Returns:
            h_adv: Perturbed normalized history tensor.
        """
        if epsilon_cars <= 0.0:
            return h.clone().detach()

        s_t = torch.from_numpy(scale).unsqueeze(-1).to(h.device).float()
        eps_norm = (epsilon_cars / s_t)

        step_alpha = (eps_norm / 4.0) if self.alpha is None else (self.alpha / s_t)

        # Initialize: optionally start from a uniform random perturbation within the L_inf ball
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
    def generate(self, model, h, c, y_norm, scale, epsilon_cars):
        if epsilon_cars <= 0.0:
            return h.clone().detach()
        s_t = torch.from_numpy(scale).unsqueeze(-1).to(h.device).float()
        eps_norm = (epsilon_cars / s_t)
        delta = (torch.rand_like(h) * 2.0 - 1.0) * eps_norm
        h_rand = torch.clamp(h + delta, min=0.0)
        h_rand = torch.minimum(h_rand, self.max_capacity / s_t)
        return h_rand.detach()


def evaluate_adversarial_robustness(model, h, c, y, scale, attacks_dict, epsilons, tolerance_cars=(1.0, 2.0)):
    """Evaluates robust metrics across multiple perturbation budgets.

    Returns:
        results: List of dicts with metrics (MAE, RMSE, Acc@tau, Coverage).
    """
    model.eval()
    y_norm = torch.from_numpy((y / scale).astype(np.float32)).to(h.device)

    results = []
    # Baseline clean evaluation (epsilon = 0.0)
    with torch.no_grad():
        clean_out = model(h, c).cpu().numpy() * scale[:, :, None]
        clean_point = clean_out[..., 1]
        clean_lo = clean_out[..., 0]
        clean_hi = clean_out[..., 2]

        err = np.abs(y - clean_point)
        row = {
            "attack": "Clean",
            "epsilon": 0.0,
            "MAE": float(np.mean(err)),
            "RMSE": float(np.sqrt(np.mean((y - clean_point) ** 2))),
            "Coverage_80": float(np.mean((y >= clean_lo) & (y <= clean_hi))),
        }
        for tau in tolerance_cars:
            row[f"Acc_within_{int(tau)}_car"] = float(np.mean(err <= tau) * 100.0)
        results.append(row)

    for eps in epsilons:
        if eps <= 0.0:
            continue
        for attack_name, attack in attacks_dict.items():
            h_adv = attack.generate(model, h, c, y_norm, scale, eps)
            with torch.no_grad():
                adv_out = model(h_adv, c).cpu().numpy() * scale[:, :, None]
                adv_point = adv_out[..., 1]
                adv_lo = adv_out[..., 0]
                adv_hi = adv_out[..., 2]

                err = np.abs(y - adv_point)
                row = {
                    "attack": attack_name,
                    "epsilon": float(eps),
                    "MAE": float(np.mean(err)),
                    "RMSE": float(np.sqrt(np.mean((y - adv_point) ** 2))),
                    "Coverage_80": float(np.mean((y >= adv_lo) & (y <= adv_hi))),
                }
                for tau in tolerance_cars:
                    row[f"Acc_within_{int(tau)}_car"] = float(np.mean(err <= tau) * 100.0)
                results.append(row)

    return results
