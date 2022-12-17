"""Implement attacks for pytorch object detection models."""
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch import nn

from .attack import ObjectDetectionAttack


class ObjectDetectionPGD(ObjectDetectionAttack):
    """Projected Gradient Descent implementation for object detection.

    :param model: victim model to attack.
    :param eps: attack budget.
    :param nb_iter: number of iterations.
    :param budget_fn: function to restrict perturbation inside budget.
    :param eps_iter: step size each iteration.
    :param clip_min: mininum value per input dimension.
    :param clip_max: maximum value per input dimension.
    """

    def __init__(
        self,
        model: nn.Module,
        eps: float,
        nb_iter: int,
        budget_fn: Callable,
        eps_iter: Optional[float] = None,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
    ) -> None:
        super().__init__()
        self.model = model
        self.eps = eps
        self.nb_iter = nb_iter
        self.budget_fn = budget_fn  # Make sure attack within budget.
        self.eps_iter = eps_iter if eps_iter else (eps / nb_iter)
        self.clip_min = clip_min
        self.clip_max = clip_max

    @torch.enable_grad()
    def __call__(
        self,
        x: torch.Tensor,
        y: Optional[List[Dict[str, torch.Tensor]]] = None,
        target_losses: Optional[List[str]] = None,
        bound: Optional[Tuple[int, int, int, int]] = None,
    ) -> torch.Tensor:
        """Performs attack when object called.

        :param x: source image.
        :param y: desired malicious boxes to generate.
        :param target_losses: desired lossed to calculate gradient.
        :param bound: desired perturbation region.
        """
        if bound:
            print(f"Perform attack inside regione {bound}")
        x1, y1, x2, y2 = bound if bound else (0, 0, x.size(2), x.size(3))

        x_adv = x.detach()
        x_adv[:, :, x1:x2, y1:y2] += torch.zeros_like(x_adv).uniform_(
            -self.eps, self.eps
        )[:, :, x1:x2, y1:y2]

        targeted = None
        if not y:
            print("No desired input detected, performing untargeted attack.")
            with torch.no_grad():
                self.model.eval()
                y: List[Dict[str, torch.Tensor]] = self.model(x)
            targeted = False
        else:
            print("Desired input detected, performing targeted attack.")
            targeted = True

        print("Starts attack...")
        self.model.train()
        for i in range(self.nb_iter):
            print(f"\rStep {i + 1}/{self.nb_iter}", end="")
            x_adv.requires_grad_(True)

            outputs = self.model(x_adv, y)
            if target_losses:
                loss = torch.stack([outputs[name] for name in target_losses]).sum()
            else:
                loss = torch.stack([l for _, l in outputs.items()]).sum()
            grad = torch.autograd.grad(loss, [x_adv])[0]
            grad = self.eps_iter * torch.sign(grad.detach())
            grad = grad * (-1 if targeted else 1)  # Descend if targeted.
            x_adv = x_adv.detach()
            x_adv[:, :, x1:x2, y1:y2] += grad[:, :, x1:x2, y1:y2]
            x_adv = self.budget_fn(x, x_adv, self.eps)
            x_adv = torch.clamp(x_adv, self.clip_min, self.clip_max)
        print("\nSuccessed!")
        self.model.eval()

        return x_adv


class ObjectDetectionLinfPGD(ObjectDetectionPGD):
    """Projected Gradient Descent implementation for object detection with L-inf-norm.

    :param model: victim model to attack.
    :param eps: attack budget.
    :param nb_iter: number of iterations.
    :param eps_iter: step size each iteration.
    :param clip_min: mininum value per input dimension.
    :param clip_max: maximum value per input dimension.
    """

    def __init__(
        self,
        model: nn.Module,
        eps: float,
        nb_iter: int,
        eps_iter: Optional[float] = None,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
    ) -> None:
        super().__init__(
            model,
            eps,
            nb_iter,
            lambda x, x_adv, eps: x_adv.max(x - eps).min(x + eps),
            eps_iter,
            clip_min,
            clip_max,
        )


class ObjectDetectionL1PGD(ObjectDetectionPGD):
    """Projected Gradient Descent implementation for object detection with L1-norm.

    :param model: victim model to attack.
    :param eps: attack budget.
    :param nb_iter: number of iterations.
    :param eps_iter: step size each iteration.
    :param clip_min: mininum value per input dimension.
    :param clip_max: maximum value per input dimension.
    """

    def __init__(
        self,
        model: nn.Module,
        eps: float,
        nb_iter: int,
        eps_iter: Optional[float] = None,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
    ) -> None:
        super().__init__(
            model,
            eps,
            nb_iter,
            lambda x, x_adv, eps: x
            + (x_adv - x) * (torch.max(1, eps / torch.abs(x_adv - x).sum())),
            eps_iter,
            clip_min,
            clip_max,
        )


class ObjectDetectionL2PGD(ObjectDetectionPGD):
    """Projected Gradient Descent implementation for object detection with L2-norm.

    :param model: victim model to attack.
    :param eps: attack budget.
    :param nb_iter: number of iterations.
    :param eps_iter: step size each iteration.
    :param clip_min: mininum value per input dimension.
    :param clip_max: maximum value per input dimension.
    """

    def __init__(
        self,
        model: nn.Module,
        eps: float,
        nb_iter: int,
        eps_iter: Optional[float] = None,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
    ) -> None:
        super().__init__(
            model,
            eps,
            nb_iter,
            lambda x, x_adv, eps: x
            + (x_adv - x) * (torch.max(1, eps / torch.square(x_adv - x).sum())),
            eps_iter,
            clip_min,
            clip_max,
        )
