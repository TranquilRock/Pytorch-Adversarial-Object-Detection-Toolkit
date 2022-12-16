""" Implements attack methods.
Currently wrap advertorch for comparison.
"""
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from advertorch import attacks
from torch import nn

from .attack import Attack, FasterRCNNAttackWithBound


class LinfPGD(Attack):
    """Projected Gradient Descent implementation, now migrate to advertorch package."""

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        eps: float,
        nb_iter: int,
        eps_iter: Optional[float] = None,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
        rand_init=True,
    ) -> None:
        super().__init__()
        if eps_iter is None:
            eps_iter = eps / nb_iter
        self.adversary = attacks.LinfPGDAttack(
            model,
            loss_fn=loss_fn,
            eps=eps,
            nb_iter=nb_iter,
            eps_iter=eps_iter,
            rand_init=rand_init,
            clip_min=clip_min,
            clip_max=clip_max,
            targeted=False,
        )

    @torch.enable_grad()
    def __call__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        return self.adversary.perturb(x, y)


class FGSM(LinfPGD):
    """Fast gradient signed method implementation."""

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        eps: float,
        nb_iter: int,
        eps_iter: Optional[float] = None,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
    ) -> None:
        super().__init__(
            model,
            loss_fn,
            eps,
            nb_iter,
            eps_iter,
            clip_min,
            clip_max,
            False,
        )


from tqdm import tqdm


class FasterRCNNLinfPGDWithBound(FasterRCNNAttackWithBound):
    """Projected Gradient Descent implementation, now migrate to advertorch package."""

    def __init__(
        self,
        model: nn.Module,
        eps: float,
        nb_iter: int,
        eps_iter: Optional[float] = None,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
    ) -> None:
        super().__init__()
        self.model = model
        self.eps = eps
        self.nb_iter = nb_iter
        self.eps_iter = eps_iter if eps_iter else (eps / nb_iter)
        self.clip_min = clip_min
        self.clip_max = clip_max

    @torch.enable_grad()
    def __call__(
        self,
        x: torch.Tensor,
        bound: Tuple[int, int, int, int],
    ) -> torch.Tensor:
        x1, y1, x2, y2 = bound

        x_adv = x.detach()
        x_adv[:, :, x1:x2, y1:y2] += torch.zeros_like(x_adv).uniform_(
            -self.eps, self.eps
        )[:, :, x1:x2, y1:y2]

        with torch.no_grad():
            self.model.eval()
            clean_outputs: List[Dict[str, torch.Tensor]] = self.model(x)
            print(clean_outputs)
            self.model.train()

        for _ in tqdm(range(self.nb_iter)):
            x_adv.requires_grad_(True)

            outputs = self.model(x_adv, clean_outputs)
            loss = (
                + outputs["loss_classifier"]
                + outputs["loss_box_reg"]
                + outputs["loss_objectness"]
                + outputs["loss_rpn_box_reg"]
            )

            grad = torch.autograd.grad(loss, [x_adv])[0]
            grad = self.eps_iter * torch.sign(grad.detach())
            x_adv = x_adv.detach()
            x_adv[:, :, x1:x2, y1:y2] += grad[:, :, x1:x2, y1:y2]
            x_adv = (x_adv.max(x - self.eps)).min(x + self.eps)
            x_adv = torch.clamp(x_adv, self.clip_min, self.clip_max)
        self.model.eval()
        return x_adv
