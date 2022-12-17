""" Implements PGD methods under different budget estimations."""
from typing import Callable, Optional

import torch
from torch import nn

from .attack import Attack


class ProjectedGradientDecentAttack(Attack):
    """Projected gradient descent attack (Madry et al, 2017).
    Performs nb_iter steps * eps_iter step size.
    Budget checked at every iteration.
    Paper: https://arxiv.org/pdf/1706.06083.pdf
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        eps: float,
        nb_iter: int,
        budget_fn: Callable,
        eps_iter: Optional[float] = None,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
        rand_init=True,
    ) -> None:
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.eps = eps
        self.nb_iter = nb_iter
        self.budget_fn = budget_fn  # Make sure attack within budget.
        self.eps_iter = eps_iter if eps_iter else (eps / nb_iter)
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.rand_init = rand_init

    @torch.enable_grad()
    def __call__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        x_adv = x.detach()
        if self.rand_init:
            x_adv += torch.zeros_like(x_adv).uniform_(-self.eps, self.eps)
        for _ in range(self.nb_iter):
            x_adv.requires_grad_(True)
            loss = self.loss_fn(self.model(x_adv), y)
            grad = torch.autograd.grad(loss, [x_adv])[0]
            grad = self.eps_iter * torch.sign(grad.detach())
            x_adv = x_adv.detach() + grad
            x_adv = (x_adv.max(x - self.eps)).min(x + self.eps)
            x_adv = self.budget_fn(x, x_adv, self.eps)
            x_adv = torch.clamp(x_adv, self.clip_min, self.clip_max)
        return x_adv


class LinfPGD(ProjectedGradientDecentAttack):
    """Projected gradient descent attack with L-inf norm budget."""

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
        super().__init__(
            model,
            loss_fn,
            eps,
            nb_iter,
            lambda x, x_adv, eps: x_adv.max(x - eps).min(x + eps),
            eps_iter,
            clip_min,
            clip_max,
            rand_init,
        )


class L2PGD(ProjectedGradientDecentAttack):
    """Projected gradient descent attack with L2 norm budget."""

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
        super().__init__(
            model,
            loss_fn,
            eps,
            nb_iter,
            lambda x, x_adv, eps: x
            + (x_adv - x) * (torch.max(1, eps / torch.square(x_adv - x).sum())),
            eps_iter,
            clip_min,
            clip_max,
            rand_init,
        )


class L1PGD(ProjectedGradientDecentAttack):
    """Projected gradient descent attack with L1 norm budget."""

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
        super().__init__(
            model,
            loss_fn,
            eps,
            nb_iter,
            lambda x, x_adv, eps: x
            + (x_adv - x) * (torch.max(1, eps / torch.abs(x_adv - x).sum())),
            eps_iter,
            clip_min,
            clip_max,
            rand_init,
        )
