""" Implements attack methods.
Currently wrap advertorch for comparison.
"""
from abc import ABC, abstractmethod
from typing import Any, Optional

import torch
from advertorch import attacks
from torch import nn


class Attack(ABC):
    """Abstract class for all attack methods."""

    @abstractmethod
    def __call__(self, x: Any, y: Any) -> Any:
        pass


class FGSM(Attack):
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
        super().__init__()
        if eps_iter is None:
            eps_iter = eps / nb_iter
        self.model = model
        self.loss_fn = loss_fn
        self.eps = eps
        self.nb_iter = nb_iter
        self.eps_iter = eps_iter
        self.clip_min = clip_min
        self.clip_max = clip_max

    @torch.enable_grad()
    def __call__(self, x, y) -> Any:
        x_adv = x.detach()
        for _ in range(self.nb_iter):
            x_adv.requires_grad_(True)
            loss = self.loss_fn(self.model(x_adv), y)
            grad = torch.autograd.grad(loss, [x_adv])[0]
            x_adv = x_adv.detach() + self.eps_iter * torch.sign(grad.detach())
            x_adv = torch.max(torch.min(x_adv, x + self.eps_iter), x - self.eps_iter)
            x_adv = torch.clamp(x_adv, self.clip_min, self.clip_max)
        return x_adv


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
            rand_init=True,
            clip_min=clip_min,
            clip_max=clip_max,
            targeted=False,
        )

    @torch.enable_grad()
    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.adversary.perturb(x, y)


@DeprecationWarning  # This will raise Err since its not callable.
@torch.enable_grad()
def linf_pgd(model, x, y, loss_fn, epsilon, alpha, num_iter) -> torch.Tensor:
    x_adv = x.detach()
    x_adv += torch.zeros_like(x_adv).uniform_(-epsilon, epsilon)
    for _ in range(num_iter):
        x_adv.requires_grad_(True)
        loss = loss_fn(model(x_adv), y)
        grad = torch.autograd.grad(loss, [x_adv])[0]
        x_adv = x_adv.detach() + alpha * torch.sign(grad.detach())
        x_adv = (x_adv.max(x - epsilon)).min(x + epsilon)
        x_adv = torch.clamp(x_adv, 0, 1)
    return x_adv
