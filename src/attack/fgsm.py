""" Implements FGSM methods under L-inf budget estimation."""

from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch import nn

from .pgd import LinfPGD


class iFGSM(LinfPGD):
    """Iterative Fast gradient signed method implementation within L-inf norm.

    FGSM in the paper: `Explaining and harnessing adversarial examples`
    https://arxiv.org/abs/1412.6572
    """

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


class FGSM(iFGSM):
    """Fast gradient signed method implementation within L-inf norm.

    FGSM in the paper: `Explaining and harnessing adversarial examples`
    https://arxiv.org/abs/1412.6572
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        eps: float,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
    ) -> None:
        super().__init__(
            model,
            loss_fn,
            eps,
            nb_iter=1,
            clip_min=clip_min,
            clip_max=clip_max,
            rand_init=False,
        )
