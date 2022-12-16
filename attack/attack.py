"""Define base class of attack methods."""
from abc import ABC, abstractmethod
from typing import Any


class Attack(ABC):
    """Abstract class for all attack methods."""

    @abstractmethod
    def __call__(self, x: Any, y: Any) -> Any:
        pass


class FasterRCNNAttackWithBound(Attack):
    """Abstract class for all attacks restricted in bound."""

    @abstractmethod
    def __call__(self, x: Any, y: Any, bound: Any) -> Any:
        pass
