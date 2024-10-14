import numpy as np

from abc import abstractmethod
from typing import Protocol
from numpy.typing import ArrayLike, NDArray

class FeasibilityEstimator(Protocol):
    @abstractmethod
    def prob(self, x: ArrayLike) -> ArrayLike:
        raise NotImplementedError

    def update(self, x: ArrayLike, y: ArrayLike):
        pass

    @property
    def is_differentiable(self) -> bool:
        return False

    def grad_prob(self, x: ArrayLike):
        raise NotImplementedError

class OracleFeasibilityEstimator(FeasibilityEstimator):
    def __init__(self, func, grad = None):
        self.func = func
        self.grad = grad

    def prob(self, x):
        return np.float_(self.func(x))

    @property
    def is_differentiable(self):
        return self.grad is not None

    def grad_prob(self, x):
        if self.grad is not None:
            return self.grad(x)
        raise NotImplementedError


__all__ = ["FeasibilityEstimator", "OracleFeasibilityEstimator"]
