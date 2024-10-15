import numpy as np

from abc import abstractmethod
from typing import Protocol, Tuple, runtime_checkable
from numpy.typing import ArrayLike

from elfi.utils import safe_div, logp

@runtime_checkable
class FeasibilityEstimator(Protocol):
    @abstractmethod
    def prob(self, x: ArrayLike) -> ArrayLike:
        raise NotImplementedError

    def log_prob(self, x: ArrayLike) -> ArrayLike:
        return logp(self.prob(x))

    def update(self, x: ArrayLike, y: ArrayLike):
        pass

    @property
    def is_differentiable(self) -> bool:
        return False

    def grad_prob(self, x: ArrayLike) -> ArrayLike:
        raise NotImplementedError

    def log_grad_prob(self, x: ArrayLike) -> ArrayLike:
        p, p_dx = self.prob_and_grad(x)
        return safe_div(p_dx, p)

    def prob_and_grad(self, x: ArrayLike) -> Tuple[ArrayLike, ArrayLike]:
        return self.prob(x), self.grad_prob(x)


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
