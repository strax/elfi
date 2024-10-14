import numpy as np

from abc import ABC, abstractmethod


class FeasibilityEstimator(ABC):
    @abstractmethod
    def predict(self, x): ...

    def update(self, x, y):
        pass

    @property
    def is_differentiable(self) -> bool:
        return False

    def predict_grad(self, x):
        raise NotImplementedError

class OracleFeasibilityEstimator(FeasibilityEstimator):
    def __init__(self, func, grad = None):
        self.func = func
        self.grad = grad

    def predict(self, x):
        return np.float_(self.func(x))

    @property
    def is_differentiable(self):
        return self.grad is not None

    def predict_grad(self, x):
        if self.grad is not None:
            return self.grad(x)
        raise NotImplementedError


__all__ = ["FeasibilityEstimator", "OracleFeasibilityEstimator"]
