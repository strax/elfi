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
    def __init__(self, func):
        self.func = func

    def predict(self, x):
        return np.float_(self.func(x))


__all__ = ["FeasibilityEstimator", "OracleFeasibilityEstimator"]
