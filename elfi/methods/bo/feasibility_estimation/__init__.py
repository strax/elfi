import numpy as np

from abc import ABC, abstractmethod


class FeasibilityEstimator(ABC):
    @abstractmethod
    def predict(self, x, t): ...

    def update(self, x, y):
        pass

class OracleFeasibilityEstimator(FeasibilityEstimator):
    def __init__(self, func):
        self.func = func

    def predict(self, x, t):
        del t
        # x has shape (...batches, param) but simulators receive a tuple or ndarray with shape (param, ...batches), so swap the position of the param axis
        x = np.swapaxes(x, -1, 0)
        return np.float_(self.func(*x))


__all__ = ["FeasibilityEstimator", "OracleFeasibilityEstimator"]
