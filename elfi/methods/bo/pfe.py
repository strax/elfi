import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class PFEstimator(ABC):
    @abstractmethod
    def predict(self, x, t):
        ...

class Oracle(PFEstimator):
    def __init__(self, func):
        self.func = func

    def predict(self, x, t):
        del t
        return self.func(x)