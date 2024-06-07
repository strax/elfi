import logging
from abc import ABC, abstractmethod
from scipy.stats.distributions import rv_continuous

logger = logging.getLogger(__name__)

class PFEstimator(ABC):
    @abstractmethod
    def predict(self, x, t):
        ...

class FixedDistribution(PFEstimator):
    def __init__(self, dist: rv_continuous):
        self.dist = dist
    
    def predict(self, x, t):
        del t
        return self.dist.pdf(x)
