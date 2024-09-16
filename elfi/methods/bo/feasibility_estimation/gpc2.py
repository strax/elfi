import numpy as np
import torch

import gpytorch
import logging

from gpytorch.models import ExactGP
from gpytorch.likelihoods import DirichletClassificationLikelihood, Likelihood
from gpytorch.means import ZeroMean, ConstantMean
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel
from gpytorch.distributions import MultivariateNormal
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.optim import LBFGS, Optimizer
from torch import Tensor
from typing import Callable
from numpy.typing import NDArray

from . import FeasibilityEstimator

logger = logging.getLogger(__name__)

def _optimize(step: Callable[[], float], optimizer: Optimizer, *, max_iter=1000, ftol=1e-6) -> tuple[float, int]:
    prev_objective = torch.inf
    for i in range(max_iter):
        objective = optimizer.step(step)
        if prev_objective - objective < ftol:
            logger.debug("Early stopping: %s < %s", prev_objective - objective, ftol)
            break
        prev_objective = objective
    return objective, i + 1

def _convert_to_tensor(input: Tensor | NDArray, *, dtype = torch.float):
    if not isinstance(input, Tensor):
        input = torch.from_numpy(input)
    return input.to(dtype)

class BinaryDirichletGPC(ExactGP):
    def __init__(self, X: Tensor, y: Tensor):
        y = y.to(torch.int)
        likelihood = DirichletClassificationLikelihood(y)
        assert likelihood.num_classes == 2
        super().__init__(X, likelihood.transformed_targets, likelihood)
        self.likelihood = likelihood
        batch_shape = torch.Size((2,))
        self.mean = ConstantMean(batch_shape=batch_shape)
        self.cov = ScaleKernel(MaternKernel(5/2, batch_shape=batch_shape), batch_shape=batch_shape)

    def forward(self, x: Tensor):
        mean, cov = self.mean(x), self.cov(x)
        return MultivariateNormal(mean, cov)

    def set_train_data(self, inputs: Tensor, targets: Tensor):
        X, y = inputs, targets.to(torch.int)

        self.likelihood = DirichletClassificationLikelihood(y)
        super().set_train_data(X, self.likelihood.transformed_targets, strict=False)


class GPCFeasibilityEstimator(FeasibilityEstimator):
    X: Tensor | None = None
    y: Tensor | None = None
    model: BinaryDirichletGPC | None = None
    optimize_after_update: bool

    def __init__(self, *, optimize_after_update = False):
        self.optimize_after_update = optimize_after_update

    def _init_model(self):
        if not (0 < self.y.count_nonzero() < self.y.numel()):
            logger.debug('Skipping model initialization due to not having seen both failed/succeeded observations')
            return
        logger.debug('(Re)initializing model')
        self.model = BinaryDirichletGPC(self.X, self.y)
        self._optimize_hyperparameters()
        return self.model

    def _optimize_hyperparameters(self, **kwargs):
        assert self.model is not None
        model, likelihood = self.model, self.model.likelihood

        optimizer = LBFGS(model.hyperparameters(), max_iter=1)
        mll = ExactMarginalLogLikelihood(likelihood, model)

        model.train()
        likelihood.train()

        def step():
            optimizer.zero_grad()
            output = model(*model.train_inputs)
            objective = -mll(output, model.train_targets).sum()
            objective.backward()
            return objective

        loss, iters = _optimize(step, optimizer, **kwargs)
        logger.debug("Optimized hyperparameters (%s iterations): %s", iters, loss)

        model.eval()
        likelihood.eval()

    @torch.no_grad
    def predict(self, x: NDArray, t: int):
        del t
        if self.model is None:
            return 1.

        x = _convert_to_tensor(np.atleast_2d(x))
        with gpytorch.settings.fast_computations():
            predictive = self.model(x)
            # Approximate eq. 8
            p_failure, _ = predictive.sample(torch.Size((256,))).softmax(1).mean(0)
            return 1. - p_failure.numpy(force=True)

    def update(self, x: NDArray, y: NDArray):
        y = np.isfinite(y)
        X = _convert_to_tensor(np.atleast_2d(x))
        y = _convert_to_tensor(np.atleast_1d(y), dtype=torch.bool)

        if self.X is None:
            self.X = X
            self.y = y
        else:
            self.X = torch.cat((self.X, X), 0)
            self.y = torch.cat((self.y, y), 0)

        if self.model is None or self.optimize_after_update:
            logger.debug("Update caused model to be recreated")
            self._init_model()
        else:
            logger.debug("Updating GPC posterior")
            # TODO: Use `self.model.get_fantasy_model` if/when it is fixed
            self.model.set_train_data(self.X, self.y)

    def optimize(self):
        # If `optimize_after_update` is set, the hyperparameters are already optimized
        if not self.optimize_after_update:
            # Recreating the whole model is a bit inefficient, but
            # hyperparameter optimization does not work if resumed from a previous state
            self._init_model()
