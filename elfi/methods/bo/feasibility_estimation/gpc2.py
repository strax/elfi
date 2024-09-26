import logging
from typing import Callable

import gpytorch
import numpy as np
import torch
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import MaternKernel, RBFKernel, RQKernel, ScaleKernel
from gpytorch.likelihoods import DirichletClassificationLikelihood
from gpytorch.means import ConstantMean, ZeroMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.models import ExactGP
from numpy.typing import NDArray
from torch import Tensor
from torch.optim import LBFGS, Optimizer

from . import FeasibilityEstimator

logger = logging.getLogger(__name__)


def _optimize(
    step: Callable[[], float], optimizer: Optimizer, *, max_iter=1000, ftol=1e-6
) -> tuple[float, int]:
    prev_objective = torch.inf
    for i in range(max_iter):
        objective = optimizer.step(step)
        if prev_objective - objective < ftol:
            logger.debug("Early stopping: %s < %s", prev_objective - objective, ftol)
            break
        prev_objective = objective
    return objective, i + 1


def _as_tensor(input: Tensor | NDArray) -> Tensor:
    if not isinstance(input, Tensor):
        input = torch.from_numpy(input)
    return input


class BinaryDirichletGPC(ExactGP):
    def __init__(self, X: Tensor, y: Tensor):
        y = y.to(torch.int)
        likelihood = DirichletClassificationLikelihood(y, dtype=torch.double)
        assert likelihood.num_classes == 2
        super().__init__(X, likelihood.transformed_targets, likelihood)
        self.likelihood = likelihood
        batch_shape = torch.Size((2,))
        self.mean = ConstantMean(batch_shape=batch_shape).double()
        self.cov = ScaleKernel(MaternKernel(5/2, batch_shape=batch_shape).double(), batch_shape=batch_shape).double()

    def forward(self, x: Tensor):
        mean, cov = self.mean(x), self.cov(x)
        return MultivariateNormal(mean, cov)

    def set_train_data(self, inputs: Tensor, targets: Tensor):
        X, y = inputs, targets.to(torch.int)

        self.likelihood = DirichletClassificationLikelihood(y, dtype=torch.double)
        super().set_train_data(X, self.likelihood.transformed_targets, strict=False)


class GPCFeasibilityEstimator(FeasibilityEstimator):
    X: Tensor | None = None
    y: Tensor | None = None
    model: BinaryDirichletGPC | None = None
    optimize_after_update: bool

    def __init__(self, *, optimize_after_update=False):
        self.optimize_after_update = optimize_after_update

    def _init_model(self):
        if not (0 < self.y.count_nonzero() < self.y.numel()):
            logger.debug(
                "Skipping model initialization due to not having seen both failed/succeeded observations"
            )
            return
        logger.debug("(Re)initializing model")
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

    @torch.inference_mode
    def predict(self, x: NDArray, t: int):
        del t
        if self.model is None:
            return 1.0

        x = _as_tensor(np.atleast_2d(x)).double()
        with gpytorch.settings.fast_computations(False, False, False):
            predictive = self.model(x)
            # Approximate eq. 8
            p_failure, _ = predictive.sample(torch.Size((256,))).softmax(1).mean(0)
            return 1.0 - p_failure.numpy(force=True)

    def update(self, x: NDArray, y: NDArray):
        y = np.isfinite(y)
        X = _as_tensor(np.atleast_2d(x)).double()
        y = _as_tensor(np.atleast_1d(y)).bool()

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
