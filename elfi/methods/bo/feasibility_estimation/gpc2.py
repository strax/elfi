import logging
import time
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
from torchmin import Minimizer

from . import FeasibilityEstimator

logger = logging.getLogger(__name__)


def _as_tensor(input: Tensor | NDArray) -> Tensor:
    if not isinstance(input, Tensor):
        input = torch.from_numpy(input)
    return input


def _approx_sigmoid_gaussian_conv(mu: Tensor, sigma2: Tensor) -> Tensor:
    return torch.sigmoid(mu / torch.sqrt(1 + torch.pi / 8 * sigma2))


class BinaryDirichletGPC(ExactGP):
    def __init__(self, X: Tensor, y: Tensor):
        y = y.to(torch.int)
        likelihood = DirichletClassificationLikelihood(y, dtype=torch.double)
        assert likelihood.num_classes == 2
        super().__init__(X, likelihood.transformed_targets, likelihood)
        self.likelihood = likelihood
        batch_shape = torch.Size((2,))
        self.mean = ConstantMean(batch_shape=batch_shape).double()
        self.cov = ScaleKernel(
            MaternKernel(5 / 2, batch_shape=batch_shape).double(),
            batch_shape=batch_shape,
        ).double()

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
    fast_predictive_integration: bool

    _prev_reopt_nobs: int = 0

    def __init__(self, *, reoptimization_interval=10, fast_predictive_integration=True):
        self.reopt_interval = reoptimization_interval
        self.fast_predictive_integration = fast_predictive_integration

    def _init_model(self):
        logger.debug("(Re)initializing model")
        self.model = BinaryDirichletGPC(self.X, self.y)
        self._optimize_hyperparameters()
        return self.model

    def _optimize_hyperparameters(self, **kwargs):
        assert self.model is not None
        model, likelihood = self.model, self.model.likelihood

        optimizer = Minimizer(model.hyperparameters(), max_iter=1000, tol=1e-8)
        mll = ExactMarginalLogLikelihood(likelihood, model)

        model.train()
        likelihood.train()

        def step():
            optimizer.zero_grad()
            output = model(*model.train_inputs)
            objective = -mll(output, model.train_targets).sum()
            return objective

        time_begin = time.monotonic()
        loss = optimizer.step(step)
        time_end = time.monotonic()
        logger.debug(
            "Optimized hyperparameters in %.4fs: MLL = %.6f",
            time_end - time_begin,
            loss,
        )
        assert torch.all(
            torch.isfinite(loss)
        ), "Optimization resulted in nonfinite parameters"

        model.eval()
        likelihood.eval()

    @torch.inference_mode
    def predict(self, x: NDArray):
        if self.model is None:
            return 1.0

        x = _as_tensor(np.atleast_2d(x)).double()
        with gpytorch.settings.fast_computations(False, False, False):
            predictive = self.model(x)
            if self.fast_predictive_integration:
                mu = predictive.mean[0] - predictive.mean[1]
                sigma2 = predictive.variance[0] + predictive.variance[1]
                p_failure = _approx_sigmoid_gaussian_conv(mu, sigma2)
            else:
                # Approximate eq. 8
                p_failure, _ = predictive.sample(torch.Size((256,))).softmax(1).mean(0)
            return 1.0 - p_failure.numpy(force=True)

    def update(self, x: NDArray, y: NDArray):
        y = np.isfinite(y)
        X = _as_tensor(np.atleast_2d(x)).double()
        y = _as_tensor(np.atleast_1d(y)).bool()

        # Update observations
        if self.X is None:
            self.X = X
            self.y = y
        else:
            self.X = torch.cat((self.X, X), 0)
            self.y = torch.cat((self.y, y), 0)

        # Dirichlet GPC needs observations from both classes to function
        if not (0 < self.y.count_nonzero() < self.y.numel()):
            logger.debug(
                "Skipping model initialization due to not having seen both failed/succeeded observations"
            )
            return

        if self.model is None or self._should_reopt():
            self._optimize()
            self._prev_reopt_nobs = self.y.numel()
        else:
            n_succeeded = torch.count_nonzero(self.y).item()
            n_failed = torch.numel(self.y) - n_succeeded
            logger.debug(
                "Updating GPC posterior (succeeded: %d, failed: %d, total: %d)",
                n_succeeded,
                n_failed,
                n_succeeded + n_failed,
            )
            # TODO: Use `self.model.get_fantasy_model` if/when it is fixed
            self.model.set_train_data(self.X, self.y)

    def _should_reopt(self) -> bool:
        nobs_since_reopt = self.y.numel() - self._prev_reopt_nobs
        return nobs_since_reopt > self.reopt_interval

    def _optimize(self):
        logger.debug("Reoptimizing")
        # Recreating the whole model is a bit inefficient, but hyperparameter optimization does not
        # work if resumed from a previous state
        self._init_model()
