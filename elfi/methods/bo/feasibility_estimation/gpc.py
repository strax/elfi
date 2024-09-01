import numpy as np

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.utils.validation import check_is_fitted, NotFittedError
from numpy.testing import assert_equal

from . import FeasibilityEstimator


class GPCFeasibilityEstimator(FeasibilityEstimator):
    _gpc: GaussianProcessClassifier
    _X: np.ndarray | None
    _y: np.ndarray | None
    _use_feasibility_threshold: bool

    def __init__(self, *, use_feasibility_threshold=False):
        self._gpc = GaussianProcessClassifier()
        self._X = None
        self._y = None
        self._use_feasibility_threshold = use_feasibility_threshold

    @property
    def _is_fitted(self) -> bool:
        try:
            check_is_fitted(self._gpc)
            return True
        except NotFittedError:
            return False

    def _append_observed(self, x, y):
        x = np.atleast_2d(x)
        if self._X is None and self._y is None:
            self._X = np.copy(x)
            self._y = np.copy(y)
        else:
            self._X = np.vstack((self._X, x))
            self._y = np.concatenate([self._y, y])

    def update(self, x, y):
        # 1 = feasible, 0 = infeasible
        f = np.where(np.isfinite(y), 1, 0).ravel()

        self._append_observed(x, f)

        # Check if we have seen both feasible and infeasible points as
        # GaussianProcessClassifier requires both to be present for fitting
        if 0 < np.count_nonzero(self._y) < np.size(self._y):
            self._gpc.fit(self._X, self._y)
            # FIXME: Remove this check
            assert_equal(np.array(self._gpc.classes_), np.array([0, 1]))

    def predict(self, x, t):
        del t

        x = np.asarray(x)
        *batch_dims, input_dim = np.shape(x)
        if self._is_fitted:
            p_feasible = self._gpc.predict_proba(x.reshape(-1, input_dim))[:, 1]
            p_feasible = p_feasible.reshape(*batch_dims, 1)
            if self._use_feasibility_threshold:
                # If feasibility thresholding is enabled, we consider points that have p(feasible) >= 0.5
                # to be certainly feasible and rescale the probability accordingly. This ensures that
                # feasibility estimation does not affect the exploration-exploitation tradeoff of the base
                # acquisition function by giving known feasible points a higher feasibility probability
                # compared to unseen points.
                p_feasible = np.minimum(p_feasible, 0.5) * 2
            return p_feasible
        # We have not yet fitted the classifier, so we assume all points to be feasible.
        return np.broadcast_to(1.0, (*batch_dims, 1))


__all__ = ["GPCFeasibilityEstimator"]
