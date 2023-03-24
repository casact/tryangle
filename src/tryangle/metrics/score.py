# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np

from tryangle.metrics.base import get_actual_expected

from sklearn.metrics._scorer import _BaseScorer
from sklearn.metrics import mean_squared_error


class AVEScore(_BaseScorer):
    """Base AvE scoring class"""

    def __init__(
        self,
        score_func=mean_squared_error,
        sign=-1,
        kwargs={"squared": False},
        weighted=False,
    ):
        self.weighted = weighted
        super().__init__(score_func, sign, kwargs)

    def _sample_weight(self, X, valuation_date):
        X_prior = X[X.triangle.valuation < valuation_date]
        current = X.triangle.latest_diagonal.to_frame().to_numpy()
        prior = X_prior.triangle.latest_diagonal.to_frame().to_numpy().reshape(-1, 1)

        if X.triangle.shape[0] > 1:
            current = current[:, :-1].reshape(-1, 1)
        else:
            current = current[:-1]

        return np.abs(current - prior)

    def _score(self, method_caller, estimator, X, y_true=None, sample_weight=None):
        """Evaluate predicted target values for X relative to y_true."""  # TODO: Update docstring

        valuation_date = X.triangle.cum_to_incr().latest_diagonal.valuation[0]
        actual, expected = get_actual_expected(estimator, X)

        if self.weighted:
            sample_weight = self._sample_weight(X, valuation_date)
            return self._sign * self._score_func(
                actual, expected, sample_weight=sample_weight, **self._kwargs
            )
        else:
            return self._sign * self._score_func(actual, expected, **self._kwargs)


class CDRScore(AVEScore):
    """Base CDR scoring class"""

    def _score(self, method_caller, estimator, X, y_true, sample_weight=None):
        """Evaluate predicted target values for X relative to y_true."""  # TODO: Update docstring

        valuation_date = y_true.triangle.latest_diagonal.valuation[0]
        X_k = X[X.triangle.valuation < valuation_date]
        X_k_1 = X

        # We only need actual as the expected will be in ibnr_k
        actual, _ = get_actual_expected(estimator, X)

        # ! TODO: Rewrite a better method that avoids fillna(0) as this could mask some error
        ibnr_k = estimator.fit_predict(X_k).ibnr_.to_frame().fillna(0).to_numpy()
        if X_k.triangle.shape[0] > 1:
            ibnr_k = ibnr_k.reshape(-1, 1)

        ibnr_k_1 = estimator.fit_predict(X_k_1).ibnr_.to_frame().fillna(0).to_numpy()
        if X_k_1.triangle.shape[0] > 1:
            ibnr_k_1 = ibnr_k_1[:, :-1].reshape(-1, 1)
        else:
            ibnr_k_1 = ibnr_k_1[:-1]

        if self.weighted:
            sample_weight = super()._sample_weight(X, valuation_date)
            return self._sign * self._score_func(
                actual + ibnr_k_1, ibnr_k, sample_weight=sample_weight, **self._kwargs
            )
        else:
            return self._sign * self._score_func(
                actual + ibnr_k_1, ibnr_k, **self._kwargs
            )


neg_ave_scorer = AVEScore()
neg_weighted_ave_scorer = AVEScore(weighted=True)
neg_cdr_scorer = CDRScore()
neg_weighted_cdr_scorer = CDRScore(weighted=True)
