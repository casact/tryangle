# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from chainladder.development import Development as DevelopmentCL
from chainladder.methods import BornhuetterFerguson as BornhuetterFergusonCL
from chainladder.methods import CapeCod as CapeCodCL
from chainladder.methods import Chainladder as ChainladderCL
from chainladder.methods import Benktander as BenktanderCL
from chainladder.workflow import VotingChainladder as VotingChainladderCL
from tryangle.core._base import TryangleData


class EstimatorMixin:
    """
    Fit and predict a chainladder estimator
    """

    def fit(self, X, y=None, sample_weight=None):
        return super().fit(X.triangle, y=None, sample_weight=X.sample_weight)

    def predict(self, X, sample_weight=None):
        return super().predict(X.triangle, sample_weight=X.sample_weight)

    def fit_predict(self, X, sample_weight=None):
        self.fit(X)
        return self.predict(X)


class TransformerMixin:
    """
    Fit and transform a chainladder transfomer
    """

    def fit(self, X, y=None, sample_weight=None):
        return super().fit(X.triangle, y=None, sample_weight=X.sample_weight)

    def transform(self, X):
        return TryangleData(super().transform(X.triangle), X.sample_weight)


class Development(TransformerMixin, DevelopmentCL):
    __doc__ = DevelopmentCL.__doc__

    def __init__(
        self,
        n_periods=-1,
        average="volume",
        sigma_interpolation="log-linear",
        drop=None,
        drop_high=None,
        drop_low=None,
        drop_valuation=None,
        fillna=None,
    ):
        super().__init__(
            n_periods=n_periods,
            average=average,
            sigma_interpolation=sigma_interpolation,
            drop=drop,
            drop_high=drop_high,
            drop_low=drop_low,
            drop_valuation=drop_valuation,
            fillna=fillna,
        )


class Chainladder(EstimatorMixin, ChainladderCL):
    __doc__ = ChainladderCL.__doc__


class BornhuetterFerguson(EstimatorMixin, BornhuetterFergusonCL):
    __doc__ = BornhuetterFergusonCL.__doc__

    def __init__(self, apriori=1.0, apriori_sigma=0.0, random_state=None):
        super().__init__(
            apriori=apriori, apriori_sigma=apriori_sigma, random_state=random_state
        )


class CapeCod(EstimatorMixin, CapeCodCL):
    __doc__ = CapeCodCL.__doc__

    def __init__(self, trend=0, decay=1):
        super().__init__(trend=trend, decay=decay)


class Benktander(EstimatorMixin, BenktanderCL):
    __doc__ = BenktanderCL.__doc__

    def __init__(self, apriori=1.0, n_iters=1, apriori_sigma=0, random_state=None):
        super().__init__(apriori=1.0, n_iters=1, apriori_sigma=0, random_state=None)


class VotingChainladder(EstimatorMixin, VotingChainladderCL):
    __doc__ = VotingChainladderCL.__doc__

    def __init__(self, estimators, weights=None, n_jobs=None, verbose=False):
        # Convert tryangle estimators to chainladder estimators
        estimators = [
            (name, globals()[f"{estimator.__class__.__name__}CL"](**estimator.__dict__))
            for name, estimator in estimators
        ]
        super().__init__(estimators, weights=None, n_jobs=None, verbose=False)
