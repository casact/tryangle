# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np

from chainladder.adjustments.bootstrap import (
    BootstrapODPSample as _cl_BootstrapODPSample,
)
from chainladder.development.clark import ClarkLDF as _cl_ClarkLDF
from chainladder.development.constant import (
    DevelopmentConstant as _cl_DevelopmentConstant,
)
from chainladder.development.development import Development as _cl_Development
from chainladder.development.incremental import (
    IncrementalAdditive as _cl_IncrementalAdditive,
)
from chainladder.methods.benktander import Benktander as _cl_Benktander
from chainladder.methods.bornferg import BornhuetterFerguson as _cl_BornhuetterFerguson
from chainladder.methods.capecod import CapeCod as _cl_CapeCod
from chainladder.methods.chainladder import Chainladder as _cl_Chainladder
from chainladder.methods.mack import MackChainladder as _cl_MackChainladder
from chainladder.workflow.voting import VotingChainladder as _cl_VotingChainladder

from tryangle.core.base import TryangleData


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


class BootstrapODPSample(TransformerMixin, _cl_BootstrapODPSample):
    __doc__ = _cl_BootstrapODPSample.__doc__

    def __init__(
        self,
        n_sims=1000,
        n_periods=-1,
        hat_adj=True,
        drop=None,
        drop_high=None,
        drop_low=None,
        drop_valuation=None,
        random_state=None,
    ):
        super().__init__(
            n_sims=n_sims,
            n_periods=n_periods,
            hat_adj=hat_adj,
            drop=drop,
            drop_high=drop_high,
            drop_low=drop_low,
            drop_valuation=drop_valuation,
            random_state=random_state,
        )

    def transform(self, X):
        """
        Parameters
        ----------
        X: Triangle
            The triangle to be transformed
        Returns
        -------
            X_new: New triangle with transformed attributes.
        """
        X_new_triangle = X.triangle.copy()
        X_new_triangle = self.resampled_triangles_
        X_new_triangle.scale_ = self.scale_
        X_new_triangle.random_state = self.random_state

        if X.sample_weight is not None:
            repeated_sample_weights = np.repeat(
                X.sample_weight.to_frame().to_numpy()[np.newaxis, np.newaxis, ...],
                10,
                axis=0,
            )
            X_new_sample_weight = X.sample_weight.copy()
            X_new_sample_weight.kdims = np.arange(10)
            X_new_sample_weight.values = repeated_sample_weights
            X_new_sample_weight._set_slicers()
        else:
            X_new_sample_weight = None

        return TryangleData(X_new_triangle, X_new_sample_weight)


class Development(TransformerMixin, _cl_Development):
    __doc__ = _cl_Development.__doc__

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


class DevelopmentConstant(TransformerMixin, _cl_DevelopmentConstant):
    __doc__ = _cl_DevelopmentConstant.__doc__

    def __init__(
        self,
        patterns=None,
        style="ldf",
        callable_axis=0,
    ):
        super().__init__(
            patterns=patterns,
            style=style,
            callable_axis=callable_axis,
        )


class IncrementalAdditive(TransformerMixin, _cl_IncrementalAdditive):
    __doc__ = _cl_IncrementalAdditive.__doc__

    def __init__(
        self,
        trend=0.0,
        n_periods=-1,
        average="volume",
        future_trend=0,
        drop=None,
        drop_high=None,
        drop_low=None,
        drop_valuation=None,
    ):
        super().__init__(
            trend=trend,
            n_periods=n_periods,
            average=average,
            future_trend=future_trend,
            drop=drop,
            drop_high=drop_high,
            drop_low=drop_low,
            drop_valuation=drop_valuation,
        )


class ClarkLDF(TransformerMixin, _cl_ClarkLDF):
    __doc__ = _cl_ClarkLDF.__doc__

    def __init__(self, growth="loglogistic"):
        super().__init__(
            growth=growth,
        )


class Chainladder(EstimatorMixin, _cl_Chainladder):
    __doc__ = _cl_Chainladder.__doc__


class MackChainladder(EstimatorMixin, _cl_MackChainladder):
    __doc__ = _cl_MackChainladder.__doc__


class BornhuetterFerguson(EstimatorMixin, _cl_BornhuetterFerguson):
    __doc__ = _cl_BornhuetterFerguson.__doc__

    def __init__(self, apriori=1.0, apriori_sigma=0.0, random_state=None):
        super().__init__(
            apriori=apriori, apriori_sigma=apriori_sigma, random_state=random_state
        )


class CapeCod(EstimatorMixin, _cl_CapeCod):
    __doc__ = _cl_CapeCod.__doc__

    def __init__(self, trend=0, decay=1):
        super().__init__(trend=trend, decay=decay)


class Benktander(EstimatorMixin, _cl_Benktander):
    __doc__ = _cl_Benktander.__doc__

    def __init__(self, apriori=1.0, n_iters=1, apriori_sigma=0, random_state=None):
        super().__init__(
            apriori=apriori,
            n_iters=n_iters,
            apriori_sigma=apriori_sigma,
            random_state=random_state,
        )


class VotingChainladder(EstimatorMixin, _cl_VotingChainladder):
    __doc__ = _cl_VotingChainladder.__doc__

    def __init__(
        self,
        estimators,
        weights=None,
        default_weighting=None,
        n_jobs=None,
        verbose=False,
    ):
        # Convert tryangle estimators to chainladder estimators
        estimators = [
            (
                name,
                globals()["_cl_" + estimator.__class__.__name__](**estimator.__dict__),
            )
            for name, estimator in estimators
        ]
        super().__init__(
            estimators=estimators,
            weights=weights,
            default_weighting=default_weighting,
            n_jobs=n_jobs,
            verbose=verbose,
        )
