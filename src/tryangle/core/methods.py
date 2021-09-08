# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import chainladder.development as cl_development
import chainladder.methods as cl_methods
import chainladder.workflow as cl_workflow
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


class Development(TransformerMixin, cl_development.Development):
    __doc__ = cl_development.Development.__doc__

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


class DevelopmentConstant(TransformerMixin, cl_development.DevelopmentConstant):
    __doc__ = cl_development.DevelopmentConstant.__doc__

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


class IncrementalAdditive(TransformerMixin, cl_development.IncrementalAdditive):
    __doc__ = cl_development.IncrementalAdditive.__doc__

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


class ClarkLDF(TransformerMixin, cl_development.ClarkLDF):
    __doc__ = cl_development.ClarkLDF.__doc__

    def __init__(self, growth="loglogistic"):
        super().__init__(
            growth=growth,
        )


class Chainladder(EstimatorMixin, cl_methods.Chainladder):
    __doc__ = cl_methods.Chainladder.__doc__


class MackChainladder(EstimatorMixin, cl_methods.MackChainladder):
    __doc__ = cl_methods.MackChainladder.__doc__


class BornhuetterFerguson(EstimatorMixin, cl_methods.BornhuetterFerguson):
    __doc__ = cl_methods.BornhuetterFerguson.__doc__

    def __init__(self, apriori=1.0, apriori_sigma=0.0, random_state=None):
        super().__init__(
            apriori=apriori, apriori_sigma=apriori_sigma, random_state=random_state
        )


class CapeCod(EstimatorMixin, cl_methods.CapeCod):
    __doc__ = cl_methods.CapeCod.__doc__

    def __init__(self, trend=0, decay=1):
        super().__init__(trend=trend, decay=decay)


class Benktander(EstimatorMixin, cl_methods.Benktander):
    __doc__ = cl_methods.Benktander.__doc__

    def __init__(self, apriori=1.0, n_iters=1, apriori_sigma=0, random_state=None):
        super().__init__(
            apriori=apriori,
            n_iters=n_iters,
            apriori_sigma=apriori_sigma,
            random_state=random_state,
        )


class VotingChainladder(EstimatorMixin, cl_workflow.VotingChainladder):
    __doc__ = cl_workflow.VotingChainladder.__doc__

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
                getattr(globals()["cl_methods"], estimator.__class__.__name__)(
                    **estimator.__dict__
                ),
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
