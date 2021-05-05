# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import pytest
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.utils.validation import check_is_fitted
from tryangle.model_selection._split import TriangleSplit
from tryangle.core._methods import CapeCod
from tryangle.metrics._score import neg_ave_scorer
from tryangle.utils.datasets import load_sample


def test_split():
    X = load_sample("swiss")
    triangle = X.triangle.copy()
    sample_weight = X.sample_weight.copy()
    tscv = TriangleSplit(n_splits=10)
    val_years = list(range(1988, 1998))

    for i, (train_idx, _) in enumerate(tscv.split(X)):
        assert (
            X[train_idx].triangle == triangle[triangle.valuation.year <= val_years[i]]
        )
        assert (
            X[train_idx].sample_weight
            == sample_weight[sample_weight.valuation.year <= val_years[i]]
        )


@pytest.mark.parametrize("SearchClass", [(GridSearchCV), (RandomizedSearchCV)])
def test_search_methods(SearchClass):
    X = load_sample("swiss")
    tscv = TriangleSplit(n_splits=5)

    model = GridSearchCV(
        CapeCod(),
        param_grid={"decay": [0.2, 0.8]},
        scoring=neg_ave_scorer,
        cv=tscv,
        n_jobs=1,
    ).fit(X, X)
    assert check_is_fitted(model) is None
