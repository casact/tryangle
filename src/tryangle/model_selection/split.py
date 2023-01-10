# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np
from sklearn.model_selection._split import TimeSeriesSplit


class TriangleSplit(TimeSeriesSplit):
    """Triangle cross-validation across calendar periods

    Splits a ``TryangleData`` instance into k sub-triangles or folds
    over the latest k calendar periods or diagonals. Thus, successive
    folds are supersets of previous folds.

    Parameters
    ----------
    n_splits : int, default=5
        Number of splits, must be at least 2.
    """

    def __init__(self, n_splits=5, *, max_train_size=None, test_size=None, gap=0):
        super().__init__(n_splits=n_splits, max_train_size=None, test_size=1, gap=0)

    def split(self, X, y=None, groups=None):
        valuation_date = X.triangle.latest_diagonal.valuation[0]
        valuation_dates = np.array(
            [
                date
                for date in X.triangle.valuation.drop_duplicates().sort_values()
                if date.date() <= valuation_date.date()
            ]
        )
        for train, test in super().split(valuation_dates):
            indices = np.arange(X.triangle.shape[2] * X.triangle.shape[3])
            test_start = test[0]
            yield (
                indices[X.triangle.valuation <= valuation_dates[test_start]],
                indices[X.triangle.valuation <= valuation_dates[test_start]],
            )
