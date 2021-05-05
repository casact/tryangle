# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import os
import pandas as pd
import pytest

from chainladder import Triangle
from tryangle.core import TryangleData
from tryangle.utils import datasets


@pytest.mark.parametrize(
    "key",
    [
        ("swiss"),
        ("cas"),
        ("sme"),
    ],
)
def test_load_dataset(key):
    # Load TryangleData
    train, test = datasets.load_sample(key), datasets.load_test_sample(key)

    # Load raw data
    train_raw = pd.read_csv(
        os.path.join(os.path.dirname(datasets.__file__), "data", f"{key}_train.csv")
    )
    test_raw = pd.read_csv(
        os.path.join(os.path.dirname(datasets.__file__), "data", f"{key}_test.csv")
    )

    max_train_lag = train_raw.lag.max()

    for X, raw in zip([train, test], [train_raw, test_raw]):
        assert isinstance(X, TryangleData)

        assert isinstance(X.triangle, Triangle)
        if X.sample_weight:
            assert isinstance(X.sample_weight, Triangle)

        assert X.triangle.shape == (1, 1, max_train_lag + 1, max_train_lag + 1)
        assert X.shape == (X.triangle.shape[2] * X.triangle.shape[3], 1)

        if X.sample_weight:
            assert (X.triangle.origin == X.sample_weight.origin).all()

        assert (X.triangle.sum().sum() - raw.claim.sum()) < 0.01
        if X.sample_weight:
            assert (X.sample_weight.sum() - raw.claim.sum()) < 0.01

    assert (train.triangle.origin == test.triangle.origin).all()
