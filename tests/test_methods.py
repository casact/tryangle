# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import pytest
from sklearn.pipeline import Pipeline
from tryangle.core._methods import (
    Benktander,
    BornhuetterFerguson,
    CapeCod,
    Chainladder,
    Development,
    VotingChainladder,
)
from tryangle.utils.datasets import load_sample

base_estimators = [Chainladder, BornhuetterFerguson, CapeCod, Benktander]

base_transformers = [Development]


@pytest.fixture
def sample_data():
    yield load_sample("swiss")


@pytest.mark.parametrize("estimator", base_estimators)
def test_estimators(estimator, sample_data):
    X = sample_data
    estimator().fit_predict(X)


@pytest.mark.parametrize("transformer", base_transformers)
def test_transformers(transformer, sample_data):
    X = sample_data
    transformer().fit_transform(X)


def test_voting_estimator(sample_data):
    X = sample_data
    estimators = [
        ("cl", Chainladder()),
        ("bf", BornhuetterFerguson()),
        ("cc", CapeCod()),
    ]
    VotingChainladder(estimators=estimators).fit_predict(X)


def test_pipeline(sample_data):
    X = sample_data
    pipe = Pipeline([("development", Development()), ("cl", Chainladder())])
    pipe.fit_predict(X)
