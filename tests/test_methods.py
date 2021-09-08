# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import pytest
import numpy as np
from sklearn.pipeline import Pipeline
from tryangle.core.methods import (
    Benktander,
    BornhuetterFerguson,
    CapeCod,
    Chainladder,
    ClarkLDF,
    MackChainladder,
    Development,
    DevelopmentConstant,
    IncrementalAdditive,
    VotingChainladder,
)
from tryangle.utils.datasets import load_sample

base_estimators = [
    Chainladder,
    MackChainladder,
    BornhuetterFerguson,
    CapeCod,
    Benktander,
]

base_transformers = [
    Development,
    DevelopmentConstant,
    IncrementalAdditive,
    ClarkLDF,
]


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
    kwargs = {}

    if isinstance(transformer(), DevelopmentConstant):
        n = len(sample_data.triangle.development)
        x = np.linspace(1, n, n)
        ldf = 1 - np.exp(-x)
        print(x)
        kwargs.update({"patterns": {(i + 1) * 12: l for i, l in enumerate(ldf)}})

    transformer(**kwargs).fit_transform(X)


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
