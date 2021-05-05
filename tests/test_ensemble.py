# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import os
from distutils import dir_util

import numpy as np
import pytest
from chainladder.utils import load_sample
from tryangle.core._base import TryangleData
from tryangle.core._methods import BornhuetterFerguson, CapeCod, Chainladder
from tryangle.ensemble._base import AutoEnsemble
from tryangle.model_selection._split import TriangleSplit


@pytest.fixture
def auto_ensemble():
    estimators = [
        ("cl", Chainladder()),
        ("bf", BornhuetterFerguson()),
        ("cc", CapeCod()),
    ]
    tscv = TriangleSplit(n_splits=3)
    yield AutoEnsemble(
        estimators=estimators,
        cv=tscv,
    )


@pytest.fixture
def sample_data():
    ia = load_sample("ia_sample")
    X = TryangleData(ia["loss"], ia["exposure"].latest_diagonal)
    yield X


@pytest.fixture  # https://stackoverflow.com/a/29631801
def data_dir(tmpdir, request):
    """
    Fixture responsible for searching a folder with the same name of test
    module and, if available, moving all contents to a temporary directory so
    tests can use them freely.
    """
    filename = request.module.__file__
    test_dir, _ = os.path.splitext(filename)

    if os.path.isdir(test_dir):
        dir_util.copy_tree(test_dir, str(tmpdir))

    return tmpdir


def test_compile(auto_ensemble, sample_data, data_dir):
    ens = auto_ensemble
    X = sample_data
    ens.compile(X)

    with open(data_dir.join("ia_sample_compile_true_actual.npy"), "rb") as f:
        true_actual = np.load(f, allow_pickle=True)
        assert ens.actual_.shape == (3, 6, 1)
        assert np.array_equal(ens.actual_, true_actual)

    with open(data_dir.join("ia_sample_compile_true_expected.npy"), "rb") as f:
        true_expected = np.load(f, allow_pickle=True)
        assert ens.expected_.shape == (3, 6, 3)
        assert np.array_equal(ens.expected_, true_expected)

    with open(data_dir.join("ia_sample_compile_true_t.npy"), "rb") as f:
        true_t = np.load(f, allow_pickle=True)
        assert ens.t_.shape == (3, 6, 1)
        assert np.array_equal(ens.t_, true_t)


def test_softmax(auto_ensemble):
    ens = auto_ensemble
    assert np.array_equal(
        ens._softmax(np.stack((-np.ones((5, 2)), np.zeros((5, 2)), np.ones((5, 2))))),
        0.5 * np.ones((3, 5, 2)),
    )


def test_softmax_gradient(auto_ensemble):
    ens = auto_ensemble
    assert np.array_equal(
        ens._softmax_gradient(0.5 * np.ones((3, 5, 2))),
        -0.25 * np.ones((3, 5, 2, 2)) + 0.5 * np.eye(2),
    )

# TODO: Expand tests on finalisation of autoensemble
