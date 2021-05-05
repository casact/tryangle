# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import pytest
import chainladder as cl
from tryangle.metrics._score import (
    neg_ave_scorer,
    neg_weighted_ave_scorer,
    neg_cdr_scorer,
    neg_weighted_cdr_scorer,
)
from tryangle.core._base import TryangleData
from tryangle.core._methods import Chainladder


@pytest.mark.parametrize(
    "scorer, true_score",
    [
        (neg_ave_scorer, -2222.357324),
        (neg_weighted_ave_scorer, -2372.370821),
        (neg_cdr_scorer, -5064.408237),
        (neg_weighted_cdr_scorer, -5891.385703),
    ],
)
def test_scorers(scorer, true_score):
    X = TryangleData(cl.load_sample('raa'))
    score = scorer(Chainladder(), X, X)
    assert (score - true_score) < 0.001
