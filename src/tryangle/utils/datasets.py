# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import os
import pandas as pd
from chainladder.core.triangle import Triangle
from tryangle.core import TryangleData


def load_sample(key, *args, **kwargs):
    """Function to load training datasets included in the tryangle package.

    Args:
        key (str): Key to identify dataset

    Returns:
        [TryangleData]: TryangleData object of the loaded dataset.
    """
    path = os.path.dirname(os.path.abspath(__file__))

    if key in ["swiss", "cas", "sme"]:
        df = pd.read_csv(os.path.join(path, "data", key.lower() + "_train.csv"))
        sample_weight = df[["origin", "premium"]].drop_duplicates()
        columns = ["claim"]
    else:
        raise KeyError(
            "No such dataset exists. Available datasets are 'swiss', 'cas' and 'sme'."
        )

    claim = Triangle(
        df,
        origin="origin",
        development="development",
        index=None,
        columns=columns,
        cumulative=True,
        *args,
        **kwargs
    )
    sample_weight = Triangle(
        sample_weight,
        origin="origin",
        index=None,
        development=None,
        columns=["premium"],
        cumulative=True,
        *args,
        **kwargs
    )

    return TryangleData(claim, sample_weight)


def load_test_sample(key, *args, **kwargs):
    """Function to load test datasets included in the tryangle package.

    Args:
        key (str): Key to identify dataset

    Returns:
        [TryangleData]: TryangleData object of the loaded dataset.
    """
    path = os.path.dirname(os.path.abspath(__file__))

    df = pd.read_csv(os.path.join(path, "data", key.lower() + "_test.csv"))

    claim = Triangle(
        df,
        origin="origin",
        development="development",
        index=None,
        columns=["claim"],
        cumulative=True,
        *args,
        **kwargs
    )

    if key.lower() in ["swiss"]:
        claim = claim[claim.development <= 240]
        claim = claim[claim.origin.year <= 1997]
        claim = claim[claim.valuation.year > 1997]
    elif key.lower() in ["cas", "sme"]:
        claim = claim[claim.development <= 63]
        claim = claim[claim.origin.year <= 2014]
        claim = claim[claim.valuation.year > 2014]

    return TryangleData(claim)


if __name__ == "__main__":
    pass
