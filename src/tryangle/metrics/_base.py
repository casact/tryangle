# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

def get_expected(estimator, X):
    """Get the expected incremental claims for the next diagonal
        given an estimator.

    Args:
        estimator : tryangle implementation of chainladder method
        X : TryangleData

    Returns:
        ndarray : Array of expected incremental claims
    """
    valuation_date = X.latest_diagonal.valuation[0]
    X_test = X[X.triangle.valuation < valuation_date]

    expected = estimator.fit_predict(X_test)

    expected = expected.full_triangle_.cum_to_incr()
    expected = expected[expected.valuation == valuation_date].latest_diagonal

    return expected.to_frame().fillna(0).to_numpy()


def get_actual_expected(estimator, X):
    """Get actual and expected given an estimator

    Args:
        estimator : tryangle implementation of chainladder method
        X : TryangleData

    Returns:
        (ndarray, ndarray) : Tuple of actual, expected incremental claims
    """
    # ! TODO: Rewrite a better method that avoids fillna(0) as this could mask some error

    actual = X.latest_diagonal.to_frame().fillna(0).to_numpy()[:-1]
    expected = get_expected(estimator, X)

    return actual, expected
