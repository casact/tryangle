"""
=============================
GridSearchCV using a pipeline
=============================

Finds the optimal development and CapeCod parameters
using the unweighted CDR score.

Since selecting development factors is a transformation,
it can be pipelined with an estimator
"""
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from tryangle import Development, CapeCod
from tryangle.metrics import neg_cdr_scorer
from tryangle.model_selection import TriangleSplit
from tryangle.utils.datasets import load_sample

X = load_sample("swiss")
tscv = TriangleSplit(n_splits=5)

param_grid = {
    "dev__n_periods": range(15, 20),
    "dev__drop_high": [True, False],
    "dev__drop_low": [True, False],
    "cc__decay": [0.25, 0.5, 0.75, 0.95],
}

pipe = Pipeline([("dev", Development()), ("cc", CapeCod())])

model = GridSearchCV(
    pipe, param_grid=param_grid, scoring=neg_cdr_scorer, cv=tscv, verbose=1, n_jobs=-1
)
model.fit(X, X)
print(model.best_params_)

# TODO add plotting
