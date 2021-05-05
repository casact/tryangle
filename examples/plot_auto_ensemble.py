"""
======================
AutoEnsemble CL and BF
======================

Finds optimal weights to combine chainladder and bornhuetter
ferguson methods to reduce prediction error.
"""
from tryangle.model_selection import TriangleSplit
from tryangle.utils.datasets import load_sample
from tryangle.ensemble import AutoEnsemble, Adam
from tryangle import Chainladder, BornhuetterFerguson

X = load_sample("swiss")

cl = Chainladder()
bf = BornhuetterFerguson(apriori=0.6)
estimators = [("cl", cl), ("bf", bf)]

tscv = TriangleSplit(n_splits=10)

model = AutoEnsemble(
    estimators=estimators,
    cv=tscv,
    optimizer=Adam(learning_rate=0.01),
    dropout=0.1,
    broad_dropout=0.1,
)

model.fit(X)

print(model.weights_)
