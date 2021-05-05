import optuna
from optuna.visualization import (
    plot_optimization_history,
    plot_contour,
    plot_param_importances,
)
from joblib import Parallel, delayed
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from tryangle import Development, CapeCod
from tryangle.model_selection import TriangleSplit
from tryangle.utils.datasets import load_sample
from tryangle.metrics import neg_weighted_cdr_scorer

X = load_sample("swiss")
tscv = TriangleSplit(n_splits=13)


def objective(trial):

    dev_params = {
        "n_periods": trial.suggest_int("dev__n_periods", 5, 19),
        "drop_low": trial.suggest_categorical("dev__drop_low", [True, False]),
        "drop_high": trial.suggest_categorical("dev__drop_high", [True, False]),
    }
    cc_params = {
        "decay": trial.suggest_float("cc_decay", 0.0, 1.0),
        "trend": trial.suggest_float("cc_trend", -1.0, 1.0),
    }

    pipe = Pipeline(
        [
            ("dev", Development(**dev_params)),
            ("cc", CapeCod(**cc_params)),
        ]
    )

    score = cross_val_score(
        pipe, X=X, y=X, scoring=neg_weighted_cdr_scorer, cv=tscv, n_jobs=1
    ).mean()

    return score


def optimize(n_trials):
    study = optuna.load_study(
        study_name="opt_ibnr", storage="sqlite:///opt_ibnr.sqlite3"
    )
    study.optimize(objective, n_trials=n_trials)


study = optuna.create_study(
    study_name="opt_ibnr", storage="sqlite:///opt_ibnr.sqlite3", direction="maximize"
)
r = Parallel(n_jobs=-1)([delayed(optimize)(1) for _ in range(100)])

print("Number of finished trials: ", len(study.trials))
print("Best trial:")
trial = study.best_trial
print("  Value: ", trial.value)
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))


study = optuna.load_study(study_name="opt_ibnr", storage="sqlite:///opt_ibnr.sqlite3")

plot_optimization_history(study)

plot_contour(study)

plot_param_importances(study)
