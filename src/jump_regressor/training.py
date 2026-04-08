from __future__ import annotations

import pickle

from sklearn.model_selection import GridSearchCV, cross_validate

from .config import MODELS_DIR


def train_with_grid_search(estimator, param_grid: dict, X, y):
    search = GridSearchCV(
        estimator,
        param_grid,
        cv=5,
        scoring="neg_mean_absolute_error",
        verbose=0,
        n_jobs=-1,
    )
    result = search.fit(X, y)
    best_params = result.best_params_

    best_estimator = estimator.set_params(**best_params)
    scoring = {
        "abs_error": "neg_mean_absolute_error",
        "squared_error": "neg_mean_squared_error",
        "r2": "r2",
    }
    cv_scores = cross_validate(best_estimator, X, y, cv=5, scoring=scoring, return_train_score=True)
    best_estimator.fit(X, y)
    return best_estimator, best_params, cv_scores


def save_model(model, model_name: str) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MODELS_DIR / f"{model_name}_Final.sav"
    with open(out_path, "wb") as f:
        pickle.dump(model, f)
