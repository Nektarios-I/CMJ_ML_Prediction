from __future__ import annotations

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import BayesianRidge, ElasticNet, LassoLars, LinearRegression, Ridge, SGDRegressor, TweedieRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

from .config import RANDOM_STATE


def get_estimators() -> dict:
    return {
        "MLP": MLPRegressor(max_iter=20000, n_iter_no_change=10, hidden_layer_sizes=(50,), activation="relu", solver="adam", random_state=RANDOM_STATE),
        "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=None, min_samples_split=2, random_state=RANDOM_STATE, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=RANDOM_STATE),
        "SVR": SVR(kernel="rbf", C=1.0, epsilon=0.1),
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "ElasticNet": ElasticNet(alpha=1.0, l1_ratio=0.5),
        "LassoLars": LassoLars(alpha=1.0),
        "BayesianRidge": BayesianRidge(),
        "TweedieRegressor": TweedieRegressor(power=1, alpha=0.5, link="auto"),
        "SGDRegressor": SGDRegressor(loss="squared_error", learning_rate="constant", alpha=0.0001, max_iter=1000, tol=1e-3, random_state=RANDOM_STATE),
    }


def get_param_grid(name: str) -> dict:
    grids = {
        "MLP": {
            "activation": ["identity", "logistic", "tanh", "relu"],
            "solver": ["lbfgs", "sgd", "adam"],
            "hidden_layer_sizes": [(i,) for i in range(1, 17)],
        },
        "RandomForest": {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 10, 20, 30],
        },
        "GradientBoosting": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 4, 5],
        },
        "SVR": {
            "C": [0.1, 1, 10],
            "epsilon": [0.01, 0.1, 1],
            "kernel": ["linear", "poly", "rbf"],
        },
        "LinearRegression": {},
        "Ridge": {"alpha": [0.1, 1.0, 10.0]},
        "ElasticNet": {"alpha": [0.1, 1.0, 10.0], "l1_ratio": [0.1, 0.5, 0.9]},
        "LassoLars": {"alpha": [0.1, 1.0, 10.0]},
        "BayesianRidge": {},
        "TweedieRegressor": {"power": [0, 1, 2], "alpha": [0.1, 0.5, 1.0], "link": ["auto", "identity", "log"]},
        "SGDRegressor": {
            "alpha": [0.0001, 0.001, 0.01],
            "loss": ["squared_epsilon_insensitive", "squared_error", "epsilon_insensitive", "huber"],
            "learning_rate": ["constant", "optimal"],
        },
    }
    return grids[name]


def get_random_forest_extended_grid() -> dict:
    return {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "bootstrap": [True, False],
    }
