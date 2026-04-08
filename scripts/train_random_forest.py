# pyright: reportMissingImports=false
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from jump_regressor.data import load_dataset, prepare_data
from jump_regressor.feature_selection import select_lasso_features
from jump_regressor.metrics import print_metrics, regression_metrics
from jump_regressor.models import get_estimators, get_random_forest_extended_grid
from jump_regressor.plotting import plot_feature_importance
from jump_regressor.training import save_model, train_with_grid_search


def main() -> None:
    df = load_dataset()
    prepared = prepare_data(df)

    selected_labels = select_lasso_features(prepared.X_train_n, prepared.y_train, prepared.labels)
    X_train = prepared.X_train_n[selected_labels]
    X_test = prepared.X_test_n[selected_labels]

    print(f"Initial Dataset: {df.shape[0]} jumps, {len(prepared.labels)} features")
    print(f"Train-Set: {X_train.shape[0]} jumps")
    print(f"Test-Set: {X_test.shape[0]} jumps")
    print(f"Lasso-selected features ({len(selected_labels)}): {selected_labels}\n")

    rf_estimator = get_estimators()["RandomForest"]
    best_model, best_params, _ = train_with_grid_search(rf_estimator, get_random_forest_extended_grid(), X_train, prepared.y_train)
    save_model(best_model, "RandomForest")

    y_pred = best_model.predict(X_test)
    metrics = regression_metrics(prepared.y_test, y_pred)

    print(f"Best params: {best_params}")
    print_metrics("RandomForest", metrics)
    plot_feature_importance(best_model, selected_labels, save_name="random_forest_importance.png")


if __name__ == "__main__":
    main()
