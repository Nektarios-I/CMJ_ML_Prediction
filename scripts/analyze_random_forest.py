# pyright: reportMissingImports=false
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from jump_regressor.data import load_dataset, prepare_data
from jump_regressor.feature_selection import select_lasso_features
from jump_regressor.plotting import plot_feature_importance


def main() -> None:
    df = load_dataset()
    prepared = prepare_data(df)

    selected_labels = select_lasso_features(prepared.X_train_n, prepared.y_train, prepared.labels)
    X_train = prepared.X_train_n[selected_labels]

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        bootstrap=True,
        random_state=42,
    )
    model.fit(X_train, prepared.y_train)

    plot_feature_importance(model, selected_labels, save_name="analysis_feature_importance.png")

    y_pred_train = model.predict(X_train)
    x_axis = list(range(0, len(y_pred_train) * 2, 2))

    plt.figure(figsize=(8, 5))
    plt.scatter(x_axis, y_pred_train, color="r", label="Predicted")
    plt.scatter(x_axis, prepared.y_train, color="b", label="Actual")
    plt.xlabel("Sample index (scaled)")
    plt.ylabel("Jump value")
    plt.title("Predicted vs Actual Values on Training Set")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
