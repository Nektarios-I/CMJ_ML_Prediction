from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from .config import FIGURES_DIR


def plot_feature_importance(estimator, feature_names: list[str], save_name: str | None = None):
    mdi_importances = pd.Series(estimator.feature_importances_, index=feature_names).sort_values(ascending=True)
    ax = mdi_importances.plot.barh(figsize=(8, 6))
    ax.set_title("Random Forest Feature Importances (MDI)")
    ax.set_xlabel("Importance")
    ax.figure.tight_layout()

    if save_name:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        ax.figure.savefig(FIGURES_DIR / save_name, dpi=150)

    plt.show()
