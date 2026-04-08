from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso


def select_lasso_features(X_train_n: pd.DataFrame, y_train: pd.Series, labels: list[str], alpha: float = 0.1) -> list[str]:
    lasso_model = Lasso(alpha=alpha)
    lasso_model.fit(X_train_n, y_train)

    lasso_betas = pd.DataFrame(np.round(lasso_model.coef_, 3), columns=["beta"], index=labels)
    selected = lasso_betas[lasso_betas["beta"] != 0].index.tolist()
    return selected
