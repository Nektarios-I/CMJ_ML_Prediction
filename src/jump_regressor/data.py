from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import DATA_CANDIDATES, RANDOM_STATE, START_FEATURE_INDEX, TARGET_COLUMN, TEST_SIZE


@dataclass
class PreparedData:
    labels: list[str]
    X_train_n: pd.DataFrame
    X_test_n: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def resolve_data_path() -> Path:
    for candidate in DATA_CANDIDATES:
        if candidate.exists():
            return candidate
    checked = "\n".join(str(path) for path in DATA_CANDIDATES)
    raise FileNotFoundError(f"Could not find features_round.csv. Checked:\n{checked}")


def load_dataset() -> pd.DataFrame:
    data_path = resolve_data_path()
    return pd.read_csv(data_path)


def prepare_data(df: pd.DataFrame) -> PreparedData:
    labels = list(df.columns)[START_FEATURE_INDEX:]
    X_train, X_test, y_train, y_test = train_test_split(
        df[labels],
        df[TARGET_COLUMN],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    X_train_n = (X_train - X_train.mean()) / X_train.std()
    X_test_n = (X_test - X_train.mean()) / X_train.std()

    return PreparedData(
        labels=labels,
        X_train_n=X_train_n,
        X_test_n=X_test_n,
        y_train=y_train,
        y_test=y_test,
    )
