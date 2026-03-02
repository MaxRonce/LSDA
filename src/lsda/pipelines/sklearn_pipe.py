"""
sklearn_pipe.py -- SciKit-Learn feature engineering pipeline (StandardScaler).
"""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd

from lsda.config import FEATURE_COLS


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
    ])


def fit_transform(pipeline: Pipeline, X_train: pd.DataFrame) -> pd.DataFrame:
    """Fit the pipeline on training data and return scaled features."""
    return pd.DataFrame(
        pipeline.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )


def transform(pipeline: Pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """Apply an already-fitted pipeline to new data."""
    return pd.DataFrame(
        pipeline.transform(X),
        columns=X.columns,
        index=X.index,
    )
