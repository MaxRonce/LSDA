"""
spark_pipe.py — Task 2b: PySpark ML feature engineering pipeline.

Mirrors the SciKit-Learn pipeline exactly:
  1. VectorAssembler — collects all 28 feature columns into a single vector.
  2. StandardScaler — normalises the assembled vector.
"""

from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler

from lsda.config import FEATURE_COLS


def build_pipeline(feature_cols: list[str] | None = None) -> Pipeline:
    """Return an un-fitted PySpark preprocessing pipeline.

    Parameters
    ----------
    feature_cols : list[str] | None
        Columns to assemble. Defaults to all 28 features.
    """
    cols = feature_cols or FEATURE_COLS

    assembler = VectorAssembler(inputCols=cols, outputCol="raw_features")
    scaler = StandardScaler(
        inputCol="raw_features",
        outputCol="features",
        withMean=True,
        withStd=True,
    )
    return Pipeline(stages=[assembler, scaler])
