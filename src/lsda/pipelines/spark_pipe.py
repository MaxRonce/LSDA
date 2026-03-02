"""
spark_pipe.py -- PySpark ML feature engineering pipeline.

VectorAssembler collects all 28 feature columns into a single vector,
then StandardScaler normalises it.
"""

from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler

from lsda.config import FEATURE_COLS


def build_pipeline(feature_cols: list[str] | None = None) -> Pipeline:
    """Return an un-fitted PySpark preprocessing pipeline."""
    cols = feature_cols or FEATURE_COLS

    assembler = VectorAssembler(inputCols=cols, outputCol="raw_features")
    scaler = StandardScaler(
        inputCol="raw_features",
        outputCol="features",
        withMean=True,
        withStd=True,
    )
    return Pipeline(stages=[assembler, scaler])
