"""
data.py -- Dataset loading utilities for both pandas and PySpark.
"""

import pandas as pd
import click

from lsda.config import (
    TRAIN_FILE, TEST_FILE, ALL_COLS, FEATURE_COLS, LABEL_COL,
)


def load_pandas(split: str = "train") -> pd.DataFrame:
    """Load the Higgs CSV into a pandas DataFrame."""
    path = TRAIN_FILE if split == "train" else TEST_FILE
    click.echo(f"Loading {split} set from {path}...")
    df = pd.read_csv(path, header=None, names=ALL_COLS)
    df[LABEL_COL] = df[LABEL_COL].astype(int)
    click.echo(f"  {len(df):,} rows, {len(df.columns)} columns loaded.")
    return df


def load_spark(spark, split: str = "train"):
    """Load the Higgs CSV into a PySpark DataFrame."""
    from pyspark.sql.types import StructType, StructField, DoubleType

    path = str(TRAIN_FILE if split == "train" else TEST_FILE)
    click.echo(f"Loading {split} set into Spark from {path}...")

    schema = StructType(
        [StructField(c, DoubleType(), nullable=False) for c in ALL_COLS]
    )
    df = spark.read.csv(path, header=False, schema=schema)
    # Cast label to integer type for classifiers
    from pyspark.sql.functions import col
    df = df.withColumn(LABEL_COL, col(LABEL_COL).cast("int"))
    click.echo(f"  {df.count():,} rows loaded into Spark.")
    return df


def get_xy_pandas(df: pd.DataFrame):
    """Split a pandas DataFrame into X (features) and y (labels)."""
    return df[FEATURE_COLS], df[LABEL_COL]


def get_spark_session(app_name: str = "LSDA", n_cores: int | None = None):
    """Create (or retrieve) a SparkSession."""
    import os, sys
    from pyspark.sql import SparkSession

    python_exec = sys.executable
    os.environ.setdefault("PYSPARK_PYTHON", python_exec)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", python_exec)

    if os.name == "nt" and "HADOOP_HOME" not in os.environ:
        winutils_path = r"C:\hadoop\bin\winutils.exe"
        if os.path.exists(winutils_path):
            os.environ["HADOOP_HOME"] = r"C:\hadoop"
            os.environ["PATH"] = r"C:\hadoop\bin;" + os.environ.get("PATH", "")

    master = f"local[{n_cores}]" if n_cores else "local[*]"
    spark = (
        SparkSession.builder
        .master(master)
        .appName(app_name)
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.hadoop.hadoop.security.authentication", "simple")
        .config("spark.hadoop.hadoop.security.authorization", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark

