"""
data.py — Dataset loading utilities for both pandas and PySpark.

Provides consistent column naming and typing across frameworks.
"""

import pandas as pd
import click

from lsda.config import (
    TRAIN_FILE, TEST_FILE, ALL_COLS, FEATURE_COLS, LABEL_COL,
)


def load_pandas(split: str = "train") -> pd.DataFrame:
    """Load the Higgs CSV into a pandas DataFrame.

    Parameters
    ----------
    split : str
        ``"train"`` (2 M rows) or ``"test"`` (500 K rows).

    Returns
    -------
    pd.DataFrame
        DataFrame with named columns (``label`` + 28 features).
    """
    path = TRAIN_FILE if split == "train" else TEST_FILE
    click.echo(f"Loading {split} set from {path} …")
    df = pd.read_csv(path, header=None, names=ALL_COLS)
    df[LABEL_COL] = df[LABEL_COL].astype(int)
    click.echo(f"  → {len(df):,} rows, {len(df.columns)} columns loaded.")
    return df


def load_spark(spark, split: str = "train"):
    """Load the Higgs CSV into a PySpark DataFrame.

    Parameters
    ----------
    spark : pyspark.sql.SparkSession
        Active Spark session.
    split : str
        ``"train"`` or ``"test"``.

    Returns
    -------
    pyspark.sql.DataFrame
        Spark DataFrame with named columns.
    """
    from pyspark.sql.types import StructType, StructField, DoubleType

    path = str(TRAIN_FILE if split == "train" else TEST_FILE)
    click.echo(f"Loading {split} set into Spark from {path} …")

    schema = StructType(
        [StructField(c, DoubleType(), nullable=False) for c in ALL_COLS]
    )
    df = spark.read.csv(path, header=False, schema=schema)
    # Cast label to integer type for classifiers
    from pyspark.sql.functions import col
    df = df.withColumn(LABEL_COL, col(LABEL_COL).cast("int"))
    click.echo(f"  → {df.count():,} rows loaded into Spark.")
    return df


def get_xy_pandas(df: pd.DataFrame):
    """Split a pandas DataFrame into X (features) and y (labels).

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
    """
    return df[FEATURE_COLS], df[LABEL_COL]


def get_spark_session(app_name: str = "LSDA", n_cores: int | None = None):
    """Create (or retrieve) a SparkSession.

    Parameters
    ----------
    app_name : str
        Application name shown in Spark UI.
    n_cores : int | None
        If given, sets ``local[k]`` master. If ``None``, uses ``local[*]``.

    Notes
    -----
    Java 17+/23 compatibility is handled via
    ``pyspark/conf/spark-defaults.conf`` (``--add-opens`` flags).
    """
    import os, sys
    from pyspark.sql import SparkSession

    # Point Spark workers to this venv's Python (avoids 'python3 not found')
    python_exec = sys.executable
    os.environ.setdefault("PYSPARK_PYTHON", python_exec)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", python_exec)

    # On Windows, Hadoop needs winutils.exe for filesystem ops (mkdir, chmod).
    # Set HADOOP_HOME automatically if the standard location exists.
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
        # ── Java 23 compatibility ─────────────────────────────────────────
        # Hadoop's UserGroupInformation calls Subject.getSubject() which was
        # removed in Java 23. Using simple auth avoids that code path.
        .config("spark.hadoop.hadoop.security.authentication", "simple")
        .config("spark.hadoop.hadoop.security.authorization", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark



