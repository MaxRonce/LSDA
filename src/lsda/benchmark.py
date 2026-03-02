"""
benchmark.py — Task 4: Computational efficiency & scalability benchmarking.

Measures training and prediction times for each classifier as a function of
the number of CPU cores:
  • SciKit-Learn: controlled via ``n_jobs``
  • PySpark: controlled via ``local[k]`` master configuration

Produces:
  • ``benchmark_results.csv`` — raw timing data
  • ``speedup_sklearn.png`` / ``speedup_spark.png`` — speedup curves
"""

import time
import json
import click
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lsda.config import (
    BENCHMARK_CORES, BENCHMARK_DIR, MODEL_NAMES, RANDOM_STATE, ensure_dirs,
)
from lsda.data import load_pandas, get_xy_pandas, load_spark, get_spark_session
from lsda.pipelines.sklearn_pipe import build_pipeline as sk_build, fit_transform as sk_fit_transform
from lsda.pipelines.spark_pipe import build_pipeline as sp_build


def run_benchmark(cores: list[int] | None = None,
                  models: list[str] | None = None) -> pd.DataFrame:
    """Run the full scalability benchmark.

    Parameters
    ----------
    cores : list[int] | None
        CPU core counts to test. Defaults to ``config.BENCHMARK_CORES``.
    models : list[str] | None
        Model keys to benchmark. Defaults to all three.

    Returns
    -------
    pd.DataFrame
        Timing results with columns:
        ``framework, model, n_cores, train_time, predict_time``.
    """
    ensure_dirs()
    cores = cores or BENCHMARK_CORES
    models = models or ["lr", "rf", "gbt"]

    rows: list[dict] = []

    click.echo("\n╔══════════════════════════════════════════════╗")
    click.echo("║        Scalability Benchmark                 ║")
    click.echo("╚══════════════════════════════════════════════╝")

    # ── sklearn benchmarks ──
    click.echo("\n── SciKit-Learn ──")
    df_train = load_pandas("train")
    df_test = load_pandas("test")
    X_train, y_train = get_xy_pandas(df_train)
    X_test, y_test = get_xy_pandas(df_test)

    pipe = sk_build()
    X_train_s = sk_fit_transform(pipe, X_train)
    from lsda.pipelines.sklearn_pipe import transform as sk_transform
    X_test_s = sk_transform(pipe, X_test)

    for key in models:
        for k in cores:
            click.echo(f"  {MODEL_NAMES[key]} | n_jobs={k}")
            train_t, pred_t = _bench_sklearn(key, X_train_s, y_train,
                                              X_test_s, n_jobs=k)
            rows.append({
                "framework": "sklearn",
                "model": MODEL_NAMES[key],
                "n_cores": k,
                "train_time": round(train_t, 3),
                "predict_time": round(pred_t, 3),
            })

    # ── PySpark benchmarks ──
    click.echo("\n── PySpark ──")
    for key in models:
        for k in cores:
            click.echo(f"  {MODEL_NAMES[key]} | local[{k}]")
            train_t, pred_t = _bench_spark(key, k)
            rows.append({
                "framework": "spark",
                "model": MODEL_NAMES[key],
                "n_cores": k,
                "train_time": round(train_t, 3),
                "predict_time": round(pred_t, 3),
            })

    results = pd.DataFrame(rows)
    csv_path = BENCHMARK_DIR / "benchmark_results.csv"
    results.to_csv(csv_path, index=False)
    click.echo(f"\n✔ Results saved → {csv_path}")

    # Plots
    _plot_speedup(results, "sklearn")
    _plot_speedup(results, "spark")

    return results


# ------------------------------------------------------------------
# Sklearn benchmark helper
# ------------------------------------------------------------------

def _bench_sklearn(key: str, X_train, y_train, X_test,
                   n_jobs: int) -> tuple[float, float]:
    from lsda.models.sklearn_models import _get_estimator
    est = _get_estimator(key, n_jobs)

    t0 = time.perf_counter()
    est.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    est.predict(X_test)
    pred_time = time.perf_counter() - t0

    return train_time, pred_time


# ------------------------------------------------------------------
# PySpark benchmark helper
# ------------------------------------------------------------------

def _bench_spark(key: str, n_cores: int) -> tuple[float, float]:
    from pyspark.ml.classification import (
        LogisticRegression, RandomForestClassifier, GBTClassifier,
    )
    spark = get_spark_session(f"LSDA-Bench-{key}-{n_cores}", n_cores)
    df_train = load_spark(spark, "train")
    df_test = load_spark(spark, "test")

    feat_pipe = sp_build()
    feat_model = feat_pipe.fit(df_train)
    df_train_feat = feat_model.transform(df_train)
    df_test_feat = feat_model.transform(df_test)

    # Cache to avoid re-reading CSV during timing
    df_train_feat.cache().count()
    df_test_feat.cache().count()

    est = _get_spark_estimator(key)

    t0 = time.perf_counter()
    model = est.fit(df_train_feat)
    train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    model.transform(df_test_feat).count()  # force evaluation
    pred_time = time.perf_counter() - t0

    df_train_feat.unpersist()
    df_test_feat.unpersist()
    spark.stop()

    return train_time, pred_time


def _get_spark_estimator(key: str):
    from pyspark.ml.classification import (
        LogisticRegression, RandomForestClassifier, GBTClassifier,
    )
    if key == "lr":
        return LogisticRegression(
            labelCol="label", featuresCol="features",
            maxIter=200, family="binomial",
        )
    elif key == "rf":
        return RandomForestClassifier(
            labelCol="label", featuresCol="features",
            numTrees=100, seed=RANDOM_STATE,
        )
    elif key == "gbt":
        return GBTClassifier(
            labelCol="label", featuresCol="features",
            maxIter=100, seed=RANDOM_STATE,
        )
    else:
        raise ValueError(f"Unknown model key: {key}")


# ------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------

def _plot_speedup(results: pd.DataFrame, framework: str) -> None:
    """Plot speedup curves (T1 / Tk) for each model in a framework."""
    df = results[results["framework"] == framework]
    if df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for metric, ax, ylabel in [
        ("train_time", axes[0], "Training Time (s)"),
        ("predict_time", axes[1], "Prediction Time (s)"),
    ]:
        for model_name in df["model"].unique():
            sub = df[df["model"] == model_name].sort_values("n_cores")
            ax.plot(sub["n_cores"], sub[metric], "o-", label=model_name)
        ax.set_xlabel("Number of CPU Cores")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{framework.upper()} — {ylabel}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Scalability Benchmark — {framework.upper()}", fontsize=14)
    fig.tight_layout()
    path = BENCHMARK_DIR / f"speedup_{framework}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    click.echo(f"  Plot saved → {path}")
