"""
evaluate.py — Task 5: Performance evaluation & framework comparison.

Loads all 6 tuned models (3 sklearn + 3 PySpark), runs predictions on the
500 K test set, and produces:
  • Accuracy and ROC-AUC scores for each model
  • Comparison table (CSV + printed)
  • Bar chart comparing frameworks side-by-side
  • Markdown summary report
"""

import json
import click
import joblib
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, roc_auc_score

from lsda.config import (
    SKLEARN_MODELS_DIR, SPARK_MODELS_DIR, EVAL_DIR, MODEL_NAMES,
    FEATURE_COLS, LABEL_COL, ensure_dirs,
)
from lsda.data import load_pandas, get_xy_pandas, load_spark, get_spark_session
from lsda.pipelines.sklearn_pipe import transform


def run_evaluation() -> pd.DataFrame:
    """Evaluate all saved models and produce comparison outputs.

    Returns
    -------
    pd.DataFrame
        Evaluation results table.
    """
    ensure_dirs()
    rows: list[dict] = []

    # ── SciKit-Learn models ──
    click.echo("\n── Evaluating SciKit-Learn models ──")
    df_test = load_pandas("test")
    X_test, y_test = get_xy_pandas(df_test)

    # Load scaler
    pipe = joblib.load(SKLEARN_MODELS_DIR / "scaler_pipeline.joblib")
    X_test_scaled = transform(pipe, X_test)

    for key in ["lr", "rf", "gbt"]:
        model_path = SKLEARN_MODELS_DIR / f"{key}_best.joblib"
        if not model_path.exists():
            click.echo(f"  ⚠ {model_path} not found — skipping")
            continue

        model = joblib.load(model_path)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        click.echo(f"  {MODEL_NAMES[key]:30s} ACC={acc:.4f}  AUC={auc:.4f}")

        rows.append({
            "framework": "sklearn",
            "model": MODEL_NAMES[key],
            "accuracy": round(acc, 5),
            "roc_auc": round(auc, 5),
        })

    # ── PySpark models ──
    click.echo("\n── Evaluating PySpark models ──")
    spark = get_spark_session("LSDA-Eval")
    df_test_spark = load_spark(spark, "test")

    # Load feature pipeline
    from pyspark.ml import PipelineModel
    feat_model_path = str(SPARK_MODELS_DIR / "feature_pipeline")
    try:
        feat_model = PipelineModel.load(feat_model_path)
        df_test_feat = feat_model.transform(df_test_spark)
    except Exception as e:
        click.echo(f"  ⚠ Could not load feature pipeline: {e}")
        spark.stop()
        return _finalize(rows)

    from pyspark.ml.evaluation import BinaryClassificationEvaluator, \
        MulticlassClassificationEvaluator

    auc_eval = BinaryClassificationEvaluator(
        labelCol="label", rawPredictionCol="rawPrediction",
        metricName="areaUnderROC",
    )
    acc_eval = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction",
        metricName="accuracy",
    )

    for key in ["lr", "rf", "gbt"]:
        model_path = str(SPARK_MODELS_DIR / f"{key}_best")
        try:
            model = _load_spark_model(key, model_path)
        except Exception:
            click.echo(f"  ⚠ {model_path} not found — skipping")
            continue

        preds = model.transform(df_test_feat)
        acc = acc_eval.evaluate(preds)
        auc = auc_eval.evaluate(preds)
        click.echo(f"  {MODEL_NAMES[key]:30s} ACC={acc:.4f}  AUC={auc:.4f}")

        rows.append({
            "framework": "spark",
            "model": MODEL_NAMES[key],
            "accuracy": round(acc, 5),
            "roc_auc": round(auc, 5),
        })

    spark.stop()
    return _finalize(rows)


def _load_spark_model(key: str, path: str):
    """Load the correct Spark model type."""
    from pyspark.ml.classification import (
        LogisticRegressionModel,
        RandomForestClassificationModel,
        GBTClassificationModel,
    )
    loaders = {
        "lr": LogisticRegressionModel,
        "rf": RandomForestClassificationModel,
        "gbt": GBTClassificationModel,
    }
    return loaders[key].load(path)


def _finalize(rows: list[dict]) -> pd.DataFrame:
    """Save results, plot comparison, write report."""
    results = pd.DataFrame(rows)
    if results.empty:
        click.echo("⚠ No models evaluated.")
        return results

    # Save CSV
    csv_path = EVAL_DIR / "evaluation_results.csv"
    results.to_csv(csv_path, index=False)
    click.echo(f"\n✔ Results saved → {csv_path}")

    # Print table
    click.echo("\n" + results.to_string(index=False))

    # Bar chart
    _plot_comparison(results)

    # Markdown report
    _write_report(results)

    return results


def _plot_comparison(results: pd.DataFrame) -> None:
    """Side-by-side bar chart of accuracy and AUC by framework."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for metric, ax, title in [
        ("accuracy", axes[0], "Accuracy"),
        ("roc_auc", axes[1], "ROC-AUC"),
    ]:
        pivot = results.pivot(index="model", columns="framework", values=metric)
        pivot.plot.bar(ax=ax, edgecolor="black", alpha=0.85)
        ax.set_title(title, fontsize=13)
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_xlabel("")
        ax.set_ylim(0.5, 1.0)
        ax.legend(title="Framework")
        ax.tick_params(axis="x", rotation=15)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Framework Comparison — Test Set Performance", fontsize=15)
    fig.tight_layout()
    path = EVAL_DIR / "comparison_chart.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    click.echo(f"  Chart saved → {path}")


def _write_report(results: pd.DataFrame) -> None:
    """Generate a markdown summary report."""
    lines = [
        "# Evaluation Report",
        "",
        "## Performance Summary",
        "",
        results.to_markdown(index=False),
        "",
        "## Analysis",
        "",
        "### Predictive Consistency",
        "Compare accuracy and ROC-AUC between sklearn and PySpark for each classifier.",
        "Both frameworks should produce similar predictive performance since the ",
        "underlying algorithms are equivalent, confirming implementation consistency.",
        "",
        "### Time-Efficiency Trade-offs",
        "PySpark introduces overhead for job scheduling and data serialisation,",
        "which can make it slower than sklearn on small-to-medium datasets",
        "that fit in memory. However, PySpark scales horizontally for datasets",
        "that exceed single-machine memory, making it suitable for truly large-scale",
        "problems beyond what sklearn can handle.",
        "",
        "---",
        "*Report generated by the LSDA benchmark pipeline.*",
    ]
    report_path = EVAL_DIR / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    click.echo(f"  Report saved → {report_path}")
