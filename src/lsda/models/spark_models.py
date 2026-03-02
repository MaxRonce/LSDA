"""
spark_models.py -- PySpark ML model training, tuning and cross-validation.

Three classifiers: Logistic Regression, Random Forest, Gradient Boosted Trees.
Uses CrossValidator with ParamGridBuilder for K-fold tuning.
Optionally uses Optuna when --optuna is passed.
"""

import time
import json
import click

from pyspark.ml.classification import (
    LogisticRegression,
    RandomForestClassifier,
    GBTClassifier,
)
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator

from lsda.config import (
    PARAM_GRIDS, OPTUNA_SEARCH_SPACES, OPTUNA_N_TRIALS, OPTUNA_TIMEOUT,
    CV_FOLDS, RANDOM_STATE, SPARK_MODELS_DIR, MODEL_NAMES, ensure_dirs,
)
from lsda.data import load_spark, get_spark_session
from lsda.pipelines.spark_pipe import build_pipeline





def train_all(models: list[str] | None = None, use_optuna: bool = False,
              n_cores: int | None = None) -> dict:
    """Train, tune, and save all requested PySpark models."""
    ensure_dirs()
    models = models or ["lr", "rf", "gbt"]

    spark = get_spark_session("LSDA-Train", n_cores)
    df_train = load_spark(spark, "train")

    # Feature pipeline
    feat_pipe = build_pipeline()
    feat_model = feat_pipe.fit(df_train)
    df_train_feat = feat_model.transform(df_train)
    # Save the fitted feature pipeline
    feat_model.write().overwrite().save(str(SPARK_MODELS_DIR / "feature_pipeline"))

    evaluator = BinaryClassificationEvaluator(
        labelCol="label", rawPredictionCol="rawPrediction",
        metricName="areaUnderROC",
    )

    results = {}
    for key in models:
        click.echo(f"\n{'='*60}")
        click.echo(f"  Training PySpark -- {MODEL_NAMES[key]}")
        click.echo(f"{'='*60}")

        if use_optuna:
            result = _train_optuna(key, df_train_feat, evaluator, spark)
        else:
            result = _train_grid(key, df_train_feat, evaluator)

        results[key] = result

    # Save summary
    summary_path = SPARK_MODELS_DIR / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    click.echo(f"\nTraining summary saved to {summary_path}")

    spark.stop()
    return results





def _get_estimator(key: str):
    if key == "lr":
        return LogisticRegression(
            labelCol="label", featuresCol="features",
            maxIter=200, family="binomial",
        )
    elif key == "rf":
        return RandomForestClassifier(
            labelCol="label", featuresCol="features",
            seed=RANDOM_STATE,
        )
    elif key == "gbt":
        return GBTClassifier(
            labelCol="label", featuresCol="features",
            seed=RANDOM_STATE,
        )
    else:
        raise ValueError(f"Unknown model key: {key}")





def _build_param_grid(key: str, estimator):
    """Build a PySpark ParamGrid from config."""
    builder = ParamGridBuilder()
    grid = PARAM_GRIDS[key]

    if key == "lr":
        builder.addGrid(estimator.regParam,
                        [1.0 / c for c in grid["C"]])
    elif key == "rf":
        builder.addGrid(estimator.numTrees, grid["n_estimators"])
        builder.addGrid(estimator.maxDepth, grid["max_depth"])
    elif key == "gbt":
        builder.addGrid(estimator.maxIter, grid["n_estimators"])
        builder.addGrid(estimator.maxDepth, grid["max_depth"])
        builder.addGrid(estimator.stepSize, grid["learning_rate"])

    return builder.build()


def _train_grid(key: str, df_train, evaluator) -> dict:
    estimator = _get_estimator(key)
    param_grid = _build_param_grid(key, estimator)

    cv = CrossValidator(
        estimator=estimator,
        estimatorParamMaps=param_grid,
        evaluator=evaluator,
        numFolds=CV_FOLDS,
        parallelism=2,
        seed=RANDOM_STATE,
    )

    click.echo(f"  CrossValidator -- {len(param_grid)} param combos x {CV_FOLDS} folds")

    t0 = time.perf_counter()
    cv_model = cv.fit(df_train)
    train_time = time.perf_counter() - t0

    best_score = max(cv_model.avgMetrics)
    click.echo(f"  Best CV AUC : {best_score:.4f}")
    click.echo(f"  Train time  : {train_time:.1f}s")

    # Save best model
    model_path = str(SPARK_MODELS_DIR / f"{key}_best")
    cv_model.bestModel.write().overwrite().save(model_path)
    click.echo(f"  Model saved to {model_path}")

    return {
        "best_params": "see saved model metadata",
        "cv_score": round(best_score, 5),
        "train_time": round(train_time, 2),
    }





def _train_optuna(key: str, df_train, evaluator, spark) -> dict:
    """Optuna-based tuning for PySpark models."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    search_space = OPTUNA_SEARCH_SPACES[key]
    # Cache the df
    df_train.cache()

    # We'll do manual k-fold splits
    splits = df_train.randomSplit(
        [1.0] * CV_FOLDS, seed=RANDOM_STATE
    )

    def objective(trial):
        params = {}
        for name, spec in search_space.items():
            kind, low, high = spec
            if kind == "int":
                params[name] = trial.suggest_int(name, low, high)
            elif kind == "float":
                params[name] = trial.suggest_float(name, low, high)
            elif kind == "log_float":
                params[name] = trial.suggest_float(name, low, high, log=True)

        # Map params to Spark estimator
        est = _get_estimator(key)
        _set_spark_params(key, est, params)

        # Manual K-fold
        auc_scores = []
        for i in range(CV_FOLDS):
            val_df = splits[i]
            train_df = df_train.subtract(val_df)
            model = est.fit(train_df)
            preds = model.transform(val_df)
            auc = evaluator.evaluate(preds)
            auc_scores.append(auc)

        return sum(auc_scores) / len(auc_scores)

    click.echo(f"  Optuna search -- {OPTUNA_N_TRIALS} trials, "
               f"{OPTUNA_TIMEOUT}s timeout")

    t0 = time.perf_counter()
    study = optuna.create_study(direction="maximize",
                                study_name=f"spark_{key}")
    study.optimize(objective, n_trials=OPTUNA_N_TRIALS, timeout=OPTUNA_TIMEOUT)
    train_time = time.perf_counter() - t0

    best = study.best_trial
    click.echo(f"  Best params : {best.params}")
    click.echo(f"  Best CV AUC : {best.value:.4f}")
    click.echo(f"  Train time  : {train_time:.1f}s")

    # Refit on full data
    est = _get_estimator(key)
    _set_spark_params(key, est, best.params)
    final_model = est.fit(df_train)

    model_path = str(SPARK_MODELS_DIR / f"{key}_best")
    final_model.write().overwrite().save(model_path)
    click.echo(f"  Model saved to {model_path}")

    df_train.unpersist()

    return {
        "best_params": best.params,
        "cv_score": round(best.value, 5),
        "train_time": round(train_time, 2),
    }


def _set_spark_params(key: str, estimator, params: dict) -> None:
    """Apply generic param dict to a Spark estimator."""
    if key == "lr":
        if "C" in params:
            estimator.setRegParam(1.0 / params["C"])
    elif key == "rf":
        if "n_estimators" in params:
            estimator.setNumTrees(params["n_estimators"])
        if "max_depth" in params:
            estimator.setMaxDepth(params["max_depth"])
    elif key == "gbt":
        if "n_estimators" in params:
            estimator.setMaxIter(params["n_estimators"])
        if "max_depth" in params:
            estimator.setMaxDepth(params["max_depth"])
        if "learning_rate" in params:
            estimator.setStepSize(params["learning_rate"])
