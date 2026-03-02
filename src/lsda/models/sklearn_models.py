"""
sklearn_models.py -- SciKit-Learn model training, tuning and cross-validation.

Supports grid search (default) and Optuna-based Bayesian optimisation.
Three classifiers: Logistic Regression, Random Forest, Gradient Boosted Trees.
"""

import time
import json
import click
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score

from lsda.config import (
    PARAM_GRIDS, OPTUNA_SEARCH_SPACES, OPTUNA_N_TRIALS, OPTUNA_TIMEOUT,
    CV_FOLDS, RANDOM_STATE, SKLEARN_MODELS_DIR, MODEL_NAMES, ensure_dirs,
)
from lsda.data import load_pandas, get_xy_pandas
from lsda.pipelines.sklearn_pipe import build_pipeline, fit_transform, transform




def train_all(models: list[str] | None = None, use_optuna: bool = False,
              n_jobs: int = -1) -> dict:
    """Train, tune, and save all requested sklearn models."""
    ensure_dirs()
    models = models or ["lr", "rf", "gbt"]

    # Load data
    df_train = load_pandas("train")
    X_train, y_train = get_xy_pandas(df_train)

    # Feature pipeline
    pipe = build_pipeline()
    X_train_scaled = fit_transform(pipe, X_train)
    # Save the scaler for later use in evaluation
    joblib.dump(pipe, SKLEARN_MODELS_DIR / "scaler_pipeline.joblib")

    results = {}
    for key in models:
        click.echo(f"\n{'='*60}")
        click.echo(f"  Training sklearn -- {MODEL_NAMES[key]}")
        click.echo(f"{'='*60}")

        if use_optuna:
            result = _train_optuna(key, X_train_scaled, y_train, n_jobs)
        else:
            result = _train_grid(key, X_train_scaled, y_train, n_jobs)

        results[key] = result

    # Save summary
    summary_path = SKLEARN_MODELS_DIR / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    click.echo(f"\nTraining summary saved to {summary_path}")

    return results





def _get_estimator(key: str, n_jobs: int = -1):
    """Instantiate a fresh estimator by key."""
    if key == "lr":
        return LogisticRegression(
            random_state=RANDOM_STATE, solver="saga", max_iter=200
        )
    elif key == "rf":
        return RandomForestClassifier(
            random_state=RANDOM_STATE, n_jobs=n_jobs,
            max_samples=0.5,
        )
    elif key == "gbt":
        return HistGradientBoostingClassifier(
            random_state=RANDOM_STATE, early_stopping=True,
        )
    else:
        raise ValueError(f"Unknown model key: {key}")


def _train_grid(key: str, X: pd.DataFrame, y: pd.Series,
                n_jobs: int) -> dict:
    """Train using GridSearchCV."""
    estimator = _get_estimator(key, n_jobs)
    param_grid = PARAM_GRIDS[key]

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True,
                         random_state=RANDOM_STATE)

    click.echo(f"  Grid search over {param_grid}")
    gs = GridSearchCV(
        estimator, param_grid, cv=cv, scoring="roc_auc",
        n_jobs=n_jobs, verbose=1, refit=True,
    )

    t0 = time.perf_counter()
    gs.fit(X, y)
    train_time = time.perf_counter() - t0

    click.echo(f"  Best params : {gs.best_params_}")
    click.echo(f"  Best CV AUC : {gs.best_score_:.4f}")
    click.echo(f"  Train time  : {train_time:.1f}s")

    # Save model
    model_path = SKLEARN_MODELS_DIR / f"{key}_best.joblib"
    joblib.dump(gs.best_estimator_, model_path)
    click.echo(f"  Model saved to {model_path}")

    # Save CV results
    cv_df = pd.DataFrame(gs.cv_results_)
    cv_df.to_csv(SKLEARN_MODELS_DIR / f"{key}_cv_results.csv", index=False)

    return {
        "best_params": gs.best_params_,
        "cv_score": round(gs.best_score_, 5),
        "train_time": round(train_time, 2),
    }





OPTUNA_SUBSAMPLE = 100_000
OPTUNA_REFIT_SUBSAMPLE = 500_000
OPTUNA_CV_FOLDS = 3

def _train_optuna(key: str, X: pd.DataFrame, y: pd.Series,
                  n_jobs: int) -> dict:
    """Train using Optuna Bayesian optimisation.

    Sub-samples rows and uses 3-fold CV during the optimisation loop,
    then refits the best params on a larger subset.
    """
    import optuna
    from lsda import config  # read at call-time so CLI overrides apply
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    search_space = config.OPTUNA_SEARCH_SPACES[key]

    # Stratified subsample for fast search
    from sklearn.model_selection import train_test_split
    n = min(OPTUNA_SUBSAMPLE, len(X))
    if n < len(X):
        X_sub, _, y_sub, _ = train_test_split(
            X, y, train_size=n, stratify=y, random_state=RANDOM_STATE,
        )
        click.echo(f"  Subsampled to {n:,} rows for Optuna search")
    else:
        X_sub, y_sub = X, y

    cv = StratifiedKFold(n_splits=OPTUNA_CV_FOLDS, shuffle=True,
                         random_state=RANDOM_STATE)

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

        est = _get_estimator(key, n_jobs)
        est.set_params(**params)
        scores = cross_val_score(est, X_sub, y_sub, cv=cv,
                                 scoring="roc_auc", n_jobs=n_jobs)
        return scores.mean()

    n_trials = config.OPTUNA_N_TRIALS
    timeout = config.OPTUNA_TIMEOUT
    click.echo(f"  Optuna search -- {n_trials} trials, {timeout}s timeout, "
               f"{OPTUNA_CV_FOLDS}-fold CV on {len(X_sub):,} rows")

    t0 = time.perf_counter()
    study = optuna.create_study(direction="maximize",
                                study_name=f"sklearn_{key}")
    study.optimize(objective, n_trials=n_trials,
                   timeout=timeout, show_progress_bar=True)
    search_time = time.perf_counter() - t0

    best = study.best_trial
    click.echo(f"  Best params : {best.params}")
    click.echo(f"  Best CV AUC (subsample): {best.value:.4f}")
    click.echo(f"  Search time : {search_time:.1f}s")

    # Refit with best params on a manageable subset
    from sklearn.model_selection import train_test_split as tts2
    n_refit = min(OPTUNA_REFIT_SUBSAMPLE, len(X))
    if n_refit < len(X):
        X_refit, _, y_refit, _ = tts2(
            X, y, train_size=n_refit, stratify=y, random_state=RANDOM_STATE,
        )
        click.echo(f"  Refitting on {n_refit:,} rows (subsampled)...")
    else:
        X_refit, y_refit = X, y
        click.echo(f"  Refitting on full {len(X):,} rows...")
    est = _get_estimator(key, n_jobs)
    est.set_params(**best.params)
    t0 = time.perf_counter()
    est.fit(X_refit, y_refit)
    refit_time = time.perf_counter() - t0
    total_time = search_time + refit_time

    model_path = SKLEARN_MODELS_DIR / f"{key}_best.joblib"
    joblib.dump(est, model_path)
    click.echo(f"  Refit time  : {refit_time:.1f}s")
    click.echo(f"  Total time  : {total_time:.1f}s")
    click.echo(f"  Model saved to {model_path}")

    return {
        "best_params": best.params,
        "cv_score": round(best.value, 5),
        "train_time": round(total_time, 2),
    }
