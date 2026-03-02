# LSDA — Higgs Boson Classification Benchmark

**Large-Scale Data Analysis** project comparing SciKit-Learn vs PySpark ML on the Higgs boson signal/background classification task.

## Setup

```bash
# 1. Create & activate virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install the project in editable mode
pip install -e .
```

## Data

Place the dataset files in the `data/` directory:
- `HIGGS-train.csv` — 2,000,000 training examples
- `HIGGS-test.csv` — 500,000 test examples

Download link: https://utbox.univ-tours.fr/s/dXYJaT2x7fkaxGm

## CLI Usage

All commands are available via `python -m lsda`:

```bash
# Exploratory Data Analysis
python -m lsda eda

# Train models (with grid search or Optuna)
python -m lsda train --framework sklearn           # sklearn only
python -m lsda train --framework spark             # PySpark only
python -m lsda train --framework all               # both frameworks
python -m lsda train --framework sklearn --optuna   # Optuna optimisation
python -m lsda train --model lr                    # single classifier

# Scalability benchmark
python -m lsda benchmark --cores 1,2,4,8

# Evaluation & comparison report
python -m lsda evaluate

# Full pipeline (all steps sequentially)
python -m lsda run-all
python -m lsda run-all --optuna --cores 1,2,4
```

## Outputs

All outputs are written to the `outputs/` directory:

| Directory          | Contents                                      |
|--------------------|-----------------------------------------------|
| `outputs/eda/`     | Distribution plots, correlation heatmap, stats |
| `outputs/models/`  | Saved sklearn (.joblib) & Spark models         |
| `outputs/benchmarks/` | Timing CSV & speedup plots                  |
| `outputs/evaluation/` | Comparison table, chart & markdown report    |

## Classifiers

| # | Classifier              | SciKit-Learn                    | PySpark ML                        |
|---|-------------------------|---------------------------------|-----------------------------------|
| 1 | Logistic Regression     | `LogisticRegression`            | `LogisticRegression`              |
| 2 | Random Forest           | `RandomForestClassifier`        | `RandomForestClassifier`          |
| 3 | Gradient Boosted Trees  | `GradientBoostingClassifier`    | `GBTClassifier`                   |

## Project Structure

```
src/lsda/
├── cli.py               # Click CLI entrypoint
├── config.py            # Paths, column names, hyperparameters
├── data.py              # Data loading (pandas + Spark)
├── eda.py               # Exploratory data analysis
├── benchmark.py         # Scalability benchmarking
├── evaluate.py          # Model evaluation & comparison
├── pipelines/
│   ├── sklearn_pipe.py  # sklearn feature pipeline
│   └── spark_pipe.py    # PySpark feature pipeline
└── models/
    ├── sklearn_models.py  # sklearn training + tuning
    └── spark_models.py    # PySpark training + tuning
```
