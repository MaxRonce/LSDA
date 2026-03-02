"""
config.py — Central configuration for paths, column names, and hyperparameter grids.

All tuneable constants live here so that every module stays in sync.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # …/LSDA
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

TRAIN_FILE = DATA_DIR / "HIGGS-train.csv"
TEST_FILE = DATA_DIR / "HIGGS-test.csv"

# Output sub-directories (created on demand)
EDA_DIR = OUTPUT_DIR / "eda"
MODELS_DIR = OUTPUT_DIR / "models"
SKLEARN_MODELS_DIR = MODELS_DIR / "sklearn"
SPARK_MODELS_DIR = MODELS_DIR / "spark"
BENCHMARK_DIR = OUTPUT_DIR / "benchmarks"
EVAL_DIR = OUTPUT_DIR / "evaluation"

# ---------------------------------------------------------------------------
# Dataset schema
# ---------------------------------------------------------------------------
# The first column is the label (1 = signal, 0 = background).
# Columns 2-22 are low-level kinematic features measured by the detector.
# Columns 23-29 are high-level physics-derived features.
LABEL_COL = "label"

LOW_LEVEL_FEATURES = [
    "lepton_pT", "lepton_eta", "lepton_phi",
    "missing_energy_magnitude", "missing_energy_phi",
    "jet1_pt", "jet1_eta", "jet1_phi", "jet1_b_tag",
    "jet2_pt", "jet2_eta", "jet2_phi", "jet2_b_tag",
    "jet3_pt", "jet3_eta", "jet3_phi", "jet3_b_tag",
    "jet4_pt", "jet4_eta", "jet4_phi", "jet4_b_tag",
]

HIGH_LEVEL_FEATURES = [
    "m_jj",       # invariant mass of jet-jet system
    "m_jjj",      # invariant mass of jet-jet-jet system
    "m_lv",       # invariant mass of lepton-neutrino system
    "m_jlv",      # invariant mass of jet-lepton-neutrino system
    "m_bb",       # invariant mass of b-tagged jet pair
    "m_wbb",      # invariant mass of W+bb system
    "m_wwbb",     # invariant mass of WW+bb system
]

# All 28 features in column order
FEATURE_COLS = LOW_LEVEL_FEATURES + HIGH_LEVEL_FEATURES
ALL_COLS = [LABEL_COL] + FEATURE_COLS

# ---------------------------------------------------------------------------
# Feature selection justification
# ---------------------------------------------------------------------------
# We keep ALL 28 features by default.
#
# Justification:
# • The 21 low-level kinematics are direct detector measurements — each one
#   encodes a potentially unique physical signature of Higgs production.
# • The 7 high-level features are physics-motivated invariant-mass
#   combinations specifically designed to separate signal from background.
#   Dropping them would discard expert domain knowledge.
# • Since the dataset has only 28 numeric features (no categoricals, no
#   high-cardinality columns) there is negligible risk of the "curse of
#   dimensionality", and tree-based models handle irrelevant features
#   gracefully via feature-importance pruning.
# • Empirically, the original paper shows that using ALL features gives the
#   best AUC, confirming that none are redundant enough to exclude a priori.
#
# If feature selection is desired (e.g. to study the value of high-level
# features), the pipeline accepts a configurable subset via the
# `--features` CLI flag.
USE_ALL_FEATURES = True

# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------
CV_FOLDS = 5
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Hyperparameter search grids (shared between sklearn & PySpark)
# ---------------------------------------------------------------------------
PARAM_GRIDS = {
    "lr": {
        "C": [0.01, 0.1, 1.0, 10.0],
        "max_iter": [200],
    },
    "rf": {
        "n_estimators": [50, 100],
        "max_depth": [8, 12],
    },
    "gbt": {
        "max_iter": [100, 200],
        "max_depth": [3, 5],
        "learning_rate": [0.05, 0.1],
    },
}

# Mapping from short name to display name
MODEL_NAMES = {
    "lr": "Logistic Regression",
    "rf": "Random Forest",
    "gbt": "Gradient Boosted Trees",
}

# ---------------------------------------------------------------------------
# Optuna tuning (optional, activated via --optuna flag)
# ---------------------------------------------------------------------------
OPTUNA_N_TRIALS = 10
OPTUNA_TIMEOUT = 60  # seconds per model

OPTUNA_SEARCH_SPACES = {
    "lr": {
        "C": ("log_float", 1e-3, 100.0),
    },
    "rf": {
        "n_estimators": ("int", 50, 150),
        "max_depth": ("int", 5, 15),
    },
    "gbt": {
        "max_iter": ("int", 50, 200),
        "max_depth": ("int", 2, 8),
        "learning_rate": ("log_float", 0.01, 0.3),
    },
}

# ---------------------------------------------------------------------------
# Benchmark core counts
# ---------------------------------------------------------------------------
BENCHMARK_CORES = [1, 2, 4, 8]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ensure_dirs() -> None:
    """Create all output directories if they do not exist."""
    for d in (EDA_DIR, SKLEARN_MODELS_DIR, SPARK_MODELS_DIR, BENCHMARK_DIR, EVAL_DIR):
        d.mkdir(parents=True, exist_ok=True)
