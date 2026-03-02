"""
config.py -- Central configuration for paths, column names, and hyperparameter grids.
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
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


# The first column is the label (1 = signal, 0 = background).
# Columns 2-22 are low-level kinematic features.
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
    "m_jj",
    "m_jjj",
    "m_lv",
    "m_jlv",
    "m_bb",
    "m_wbb",
    "m_wwbb",
]

# All 28 features in column order
FEATURE_COLS = LOW_LEVEL_FEATURES + HIGH_LEVEL_FEATURES
ALL_COLS = [LABEL_COL] + FEATURE_COLS

USE_ALL_FEATURES = True


CV_FOLDS = 5
RANDOM_STATE = 42


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


BENCHMARK_CORES = [1, 2, 4, 8]



def ensure_dirs() -> None:
    """Create all output directories if they do not exist."""
    for d in (EDA_DIR, SKLEARN_MODELS_DIR, SPARK_MODELS_DIR, BENCHMARK_DIR, EVAL_DIR):
        d.mkdir(parents=True, exist_ok=True)
