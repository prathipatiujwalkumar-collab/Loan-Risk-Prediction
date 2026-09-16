"""
Configuration settings for the Loan Risk Prediction project.

Author: Ujwal
Project: Intelligent Loan Risk Assessment Platform
"""

from pathlib import Path

# ==================================================
# Project Root
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ==================================================
# Data Directories
# ==================================================

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

# ==================================================
# Artifact Directories
# ==================================================

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

MODELS_DIR = ARTIFACTS_DIR / "models"
ENCODERS_DIR = ARTIFACTS_DIR / "encoders"
SCALERS_DIR = ARTIFACTS_DIR / "scalers"
REPORTS_DIR = ARTIFACTS_DIR / "reports"

# ==================================================
# Logs
# ==================================================

LOG_DIR = PROJECT_ROOT / "logs"

# Compatibility alias used by logger tests/modules
LOG_FILE = LOG_DIR / "application.log"

# ==================================================
# MLflow
# ==================================================

MLFLOW_DIR = PROJECT_ROOT / "mlruns"
MLFLOW_EXPERIMENT = "Loan Risk Prediction"

# ==================================================
# Dataset Files
# ==================================================

TRAIN_FILE = RAW_DATA_DIR / "application_train.csv"

# ==================================================
# Dataset Configuration
# ==================================================

TARGET_COLUMN = "TARGET"

# ==================================================
# Train/Test Split
# ==================================================

TEST_SIZE = 0.20
RANDOM_STATE = 42
STRATIFY = True

# ==================================================
# Model Configuration
# ==================================================

MAX_ITER = 2000
CLASS_WEIGHT = "balanced"
N_JOBS = -1

# ==================================================
# Model Paths
# ==================================================

PRODUCTION_MODEL = MODELS_DIR / "production_model.joblib"
MODEL_PATH = PRODUCTION_MODEL

# ==================================================
# Reports
# ==================================================

REPORT_FILE = REPORTS_DIR / "evaluation_report.txt"

# ==================================================
# Cross Validation / Evaluation
# ==================================================

CV_FOLDS = 3
SCORING = "roc_auc"

# ==================================================
# Hyperparameter Tuning
# ==================================================

N_ITER_SEARCH = 3

# Scoring metric used during hyperparameter optimization
TUNING_SCORING = SCORING

# ==================================================
# Default Model Configuration
# ==================================================

RF_N_ESTIMATORS = 200
RF_MAX_DEPTH = 15

# ==================================================
# Logistic Regression Search Space
# ==================================================

LR_C_MIN = 0.01
LR_C_RANGE = 10

# ==================================================
# Random Forest Search Space
# ==================================================

RF_ESTIMATOR_MIN = 100
RF_ESTIMATOR_MAX = 500

RF_DEPTH_MIN = 5
RF_DEPTH_MAX = 30

RF_MIN_SPLIT_MIN = 2
RF_MIN_SPLIT_MAX = 10

RF_MIN_LEAF_MIN = 1
RF_MIN_LEAF_MAX = 5

# ==================================================
# Create Required Directories
# ==================================================

PROJECT_DIRECTORIES = [
    RAW_DATA_DIR,
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    EXTERNAL_DATA_DIR,
    ARTIFACTS_DIR,
    MODELS_DIR,
    ENCODERS_DIR,
    SCALERS_DIR,
    REPORTS_DIR,
    LOG_DIR,
    MLFLOW_DIR,
]

for directory in PROJECT_DIRECTORIES:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )