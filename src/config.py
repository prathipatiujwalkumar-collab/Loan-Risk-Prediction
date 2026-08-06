"""
Configuration settings for the Loan Risk Prediction project.

This module centralizes all project paths and constants so that
the rest of the application can import them from a single location.
"""

from pathlib import Path

# --------------------------------------------------
# Project Root
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --------------------------------------------------
# Data Directories
# --------------------------------------------------

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

# --------------------------------------------------
# Artifacts
# --------------------------------------------------

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

MODELS_DIR = ARTIFACTS_DIR / "models"
ENCODERS_DIR = ARTIFACTS_DIR / "encoders"
SCALERS_DIR = ARTIFACTS_DIR / "scalers"
REPORTS_DIR = ARTIFACTS_DIR / "reports"

# --------------------------------------------------
# Logs
# --------------------------------------------------

LOG_DIR = PROJECT_ROOT / "logs"

# --------------------------------------------------
# MLflow
# --------------------------------------------------

MLFLOW_DIR = PROJECT_ROOT / "mlruns"

# --------------------------------------------------
# Dataset Files
# --------------------------------------------------

TRAIN_FILE = RAW_DATA_DIR / "application_train.csv"
# ==========================================
# Dataset Configuration
# ==========================================

TARGET_COLUMN = "TARGET"

# ==========================================
# Train/Test Split
# ==========================================

TEST_SIZE = 0.20
RANDOM_STATE = 42
STRATIFY = True
# ==========================================
# Model Configuration
# ==========================================

MODEL_NAME = "LogisticRegression"

MAX_ITER = 2000

N_JOBS = -1

MODEL_PATH = "models/logistic_regression.joblib"
# ==========================================
# Model Configuration
# ==========================================

RANDOM_STATE = 42

MAX_ITER = 2000

CLASS_WEIGHT = "balanced"

MODEL_PATH = "models/model.joblib"