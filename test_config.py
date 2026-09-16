"""
Tests for project configuration.

Author: Ujwal
Project: Intelligent Loan Risk Assessment Platform
"""

from pathlib import Path

from src.config import (
    PROJECT_ROOT,
    DATA_DIR,
    RAW_DATA_DIR,
    TRAIN_FILE,
    TARGET_COLUMN,
    TEST_SIZE,
    RANDOM_STATE,
    CV_FOLDS,
    SCORING,
    MODELS_DIR,
    LOG_DIR,
    MLFLOW_DIR,
)


def test_project_root_exists():
    """Test that the project root directory exists."""

    assert PROJECT_ROOT.exists()
    assert PROJECT_ROOT.is_dir()


def test_data_directories_exist():
    """Test that required data directories exist."""

    assert DATA_DIR.exists()
    assert RAW_DATA_DIR.exists()


def test_training_file_exists():
    """Test that the training dataset exists."""

    assert isinstance(TRAIN_FILE, Path)
    assert TRAIN_FILE.exists()
    assert TRAIN_FILE.is_file()


def test_training_file_name():
    """Test that the configured training file is correct."""

    assert TRAIN_FILE.name == "application_train.csv"


def test_target_column():
    """Test that the target column is configured correctly."""

    assert TARGET_COLUMN == "TARGET"


def test_train_test_configuration():
    """Test train/test split configuration."""

    assert 0 < TEST_SIZE < 1
    assert isinstance(RANDOM_STATE, int)


def test_cross_validation_configuration():
    """Test cross-validation configuration."""

    assert isinstance(CV_FOLDS, int)
    assert CV_FOLDS >= 2
    assert SCORING == "roc_auc"


def test_artifact_directories_exist():
    """Test that model and logging directories exist."""

    assert MODELS_DIR.exists()
    assert LOG_DIR.exists()
    assert MLFLOW_DIR.exists()
