"""
Unit tests for CrossValidator.

Author: Ujwal
Project: Intelligent Loan Risk Assessment Platform
"""

import numpy as np
import pytest

from src.data_loader import DataLoader
from src.data_validator import DataValidator
from src.feature_engineering import FeatureEngineer
from src.data_splitter import DataSplitter
from src.trainer import ModelTrainer
from src.cross_validator import CrossValidator


# ==================================================
# Fixtures
# ==================================================

@pytest.fixture(scope="module")
def split_data():
    """Load, validate, engineer, and split the dataset."""

    loader = DataLoader()
    df = loader.load_training_data()

    validator = DataValidator()
    validator.validate(df)

    engineer = FeatureEngineer()
    df = engineer.transform(df)

    splitter = DataSplitter()

    return splitter.split(df)


@pytest.fixture(scope="module")
def training_pipeline(split_data):
    """Build the model training pipeline."""

    X_train, X_test, y_train, y_test = split_data

    trainer = ModelTrainer()

    return trainer.build_pipeline(X_train)


@pytest.fixture
def cross_validator():
    """Create a CrossValidator instance."""

    return CrossValidator()


# ==================================================
# Initialization
# ==================================================

def test_cross_validator_initialization(cross_validator):
    """Test CrossValidator initialization."""

    assert cross_validator is not None


# ==================================================
# Evaluation
# ==================================================

def test_cross_validation_returns_results(
    cross_validator,
    training_pipeline,
    split_data,
):
    """Test that cross-validation returns a result dictionary."""

    X_train, X_test, y_train, y_test = split_data

    results = cross_validator.evaluate(
        training_pipeline,
        X_train,
        y_train,
    )

    assert isinstance(results, dict)


# ==================================================
# Result Keys
# ==================================================

def test_cross_validation_result_keys(
    cross_validator,
    training_pipeline,
    split_data,
):
    """Test that the expected evaluation results are returned."""

    X_train, X_test, y_train, y_test = split_data

    results = cross_validator.evaluate(
        training_pipeline,
        X_train,
        y_train,
    )

    assert "scores" in results
    assert "mean" in results
    assert "std" in results


# ==================================================
# Fold Scores
# ==================================================

def test_cross_validation_fold_scores(
    cross_validator,
    training_pipeline,
    split_data,
):
    """Test that fold scores are generated."""

    X_train, X_test, y_train, y_test = split_data

    results = cross_validator.evaluate(
        training_pipeline,
        X_train,
        y_train,
    )

    scores = results["scores"]

    assert scores is not None
    assert len(scores) > 0


# ==================================================
# Score Values
# ==================================================

def test_cross_validation_scores_are_valid(
    cross_validator,
    training_pipeline,
    split_data,
):
    """Test that ROC-AUC scores are valid probabilities."""

    X_train, X_test, y_train, y_test = split_data

    results = cross_validator.evaluate(
        training_pipeline,
        X_train,
        y_train,
    )

    scores = np.asarray(results["scores"])

    assert np.all(np.isfinite(scores))
    assert np.all(scores >= 0.0)
    assert np.all(scores <= 1.0)


# ==================================================
# Mean Score
# ==================================================

def test_cross_validation_mean_is_valid(
    cross_validator,
    training_pipeline,
    split_data,
):
    """Test that the mean ROC-AUC score is valid."""

    X_train, X_test, y_train, y_test = split_data

    results = cross_validator.evaluate(
        training_pipeline,
        X_train,
        y_train,
    )

    mean_score = results["mean"]

    assert np.isfinite(mean_score)
    assert 0.0 <= mean_score <= 1.0


# ==================================================
# Standard Deviation
# ==================================================

def test_cross_validation_std_is_valid(
    cross_validator,
    training_pipeline,
    split_data,
):
    """Test that the standard deviation is valid."""

    X_train, X_test, y_train, y_test = split_data

    results = cross_validator.evaluate(
        training_pipeline,
        X_train,
        y_train,
    )

    std_score = results["std"]

    assert np.isfinite(std_score)
    assert std_score >= 0.0


# ==================================================
# Mean Matches Fold Scores
# ==================================================

def test_mean_matches_fold_scores(
    cross_validator,
    training_pipeline,
    split_data,
):
    """Test that the reported mean matches the fold scores."""

    X_train, X_test, y_train, y_test = split_data

    results = cross_validator.evaluate(
        training_pipeline,
        X_train,
        y_train,
    )

    scores = np.asarray(results["scores"])

    expected_mean = np.mean(scores)

    assert np.isclose(
        results["mean"],
        expected_mean,
    )