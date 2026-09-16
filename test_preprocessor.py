"""
Unit tests for Preprocessor.

Author: Ujwal
Project: Intelligent Loan Risk Assessment Platform
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import ARTIFACTS_DIR
from src.data_loader import DataLoader
from src.data_validator import DataValidator
from src.feature_engineering import FeatureEngineer
from src.data_splitter import DataSplitter
from src.preprocessor import Preprocessor


# ==================================================
# Fixtures
# ==================================================

@pytest.fixture(scope="module")
def dataset():
    """Load and validate the training dataset."""

    loader = DataLoader()
    df = loader.load_training_data()

    validator = DataValidator()
    validator.validate(df)

    engineer = FeatureEngineer()
    df = engineer.transform(df)

    return df


@pytest.fixture(scope="module")
def split_data(dataset):
    """Create train/test datasets."""

    splitter = DataSplitter()

    return splitter.split(dataset)


@pytest.fixture
def preprocessor():
    """Create a fresh Preprocessor instance."""

    return Preprocessor()


# ==================================================
# Initialization
# ==================================================

def test_preprocessor_initialization(preprocessor):
    """Test that Preprocessor initializes successfully."""

    assert preprocessor is not None


# ==================================================
# Summary
# ==================================================

def test_preprocessor_summary(preprocessor, split_data):
    """Test that the preprocessing summary is generated."""

    X_train, X_test, y_train, y_test = split_data

    summary = preprocessor.summary(X_train)

    assert isinstance(summary, dict)
    assert len(summary) > 0


# ==================================================
# Fit Transform
# ==================================================

def test_fit_transform_returns_data(preprocessor, split_data):
    """Test that training data can be fitted and transformed."""

    X_train, X_test, y_train, y_test = split_data

    X_train_processed = preprocessor.fit_transform(X_train)

    assert X_train_processed is not None
    assert X_train_processed.shape[0] == X_train.shape[0]
    assert X_train_processed.shape[1] > 0


# ==================================================
# Test Data Transformation
# ==================================================

def test_transform_test_data(preprocessor, split_data):
    """Test that test data can be transformed."""

    X_train, X_test, y_train, y_test = split_data

    preprocessor.fit_transform(X_train)

    X_test_processed = preprocessor.transform(X_test)

    assert X_test_processed is not None
    assert X_test_processed.shape[0] == X_test.shape[0]


# ==================================================
# Feature Count Consistency
# ==================================================

def test_train_test_feature_count_matches(preprocessor, split_data):
    """Test that train and test data produce the same number
    of transformed features.
    """

    X_train, X_test, y_train, y_test = split_data

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    assert X_train_processed.shape[1] == X_test_processed.shape[1]


# ==================================================
# Feature Names
# ==================================================

def test_feature_names_generated(preprocessor, split_data):
    """Test that transformed feature names are available."""

    X_train, X_test, y_train, y_test = split_data

    X_train_processed = preprocessor.fit_transform(X_train)

    feature_names = preprocessor.get_feature_names()

    assert feature_names is not None
    assert len(feature_names) == X_train_processed.shape[1]


# ==================================================
# Feature Names Are Unique
# ==================================================

def test_feature_names_are_unique(preprocessor, split_data):
    """Test that transformed feature names are unique."""

    X_train, X_test, y_train, y_test = split_data

    preprocessor.fit_transform(X_train)

    feature_names = preprocessor.get_feature_names()

    assert len(feature_names) == len(set(feature_names))


# ==================================================
# Processed Data Is Numeric
# ==================================================

def test_processed_data_is_numeric(preprocessor, split_data):
    """Test that the processed feature matrix is numeric."""

    X_train, X_test, y_train, y_test = split_data

    X_train_processed = preprocessor.fit_transform(X_train)

    # Convert sparse matrices to dense only for this check.
    if hasattr(X_train_processed, "toarray"):
        values = X_train_processed.toarray()
    else:
        values = np.asarray(X_train_processed)

    assert np.issubdtype(values.dtype, np.number)


# ==================================================
# Save Preprocessor
# ==================================================

def test_preprocessor_save(preprocessor, split_data, tmp_path):
    """Test that a fitted preprocessor can be saved."""

    X_train, X_test, y_train, y_test = split_data

    preprocessor.fit_transform(X_train)

    save_path = tmp_path / "preprocessor.joblib"

    preprocessor.save(save_path)

    assert save_path.exists()
    assert save_path.is_file()
    assert save_path.stat().st_size > 0