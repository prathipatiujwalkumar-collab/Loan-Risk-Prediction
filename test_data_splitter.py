"""
Unit tests for DataSplitter.

Author: Ujwal
Project: Intelligent Loan Risk Assessment Platform
"""

import pandas as pd

from src.data_loader import DataLoader
from src.data_validator import DataValidator
from src.feature_engineering import FeatureEngineer
from src.data_splitter import DataSplitter


# ==================================================
# Fixtures / Test Data Preparation
# ==================================================

def prepare_data():
    """
    Load, validate, and transform the training dataset.
    """

    loader = DataLoader()
    df = loader.load_training_data()

    validator = DataValidator()
    validator.validate(df)

    engineer = FeatureEngineer()
    df = engineer.transform(df)

    return df


# ==================================================
# Test Initialization
# ==================================================

def test_data_splitter_initialization():
    """
    Test that DataSplitter can be initialized.
    """

    splitter = DataSplitter()

    assert splitter is not None


# ==================================================
# Test Split Output
# ==================================================

def test_data_split_returns_four_outputs():
    """
    Test that the splitter returns
    X_train, X_test, y_train, and y_test.
    """

    df = prepare_data()

    splitter = DataSplitter()

    result = splitter.split(df)

    assert len(result) == 4


# ==================================================
# Test Feature Shapes
# ==================================================

def test_train_test_feature_shapes():
    """
    Test that training and testing features
    contain rows and columns.
    """

    df = prepare_data()

    splitter = DataSplitter()

    X_train, X_test, y_train, y_test = splitter.split(df)

    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0

    assert X_train.shape[1] == X_test.shape[1]


# ==================================================
# Test Target Shapes
# ==================================================

def test_train_test_target_shapes():
    """
    Test that target arrays contain observations
    corresponding to their feature datasets.
    """

    df = prepare_data()

    splitter = DataSplitter()

    X_train, X_test, y_train, y_test = splitter.split(df)

    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)


# ==================================================
# Test Train/Test Separation
# ==================================================

def test_train_test_row_counts():
    """
    Test that the complete dataset is divided
    between training and testing sets.
    """

    df = prepare_data()

    splitter = DataSplitter()

    X_train, X_test, y_train, y_test = splitter.split(df)

    total_rows = len(X_train) + len(X_test)

    assert total_rows == len(df)


# ==================================================
# Test Target Distribution
# ==================================================

def test_target_distribution_is_preserved():
    """
    Test that both training and testing sets contain
    the target classes.
    """

    df = prepare_data()

    splitter = DataSplitter()

    X_train, X_test, y_train, y_test = splitter.split(df)

    assert len(y_train.unique()) >= 2
    assert len(y_test.unique()) >= 2


# ==================================================
# Test No Missing Targets
# ==================================================

def test_target_values_are_not_missing():
    """
    Test that the split does not produce missing
    target values.
    """

    df = prepare_data()

    splitter = DataSplitter()

    X_train, X_test, y_train, y_test = splitter.split(df)

    assert not y_train.isna().any()
    assert not y_test.isna().any()