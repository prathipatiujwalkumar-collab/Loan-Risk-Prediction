"""
Unit tests for DataValidator.

Author: Ujwal
Project: Intelligent Loan Risk Assessment Platform
"""

import pandas as pd
import pytest

from src.data_loader import DataLoader
from src.data_validator import DataValidator


# ==================================================
# Fixtures
# ==================================================

@pytest.fixture
def dataset():
    """
    Load the training dataset for validation tests.
    """

    loader = DataLoader()

    return loader.load_training_data()


@pytest.fixture
def validator():
    """
    Create a DataValidator instance.
    """

    return DataValidator()


# ==================================================
# Initialization
# ==================================================

def test_data_validator_initialization():
    """
    Test that DataValidator can be initialized.
    """

    validator = DataValidator()

    assert validator is not None


# ==================================================
# Validation
# ==================================================

def test_dataset_validation_passes(dataset, validator):
    """
    Test that the current training dataset passes
    all validation rules.
    """

    result = validator.validate(dataset)

    assert result is None or result is True


# ==================================================
# Dataset Type
# ==================================================

def test_validator_accepts_dataframe(dataset, validator):
    """
    Test that the validator successfully accepts
    a pandas DataFrame.
    """

    assert isinstance(dataset, pd.DataFrame)

    validator.validate(dataset)


# ==================================================
# Dataset Is Not Empty
# ==================================================

def test_dataset_is_not_empty(dataset):
    """
    Test that the training dataset contains rows
    and columns.
    """

    assert not dataset.empty
    assert dataset.shape[0] > 0
    assert dataset.shape[1] > 0


# ==================================================
# Target Column
# ==================================================

def test_target_column_exists(dataset):
    """
    Test that the required TARGET column exists.
    """

    assert "TARGET" in dataset.columns


# ==================================================
# Target Values
# ==================================================

def test_target_contains_binary_values(dataset):
    """
    Test that TARGET contains only binary classification
    values.
    """

    target_values = set(dataset["TARGET"].dropna().unique())

    assert target_values.issubset({0, 1})


# ==================================================
# Expected Dataset Shape
# ==================================================

def test_training_dataset_shape(dataset):
    """
    Test the expected Home Credit training dataset shape.
    """

    assert dataset.shape == (307511, 122)


# ==================================================
# Validator Rejects Invalid Input
# ==================================================

def test_validator_rejects_invalid_input(validator):
    """
    Test that invalid input is rejected by the validator.
    """

    with pytest.raises(Exception):
        validator.validate(None)