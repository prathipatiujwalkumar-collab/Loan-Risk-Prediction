"""
Pytest test module for DataLoader.

Author: Ujwal
Project: Intelligent Loan Risk Assessment Platform
"""

# ==================================================
# Imports
# ==================================================

import pandas as pd

from src.data_loader import DataLoader
from src.config import TRAIN_FILE


# ==================================================
# Fixtures
# ==================================================

def test_data_loader_initialization():
    """
    Test that DataLoader can be initialized successfully.
    """

    loader = DataLoader()

    assert loader is not None


# ==================================================
# Test Training Data Loading
# ==================================================

def test_load_training_data_returns_dataframe():
    """
    Test that training data is loaded as a pandas DataFrame.
    """

    loader = DataLoader()

    df = loader.load_training_data()

    assert isinstance(df, pd.DataFrame)


# ==================================================
# Test Training Data Is Not Empty
# ==================================================

def test_training_data_is_not_empty():
    """
    Test that the loaded training dataset contains data.
    """

    loader = DataLoader()

    df = loader.load_training_data()

    assert not df.empty


# ==================================================
# Test Training File Exists
# ==================================================

def test_training_file_exists():
    """
    Test that the configured training file exists.
    """

    assert TRAIN_FILE.exists()


# ==================================================
# Test Dataset Shape
# ==================================================

def test_training_data_has_expected_shape():
    """
    Test the expected structure of the Home Credit dataset.

    The project currently uses:
    307511 rows
    122 columns
    """

    loader = DataLoader()

    df = loader.load_training_data()

    assert df.shape == (307511, 122)


# ==================================================
# Test Target Column
# ==================================================

def test_training_data_contains_target_column():
    """
    Test that the target column is present.
    """

    loader = DataLoader()

    df = loader.load_training_data()

    assert "TARGET" in df.columns