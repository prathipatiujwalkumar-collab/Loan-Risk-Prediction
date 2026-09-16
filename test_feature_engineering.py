"""
Unit tests for Feature Engineering.

Author: Ujwal
Project: Intelligent Loan Risk Assessment Platform
"""

import pandas as pd

from src.data_loader import DataLoader
from src.data_validator import DataValidator
from src.feature_engineering import FeatureEngineer


# ==================================================
# Expected Engineered Features
# ==================================================

ENGINEERED_FEATURES = [
    "CREDIT_INCOME_RATIO",
    "ANNUITY_INCOME_RATIO",
    "CREDIT_GOODS_RATIO",
    "EMPLOYMENT_AGE_RATIO",
    "CHILDREN_RATIO",
    "INCOME_PER_FAMILY_MEMBER",
    "ANNUITY_CREDIT_RATIO",
    "GOODS_INCOME_RATIO",
    "YEARS_EMPLOYED",
    "AGE_YEARS",
    "INCOME_PER_CHILD",
    "CREDIT_PER_FAMILY_MEMBER",
]


# ==================================================
# Data Preparation
# ==================================================

def prepare_data():
    """
    Load and validate the training dataset.
    """

    loader = DataLoader()

    df = loader.load_training_data()

    validator = DataValidator()

    validator.validate(df)

    return df


# ==================================================
# Test Initialization
# ==================================================

def test_feature_engineer_initialization():
    """
    Test that FeatureEngineer can be initialized.
    """

    engineer = FeatureEngineer()

    assert engineer is not None


# ==================================================
# Test Transformation
# ==================================================

def test_feature_engineering_returns_dataframe():
    """
    Test that feature engineering returns a DataFrame.
    """

    df = prepare_data()

    engineer = FeatureEngineer()

    engineered_df = engineer.transform(df)

    assert isinstance(engineered_df, pd.DataFrame)


# ==================================================
# Test Dataset Preservation
# ==================================================

def test_feature_engineering_preserves_row_count():
    """
    Feature engineering should not change the number
    of observations.
    """

    df = prepare_data()

    engineer = FeatureEngineer()

    engineered_df = engineer.transform(df)

    assert len(engineered_df) == len(df)


# ==================================================
# Test Engineered Features
# ==================================================

def test_engineered_features_are_created():
    """
    Test that all expected engineered features exist.
    """

    df = prepare_data()

    engineer = FeatureEngineer()

    engineered_df = engineer.transform(df)

    for feature in ENGINEERED_FEATURES:

        assert feature in engineered_df.columns


# ==================================================
# Test Engineered Feature Count
# ==================================================

def test_engineered_feature_count():
    """
    Test that the expected number of engineered
    features are created.
    """

    df = prepare_data()

    engineer = FeatureEngineer()

    engineered_df = engineer.transform(df)

    created_features = [
        feature
        for feature in ENGINEERED_FEATURES
        if feature in engineered_df.columns
    ]

    assert len(created_features) == len(ENGINEERED_FEATURES)


# ==================================================
# Test No Infinite Values
# ==================================================

def test_engineered_features_have_no_infinite_values():
    """
    Test that engineered features do not contain
    positive or negative infinity.
    """

    df = prepare_data()

    engineer = FeatureEngineer()

    engineered_df = engineer.transform(df)

    features = engineered_df[ENGINEERED_FEATURES]

    assert not features.isin([float("inf"), float("-inf")]).any().any()


# ==================================================
# Test Feature Data Types
# ==================================================

def test_engineered_features_are_numeric():
    """
    Test that all engineered features are numeric.
    """

    df = prepare_data()

    engineer = FeatureEngineer()

    engineered_df = engineer.transform(df)

    features = engineered_df[ENGINEERED_FEATURES]

    assert all(
        pd.api.types.is_numeric_dtype(features[column])
        for column in features.columns
    )


# ==================================================
# Test Feature Engineering Does Not Remove Target
# ==================================================

def test_target_column_is_preserved():
    """
    Test that the TARGET column remains available
    after feature engineering.
    """

    df = prepare_data()

    engineer = FeatureEngineer()

    engineered_df = engineer.transform(df)

    assert "TARGET" in engineered_df.columns