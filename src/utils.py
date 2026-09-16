"""
Production-grade Utility Module

Author: Ujwal
Project: Intelligent Loan Risk Assessment Platform

Common helper functions used across the project.
"""

# =====================================================
# Standard Library Imports
# =====================================================

from pathlib import Path

# =====================================================
# Third-Party Imports
# =====================================================

import joblib
import pandas as pd
import matplotlib.pyplot as plt

# =====================================================
# DataFrame Utilities
# =====================================================

def get_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return missing value statistics.
    """

    missing = pd.DataFrame(
        {
            "Missing Values": df.isnull().sum(),
            "Percentage": (
                df.isnull().mean() * 100
            ).round(2),
        }
    )

    return (
        missing[missing["Missing Values"] > 0]
        .sort_values(
            by="Percentage",
            ascending=False,
        )
    )


def summarize_dataframe(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return dataframe summary.
    """

    return pd.DataFrame(
        {
            "Data Type": df.dtypes,
            "Missing": df.isnull().sum(),
            "Unique": df.nunique(),
        }
    )


# =====================================================
# Visualization Utilities
# =====================================================

def plot_target_distribution(
    df: pd.DataFrame,
    target: str = "TARGET",
):
    """
    Plot target distribution.
    """

    plt.figure(figsize=(6, 4))

    df[target].value_counts().plot(
        kind="bar"
    )

    plt.title("Target Distribution")
    plt.xlabel(target)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


def plot_missing_values(
    df: pd.DataFrame,
    top_n: int = 20,
):
    """
    Plot columns with most missing values.
    """

    missing = (
        df.isnull()
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
    )

    plt.figure(figsize=(10, 6))

    missing.plot(kind="bar")

    plt.title(f"Top {top_n} Missing Columns")
    plt.ylabel("Missing Values")
    plt.tight_layout()
    plt.show()


# =====================================================
# Feature Utilities
# =====================================================

def identify_numerical_columns(
    df: pd.DataFrame,
):
    """
    Return numerical columns.
    """

    return df.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()


def identify_categorical_columns(
    df: pd.DataFrame,
):
    """
    Return categorical columns.
    """

    return df.select_dtypes(
        include=["object"]
    ).columns.tolist()


# =====================================================
# Model Utilities
# =====================================================

def save_model(
    model,
    filepath,
):
    """
    Save any sklearn model.
    """

    filepath = Path(filepath)

    filepath.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        filepath,
    )


def load_model(
    filepath,
):
    """
    Load any saved model.
    """

    return joblib.load(filepath)