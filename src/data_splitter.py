"""
Production-grade Data Splitter Module

Author: Ujwal
Project: Intelligent Loan Risk Assessment Platform
"""

# ==================================================
# Standard Library Imports
# ==================================================

import sys

# ==================================================
# Third-Party Imports
# ==================================================

import pandas as pd
from sklearn.model_selection import train_test_split

# ==================================================
# Local Imports
# ==================================================

from src.logger import get_logger
from src.exceptions import LoanRiskException
from src.config import (
    TARGET_COLUMN,
    TEST_SIZE,
    RANDOM_STATE,
    STRATIFY,
)

# ==================================================
# Logger
# ==================================================

logger = get_logger(__name__)


class DataSplitter:
    """
    Splits the dataset into training and testing sets.
    """

    def __init__(self) -> None:

        logger.info("DataSplitter initialized.")

    def split(
        self,
        df: pd.DataFrame,
    ) -> tuple:
        """
        Split the dataset into training and testing sets.

        Parameters
        ----------
        df : pd.DataFrame
            Complete dataset including the target column.

        Returns
        -------
        tuple
            X_train, X_test, y_train, y_test
        """

        try:

            logger.info("Starting train-test split...")

            if df.empty:
                raise ValueError("Input dataframe is empty.")

            if TARGET_COLUMN not in df.columns:
                raise ValueError(
                    f"Target column '{TARGET_COLUMN}' not found."
                )

            X = df.drop(columns=[TARGET_COLUMN])
            y = df[TARGET_COLUMN]

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=TEST_SIZE,
                random_state=RANDOM_STATE,
                stratify=y if STRATIFY else None,
            )

            logger.info("Train-test split completed successfully.")

            logger.info(f"X_train shape : {X_train.shape}")
            logger.info(f"X_test shape  : {X_test.shape}")
            logger.info(f"y_train shape : {y_train.shape}")
            logger.info(f"y_test shape  : {y_test.shape}")

            logger.info(
                f"Training samples : {len(X_train)}"
            )

            logger.info(
                f"Testing samples : {len(X_test)}"
            )

            return (
                X_train,
                X_test,
                y_train,
                y_test,
            )

        except Exception as error:

            logger.exception(error)

            raise LoanRiskException(error, sys)