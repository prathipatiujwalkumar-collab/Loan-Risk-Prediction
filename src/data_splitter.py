"""
Production-grade Data Splitter.

Author: Ujwal
Project: Intelligent Loan Risk Assessment Platform
"""

# ==========================
# Standard Library
# ==========================

import sys

# ==========================
# Third Party
# ==========================

import pandas as pd
from sklearn.model_selection import train_test_split

# ==========================
# Local Imports
# ==========================

from src.logger import get_logger
from src.exceptions import LoanRiskException
from src.config import (
    TARGET_COLUMN,
    TEST_SIZE,
    RANDOM_STATE,
    STRATIFY,
)

logger = get_logger("DataSplitter")


class DataSplitter:
    """
    Splits dataset into train and test sets.
    """

    def __init__(self):

        logger.info("DataSplitter initialized.")

    def split(self, df: pd.DataFrame):

        try:

            logger.info("Starting train-test split...")

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

            return X_train, X_test, y_train, y_test

        except Exception as error:

            logger.exception(error)
            raise LoanRiskException(error, sys)