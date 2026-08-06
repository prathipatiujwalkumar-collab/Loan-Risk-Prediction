"""
Production-grade data validation module.

Author: Ujwal
Project: Intelligent Loan Risk Assessment Platform
"""

import sys
import pandas as pd

from src.logger import get_logger
from src.exceptions import LoanRiskException


logger = get_logger("DataValidator")


class DataValidator:
    """
    Validates datasets before preprocessing.
    """

    def __init__(self):
        logger.info("DataValidator initialized.")

    def validate_empty_dataframe(self, df: pd.DataFrame):
        """
        Ensure dataframe is not empty.
        """

        try:

            if df.empty:
                raise ValueError("Input DataFrame is empty.")

            logger.info("Empty DataFrame validation passed.")

        except Exception as error:
            logger.exception(error)
            raise LoanRiskException(error, sys)

    def validate_target_column(self, df: pd.DataFrame):
        """
        Ensure TARGET column exists.
        """

        try:

            if "TARGET" not in df.columns:
                raise ValueError("TARGET column is missing.")

            logger.info("TARGET column validation passed.")

        except Exception as error:
            logger.exception(error)
            raise LoanRiskException(error, sys)

    def validate_duplicate_rows(self, df: pd.DataFrame):
        """
        Check duplicate rows.
        """

        try:

            duplicates = df.duplicated().sum()

            logger.info(f"Duplicate rows found: {duplicates}")

            return duplicates

        except Exception as error:
            logger.exception(error)
            raise LoanRiskException(error, sys)

    def validate_duplicate_columns(self, df: pd.DataFrame):
        """
        Check duplicate column names.
        """

        try:

            duplicate_columns = df.columns[df.columns.duplicated()].tolist()

            if duplicate_columns:

                raise ValueError(
                    f"Duplicate columns found: {duplicate_columns}"
                )

            logger.info("Duplicate column validation passed.")

        except Exception as error:
            logger.exception(error)
            raise LoanRiskException(error, sys)

    def validate(self, df: pd.DataFrame):
        """
        Run all validations.
        """

        self.validate_empty_dataframe(df)
        self.validate_target_column(df)
        self.validate_duplicate_columns(df)
        self.validate_duplicate_rows(df)

        logger.info("All dataset validations passed successfully.")