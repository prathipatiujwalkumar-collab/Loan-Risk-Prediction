"""
Production-grade Data Validator Module

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

# ==================================================
# Local Imports
# ==================================================

from src.logger import get_logger
from src.exceptions import LoanRiskException
from src.config import TARGET_COLUMN

# ==================================================
# Logger
# ==================================================

logger = get_logger(__name__)


class DataValidator:
    """
    Validates datasets before preprocessing and model training.
    """

    def __init__(self) -> None:

        logger.info("DataValidator initialized.")

    def validate_empty_dataframe(
        self,
        df: pd.DataFrame,
    ) -> None:
        """
        Ensure the dataframe is not empty.
        """

        try:

            if df.empty:
                raise ValueError("Input DataFrame is empty.")

            logger.info("Empty DataFrame validation passed.")

        except Exception as error:

            logger.exception(error)

            raise LoanRiskException(error, sys)

    def validate_target_column(
        self,
        df: pd.DataFrame,
    ) -> None:
        """
        Ensure the target column exists.
        """

        try:

            if TARGET_COLUMN not in df.columns:

                raise ValueError(
                    f"Target column '{TARGET_COLUMN}' is missing."
                )

            logger.info("Target column validation passed.")

        except Exception as error:

            logger.exception(error)

            raise LoanRiskException(error, sys)

    def validate_missing_target(
        self,
        df: pd.DataFrame,
    ) -> None:
        """
        Ensure the target column contains no missing values.
        """

        try:

            missing = df[TARGET_COLUMN].isnull().sum()

            if missing > 0:

                raise ValueError(
                    f"Target column contains {missing} missing values."
                )

            logger.info("Target missing value validation passed.")

        except Exception as error:

            logger.exception(error)

            raise LoanRiskException(error, sys)

    def validate_duplicate_columns(
        self,
        df: pd.DataFrame,
    ) -> None:
        """
        Ensure duplicate column names do not exist.
        """

        try:

            duplicate_columns = (
                df.columns[df.columns.duplicated()]
                .tolist()
            )

            if duplicate_columns:

                raise ValueError(
                    f"Duplicate columns found: {duplicate_columns}"
                )

            logger.info("Duplicate column validation passed.")

        except Exception as error:

            logger.exception(error)

            raise LoanRiskException(error, sys)

    def validate_duplicate_rows(
        self,
        df: pd.DataFrame,
    ) -> int:
        """
        Count duplicate rows.

        Returns
        -------
        int
            Number of duplicate rows.
        """

        try:

            duplicates = int(df.duplicated().sum())

            logger.info(
                f"Duplicate rows found: {duplicates}"
            )

            return duplicates

        except Exception as error:

            logger.exception(error)

            raise LoanRiskException(error, sys)

    def validate_missing_values(
        self,
        df: pd.DataFrame,
    ) -> pd.Series:
        """
        Return missing values for every column.

        This is informative only and does not fail validation.
        """

        try:

            missing = df.isnull().sum()

            logger.info(
                f"Columns with missing values: {(missing > 0).sum()}"
            )

            return missing

        except Exception as error:

            logger.exception(error)

            raise LoanRiskException(error, sys)

    def validate(
        self,
        df: pd.DataFrame,
    ) -> None:
        """
        Run all validation checks.
        """

        try:

            self.validate_empty_dataframe(df)

            self.validate_target_column(df)

            self.validate_missing_target(df)

            self.validate_duplicate_columns(df)

            self.validate_duplicate_rows(df)

            self.validate_missing_values(df)

            logger.info(
                "All dataset validations passed successfully."
            )

        except Exception as error:

            logger.exception(error)

            raise LoanRiskException(error, sys)