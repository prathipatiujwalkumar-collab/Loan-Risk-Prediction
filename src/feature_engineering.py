"""
Production-grade Feature Engineering Module

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

import numpy as np
import pandas as pd

# ==================================================
# Local Imports
# ==================================================

from src.logger import get_logger
from src.exceptions import LoanRiskException

# ==================================================
# Logger
# ==================================================

logger = get_logger("FeatureEngineer")


class FeatureEngineer:
    """
    Performs business-driven feature engineering for
    Home Credit Loan Default Prediction.
    """

    REQUIRED_COLUMNS = [
        "AMT_CREDIT",
        "AMT_INCOME_TOTAL",
        "AMT_ANNUITY",
        "AMT_GOODS_PRICE",
        "DAYS_EMPLOYED",
        "DAYS_BIRTH",
        "CNT_CHILDREN",
        "CNT_FAM_MEMBERS",
    ]

    def __init__(self):
        logger.info("FeatureEngineer initialized.")

    # ==================================================
    # Utility Functions
    # ==================================================

    @staticmethod
    def _safe_divide(
        numerator: pd.Series,
        denominator: pd.Series,
    ) -> pd.Series:
        """
        Safely divide two Series.

        Any zero denominator is converted to NaN,
        preventing divide-by-zero warnings.
        """

        denominator = denominator.replace(0, np.nan)

        return numerator / denominator

    @staticmethod
    def _validate_columns(df: pd.DataFrame, required_columns: list):
        """
        Ensure every required column exists.
        """

        missing = [
            col
            for col in required_columns
            if col not in df.columns
        ]

        if missing:
            raise ValueError(
                f"Missing required columns: {missing}"
            )

    # ==================================================
    # Main Feature Engineering
    # ==================================================

    def transform(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Perform business feature engineering.

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        pd.DataFrame
        """

        try:

            if df.empty:
                raise ValueError(
                    "Input DataFrame is empty."
                )

            logger.info(
                "Starting feature engineering..."
            )

            data = df.copy()

            # ------------------------------------------
            # Validate Required Columns
            # ------------------------------------------

            self._validate_columns(
                data,
                self.REQUIRED_COLUMNS,
            )

            # ------------------------------------------
            # Replace Placeholder Values
            # ------------------------------------------

            # Home Credit dataset uses 365243 to represent
            # missing employment days.

            data["DAYS_EMPLOYED"] = (
                data["DAYS_EMPLOYED"]
                .replace(365243, np.nan)
            )

            # ------------------------------------------
            # Financial Ratios
            # ------------------------------------------

            data["CREDIT_INCOME_RATIO"] = (
                self._safe_divide(
                    data["AMT_CREDIT"],
                    data["AMT_INCOME_TOTAL"],
                )
            )

            data["ANNUITY_INCOME_RATIO"] = (
                self._safe_divide(
                    data["AMT_ANNUITY"],
                    data["AMT_INCOME_TOTAL"],
                )
            )

            data["ANNUITY_CREDIT_RATIO"] = (
                self._safe_divide(
                    data["AMT_ANNUITY"],
                    data["AMT_CREDIT"],
                )
            )

            data["GOODS_INCOME_RATIO"] = (
                self._safe_divide(
                    data["AMT_GOODS_PRICE"],
                    data["AMT_INCOME_TOTAL"],
                )
            )

            data["CREDIT_GOODS_RATIO"] = (
                self._safe_divide(
                    data["AMT_CREDIT"],
                    data["AMT_GOODS_PRICE"],
                )
            )
            # ------------------------------------------
            # Employment & Family Features
            # ------------------------------------------

            data["EMPLOYMENT_AGE_RATIO"] = (
                self._safe_divide(
                    data["DAYS_EMPLOYED"],
                    data["DAYS_BIRTH"],
                )
            )

            data["CHILDREN_RATIO"] = (
                self._safe_divide(
                    data["CNT_CHILDREN"],
                    data["CNT_FAM_MEMBERS"],
                )
            )

            data["INCOME_PER_FAMILY_MEMBER"] = (
                self._safe_divide(
                    data["AMT_INCOME_TOTAL"],
                    data["CNT_FAM_MEMBERS"],
                )
            )

            # ------------------------------------------
            # Additional High-Value Features
            # ------------------------------------------

            data["YEARS_EMPLOYED"] = (
                np.abs(data["DAYS_EMPLOYED"]) / 365.25
            )

            data["AGE_YEARS"] = (
                np.abs(data["DAYS_BIRTH"]) / 365.25
            )

            data["INCOME_PER_CHILD"] = (
                self._safe_divide(
                    data["AMT_INCOME_TOTAL"],
                    data["CNT_CHILDREN"] + 1,
                )
            )

            data["CREDIT_PER_FAMILY_MEMBER"] = (
                self._safe_divide(
                    data["AMT_CREDIT"],
                    data["CNT_FAM_MEMBERS"],
                )
            )

            # ------------------------------------------
            # Handle Infinite Values
            # ------------------------------------------

            data.replace(
                [np.inf, -np.inf],
                np.nan,
                inplace=True,
            )

            # ------------------------------------------
            # Logging
            # ------------------------------------------

            original_columns = df.shape[1]
            new_columns = data.shape[1]
            added_features = new_columns - original_columns

            logger.info(
                "Feature engineering completed successfully."
            )

            logger.info(
                f"Original Features : {original_columns}"
            )

            logger.info(
                f"Engineered Features Added : {added_features}"
            )

            logger.info(
                f"Final Feature Count : {new_columns}"
            )

            logger.info(
                f"Dataset Shape : {data.shape}"
            )

            return data

        except Exception as error:

            logger.exception(error)

            raise LoanRiskException(error, sys)