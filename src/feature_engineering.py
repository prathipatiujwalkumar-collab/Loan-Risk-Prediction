"""
Production-grade feature engineering module.

Author: Ujwal
Project: Intelligent Loan Risk Assessment Platform
"""

# ===========================
# Standard Library Imports
# ===========================

import sys

# ===========================
# Third-Party Imports
# ===========================

import numpy as np
import pandas as pd

# ===========================
# Local Imports
# ===========================

from src.logger import get_logger
from src.exceptions import LoanRiskException

# ===========================
# Logger
# ===========================

logger = get_logger("FeatureEngineer")


class FeatureEngineer:
    """
    Creates business-driven features for loan risk prediction.
    """

    def __init__(self):

        logger.info("FeatureEngineer initialized.")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create business-driven features for loan risk prediction.
        """

        try:

            data = df.copy()

            logger.info("Starting feature engineering...")

            required_columns = [
                "AMT_CREDIT",
                "AMT_INCOME_TOTAL",
                "AMT_ANNUITY",
                "AMT_GOODS_PRICE",
                "DAYS_EMPLOYED",
                "DAYS_BIRTH",
                "CNT_CHILDREN",
                "CNT_FAM_MEMBERS"
            ]

            missing_columns = [
                col for col in required_columns
                if col not in data.columns
            ]

            if missing_columns:
                raise ValueError(
                    f"Missing required columns: {missing_columns}"
                )

            # ----------------------------------------
            # Clean Known Placeholder Values
            # ----------------------------------------

            data["DAYS_EMPLOYED"] = (
                data["DAYS_EMPLOYED"]
                .replace(365243, np.nan)
            )

            # ----------------------------------------
            # Business Features
            # ----------------------------------------

            data["CREDIT_INCOME_RATIO"] = np.where(
                data["AMT_INCOME_TOTAL"] != 0,
                data["AMT_CREDIT"] / data["AMT_INCOME_TOTAL"],
                np.nan
            )

            data["ANNUITY_INCOME_RATIO"] = np.where(
                data["AMT_INCOME_TOTAL"] != 0,
                data["AMT_ANNUITY"] / data["AMT_INCOME_TOTAL"],
                np.nan
            )

            data["CREDIT_GOODS_RATIO"] = np.where(
                data["AMT_GOODS_PRICE"] != 0,
                data["AMT_CREDIT"] / data["AMT_GOODS_PRICE"],
                np.nan
            )

            data["EMPLOYMENT_AGE_RATIO"] = np.where(
                data["DAYS_BIRTH"] != 0,
                data["DAYS_EMPLOYED"] / data["DAYS_BIRTH"],
                np.nan
            )

            data["CHILDREN_RATIO"] = np.where(
                data["CNT_FAM_MEMBERS"] != 0,
                data["CNT_CHILDREN"] / data["CNT_FAM_MEMBERS"],
                np.nan
            )

            data["INCOME_PER_FAMILY_MEMBER"] = np.where(
                data["CNT_FAM_MEMBERS"] != 0,
                data["AMT_INCOME_TOTAL"] / data["CNT_FAM_MEMBERS"],
                np.nan
            )

            data["ANNUITY_CREDIT_RATIO"] = np.where(
                data["AMT_CREDIT"] != 0,
                data["AMT_ANNUITY"] / data["AMT_CREDIT"],
                np.nan
            )

            data["GOODS_INCOME_RATIO"] = np.where(
                data["AMT_INCOME_TOTAL"] != 0,
                data["AMT_GOODS_PRICE"] / data["AMT_INCOME_TOTAL"],
                np.nan
            )

            logger.info("Feature engineering completed successfully.")
            logger.info("Features added: 8")
            logger.info(f"New dataset shape: {data.shape}")

            return data

        except Exception as error:

            logger.exception(error)
            raise LoanRiskException(error, sys)