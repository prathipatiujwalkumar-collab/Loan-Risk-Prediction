"""
Production-grade Model Factory Module

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

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# ==================================================
# Local Imports
# ==================================================

from src.logger import get_logger
from src.exceptions import LoanRiskException
from src.config import (
    RANDOM_STATE,
    MAX_ITER,
    CLASS_WEIGHT,
    RF_N_ESTIMATORS,
    RF_MAX_DEPTH,
    N_JOBS,
)

# ==================================================
# Logger
# ==================================================

logger = get_logger("ModelFactory")


class ModelFactory:
    """
    Factory responsible for creating all machine
    learning models used throughout the project.
    """

    def __init__(self) -> None:

        logger.info("ModelFactory initialized.")

    def get_models(self) -> dict:
        """
        Return all available machine learning models.

        Returns
        -------
        dict
            Dictionary containing model names and
            initialized sklearn estimators.
        """

        try:

            logger.info(
                "Loading available machine learning models."
            )

            models = {

                "Logistic Regression": LogisticRegression(
                    random_state=RANDOM_STATE,
                    max_iter=MAX_ITER,
                    class_weight=CLASS_WEIGHT,
                    solver="lbfgs",
                ),

                "Random Forest": RandomForestClassifier(
                    n_estimators=RF_N_ESTIMATORS,
                    max_depth=RF_MAX_DEPTH,
                    random_state=RANDOM_STATE,
                    n_jobs=N_JOBS,
                    class_weight=CLASS_WEIGHT,
                ),
            }

            logger.info(
                f"{len(models)} models loaded successfully."
            )

            return models

        except Exception as error:

            logger.exception(error)

            raise LoanRiskException(error, sys)