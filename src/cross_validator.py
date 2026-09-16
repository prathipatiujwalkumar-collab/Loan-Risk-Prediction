"""
Production-grade Cross Validation Module

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
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
)

# ==================================================
# Local Imports
# ==================================================

from src.logger import get_logger
from src.exceptions import LoanRiskException
from src.config import (
    RANDOM_STATE,
    CV_FOLDS,
    SCORING,
)

# ==================================================
# Logger
# ==================================================

logger = get_logger(__name__)


class CrossValidator:
    """
    Performs Stratified K-Fold Cross Validation on an
    entire preprocessing + modeling pipeline.
    """

    def __init__(self) -> None:

        self.cv = StratifiedKFold(
            n_splits=CV_FOLDS,
            shuffle=True,
            random_state=RANDOM_STATE,
        )

        logger.info(
            f"CrossValidator initialized with "
            f"{CV_FOLDS}-Fold Stratified Cross Validation."
        )

    def evaluate(
        self,
        pipeline: Pipeline,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> dict:

        try:

            if pipeline is None:
                raise ValueError("Pipeline cannot be None.")

            if X.empty:
                raise ValueError("Feature dataframe is empty.")

            if y.empty:
                raise ValueError("Target vector is empty.")

            logger.info(
                f"Starting {CV_FOLDS}-Fold Cross Validation..."
            )

            scores = cross_val_score(
                estimator=pipeline,
                X=X,
                y=y,
                cv=self.cv,
                scoring=SCORING,
                n_jobs=-1,
            )

            results = {
                "scores": scores.tolist(),
                "mean": float(scores.mean()),
                "std": float(scores.std()),
            }

            logger.info(
                f"Fold Scores : {results['scores']}"
            )

            logger.info(
                f"Mean {SCORING}: {results['mean']:.4f}"
            )

            logger.info(
                f"Std {SCORING}: {results['std']:.4f}"
            )

            logger.info(
                "Cross Validation completed successfully."
            )

            return results

        except Exception as error:

            logger.exception(error)

            raise LoanRiskException(error, sys)