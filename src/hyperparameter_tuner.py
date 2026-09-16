"""
Production-grade Hyperparameter Tuning Module

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
    RandomizedSearchCV,
    StratifiedKFold,
)

# ==================================================
# Local Imports
# ==================================================

from src.logger import get_logger
from src.exceptions import LoanRiskException

from src.config import (
    RANDOM_STATE,
    N_ITER_SEARCH,
    CV_FOLDS,
    TUNING_SCORING,
)

# ==================================================
# Logger
# ==================================================

logger = get_logger("HyperparameterTuner")


class HyperparameterTuner:
    """
    Performs hyperparameter optimization using
    RandomizedSearchCV.
    """

    def __init__(self) -> None:

        self.cv = StratifiedKFold(
            n_splits=CV_FOLDS,
            shuffle=True,
            random_state=RANDOM_STATE,
        )

        logger.info(
            f"HyperparameterTuner initialized "
            f"({CV_FOLDS}-Fold Stratified CV)."
        )

    def tune(
        self,
        pipeline: Pipeline,
        parameters: dict,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> RandomizedSearchCV:

        """
        Tune hyperparameters using RandomizedSearchCV.
        """

        try:

            if pipeline is None:
                raise ValueError("Pipeline cannot be None.")

            if not parameters:
                raise ValueError(
                    "Parameter search space is empty."
                )

            if X_train.empty:
                raise ValueError(
                    "Training dataset is empty."
                )

            if y_train.empty:
                raise ValueError(
                    "Training labels are empty."
                )

            logger.info(
                "Starting hyperparameter tuning..."
            )

            logger.info(
                f"Scoring Metric : {TUNING_SCORING}"
            )

            logger.info(
                f"Iterations : {N_ITER_SEARCH}"
            )

            logger.info(
                f"CV Folds : {CV_FOLDS}"
            )

            search = RandomizedSearchCV(

                estimator=pipeline,

                param_distributions=parameters,

                n_iter=N_ITER_SEARCH,

                scoring=TUNING_SCORING,

                cv=self.cv,

                random_state=RANDOM_STATE,

                n_jobs=-1,

                verbose=2,

                refit=True,

                return_train_score=True,

                error_score="raise",

            )

            search.fit(
                X_train,
                y_train,
            )

            logger.info(
                "Hyperparameter tuning completed."
            )

            logger.info(
                f"Best Score : {search.best_score_:.4f}"
            )

            logger.info(
                f"Best Parameters : {search.best_params_}"
            )

            logger.info(
                f"Best Estimator : "
                f"{type(search.best_estimator_.named_steps['model']).__name__}"
            )

            return search

        except Exception as error:

            logger.exception(error)

            raise LoanRiskException(error, sys)