"""
Production-grade Model Evaluation Module

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

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

# ==================================================
# Local Imports
# ==================================================

from src.logger import get_logger
from src.exceptions import LoanRiskException

# ==================================================
# Logger
# ==================================================

logger = get_logger(__name__)


class Evaluator:
    """
    Evaluates binary classification models using
    standard machine learning metrics.
    """

    def __init__(self) -> None:

        logger.info("Evaluator initialized.")

    def evaluate(
        self,
        y_true,
        y_pred,
        y_prob,
    ) -> dict:
        """
        Evaluate model performance.

        Parameters
        ----------
        y_true
            Actual labels.

        y_pred
            Predicted labels.

        y_prob
            Predicted probabilities.

        Returns
        -------
        dict
            Evaluation metrics.
        """

        try:

            y_true = np.asarray(y_true)
            y_pred = np.asarray(y_pred)
            y_prob = np.asarray(y_prob)

            if len(y_true) != len(y_pred):

                raise ValueError(
                    "y_true and y_pred have different lengths."
                )

            if y_prob.ndim != 2 or y_prob.shape[1] < 2:

                raise ValueError(
                    "Predicted probabilities must contain two columns."
                )

            metrics = {

                "Accuracy": float(
                    accuracy_score(
                        y_true,
                        y_pred,
                    )
                ),

                "Precision": float(
                    precision_score(
                        y_true,
                        y_pred,
                        zero_division=0,
                    )
                ),

                "Recall": float(
                    recall_score(
                        y_true,
                        y_pred,
                        zero_division=0,
                    )
                ),

                "F1": float(
                    f1_score(
                        y_true,
                        y_pred,
                        zero_division=0,
                    )
                ),

                "ROC_AUC": float(
                    roc_auc_score(
                        y_true,
                        y_prob[:, 1],
                    )
                ),

                "Confusion Matrix": confusion_matrix(
                    y_true,
                    y_pred,
                ),

                "Classification Report": classification_report(
                    y_true,
                    y_pred,
                    zero_division=0,
                ),
            }

            logger.info(
                f"Accuracy : {metrics['Accuracy']:.4f}"
            )

            logger.info(
                f"Precision : {metrics['Precision']:.4f}"
            )

            logger.info(
                f"Recall : {metrics['Recall']:.4f}"
            )

            logger.info(
                f"F1 Score : {metrics['F1']:.4f}"
            )

            logger.info(
                f"ROC-AUC : {metrics['ROC_AUC']:.4f}"
            )

            logger.info("Evaluation completed successfully.")

            return metrics

        except Exception as error:

            logger.exception(error)

            raise LoanRiskException(error, sys)