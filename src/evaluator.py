"""
Model Evaluation Module
"""

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from src.logger import get_logger

logger = get_logger("Evaluator")


class Evaluator:

    def evaluate(self, y_true, y_pred, y_prob):

        try:

            metrics = {

                "Accuracy": accuracy_score(
                    y_true,
                    y_pred,
                ),

                "Precision": precision_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                ),

                "Recall": recall_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                ),

                "F1": f1_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                ),

                "ROC_AUC": roc_auc_score(
                    y_true,
                    y_prob[:, 1],
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

            logger.info("Evaluation completed successfully.")

            return metrics

        except Exception as error:

            logger.exception(error)
            raise