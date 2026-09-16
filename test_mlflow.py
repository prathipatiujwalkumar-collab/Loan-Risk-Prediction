"""
Test Script for MLflow Tracker

Author: Ujwal
Project: Intelligent Loan Risk Assessment Platform
"""

from sklearn.linear_model import LogisticRegression

from src.mlflow_tracker import MLflowTracker


def main():

    print("=" * 60)
    print("MLFLOW TRACKER TEST")
    print("=" * 60)

    tracker = MLflowTracker()

    model = LogisticRegression()

    with tracker.start_run("Test Run"):

        tracker.log_params(
            {
                "model": "LogisticRegression",
                "solver": "lbfgs",
                "max_iter": 2000,
            }
        )

        tracker.log_metrics(
            {
                "Accuracy": 0.91,
                "ROC_AUC": 0.95,
            }
        )

        tracker.log_tags(
            {
                "stage": "testing",
                "author": "Ujwal",
            }
        )

        # This will be skipped gracefully because
        # the model is not fitted.
        tracker.log_model(model)

    tracker.end_run()

    print()

    print("MLflow Tracker Test Passed Successfully")

    print("=" * 60)


if __name__ == "__main__":

    main()