"""
Test script for ReportGenerator.

Author: Ujwal
Project: Intelligent Loan Risk Assessment Platform
"""

from src.report_generator import ReportGenerator


def main():

    # ==================================================
    # Initialize
    # ==================================================

    generator = ReportGenerator()

    # ==================================================
    # Sample Evaluation Results
    # ==================================================

    metrics = {
        "Accuracy": 0.91,
        "Precision": 0.87,
        "Recall": 0.82,
        "F1": 0.84,
        "ROC_AUC": 0.95,
        "Confusion Matrix": [
            [56000, 1200],
            [3500, 5000],
        ],
    }

    # ==================================================
    # Generate Report
    # ==================================================

    report_path = generator.generate(
        model_name="LogisticRegression",
        metrics=metrics,
        filename="logistic_regression_report.json",
    )

    print("\nReport Generated")
    print("=" * 50)
    print(report_path)

    # ==================================================
    # Load Report
    # ==================================================

    report = generator.load(
        "logistic_regression_report.json"
    )

    print("\nLoaded Report")
    print("=" * 50)

    print(f"Project   : {report['project']}")
    print(f"Model     : {report['model']}")
    print(f"Timestamp : {report['timestamp']}")

    print("\nMetrics")
    print("=" * 50)

    for name, value in report["metrics"].items():

        print(f"{name}: {value}")

    print("\nReport Generator Test Successful")


if __name__ == "__main__":
    main()