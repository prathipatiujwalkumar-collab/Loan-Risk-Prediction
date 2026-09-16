"""
Production-grade MLflow Tracking Module

Author: Ujwal
Project: Intelligent Loan Risk Assessment Platform
"""

# ==================================================
# Standard Library Imports
# ==================================================
import os
import sys
from pathlib import Path
from typing import Any

# ==================================================
# Third-Party Imports
# ==================================================

import mlflow
import mlflow.sklearn

# ==================================================
# Local Imports
# ==================================================

from src.logger import get_logger
from src.exceptions import LoanRiskException

# ==================================================
# Logger
# ==================================================

logger = get_logger("MLflowTracker")


class MLflowTracker:
    """
    Handles MLflow experiment tracking.
    """

    def __init__(
        self,
        experiment_name: str = "Loan Risk Prediction",
    ) -> None:

        try:

            tracking_uri = os.getenv(
                "MLFLOW_TRACKING_URI",
                "http://localhost:5000",
            )

            mlflow.set_tracking_uri(
                tracking_uri
            )

            logger.info(
                f"MLflow tracking URI: {tracking_uri}"
            )

            mlflow.set_experiment(
                experiment_name
            )

        except Exception as error:

            logger.exception(error)

            raise LoanRiskException(error, sys)

    # -------------------------------------------------

    def start_run(
        self,
        run_name: str | None = None,
    ):

        try:

            logger.info(
                f"Starting MLflow run: {run_name}"
            )

            return mlflow.start_run(
                run_name=run_name
            )

        except Exception as error:

            logger.exception(error)

            raise LoanRiskException(error, sys)

    # -------------------------------------------------

    def end_run(self) -> None:

        try:

            if mlflow.active_run():

                mlflow.end_run()

                logger.info("MLflow run ended.")

        except Exception as error:

            logger.exception(error)

            raise LoanRiskException(error, sys)

    # -------------------------------------------------

    def log_params(
        self,
        params: dict,
    ) -> None:

        try:

            mlflow.log_params(params)

            logger.info("Parameters logged.")

        except Exception as error:

            logger.exception(error)

            raise LoanRiskException(error, sys)

    # -------------------------------------------------

    def log_metrics(
        self,
        metrics: dict,
    ) -> None:

        try:

            for key, value in metrics.items():

                try:

                    mlflow.log_metric(
                        key,
                        float(value),
                    )

                except (ValueError, TypeError):

                    continue

            logger.info("Metrics logged.")

        except Exception as error:

            logger.exception(error)

            raise LoanRiskException(error, sys)

    # -------------------------------------------------

    def log_tags(
        self,
        tags: dict,
    ) -> None:

        try:

            mlflow.set_tags(tags)

            logger.info("Tags logged.")

        except Exception as error:

            logger.warning(
                f"Unable to log tags: {error}"
            )

    # -------------------------------------------------

    def log_model(
        self,
        pipeline: Any,
    ) -> None:

        """
        Log sklearn pipeline.
        """

        try:

            mlflow.sklearn.log_model(
                sk_model=pipeline,
                name="model",
                serialization_format="cloudpickle",
            )

            logger.info("Model logged.")

        except Exception as error:

            logger.warning(
                f"MLflow model logging skipped: {error}"
            )

    # -------------------------------------------------

    def log_artifact(
        self,
        file_path: str | Path,
    ) -> None:

        try:

            mlflow.log_artifact(str(file_path))

            logger.info(
                f"Artifact logged: {file_path}"
            )

        except Exception as error:

            logger.warning(
                f"Artifact logging skipped: {error}"
            )

    # -------------------------------------------------

    def log_dict(
        self,
        dictionary: dict,
        artifact_file: str,
    ) -> None:

        try:

            mlflow.log_dict(
                dictionary,
                artifact_file,
            )

            logger.info(
                f"Dictionary logged: {artifact_file}"
            )

        except Exception as error:

            logger.warning(
                f"Dictionary logging skipped: {error}"
            )