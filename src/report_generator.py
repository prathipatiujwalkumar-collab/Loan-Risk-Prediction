"""
Production-grade Report Generator Module

Author: Ujwal
Project: Intelligent Loan Risk Assessment Platform
"""

# ==================================================
# Standard Library Imports
# ==================================================

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ==================================================
# Local Imports
# ==================================================

from src.logger import get_logger
from src.exceptions import LoanRiskException
from src.config import REPORTS_DIR

# ==================================================
# Logger
# ==================================================

logger = get_logger(__name__)


class ReportGenerator:
    """
    Generates structured JSON reports for model experiments
    and evaluation results.
    """

    def __init__(
        self,
        reports_dir: Path = REPORTS_DIR,
    ) -> None:
        """
        Initialize the ReportGenerator.

        Parameters
        ----------
        reports_dir : Path
            Directory where reports will be stored.
        """

        self.reports_dir = Path(reports_dir)

        self.reports_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        logger.info(
            f"ReportGenerator initialized at "
            f"{self.reports_dir}"
        )

    # ==================================================
    # JSON Serialization Helper
    # ==================================================

    @staticmethod
    def _serialize_value(value: Any) -> Any:
        """
        Convert common ML/Python objects into JSON-safe values.
        """

        # NumPy arrays / scalar objects
        if hasattr(value, "tolist"):

            return value.tolist()

        # Path objects
        if isinstance(value, Path):

            return str(value)

        # Dictionaries
        if isinstance(value, dict):

            return {
                str(key): ReportGenerator._serialize_value(
                    item
                )
                for key, item in value.items()
            }

        # Lists / tuples
        if isinstance(value, (list, tuple)):

            return [
                ReportGenerator._serialize_value(item)
                for item in value
            ]

        # Sets
        if isinstance(value, set):

            return [
                ReportGenerator._serialize_value(item)
                for item in value
            ]

        # Primitive JSON values
        if value is None or isinstance(
            value,
            (str, int, float, bool),
        ):

            return value

        # Fallback
        return str(value)

    # ==================================================
    # Filename Validation
    # ==================================================

    @staticmethod
    def _validate_filename(filename: str) -> str:
        """
        Validate the report filename.
        """

        if not filename:

            raise ValueError(
                "Report filename cannot be empty."
            )

        path = Path(filename)

        if path.name != filename:

            raise ValueError(
                "Filename must not contain directory paths."
            )

        if path.suffix.lower() != ".json":

            raise ValueError(
                "Report filename must have a .json extension."
            )

        return filename

    # ==================================================
    # Generate Report
    # ==================================================

    def generate(
        self,
        model_name: str,
        metrics: dict,
        filename: str = "experiment_report.json",
        overwrite: bool = True,
    ) -> Path:
        """
        Generate and save a JSON model evaluation report.

        Parameters
        ----------
        model_name : str
            Name of the trained model.

        metrics : dict
            Evaluation metrics.

        filename : str
            Output JSON filename.

        overwrite : bool
            Whether an existing report can be overwritten.

        Returns
        -------
        Path
            Path to the generated report.
        """

        try:

            # ------------------------------------------
            # Validate Inputs
            # ------------------------------------------

            if not model_name:

                raise ValueError(
                    "Model name cannot be empty."
                )

            if not isinstance(metrics, dict):

                raise TypeError(
                    "Metrics must be provided as a dictionary."
                )

            filename = self._validate_filename(
                filename
            )

            report_path = (
                self.reports_dir / filename
            )

            # ------------------------------------------
            # Existing File Check
            # ------------------------------------------

            if (
                report_path.exists()
                and not overwrite
            ):

                raise FileExistsError(
                    f"Report already exists: "
                    f"{report_path}"
                )

            # ------------------------------------------
            # Build Report
            # ------------------------------------------

            report = {
                "project": (
                    "Intelligent Loan Risk "
                    "Assessment Platform"
                ),
                "model": model_name,
                "timestamp": datetime.now(
                    timezone.utc
                ).isoformat(),
                "metrics": self._serialize_value(
                    metrics
                ),
            }

            # ------------------------------------------
            # Write JSON
            # ------------------------------------------

            with report_path.open(
                "w",
                encoding="utf-8",
            ) as file:

                json.dump(
                    report,
                    file,
                    indent=4,
                    ensure_ascii=False,
                )

            logger.info(
                f"Experiment report saved to "
                f"{report_path}"
            )

            return report_path

        except Exception as error:

            logger.exception(error)

            raise LoanRiskException(
                error,
                sys,
            )

    # ==================================================
    # Load Report
    # ==================================================

    def load(
        self,
        filename: str,
    ) -> dict:
        """
        Load an existing JSON report.

        Parameters
        ----------
        filename : str
            Report filename.

        Returns
        -------
        dict
            Report contents.
        """

        try:

            filename = self._validate_filename(
                filename
            )

            report_path = (
                self.reports_dir / filename
            )

            if not report_path.exists():

                raise FileNotFoundError(
                    f"Report not found: "
                    f"{report_path}"
                )

            with report_path.open(
                "r",
                encoding="utf-8",
            ) as file:

                report = json.load(file)

            logger.info(
                f"Report loaded successfully from "
                f"{report_path}"
            )

            return report

        except Exception as error:

            logger.exception(error)

            raise LoanRiskException(
                error,
                sys,
            )