"""
Production-grade Model Registry Module

Author: Ujwal
Project: Intelligent Loan Risk Assessment Platform
"""

# ==================================================
# Standard Library Imports
# ==================================================

import sys
import shutil
from pathlib import Path
from typing import Any

# ==================================================
# Third-Party Imports
# ==================================================

import joblib

# ==================================================
# Local Imports
# ==================================================

from src.logger import get_logger
from src.exceptions import LoanRiskException
from src.config import MODELS_DIR

# ==================================================
# Logger
# ==================================================

logger = get_logger("ModelRegistry")


class ModelRegistry:
    """
    Handles model versioning and production deployment.
    """

    def __init__(
        self,
        registry_dir: Path = MODELS_DIR,
    ) -> None:

        self.registry_dir = Path(registry_dir)

        self.registry_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        logger.info(
            f"Model registry initialized at {self.registry_dir}"
        )

    # ==================================================

    def register_model(
        self,
        model: Any,
        model_name: str,
        version: str = "v1",
        overwrite: bool = False,
    ) -> Path:

        """
        Register a trained model.

        Returns
        -------
        Path
            Saved model path.
        """

        try:

            model_path = (
                self.registry_dir
                / f"{model_name}_{version}.joblib"
            )

            if model_path.exists() and not overwrite:

                raise FileExistsError(
                    f"Model already exists:\n{model_path}"
                )

            joblib.dump(
                model,
                model_path,
            )

            logger.info(
                f"Model registered successfully:\n{model_path}"
            )

            return model_path

        except Exception as error:

            logger.exception(error)

            raise LoanRiskException(error, sys)

    # ==================================================

    def load_model(
        self,
        model_name: str,
        version: str = "v1",
    ) -> Any:

        """
        Load a registered model.
        """

        try:

            model_path = (
                self.registry_dir
                / f"{model_name}_{version}.joblib"
            )

            if not model_path.exists():

                raise FileNotFoundError(
                    f"Model not found:\n{model_path}"
                )

            model = joblib.load(model_path)

            logger.info(
                f"Loaded model:\n{model_path}"
            )

            return model

        except Exception as error:

            logger.exception(error)

            raise LoanRiskException(error, sys)

    # ==================================================

    def load_production_model(self):

        """
        Load current production model.
        """

        try:

            production_path = (
                self.registry_dir
                / "production_model.joblib"
            )

            if not production_path.exists():

                raise FileNotFoundError(
                    "Production model not found."
                )

            model = joblib.load(
                production_path
            )

            logger.info(
                "Production model loaded successfully."
            )

            return model

        except Exception as error:

            logger.exception(error)

            raise LoanRiskException(error, sys)

    # ==================================================

    def list_models(self) -> list[Path]:

        """
        List all registered models.
        """

        try:

            models = sorted(
                self.registry_dir.glob("*.joblib")
            )

            logger.info(
                f"{len(models)} registered models found."
            )

            return models

        except Exception as error:

            logger.exception(error)

            raise LoanRiskException(error, sys)

    # ==================================================

    def delete_model(
        self,
        model_name: str,
        version: str = "v1",
    ) -> None:

        """
        Delete a registered model.
        """

        try:

            model_path = (
                self.registry_dir
                / f"{model_name}_{version}.joblib"
            )

            if model_path.exists():

                model_path.unlink()

                logger.info(
                    f"Deleted model:\n{model_path}"
                )

            else:

                logger.warning(
                    f"Model not found:\n{model_path}"
                )

        except Exception as error:

            logger.exception(error)

            raise LoanRiskException(error, sys)

    # ==================================================

    def promote_model(
        self,
        model_name: str,
        version: str,
    ) -> Path:

        """
        Promote model to production.
        """

        try:

            source = (
                self.registry_dir
                / f"{model_name}_{version}.joblib"
            )

            if not source.exists():

                raise FileNotFoundError(
                    f"Model not found:\n{source}"
                )

            destination = (
                self.registry_dir
                / "production_model.joblib"
            )

            shutil.copy2(
                source,
                destination,
            )

            logger.info(
                f"{model_name} ({version}) promoted to production."
            )

            return destination

        except Exception as error:

            logger.exception(error)

            raise LoanRiskException(error, sys)

    # ==================================================

    def production_model_exists(self) -> bool:

        """
        Check whether a production model exists.
        """

        return (
            self.registry_dir
            / "production_model.joblib"
        ).exists()

    # ==================================================

    def get_latest_version(
        self,
        model_name: str,
    ) -> str | None:

        """
        Return latest registered version.
        """

        versions = sorted(
            self.registry_dir.glob(
                f"{model_name}_v*.joblib"
            )
        )

        if not versions:

            return None

        latest = versions[-1]

        return latest.stem.split("_")[-1]