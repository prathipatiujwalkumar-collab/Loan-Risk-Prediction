"""
Production-grade Data Loader Module

Author: Ujwal
Project: Intelligent Loan Risk Assessment Platform
"""

# ==================================================
# Standard Library Imports
# ==================================================

import sys
from pathlib import Path

# ==================================================
# Third-Party Imports
# ==================================================

import pandas as pd

# ==================================================
# Local Imports
# ==================================================

from src.config import TRAIN_FILE
from src.exceptions import LoanRiskException
from src.logger import get_logger

# ==================================================
# Logger
# ==================================================

logger = get_logger(__name__)


class DataLoader:
    """
    Loads datasets from disk with validation,
    logging, and exception handling.
    """

    def __init__(self) -> None:

        logger.info("DataLoader initialized.")

    def load_csv(
        self,
        file_path: Path,
    ) -> pd.DataFrame:
        """
        Load a CSV dataset.

        Parameters
        ----------
        file_path : Path
            Path to the dataset.

        Returns
        -------
        pd.DataFrame
            Loaded dataframe.
        """

        try:

            file_path = Path(file_path)

            logger.info(f"Loading dataset: {file_path}")

            if not file_path.exists():

                raise FileNotFoundError(
                    f"Dataset not found: {file_path}"
                )

            df = pd.read_csv(file_path)

            if df.empty:

                raise ValueError(
                    f"Dataset is empty: {file_path}"
                )

            logger.info(
                f"Successfully loaded dataset with shape {df.shape}"
            )

            return df

        except Exception as error:

            logger.exception(error)

            raise LoanRiskException(error, sys)

    def load_training_data(self) -> pd.DataFrame:
        """
        Load the project's training dataset.
        """

        return self.load_csv(TRAIN_FILE)

    def load_dataframe(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Validate an already loaded dataframe.

        Useful for APIs and Streamlit uploads.
        """

        try:

            if dataframe.empty:

                raise ValueError(
                    "Input dataframe is empty."
                )

            logger.info(
                f"Dataframe accepted with shape {dataframe.shape}"
            )

            return dataframe

        except Exception as error:

            logger.exception(error)

            raise LoanRiskException(error, sys)