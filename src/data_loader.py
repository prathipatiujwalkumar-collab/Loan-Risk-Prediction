"""
Data loading module for the Loan Risk Prediction project.

This module provides the DataLoader class responsible for loading
datasets from disk with logging and exception handling.
"""

# --------------------------------------------------
# Standard Library Imports
# --------------------------------------------------

import sys
from pathlib import Path

# --------------------------------------------------
# Third-Party Imports
# --------------------------------------------------

import pandas as pd

# --------------------------------------------------
# Local Imports
# --------------------------------------------------

from src.config import TRAIN_FILE
from src.exceptions import LoanRiskException
from src.logger import get_logger


class DataLoader:
    """
    Handles loading datasets from disk.

    Methods
    -------
    load_csv(file_path)
        Load any CSV file.

    load_training_data()
        Load the training dataset.
    """

    def __init__(self) -> None:
        """Initialize the DataLoader."""

        self.logger = get_logger(self.__class__.__name__)

    def load_csv(self, file_path: Path) -> pd.DataFrame:
        """
        Load a CSV file.

        Parameters
        ----------
        file_path : Path
            Path to the CSV file.

        Returns
        -------
        pd.DataFrame
            Loaded dataframe.
        """

        try:

            self.logger.info(f"Loading dataset: {file_path}")

            if not file_path.exists():
                raise FileNotFoundError(
                    f"Dataset not found: {file_path}"
                )

            df = pd.read_csv(file_path)

            if df.empty:
                raise ValueError(
                    f"Dataset is empty: {file_path}"
                )

            self.logger.info(
                f"Successfully loaded dataset with shape {df.shape}"
            )

            return df

        except Exception as error:
            self.logger.error(str(error))
            raise LoanRiskException(error, sys)

    def load_training_data(self) -> pd.DataFrame:
        """
        Load training dataset.
        """

        return self.load_csv(TRAIN_FILE)
