"""
Production-grade Predictor Module

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

import joblib
import pandas as pd

from sklearn.utils.validation import check_is_fitted

# ==================================================
# Local Imports
# ==================================================

from src.logger import get_logger
from src.exceptions import LoanRiskException
from src.config import MODEL_PATH

# ==================================================
# Logger
# ==================================================

logger = get_logger(__name__)


class Predictor:
    """
    Production prediction service responsible for loading
    the trained ML pipeline and generating predictions.

    Responsibilities
    ----------------
    - Load the production model
    - Validate model state
    - Validate prediction input
    - Generate class predictions
    - Generate prediction probabilities
    - Support single-record predictions
    """

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
    ) -> None:
        """
        Initialize the Predictor.

        Parameters
        ----------
        model_path : Path
            Path to the serialized production model.
        """

        self.model_path = Path(model_path)
        self.pipeline = None

        logger.info("Predictor initialized.")

    # ==================================================
    # Model Loading
    # ==================================================

    def load_model(self):
        """
        Load and validate the production ML pipeline.

        Returns
        -------
        object
            Loaded and fitted sklearn pipeline.
        """

        try:

            # --------------------------------------------------
            # Validate model path
            # --------------------------------------------------

            if not self.model_path.exists():

                raise FileNotFoundError(
                    f"Production model not found:\n"
                    f"{self.model_path}"
                )

            logger.info(
                f"Loading production model from "
                f"{self.model_path}"
            )

            # --------------------------------------------------
            # Load serialized model
            # --------------------------------------------------

            self.pipeline = joblib.load(
                self.model_path
            )

            # --------------------------------------------------
            # Validate fitted state
            # --------------------------------------------------

            check_is_fitted(
                self.pipeline
            )

            # --------------------------------------------------
            # Validate prediction interface
            # --------------------------------------------------

            if not hasattr(
                self.pipeline,
                "predict",
            ):

                raise TypeError(
                    "Production model does not implement "
                    "'predict()'."
                )

            # --------------------------------------------------
            # Validate probability interface
            # --------------------------------------------------

            if not hasattr(
                self.pipeline,
                "predict_proba",
            ):

                logger.warning(
                    "Production model does not implement "
                    "'predict_proba()'."
                )

            # --------------------------------------------------
            # Model metadata
            # --------------------------------------------------

            model_name = type(
                self.pipeline
            ).__name__

            logger.info(
                f"Production model loaded successfully. "
                f"Type: {model_name}"
            )

            return self.pipeline

        except Exception as error:

            logger.exception(error)

            raise LoanRiskException(
                error,
                sys,
            )

    # ==================================================
    # Ensure Model Loaded
    # ==================================================

    def _ensure_model_loaded(self) -> None:
        """
        Load the production model if it has not already
        been loaded.
        """

        if self.pipeline is None:
            self.load_model()

    # ==================================================
    # Input Validation
    # ==================================================

    @staticmethod
    def _validate_input(
        X: pd.DataFrame,
    ) -> None:
        """
        Validate prediction input.

        Parameters
        ----------
        X : pd.DataFrame
            Model-ready feature DataFrame.
        """

        if not isinstance(
            X,
            pd.DataFrame,
        ):

            raise ValueError(
                "Prediction input must be a "
                "pandas DataFrame."
            )

        if X.empty:

            raise ValueError(
                "Prediction input DataFrame is empty."
            )

        if X.shape[1] == 0:

            raise ValueError(
                "Prediction input contains no features."
            )

    # ==================================================
    # Prediction
    # ==================================================

    def predict(
        self,
        X: pd.DataFrame,
    ):
        """
        Generate class predictions.

        Parameters
        ----------
        X : pd.DataFrame
            Model-ready feature DataFrame.

        Returns
        -------
        numpy.ndarray
            Predicted classes.
        """

        try:

            self._ensure_model_loaded()

            self._validate_input(X)

            logger.info(
                f"Generating predictions for "
                f"{len(X)} records."
            )

            predictions = self.pipeline.predict(
                X
            )

            logger.info(
                f"Generated {len(predictions)} predictions."
            )

            return predictions

        except Exception as error:

            logger.exception(error)

            raise LoanRiskException(
                error,
                sys,
            )

    # ==================================================
    # Prediction Probabilities
    # ==================================================

    def predict_proba(
        self,
        X: pd.DataFrame,
    ):
        """
        Generate class probabilities.

        Parameters
        ----------
        X : pd.DataFrame
            Model-ready feature DataFrame.

        Returns
        -------
        numpy.ndarray
            Prediction probabilities for each class.
        """

        try:

            self._ensure_model_loaded()

            self._validate_input(X)

            # --------------------------------------------------
            # Ensure probability prediction is supported
            # --------------------------------------------------

            if not hasattr(
                self.pipeline,
                "predict_proba",
            ):

                raise AttributeError(
                    "Production model does not support "
                    "probability predictions."
                )

            logger.info(
                f"Generating prediction probabilities "
                f"for {len(X)} records."
            )

            probabilities = (
                self.pipeline.predict_proba(X)
            )

            logger.info(
                "Prediction probabilities generated."
            )

            return probabilities

        except Exception as error:

            logger.exception(error)

            raise LoanRiskException(
                error,
                sys,
            )

    # ==================================================
    # Single Prediction
    # ==================================================

    def predict_single(
        self,
        sample: pd.DataFrame,
    ):
        """
        Generate prediction and probability for
        exactly one loan application.

        Parameters
        ----------
        sample : pd.DataFrame
            One-row model-ready DataFrame.

        Returns
        -------
        tuple
            (
                prediction,
                probability_array
            )
        """

        try:

            self._ensure_model_loaded()

            self._validate_input(sample)

            # --------------------------------------------------
            # Enforce single-record contract
            # --------------------------------------------------

            if len(sample) != 1:

                raise ValueError(
                    "predict_single() expects exactly "
                    "one input record."
                )

            # --------------------------------------------------
            # Generate prediction
            # --------------------------------------------------

            prediction = self.pipeline.predict(
                sample
            )[0]

            # --------------------------------------------------
            # Generate probability
            # --------------------------------------------------

            if not hasattr(
                self.pipeline,
                "predict_proba",
            ):

                raise AttributeError(
                    "Production model does not support "
                    "probability predictions."
                )

            probability = (
                self.pipeline.predict_proba(
                    sample
                )[0]
            )

            logger.info(
                "Single customer prediction completed."
            )

            return (
                int(prediction),
                probability,
            )

        except Exception as error:

            logger.exception(error)

            raise LoanRiskException(
                error,
                sys,
            )