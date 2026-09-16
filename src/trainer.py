"""
Production-grade Model Trainer Module

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

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_is_fitted

# ==================================================
# Local Imports
# ==================================================

from src.logger import get_logger
from src.exceptions import LoanRiskException
from src.preprocessor import Preprocessor
from src.config import (
    RANDOM_STATE,
    MAX_ITER,
    CLASS_WEIGHT,
    MODEL_PATH,
)

# ==================================================
# Logger
# ==================================================

logger = get_logger(__name__)


class ModelTrainer:
    """
    Handles model training, prediction, saving, and loading.

    Supports any compatible Scikit-Learn estimator.
    """

    def __init__(self, model=None) -> None:
        """
        Initialize ModelTrainer.

        Parameters
        ----------
        model : sklearn estimator, optional
            Machine learning model.
            Defaults to Logistic Regression.
        """

        logger.info("ModelTrainer initialized.")

        if model is None:

            model = LogisticRegression(
                random_state=RANDOM_STATE,
                max_iter=MAX_ITER,
                class_weight=CLASS_WEIGHT,
                solver="lbfgs",
            )

        self.model = model
        self.pipeline = None

    # ==================================================
    # Build Pipeline
    # ==================================================

    def build_pipeline(
        self,
        X_train: pd.DataFrame,
    ) -> Pipeline:
        """
        Build the preprocessing + model pipeline.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training feature dataframe.

        Returns
        -------
        Pipeline
            Complete Scikit-Learn pipeline.
        """

        try:

            if X_train is None or X_train.empty:

                raise ValueError(
                    "X_train cannot be None or empty."
                )

            preprocessor = Preprocessor()

            transformer = preprocessor.build(
                X_train
            )

            self.pipeline = Pipeline(
                steps=[
                    (
                        "preprocessor",
                        transformer,
                    ),
                    (
                        "model",
                        self.model,
                    ),
                ]
            )

            logger.info(
                "Pipeline built successfully."
            )

            logger.info(
                f"Model: "
                f"{self.model.__class__.__name__}"
            )

            return self.pipeline

        except Exception as error:

            logger.exception(error)

            raise LoanRiskException(
                error,
                sys,
            )

    # ==================================================
    # Train
    # ==================================================

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> Pipeline:
        """
        Train the complete preprocessing + model pipeline.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training features.

        y_train : pd.Series
            Training target.

        Returns
        -------
        Pipeline
            Fitted pipeline.
        """

        try:

            if X_train is None or X_train.empty:

                raise ValueError(
                    "X_train cannot be None or empty."
                )

            if y_train is None or y_train.empty:

                raise ValueError(
                    "y_train cannot be None or empty."
                )

            if len(X_train) != len(y_train):

                raise ValueError(
                    "X_train and y_train must contain "
                    "the same number of samples."
                )

            if self.pipeline is None:

                self.build_pipeline(X_train)

            logger.info(
                "Model training started..."
            )

            self.pipeline.fit(
                X_train,
                y_train,
            )

            # Verify that the complete pipeline
            # was successfully fitted.
            check_is_fitted(
                self.pipeline
            )

            logger.info(
                "Model training completed successfully."
            )

            logger.info(
                f"Training samples: {len(X_train)}"
            )

            logger.info(
                f"Training features: "
                f"{X_train.shape[1]}"
            )

            return self.pipeline

        except Exception as error:

            logger.exception(error)

            raise LoanRiskException(
                error,
                sys,
            )

    # ==================================================
    # Prediction
    # ==================================================

    def _validate_pipeline(self) -> None:
        """
        Ensure that the training pipeline exists
        and has been fitted.
        """

        if self.pipeline is None:

            raise ValueError(
                "Pipeline has not been initialized."
            )

        check_is_fitted(
            self.pipeline
        )

    # ==================================================
    # Predict
    # ==================================================

    def predict(
        self,
        X: pd.DataFrame,
    ):
        """
        Predict class labels.

        Parameters
        ----------
        X : pd.DataFrame
            Input features.

        Returns
        -------
        ndarray
            Predicted class labels.
        """

        try:

            if X is None or X.empty:

                raise ValueError(
                    "Prediction input cannot be "
                    "None or empty."
                )

            self._validate_pipeline()

            predictions = self.pipeline.predict(X)

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
    # Predict Probabilities
    # ==================================================

    def predict_proba(
        self,
        X: pd.DataFrame,
    ):
        """
        Predict class probabilities.

        Parameters
        ----------
        X : pd.DataFrame
            Input features.

        Returns
        -------
        ndarray
            Predicted class probabilities.
        """

        try:

            if X is None or X.empty:

                raise ValueError(
                    "Prediction input cannot be "
                    "None or empty."
                )

            self._validate_pipeline()

            if not hasattr(
                self.pipeline,
                "predict_proba",
            ):

                raise AttributeError(
                    f"{self.model.__class__.__name__} "
                    "does not support predict_proba()."
                )

            probabilities = (
                self.pipeline.predict_proba(X)
            )

            logger.info(
                "Prediction probabilities "
                "generated successfully."
            )

            return probabilities

        except Exception as error:

            logger.exception(error)

            raise LoanRiskException(
                error,
                sys,
            )

    # ==================================================
    # Save
    # ==================================================

    # ==================================================
# Save Model
# ==================================================

    def save(
        self,
        filepath: str | Path = MODEL_PATH,
    ) -> Path:
        """
        Save the complete fitted preprocessing + model
        pipeline to disk.

        Parameters
        ----------
        filepath : str or Path
            Destination path for the production model.

        Returns
        -------
        Path
            Path of the saved model.
        """

        try:

            filepath = Path(filepath)

            # --------------------------------------------------
            # Validate pipeline
            # --------------------------------------------------

            self._validate_pipeline()

            # --------------------------------------------------
            # Create parent directory if necessary
            # --------------------------------------------------

            filepath.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            # --------------------------------------------------
            # Save COMPLETE fitted pipeline
            # --------------------------------------------------

            joblib.dump(
                self.pipeline,
                filepath,
            )

            logger.info(
                f"Production pipeline saved successfully: "
                f"{filepath}"
            )

            logger.info(
                f"Saved model type: "
                f"{self.model.__class__.__name__}"
            )

            logger.info(
                "Saved object contains preprocessing "
                "and fitted estimator."
            )

            return filepath

        except Exception as error:

            logger.exception(error)

            raise LoanRiskException(
                error,
                sys,
            )
    # ==================================================
    # Load
    # ==================================================

    def load(
        self,
        filepath: str | Path = MODEL_PATH,
    ) -> Pipeline:
        """
        Load a previously trained pipeline.

        Parameters
        ----------
        filepath : str or Path
            Path to saved pipeline.

        Returns
        -------
        Pipeline
            Loaded fitted pipeline.
        """

        try:

            filepath = Path(filepath)

            if not filepath.exists():

                raise FileNotFoundError(
                    f"Model file not found: {filepath}"
                )

            loaded_pipeline = joblib.load(
                filepath
            )

            check_is_fitted(
                loaded_pipeline
            )

            self.pipeline = loaded_pipeline

            logger.info(
                f"Model loaded successfully from "
                f"{filepath}"
            )

            return self.pipeline

        except Exception as error:

            logger.exception(error)

            raise LoanRiskException(
                error,
                sys,
            )