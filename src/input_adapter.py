"""
Production-grade Input Adapter Module

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
import numpy as np
import pandas as pd

# ==================================================
# Local Imports
# ==================================================

from src.logger import get_logger
from src.exceptions import LoanRiskException
from src.feature_engineering import FeatureEngineer
from src.config import MODEL_PATH

# ==================================================
# Logger
# ==================================================

logger = get_logger("InputAdapter")


class InputAdapter:
    """
    Converts API/user input into the feature format expected
    by the trained ML pipeline.

    Responsibilities
    ----------------
    - Validate incoming input
    - Convert supported input formats to DataFrame
    - Apply feature engineering
    - Extract the production model feature schema
    - Align input features with the training schema
    - Represent missing features using NumPy NaN
    - Preserve exact training column order
    - Return model-ready features
    """

    REQUIRED_COLUMNS = [
        "AMT_CREDIT",
        "AMT_INCOME_TOTAL",
        "AMT_ANNUITY",
        "AMT_GOODS_PRICE",
        "DAYS_EMPLOYED",
        "DAYS_BIRTH",
        "CNT_CHILDREN",
        "CNT_FAM_MEMBERS",
    ]

    def __init__(self) -> None:
        """
        Initialize the InputAdapter.
        """

        self.feature_engineer = FeatureEngineer()

        self.model_path = Path(MODEL_PATH)

        # Production sklearn Pipeline.
        self.model = None

        # Exact feature schema expected by the fitted
        # ColumnTransformer.
        self.expected_columns = None

        logger.info("InputAdapter initialized.")

    # ==================================================
    # Validation
    # ==================================================

    def _validate_input(
        self,
        data: pd.DataFrame,
    ) -> None:
        """
        Validate incoming prediction data.

        Raises
        ------
        ValueError
            If the input is empty, required columns are missing,
            or a complete row contains no usable information.
        """

        if data.empty:
            raise ValueError(
                "Prediction input cannot be empty."
            )

        missing_columns = [
            column
            for column in self.REQUIRED_COLUMNS
            if column not in data.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing required input columns: "
                f"{missing_columns}"
            )

        if data.isnull().all(axis=1).any():
            raise ValueError(
                "Prediction input contains an entirely "
                "empty row."
            )

    # ==================================================
    # Input Conversion
    # ==================================================

    def _to_dataframe(
        self,
        data,
    ) -> pd.DataFrame:
        """
        Convert supported input formats into a DataFrame.

        Supported formats
        -----------------
        - pandas DataFrame
        - dictionary
        - list of dictionaries
        """

        if isinstance(data, pd.DataFrame):
            return data.copy()

        if isinstance(data, dict):
            return pd.DataFrame([data])

        if isinstance(data, list):
            return pd.DataFrame(data)

        raise TypeError(
            "Input must be a pandas DataFrame, "
            "dictionary, or list of dictionaries."
        )

    # ==================================================
    # Model Loading
    # ==================================================

    def _load_model(self):
        """
        Load the production model if it has not already
        been loaded.

        Returns
        -------
        sklearn.pipeline.Pipeline
            Fitted production pipeline.
        """

        try:

            if self.model is not None:
                return self.model

            if not self.model_path.exists():
                raise FileNotFoundError(
                    f"Production model not found:\n"
                    f"{self.model_path}"
                )

            logger.info(
                f"Loading production model from "
                f"{self.model_path}"
            )

            model = joblib.load(self.model_path)

            # --------------------------------------------------
            # Validate Pipeline structure
            # --------------------------------------------------

            if not hasattr(model, "named_steps"):
                raise TypeError(
                    "Production model must be a "
                    "scikit-learn Pipeline."
                )

            if "preprocessor" not in model.named_steps:
                raise ValueError(
                    "Production model pipeline does not "
                    "contain a 'preprocessor' step."
                )

            preprocessor = model.named_steps["preprocessor"]

            if not hasattr(
                preprocessor,
                "transformers_",
            ):
                raise ValueError(
                    "Production model preprocessor is not "
                    "fitted."
                )

            self.model = model

            logger.info(
                "Production model loaded successfully."
            )

            return self.model

        except Exception as error:

            logger.exception(error)

            raise LoanRiskException(
                error,
                sys,
            )

    # ==================================================
    # Model Schema
    # ==================================================

    def _load_model_schema(self) -> list[str]:
        """
        Extract the exact feature schema expected by the
        fitted production model's ColumnTransformer.

        Returns
        -------
        list[str]
            Ordered list of training features.
        """

        try:

            if self.expected_columns is not None:
                return self.expected_columns

            model = self._load_model()

            preprocessor = model.named_steps[
                "preprocessor"
            ]

            expected_columns = []

            # --------------------------------------------------
            # Extract columns used by each transformer
            # --------------------------------------------------

            for (
                transformer_name,
                transformer,
                columns,
            ) in preprocessor.transformers_:

                # Remainder does not represent explicitly
                # declared training features.
                if transformer_name == "remainder":
                    continue

                if columns is None:
                    continue

                if isinstance(columns, (list, tuple)):
                    expected_columns.extend(columns)

                else:

                    try:
                        expected_columns.extend(
                            list(columns)
                        )

                    except TypeError:
                        expected_columns.append(
                            columns
                        )

            # --------------------------------------------------
            # Remove duplicates while preserving order
            # --------------------------------------------------

            self.expected_columns = list(
                dict.fromkeys(expected_columns)
            )

            # --------------------------------------------------
            # Validate extracted schema
            # --------------------------------------------------

            if not self.expected_columns:
                raise ValueError(
                    "Could not determine the feature schema "
                    "from the production model."
                )

            logger.info(
                f"Production model expects "
                f"{len(self.expected_columns)} features."
            )

            return self.expected_columns

        except Exception as error:

            logger.exception(error)

            raise LoanRiskException(
                error,
                sys,
            )

    # ==================================================
    # Feature Alignment
    # ==================================================

    def _align_to_model_schema(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Align feature-engineered data with the exact schema
        expected by the production model.

        Missing model features are represented using
        NumPy NaN so that the fitted preprocessing pipeline
        can handle them using its training-time imputers.

        Extra columns are removed.

        The final column order exactly matches the training
        schema.

        Notes
        -----
        Missing columns are added in a single DataFrame
        operation using pd.concat() to avoid DataFrame
        fragmentation caused by repeated column insertion.
        """

        try:

            expected_columns = (
                self._load_model_schema()
            )

            # --------------------------------------------------
            # Identify columns
            # --------------------------------------------------

            existing_columns = [
                column
                for column in expected_columns
                if column in data.columns
            ]

            missing_columns = [
                column
                for column in expected_columns
                if column not in data.columns
            ]

            extra_columns = [
                column
                for column in data.columns
                if column not in expected_columns
            ]

            # --------------------------------------------------
            # Log unexpected columns
            # --------------------------------------------------

            if extra_columns:

                logger.info(
                    f"Removing {len(extra_columns)} "
                    f"unexpected prediction features."
                )

            # --------------------------------------------------
            # Keep existing model features
            # --------------------------------------------------

            aligned_existing = data[
                existing_columns
            ].copy()

            # --------------------------------------------------
            # Add missing features using NumPy NaN
            #
            # IMPORTANT:
            # Do NOT use pd.NA here.
            #
            # sklearn's preprocessing pipeline expects values
            # that can be converted to NumPy numeric arrays.
            # pd.NA can produce:
            #
            # TypeError:
            # float() argument must be a string or a real
            # number, not 'NAType'
            #
            # np.nan is compatible with sklearn's SimpleImputer.
            # --------------------------------------------------

            if missing_columns:

                missing_data = pd.DataFrame(
                    np.nan,
                    index=data.index,
                    columns=missing_columns,
                )

                aligned_data = pd.concat(
                    [
                        aligned_existing,
                        missing_data,
                    ],
                    axis=1,
                )

            else:

                aligned_data = aligned_existing

            # --------------------------------------------------
            # Match exact training column order
            # --------------------------------------------------

            aligned_data = aligned_data[
                expected_columns
            ].copy()

            # --------------------------------------------------
            # Defensive NA normalization
            #
            # Convert any remaining pandas nullable NA values
            # to NumPy NaN.
            # --------------------------------------------------

            aligned_data = aligned_data.replace(
                {pd.NA: np.nan}
            )

            # --------------------------------------------------
            # Validation
            # --------------------------------------------------

            if list(aligned_data.columns) != expected_columns:
                raise ValueError(
                    "Prediction feature order does not match "
                    "the production model schema."
                )

            if aligned_data.shape[1] != len(
                expected_columns
            ):
                raise ValueError(
                    "Prediction feature count does not match "
                    "the production model schema."
                )

            # --------------------------------------------------
            # Logging
            # --------------------------------------------------

            logger.info(
                "Feature alignment completed."
            )

            logger.info(
                f"Original adapted shape : "
                f"{data.shape}"
            )

            logger.info(
                f"Aligned model shape    : "
                f"{aligned_data.shape}"
            )

            logger.info(
                f"Missing model features "
                f"filled with NaN       : "
                f"{len(missing_columns)}"
            )

            logger.info(
                f"Extra features removed : "
                f"{len(extra_columns)}"
            )

            logger.info(
                f"Final model feature count : "
                f"{len(aligned_data.columns)}"
            )

            return aligned_data

        except Exception as error:

            logger.exception(error)

            raise LoanRiskException(
                error,
                sys,
            )

    # ==================================================
    # Main Adapter
    # ==================================================

    def adapt(
        self,
        data,
    ) -> pd.DataFrame:
        """
        Convert raw prediction input into model-ready features.

        Parameters
        ----------
        data
            Raw input from API or application.

        Returns
        -------
        pd.DataFrame
            Feature-engineered and schema-aligned input.
        """

        try:

            logger.info(
                "Starting input adaptation..."
            )

            # ------------------------------------------
            # Convert input to DataFrame
            # ------------------------------------------

            dataframe = self._to_dataframe(data)

            logger.info(
                f"Input converted to DataFrame: "
                f"{dataframe.shape}"
            )

            # ------------------------------------------
            # Validate
            # ------------------------------------------

            self._validate_input(
                dataframe
            )

            # ------------------------------------------
            # Feature Engineering
            # ------------------------------------------

            adapted_data = (
                self.feature_engineer.transform(
                    dataframe
                )
            )

            # ------------------------------------------
            # Align with Production Model
            # ------------------------------------------

            aligned_data = (
                self._align_to_model_schema(
                    adapted_data
                )
            )

            # ------------------------------------------
            # Final Safety Check
            # ------------------------------------------

            # pd.NA must never reach sklearn.
            if aligned_data.map(
                lambda value: value is pd.NA
            ).any().any():

                raise ValueError(
                    "Prediction input contains pd.NA "
                    "after schema alignment."
                )

            logger.info(
                "Input adaptation completed successfully."
            )

            logger.info(
                f"Final model input shape: "
                f"{aligned_data.shape}"
            )

            return aligned_data

        except Exception as error:

            logger.exception(error)

            raise LoanRiskException(
                error,
                sys,
            )

    # ==================================================
    # Single Prediction Helper
    # ==================================================

    def adapt_single(
        self,
        data: dict,
    ) -> pd.DataFrame:
        """
        Adapt a single loan application.

        Parameters
        ----------
        data : dict
            Single customer/application input.

        Returns
        -------
        pd.DataFrame
            One-row model-ready DataFrame.
        """

        try:

            adapted_data = self.adapt(
                data
            )

            if len(adapted_data) != 1:

                raise ValueError(
                    "adapt_single() expects exactly "
                    "one input record."
                )

            return adapted_data

        except Exception as error:

            logger.exception(error)

            raise LoanRiskException(
                error,
                sys,
            )