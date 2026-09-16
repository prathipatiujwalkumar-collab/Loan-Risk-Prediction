"""
Production-grade Preprocessing Module

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

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.utils.validation import check_is_fitted

# ==================================================
# Local Imports
# ==================================================

from src.logger import get_logger
from src.exceptions import LoanRiskException
from src.config import TARGET_COLUMN, ARTIFACTS_DIR

# ==================================================
# Logger
# ==================================================

logger = get_logger(__name__)


class Preprocessor:
    """
    Builds and manages the preprocessing pipeline.

    Responsibilities
    -----------------
    - Detect numerical and categorical features
    - Handle missing values
    - Scale numerical variables
    - Encode categorical variables
    - Transform training and unseen data
    - Save and load fitted preprocessors
    - Provide transformed feature names
    """

    def __init__(self) -> None:

        self.transformer = None

        logger.info("Preprocessor initialized.")

    # ==================================================
    # Validation
    # ==================================================

    @staticmethod
    def _validate_dataframe(df: pd.DataFrame) -> None:
        """
        Validate that the input is a non-empty DataFrame.
        """

        if not isinstance(df, pd.DataFrame):

            raise TypeError(
                "Input must be a pandas DataFrame."
            )

        if df.empty:

            raise ValueError(
                "Input DataFrame is empty."
            )

    # ==================================================
    # Feature Detection
    # ==================================================

    def get_feature_types(
        self,
        df: pd.DataFrame,
    ) -> tuple[list[str], list[str]]:
        """
        Identify numerical and categorical features.

        The target column is excluded automatically.
        """

        try:

            self._validate_dataframe(df)

            feature_df = df.drop(
                columns=[TARGET_COLUMN],
                errors="ignore",
            )

            numerical_features = (
                feature_df
                .select_dtypes(include=["number"])
                .columns
                .tolist()
            )

            categorical_features = (
                feature_df
                .select_dtypes(
                    include=["object", "category", "string"]
                )
                .columns
                .tolist()
            )

            logger.info(
                f"Detected {len(numerical_features)} "
                f"numerical features."
            )

            logger.info(
                f"Detected {len(categorical_features)} "
                f"categorical features."
            )

            return (
                numerical_features,
                categorical_features,
            )

        except Exception as error:

            logger.exception(error)

            raise LoanRiskException(error, sys)

    # ==================================================
    # Build Pipeline
    # ==================================================

    def build(
        self,
        df: pd.DataFrame,
    ) -> ColumnTransformer:
        """
        Build the preprocessing pipeline.

        The pipeline is not fitted by this method.
        """

        try:

            self._validate_dataframe(df)

            (
                numerical_features,
                categorical_features,
            ) = self.get_feature_types(df)

            if not numerical_features and not categorical_features:

                raise ValueError(
                    "No usable features found in the DataFrame."
                )

            # ------------------------------------------
            # Numerical Pipeline
            # ------------------------------------------

            numerical_pipeline = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="median"
                        ),
                    ),
                    (
                        "scaler",
                        StandardScaler(),
                    ),
                ]
            )

            # ------------------------------------------
            # Categorical Pipeline
            # ------------------------------------------

            categorical_pipeline = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="most_frequent"
                        ),
                    ),
                    (
                        "encoder",
                        OneHotEncoder(
                            handle_unknown="ignore",
                            sparse_output=False,
                        ),
                    ),
                ]
            )

            # ------------------------------------------
            # Column Transformer
            # ------------------------------------------

            transformers = []

            if numerical_features:

                transformers.append(
                    (
                        "numerical",
                        numerical_pipeline,
                        numerical_features,
                    )
                )

            if categorical_features:

                transformers.append(
                    (
                        "categorical",
                        categorical_pipeline,
                        categorical_features,
                    )
                )

            self.transformer = ColumnTransformer(
                transformers=transformers,
                remainder="drop",
                verbose_feature_names_out=True,
            )

            logger.info(
                "Preprocessing pipeline built successfully."
            )

            return self.transformer

        except Exception as error:

            logger.exception(error)

            raise LoanRiskException(error, sys)

    # ==================================================
    # Fit + Transform
    # ==================================================

    def fit_transform(
        self,
        X_train: pd.DataFrame,
    ):
        """
        Fit the preprocessor using training data only
        and transform the training data.
        """

        try:

            self._validate_dataframe(X_train)

            if self.transformer is None:

                self.build(X_train)

            logger.info(
                "Fitting preprocessor on training data."
            )

            transformed_data = (
                self.transformer.fit_transform(X_train)
            )

            logger.info(
                "Training data preprocessing completed."
            )

            return transformed_data

        except Exception as error:

            logger.exception(error)

            raise LoanRiskException(error, sys)

    # ==================================================
    # Transform
    # ==================================================

    def transform(
        self,
        X: pd.DataFrame,
    ):
        """
        Transform unseen data using an already-fitted
        preprocessor.
        """

        try:

            self._validate_dataframe(X)

            if self.transformer is None:

                raise ValueError(
                    "Preprocessor has not been fitted."
                )

            check_is_fitted(self.transformer)

            logger.info(
                f"Transforming dataset with shape {X.shape}."
            )

            transformed_data = (
                self.transformer.transform(X)
            )

            logger.info(
                "Data transformation completed successfully."
            )

            return transformed_data

        except Exception as error:

            logger.exception(error)

            raise LoanRiskException(error, sys)

    # ==================================================
    # Feature Names
    # ==================================================

    def get_feature_names(self) -> list[str]:
        """
        Return names of transformed features.
        """

        try:

            if self.transformer is None:

                raise ValueError(
                    "Preprocessor has not been built."
                )

            check_is_fitted(self.transformer)

            feature_names = (
                self.transformer
                .get_feature_names_out()
                .tolist()
            )

            logger.info(
                f"Generated {len(feature_names)} "
                f"transformed feature names."
            )

            return feature_names

        except Exception as error:

            logger.exception(error)

            raise LoanRiskException(error, sys)

    # ==================================================
    # Numerical Features
    # ==================================================

    def get_numerical_features(
        self,
        df: pd.DataFrame,
    ) -> list[str]:
        """
        Return numerical feature names.
        """

        try:

            numerical, _ = self.get_feature_types(df)

            return numerical

        except Exception as error:

            logger.exception(error)

            raise LoanRiskException(error, sys)

    # ==================================================
    # Categorical Features
    # ==================================================

    def get_categorical_features(
        self,
        df: pd.DataFrame,
    ) -> list[str]:
        """
        Return categorical feature names.
        """

        try:

            _, categorical = self.get_feature_types(df)

            return categorical

        except Exception as error:

            logger.exception(error)

            raise LoanRiskException(error, sys)

    # ==================================================
    # Summary
    # ==================================================

    def summary(
        self,
        df: pd.DataFrame,
    ) -> dict:
        """
        Return and log preprocessing summary.
        """

        try:

            numerical, categorical = (
                self.get_feature_types(df)
            )

            summary = {
                "total_features": (
                    len(numerical) +
                    len(categorical)
                ),
                "numerical_features": len(numerical),
                "categorical_features": len(categorical),
            }

            logger.info("=" * 60)
            logger.info("PREPROCESSOR SUMMARY")
            logger.info("=" * 60)
            logger.info(
                f"Total Features      : "
                f"{summary['total_features']}"
            )
            logger.info(
                f"Numerical Features  : "
                f"{summary['numerical_features']}"
            )
            logger.info(
                f"Categorical Features: "
                f"{summary['categorical_features']}"
            )
            logger.info("=" * 60)

            return summary

        except Exception as error:

            logger.exception(error)

            raise LoanRiskException(error, sys)

    # ==================================================
    # Save
    # ==================================================

    def save(
        self,
        filepath: str | Path,
    ) -> Path:
        """
        Save the fitted preprocessor to disk.
        """

        try:

            if self.transformer is None:

                raise ValueError(
                    "No preprocessor has been built."
                )

            check_is_fitted(self.transformer)

            filepath = Path(filepath)

            filepath.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            joblib.dump(
                self.transformer,
                filepath,
            )

            logger.info(
                f"Preprocessor saved to {filepath}"
            )

            return filepath

        except Exception as error:

            logger.exception(error)

            raise LoanRiskException(error, sys)

    # ==================================================
    # Load
    # ==================================================

    def load(
        self,
        filepath: str | Path,
    ):
        """
        Load a fitted preprocessor from disk.
        """

        try:

            filepath = Path(filepath)

            if not filepath.exists():

                raise FileNotFoundError(
                    f"Preprocessor not found: {filepath}"
                )

            loaded_transformer = joblib.load(filepath)

            check_is_fitted(loaded_transformer)

            self.transformer = loaded_transformer

            logger.info(
                f"Preprocessor loaded from {filepath}"
            )

            return self.transformer

        except Exception as error:

            logger.exception(error)

            raise LoanRiskException(error, sys)