"""
Production-grade preprocessing module.

Author: Ujwal
Project: Intelligent Loan Risk Assessment Platform
"""

# ===========================
# Standard Library Imports
# ===========================

import sys
import joblib
from pathlib import Path
# ===========================
# Third-Party Imports
# ===========================

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# ===========================
# Local Imports
# ===========================

from src.logger import get_logger
from src.exceptions import LoanRiskException

# ===========================
# Logger
# ===========================

logger = get_logger("Preprocessor")


class Preprocessor:
    """
    Builds the preprocessing pipeline.
    """

    def __init__(self):

        logger.info("Preprocessor initialized.")
        self.transformer = None

    def get_feature_types(self, df: pd.DataFrame):
        """
        Automatically identify numerical and categorical columns.
        """

        try:

            numerical_features = df.select_dtypes(
                include=["int64", "float64"]
            ).columns.tolist()

            categorical_features = df.select_dtypes(
                include=["object"]
            ).columns.tolist()

            # Remove TARGET if present
            if "TARGET" in numerical_features:
                numerical_features.remove("TARGET")

            logger.info(
                f"Numerical Features : {len(numerical_features)}"
            )

            logger.info(
                f"Categorical Features : {len(categorical_features)}"
            )

            return numerical_features, categorical_features

        except Exception as error:

            logger.exception(error)
            raise LoanRiskException(error, sys)

    def build(self, df: pd.DataFrame):

        try:

            numerical_features, categorical_features = (
                self.get_feature_types(df)
            )

            logger.info(f"Numerical Features : {len(numerical_features)}")
            logger.info(f"Categorical Features : {len(categorical_features)}")

            # -----------------------------
            # Numerical Pipeline
            # -----------------------------
            numerical_pipeline = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="median")
                    ),
                    (
                        "scaler",
                        StandardScaler()
                    )
                ]
            )

            # -----------------------------
            # Categorical Pipeline
            # -----------------------------
            categorical_pipeline = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="most_frequent")
                    ),
                    (
                        "encoder",
                        OneHotEncoder(
                            handle_unknown="ignore"
                        )
                    )
                ]
            )

            self.transformer = ColumnTransformer(
                transformers=[
                    (
                        "numerical",
                        numerical_pipeline,
                        numerical_features,
                    ),
                    (
                        "categorical",
                        categorical_pipeline,
                        categorical_features,
                    ),
                ]
            )

            logger.info("Preprocessing pipeline built.")

            return self.transformer

        except Exception as error:

            logger.exception(error)
            raise LoanRiskException(error, sys)
    def fit_transform(self, X_train):

        try:

            if self.transformer is None:
                self.build(X_train)

            logger.info("Fitting preprocessor on training data.")

            return self.transformer.fit_transform(X_train)

        except Exception as error:

            logger.exception(error)
            raise LoanRiskException(error, sys)
    
    def transform(self, X_test):

        try:

            if self.transformer is None:
                raise ValueError(
                    "Preprocessor has not been fitted yet."
                )

            logger.info("Transforming new data.")

            return self.transformer.transform(X_test)

        except Exception as error:

            logger.exception(error)
            raise LoanRiskException(error, sys)

    def get_numerical_features(self, df):
        return df.select_dtypes(
            include=["int64", "float64"]
        ).columns.tolist()
    def get_categorical_features(self, df):
        return df.select_dtypes(
            include=["object"]
        ).columns.tolist()
    def summary(self, df):

        numerical, categorical = self.get_feature_types(df)

        logger.info("=" * 50)
        logger.info("PREPROCESSOR SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Numerical Features   : {len(numerical)}")
        logger.info(f"Categorical Features : {len(categorical)}")
        logger.info("=" * 50)
    def get_feature_names(self):

        if self.transformer is None:
            raise ValueError("Preprocessor has not been fitted.")

        return self.transformer.get_feature_names_out()
    
    def get_feature_names(self):
        """
        Returns the feature names after preprocessing.
        """

        try:

            if self.transformer is None:
                raise ValueError("Preprocessor has not been fitted.")

            return self.transformer.get_feature_names_out()

        except Exception as error:

            logger.exception(error)
            raise LoanRiskException(error, sys)
    def save(self, filepath: str):
        """
        Save fitted preprocessor.
        """

        try:

            if self.transformer is None:
                raise ValueError("Nothing to save. Fit the preprocessor first.")

            Path(filepath).parent.mkdir(parents=True, exist_ok=True)

            joblib.dump(self.transformer, filepath)

            logger.info(f"Preprocessor saved to {filepath}")

        except Exception as error:

            logger.exception(error)
            raise LoanRiskException(error, sys)
    def load(self, filepath: str):
        """
        Load fitted preprocessor.
        """

        try:

            self.transformer = joblib.load(filepath)

            logger.info(f"Preprocessor loaded from {filepath}")

            return self.transformer

        except Exception as error:

            logger.exception(error)
            raise LoanRiskException(error, sys)