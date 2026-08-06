"""
Generic Production Model Trainer
"""

import sys
import joblib
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from src.logger import get_logger
from src.exceptions import LoanRiskException
from src.preprocessor import Preprocessor
from src.config import (
    RANDOM_STATE,
    MAX_ITER,
    CLASS_WEIGHT,
    MODEL_PATH,
)

logger = get_logger("ModelTrainer")


class ModelTrainer:

    def __init__(self, model=None):

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

    def build_pipeline(self, X_train):

        try:

            preprocessor = Preprocessor()

            transformer = preprocessor.build(X_train)

            self.pipeline = Pipeline(
                [
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
                f"Pipeline built using {self.model.__class__.__name__}"
            )

            return self.pipeline

        except Exception as error:

            logger.exception(error)
            raise LoanRiskException(error, sys)

    def train(self, X_train, y_train):

        try:

            if self.pipeline is None:

                self.build_pipeline(X_train)

            logger.info("Training started...")

            self.pipeline.fit(
                X_train,
                y_train,
            )

            logger.info("Training completed.")

            return self.pipeline

        except Exception as error:

            logger.exception(error)
            raise LoanRiskException(error, sys)

    def predict(self, X):

        return self.pipeline.predict(X)

    def predict_proba(self, X):

        return self.pipeline.predict_proba(X)

    def save(self, filepath=MODEL_PATH):

    try:

        if self.pipeline is None:
            raise ValueError(
                "No trained pipeline found. Train the model before saving."
            )

        Path(filepath).parent.mkdir(
            parents=True,
            exist_ok=True
        )

        joblib.dump(
            self.pipeline,
            filepath
        )

        logger.info(
            f"Model saved to {filepath}"
        )

    except Exception as error:

        logger.exception(error)
        raise LoanRiskException(error, sys)

    def load(self, filepath=MODEL_PATH):

        try:

            self.pipeline = joblib.load(filepath)

            logger.info(
                f"Model loaded from {filepath}"
            )

            return self.pipeline

        except Exception as error:

            logger.exception(error)
            raise LoanRiskException(error, sys)