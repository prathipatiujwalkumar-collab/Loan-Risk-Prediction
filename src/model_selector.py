"""
Production-grade Model Selector Module

Author: Ujwal
Project: Intelligent Loan Risk Assessment Platform
"""

# ==================================================
# Standard Library Imports
# ==================================================

import sys
import time

# ==================================================
# Local Imports
# ==================================================

from src.mlflow_tracker import MLflowTracker
from src.logger import get_logger
from src.exceptions import LoanRiskException
from src.trainer import ModelTrainer
from src.evaluator import Evaluator
from src.hyperparameter_tuner import HyperparameterTuner
from src.parameter_grid import ParameterGridFactory

# ==================================================
# Logger
# ==================================================

logger = get_logger("ModelSelector")


class ModelSelector:
    """
    Train multiple machine learning models, perform
    hyperparameter tuning, evaluate them using ROC-AUC,
    log experiments with MLflow, and select the best model.
    """

    def __init__(self) -> None:

        self.results = {}
        self.best_model = None
        self.best_score = float("-inf")

        logger.info("ModelSelector initialized.")

    # ==================================================
    # Train Models
    # ==================================================

    def train_models(
        self,
        models: dict,
        X_train,
        y_train,
        X_test,
        y_test,
    ):

        try:

            logger.info("=" * 70)
            logger.info("Starting Model Selection")
            logger.info("=" * 70)

            # Reset in case object is reused
            self.results = {}
            self.best_model = None
            self.best_score = float("-inf")

            # --------------------------------------------------
            # Initialize experiment components
            # --------------------------------------------------

            tracker = MLflowTracker()
            evaluator = Evaluator()
            tuner = HyperparameterTuner()

            # --------------------------------------------------
            # Train each available model
            # --------------------------------------------------

            for model_name, model in models.items():

                logger.info("-" * 70)
                logger.info(f"Training: {model_name}")

                start_time = time.time()

                with tracker.start_run(
                    run_name=f"{model_name}_tuned"
                ):

                    # --------------------------------------------------
                    # MLflow tags
                    # --------------------------------------------------

                    tracker.log_tags(
                        {
                            "project": "Loan Risk Prediction",
                            "model": model_name,
                            "tuning": "RandomizedSearchCV",
                            "cv": f"{tuner.cv.n_splits}-Fold Stratified CV",
                            "scoring": "roc_auc",
                        }
                    )

                    # --------------------------------------------------
                    # Create ModelTrainer
                    # --------------------------------------------------

                    trainer = ModelTrainer(
                        model=model
                    )

                    # --------------------------------------------------
                    # Build preprocessing + model pipeline
                    # --------------------------------------------------

                    pipeline = trainer.build_pipeline(
                        X_train
                    )

                    logger.info(
                        f"Pipeline created for {model_name}."
                    )

                    # --------------------------------------------------
                    # Load model-specific parameter grid
                    # --------------------------------------------------

                    parameters = (
                        ParameterGridFactory.get_grid(
                            model_name
                        )
                    )

                    logger.info(
                        f"Hyperparameter search space "
                        f"loaded for {model_name}."
                    )

                    logger.info(
                        f"Parameters: "
                        f"{list(parameters.keys())}"
                    )

                    # --------------------------------------------------
                    # Hyperparameter tuning
                    # --------------------------------------------------

                    search = tuner.tune(
                        pipeline=pipeline,
                        parameters=parameters,
                        X_train=X_train,
                        y_train=y_train,
                    )

                    # --------------------------------------------------
                    # Replace trainer pipeline with the best
                    # tuned pipeline.
                    #
                    # IMPORTANT:
                    # We intentionally DO NOT replace
                    # trainer.model because ModelTrainer.save()
                    # relies on self.model.
                    # --------------------------------------------------

                    trainer.pipeline = (
                        search.best_estimator_
                    )

                    logger.info(
                        "Best tuned pipeline assigned "
                        "to ModelTrainer."
                    )

                    logger.info(
                        f"Best Parameters : "
                        f"{search.best_params_}"
                    )

                    logger.info(
                        f"Best CV ROC-AUC : "
                        f"{search.best_score_:.5f}"
                    )

                    # --------------------------------------------------
                    # Prediction using tuned pipeline
                    # --------------------------------------------------

                    predictions = trainer.predict(
                        X_test
                    )

                    probabilities = (
                        trainer.predict_proba(
                            X_test
                        )
                    )

                    # --------------------------------------------------
                    # Evaluate tuned model
                    # --------------------------------------------------

                    metrics = evaluator.evaluate(
                        y_test,
                        predictions,
                        probabilities,
                    )

                    # Store evaluation results
                    self.results[model_name] = metrics

                    # --------------------------------------------------
                    # MLflow logging
                    # --------------------------------------------------

                    # Log ONLY the parameters selected by
                    # RandomizedSearchCV.
                    tracker.log_params(
                        search.best_params_
                    )

                    # Log test-set metrics
                    tracker.log_metrics(
                        metrics
                    )

                    # Log tuned production pipeline
                    tracker.log_model(
                        trainer.pipeline
                    )

                # --------------------------------------------------
                # Training duration
                # --------------------------------------------------

                elapsed = time.time() - start_time

                score = metrics["ROC_AUC"]

                logger.info(
                    f"Training Time : "
                    f"{elapsed:.2f} sec"
                )

                logger.info(
                    f"CV ROC-AUC    : "
                    f"{search.best_score_:.5f}"
                )

                logger.info(
                    f"Test ROC-AUC  : "
                    f"{score:.5f}"
                )

                # --------------------------------------------------
                # Select best model
                # --------------------------------------------------

                if score > self.best_score:

                    self.best_score = score
                    self.best_model = trainer

                    logger.info(
                        f"New best model selected: "
                        f"{model_name}"
                    )

            # ==================================================
            # Final Validation
            # ==================================================

            if self.best_model is None:

                raise ValueError(
                    "No valid model was selected."
                )

            logger.info("=" * 70)
            logger.info("Best Model Selected")
            logger.info("=" * 70)

            logger.info(
                f"Model   : "
                f"{type(self.best_model.model).__name__}"
            )

            logger.info(
                f"ROC-AUC : "
                f"{self.best_score:.5f}"
            )

            # ==================================================
            # Save Production Model
            # ==================================================

            self.best_model.save()

            logger.info(
                "Production model saved successfully."
            )

            logger.info(
                "Model selection completed."
            )

            return (
                self.best_model,
                self.results,
                self.best_score,
            )

        except Exception as error:

            logger.exception(error)

            raise LoanRiskException(
                error,
                sys,
            )