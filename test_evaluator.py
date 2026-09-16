"""
Pytest tests for Evaluator.

Author: Ujwal
Project: Intelligent Loan Risk Assessment Platform
"""

import pandas as pd

from src.data_loader import DataLoader
from src.data_validator import DataValidator
from src.feature_engineering import FeatureEngineer
from src.data_splitter import DataSplitter
from src.trainer import ModelTrainer
from src.evaluator import Evaluator


def prepare_model():
    """
    Prepare trained model and test data for evaluator tests.
    """

    loader = DataLoader()
    df = loader.load_training_data()

    validator = DataValidator()
    validator.validate(df)

    engineer = FeatureEngineer()
    df = engineer.transform(df)

    splitter = DataSplitter()

    X_train, X_test, y_train, y_test = splitter.split(df)

    trainer = ModelTrainer()

    trainer.train(
        X_train,
        y_train,
    )

    predictions = trainer.predict(X_test)
    probabilities = trainer.predict_proba(X_test)

    return (
        y_test,
        predictions,
        probabilities,
    )


def test_evaluator_initialization():
    """
    Test Evaluator initialization.
    """

    evaluator = Evaluator()

    assert evaluator is not None


def test_evaluator_returns_results():
    """
    Test that Evaluator returns evaluation results.
    """

    (
        y_test,
        predictions,
        probabilities,
    ) = prepare_model()

    evaluator = Evaluator()

    results = evaluator.evaluate(
        y_test,
        predictions,
        probabilities,
    )

    assert results is not None
    assert isinstance(results, dict)


def test_evaluator_results_not_empty():
    """
    Test that evaluation results contain metrics.
    """

    (
        y_test,
        predictions,
        probabilities,
    ) = prepare_model()

    evaluator = Evaluator()

    results = evaluator.evaluate(
        y_test,
        predictions,
        probabilities,
    )

    assert len(results) > 0


def test_evaluator_predictions_match_targets():
    """
    Test that predictions and targets have matching lengths.
    """

    (
        y_test,
        predictions,
        probabilities,
    ) = prepare_model()

    assert len(y_test) == len(predictions)


def test_evaluator_probability_shape():
    """
    Test that prediction probabilities match test samples.
    """

    (
        y_test,
        predictions,
        probabilities,
    ) = prepare_model()

    assert probabilities.shape[0] == len(y_test)


def test_evaluator_metrics_are_numeric():
    """
    Test that returned metric values are numeric or structured.
    """

    (
        y_test,
        predictions,
        probabilities,
    ) = prepare_model()

    evaluator = Evaluator()

    results = evaluator.evaluate(
        y_test,
        predictions,
        probabilities,
    )

    assert isinstance(results, dict)

    for value in results.values():

        assert value is not None