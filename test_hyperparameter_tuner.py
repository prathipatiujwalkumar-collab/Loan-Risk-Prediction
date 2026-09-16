"""
Pytest tests for HyperparameterTuner.

Author: Ujwal
Project: Intelligent Loan Risk Assessment Platform
"""

import pytest
import pandas as pd

from src.data_loader import DataLoader
from src.data_validator import DataValidator
from src.feature_engineering import FeatureEngineer
from src.data_splitter import DataSplitter
from src.trainer import ModelTrainer
from src.hyperparameter_tuner import HyperparameterTuner
from src.parameter_grid import ParameterGridFactory
from src.exceptions import LoanRiskException


def prepare_training_data():
    """
    Prepare training data for hyperparameter tuning tests.
    """

    loader = DataLoader()

    df = loader.load_training_data()

    validator = DataValidator()
    validator.validate(df)

    engineer = FeatureEngineer()

    df = engineer.transform(df)

    splitter = DataSplitter()

    X_train, X_test, y_train, y_test = splitter.split(df)

    return X_train, y_train


def test_hyperparameter_tuner_initialization():
    """
    Test HyperparameterTuner initialization.
    """

    tuner = HyperparameterTuner()

    assert tuner is not None
    assert tuner.cv is not None


def test_hyperparameter_tuner_cv_configuration():
    """
    Test that StratifiedKFold CV is configured.
    """

    tuner = HyperparameterTuner()

    assert tuner.cv.n_splits >= 2
    assert tuner.cv.shuffle is True


def test_parameter_grid_for_logistic_regression():
    """
    Test that Logistic Regression parameters
    can be retrieved for tuning.
    """

    parameters = ParameterGridFactory.get_grid(
        "Logistic Regression"
    )

    assert isinstance(parameters, dict)
    assert len(parameters) > 0

    assert "model__C" in parameters
    assert "model__solver" in parameters
    assert "model__class_weight" in parameters


def test_parameter_grid_for_random_forest():
    """
    Test that Random Forest parameters
    can be retrieved for tuning.
    """

    parameters = ParameterGridFactory.get_grid(
        "Random Forest"
    )

    assert isinstance(parameters, dict)
    assert len(parameters) > 0

    assert "model__n_estimators" in parameters
    assert "model__max_depth" in parameters
    assert "model__class_weight" in parameters


def test_hyperparameter_tuning():
    """
    Test complete hyperparameter tuning workflow.

    This is an integration test and may take several minutes
    because RandomizedSearchCV performs cross-validation.
    """

    X_train, y_train = prepare_training_data()

    trainer = ModelTrainer()

    pipeline = trainer.build_pipeline(X_train)

    parameters = ParameterGridFactory.get_grid(
        "Logistic Regression"
    )

    tuner = HyperparameterTuner()

    search = tuner.tune(
        pipeline=pipeline,
        parameters=parameters,
        X_train=X_train,
        y_train=y_train,
    )

    assert search is not None

    assert hasattr(
        search,
        "best_score_"
    )

    assert hasattr(
        search,
        "best_params_"
    )

    assert hasattr(
        search,
        "best_estimator_"
    )


def test_hyperparameter_tuning_returns_valid_score():
    """
    Test that hyperparameter tuning produces
    a valid cross-validation score.
    """

    X_train, y_train = prepare_training_data()

    trainer = ModelTrainer()

    pipeline = trainer.build_pipeline(X_train)

    parameters = ParameterGridFactory.get_grid(
        "Logistic Regression"
    )

    tuner = HyperparameterTuner()

    search = tuner.tune(
        pipeline=pipeline,
        parameters=parameters,
        X_train=X_train,
        y_train=y_train,
    )

    assert isinstance(
        search.best_score_,
        float
    )

    assert 0.0 <= search.best_score_ <= 1.0


def test_hyperparameter_tuner_rejects_none_pipeline():
    """
    Test validation when pipeline is None.
    """

    tuner = HyperparameterTuner()

    with pytest.raises(LoanRiskException):

        tuner.tune(
            pipeline=None,
            parameters={
                "model__C": [0.1, 1.0]
            },
            X_train=pd.DataFrame(
                {"feature": [1, 2, 3, 4]}
            ),
            y_train=pd.Series(
                [0, 1, 0, 1]
            ),
        )


def test_hyperparameter_tuner_rejects_empty_parameters():
    """
    Test validation when parameter search space is empty.
    """

    X_train = pd.DataFrame(
        {"feature": [1, 2, 3, 4]}
    )

    y_train = pd.Series(
        [0, 1, 0, 1]
    )

    trainer = ModelTrainer()

    pipeline = trainer.build_pipeline(
        X_train
    )

    tuner = HyperparameterTuner()

    with pytest.raises(LoanRiskException):

        tuner.tune(
            pipeline=pipeline,
            parameters={},
            X_train=X_train,
            y_train=y_train,
        )


def test_hyperparameter_tuner_rejects_empty_training_data():
    """
    Test validation when training data is empty.
    """

    X_train = pd.DataFrame()

    y_train = pd.Series(
        dtype="int64"
    )

    trainer = ModelTrainer()

    # We don't need a functional pipeline because
    # empty X_train should be rejected first.
    pipeline = None

    tuner = HyperparameterTuner()

    with pytest.raises(LoanRiskException):

        tuner.tune(
            pipeline=pipeline,
            parameters={
                "model__C": [0.1, 1.0]
            },
            X_train=X_train,
            y_train=y_train,
        )