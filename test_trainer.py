"""
Tests for ModelTrainer.

Author: Ujwal
Project: Intelligent Loan Risk Assessment Platform
"""

import numpy as np

from src.data_loader import DataLoader
from src.data_validator import DataValidator
from src.feature_engineering import FeatureEngineer
from src.data_splitter import DataSplitter
from src.model_factory import ModelFactory
from src.trainer import ModelTrainer


def prepare_data():
    """Prepare training and testing data."""

    loader = DataLoader()
    df = loader.load_training_data()

    validator = DataValidator()
    validator.validate(df)

    engineer = FeatureEngineer()
    df = engineer.transform(df)

    splitter = DataSplitter()

    return splitter.split(df)


def test_model_trainer_initialization():
    """Test ModelTrainer initialization."""

    model_factory = ModelFactory()
    models = model_factory.get_models()

    model = models["Logistic Regression"]

    trainer = ModelTrainer(model=model)

    assert trainer is not None
    assert trainer.model is model


def test_model_training():
    """Test that the model can be trained."""

    X_train, X_test, y_train, y_test = prepare_data()

    model_factory = ModelFactory()
    model = model_factory.get_models()["Logistic Regression"]

    trainer = ModelTrainer(model=model)

    trainer.train(X_train, y_train)

    assert trainer.model is not None


def test_model_prediction():
    """Test prediction generation."""

    X_train, X_test, y_train, y_test = prepare_data()

    model_factory = ModelFactory()
    model = model_factory.get_models()["Logistic Regression"]

    trainer = ModelTrainer(model=model)

    trainer.train(X_train, y_train)

    predictions = trainer.predict(X_test)

    assert predictions is not None
    assert len(predictions) == len(X_test)


def test_model_probability_prediction():
    """Test probability prediction."""

    X_train, X_test, y_train, y_test = prepare_data()

    model_factory = ModelFactory()
    model = model_factory.get_models()["Logistic Regression"]

    trainer = ModelTrainer(model=model)

    trainer.train(X_train, y_train)

    probabilities = trainer.predict_proba(X_test)

    assert probabilities is not None
    assert probabilities.shape[0] == len(X_test)
    assert probabilities.shape[1] == 2


def test_model_prediction_values():
    """Test that predictions contain valid binary classes."""

    X_train, X_test, y_train, y_test = prepare_data()

    model_factory = ModelFactory()
    model = model_factory.get_models()["Logistic Regression"]

    trainer = ModelTrainer(model=model)

    trainer.train(X_train, y_train)

    predictions = trainer.predict(X_test)

    unique_predictions = np.unique(predictions)

    assert set(unique_predictions).issubset({0, 1})


def test_model_save_and_load():
    """Test model persistence."""

    X_train, X_test, y_train, y_test = prepare_data()

    model_factory = ModelFactory()
    model = model_factory.get_models()["Logistic Regression"]

    trainer = ModelTrainer(model=model)

    trainer.train(X_train, y_train)

    model_path = trainer.save()

    assert model_path is not None

    loaded_trainer = ModelTrainer()

    loaded_trainer.load(model_path)

    loaded_predictions = loaded_trainer.predict(X_test)

    assert loaded_predictions is not None
    assert len(loaded_predictions) == len(X_test)


def test_loaded_model_predictions_match():
    """Test that saved and loaded models produce the same predictions."""

    X_train, X_test, y_train, y_test = prepare_data()

    model_factory = ModelFactory()
    model = model_factory.get_models()["Logistic Regression"]

    trainer = ModelTrainer(model=model)

    trainer.train(X_train, y_train)

    original_predictions = trainer.predict(X_test)

    model_path = trainer.save()

    loaded_trainer = ModelTrainer()
    loaded_trainer.load(model_path)

    loaded_predictions = loaded_trainer.predict(X_test)

    assert np.array_equal(
        original_predictions,
        loaded_predictions,
    )