"""
Tests for ModelFactory.

Author: Ujwal
Project: Intelligent Loan Risk Assessment Platform
"""

from src.model_factory import ModelFactory


def test_model_factory_initialization():
    """Test ModelFactory initialization."""

    factory = ModelFactory()

    assert factory is not None


def test_get_models_returns_dictionary():
    """Test that get_models returns a dictionary."""

    factory = ModelFactory()

    models = factory.get_models()

    assert isinstance(models, dict)


def test_models_are_not_empty():
    """Test that supported models are available."""

    factory = ModelFactory()

    models = factory.get_models()

    assert len(models) > 0


def test_logistic_regression_model_exists():
    """Test Logistic Regression model."""

    factory = ModelFactory()

    models = factory.get_models()

    assert "Logistic Regression" in models


def test_random_forest_model_exists():
    """Test Random Forest model."""

    factory = ModelFactory()

    models = factory.get_models()

    assert "Random Forest" in models


def test_model_objects_are_valid():
    """Test that all returned models are valid objects."""

    factory = ModelFactory()

    models = factory.get_models()

    for name, model in models.items():

        assert name is not None
        assert model is not None


def test_expected_number_of_models():
    """Test expected number of supported models."""

    factory = ModelFactory()

    models = factory.get_models()

    assert len(models) == 2