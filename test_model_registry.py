"""
Tests for ModelRegistry.

Author: Ujwal
Project: Intelligent Loan Risk Assessment Platform
"""

from sklearn.linear_model import LogisticRegression

from src.model_registry import ModelRegistry


def test_model_registry_initialization():
    """Test ModelRegistry initialization."""

    registry = ModelRegistry()

    assert registry is not None


def test_register_model():
    """Test registering a model."""

    registry = ModelRegistry()

    model = LogisticRegression()

    registry.register_model(
        model=model,
        model_name="logistic_regression",
        version="v1",
        overwrite=True,
    )

    models = registry.list_models()

    assert len(models) > 0


def test_registered_model_exists():
    """Test that a registered model can be listed."""

    registry = ModelRegistry()

    model = LogisticRegression()

    registry.register_model(
        model=model,
        model_name="logistic_regression",
        version="v1",
        overwrite=True,
    )

    models = registry.list_models()

    model_names = [item.name for item in models]

    assert "logistic_regression_v1.joblib" in model_names



def test_promote_model_to_production():
    """Test promoting a registered model to production."""

    registry = ModelRegistry()

    model = LogisticRegression()

    registry.register_model(
        model=model,
        model_name="logistic_regression",
        version="v1",
        overwrite=True,
    )

    registry.promote_model(
        "logistic_regression",
        "v1",
    )

    assert registry.production_model_exists()


def test_load_registered_model():
    """Test loading a registered model."""

    registry = ModelRegistry()

    model = LogisticRegression()

    registry.register_model(
        model=model,
        model_name="logistic_regression",
        version="v1",
        overwrite=True,
    )

    loaded_model = registry.load_model(
        "logistic_regression",
        "v1",
    )

    assert loaded_model is not None
    assert isinstance(
        loaded_model,
        LogisticRegression,
    )


def test_load_production_model():
    """Test loading the production model."""

    registry = ModelRegistry()

    model = LogisticRegression()

    registry.register_model(
        model=model,
        model_name="logistic_regression",
        version="v1",
        overwrite=True,
    )

    registry.promote_model(
        "logistic_regression",
        "v1",
    )

    production_model = registry.load_production_model()

    assert production_model is not None
    assert isinstance(
        production_model,
        LogisticRegression,
    )


def test_get_latest_version():
    """Test retrieving the latest model version."""

    registry = ModelRegistry()

    model = LogisticRegression()

    registry.register_model(
        model=model,
        model_name="logistic_regression",
        version="v1",
        overwrite=True,
    )

    latest = registry.get_latest_version(
        "logistic_regression"
    )

    assert latest == "v1"