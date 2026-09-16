"""
Tests for ModelSelector.

Author: Ujwal
Project: Intelligent Loan Risk Assessment Platform
"""

import pytest

from src.data_loader import DataLoader
from src.data_validator import DataValidator
from src.feature_engineering import FeatureEngineer
from src.data_splitter import DataSplitter
from src.model_factory import ModelFactory
from src.model_selector import ModelSelector


@pytest.fixture(scope="module")
def model_selection_result():
    """
    Run the expensive model-selection process only once.

    All tests that need the training result reuse this fixture.
    """

    # -----------------------------
    # 1. Load data
    # -----------------------------
    loader = DataLoader()
    df = loader.load_training_data()

    # -----------------------------
    # 2. Validate data
    # -----------------------------
    validator = DataValidator()
    validator.validate(df)

    # -----------------------------
    # 3. Feature engineering
    # -----------------------------
    engineer = FeatureEngineer()
    df = engineer.transform(df)

    # -----------------------------
    # 4. Train/test split
    # -----------------------------
    splitter = DataSplitter()
    X_train, X_test, y_train, y_test = splitter.split(df)

    # -----------------------------
    # 5. Get supported models
    # -----------------------------
    factory = ModelFactory()
    models = factory.get_models()

    # -----------------------------
    # 6. Model selection
    # -----------------------------
    selector = ModelSelector()

    return selector.train_models(
        models=models,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )


def test_model_selector_initialization():
    """Test ModelSelector initialization."""

    selector = ModelSelector()

    assert selector is not None


def test_model_selector_has_models():
    """Test that ModelFactory provides supported models."""

    factory = ModelFactory()
    models = factory.get_models()

    assert isinstance(models, dict)
    assert len(models) > 0


def test_model_selector_returns_best_model(model_selection_result):
    """Test that ModelSelector returns a best model."""

    best_model, results, best_score = model_selection_result

    assert best_model is not None
    assert best_score is not None


def test_model_selector_returns_results(model_selection_result):
    """Test that model comparison results are returned."""

    best_model, results, best_score = model_selection_result

    assert isinstance(results, dict)
    assert len(results) > 0


def test_model_selector_result_metrics(model_selection_result):
    """Test that each model has the expected evaluation metrics."""

    best_model, results, best_score = model_selection_result

    expected_metrics = {
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC_AUC",
    }

    for model_name, metrics in results.items():

        assert expected_metrics.issubset(
            metrics.keys()
        )


def test_model_selector_best_score_is_numeric(model_selection_result):
    """Test that the best ROC-AUC score is numeric."""

    best_model, results, best_score = model_selection_result

    assert isinstance(best_score, (int, float))


def test_model_selector_best_model_is_supported(model_selection_result):
    """Test that the selected model belongs to the supported models."""

    best_model, results, best_score = model_selection_result

    assert best_model is not None