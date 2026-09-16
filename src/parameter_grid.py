"""
Production-grade Parameter Grid Module

Author: Ujwal
Project: Intelligent Loan Risk Assessment Platform

Provides centralized hyperparameter search spaces for
supported machine learning models.
"""

# ==================================================
# Standard Library Imports
# ==================================================

import sys

# ==================================================
# Third-Party Imports
# ==================================================

from sklearn.model_selection import ParameterGrid as SklearnParameterGrid

# ==================================================
# Local Imports
# ==================================================

from src.logger import get_logger
from src.exceptions import LoanRiskException

# ==================================================
# Logger
# ==================================================

logger = get_logger("ParameterGrid")


# ==================================================
# Parameter Grid Factory
# ==================================================

class ParameterGridFactory:
    """
    Factory for creating model-specific hyperparameter grids.
    """

    @staticmethod
    def logistic_regression():
        """
        Return hyperparameter grid for Logistic Regression.
        """

        return {
            "model__C": [
                0.01,
                0.1,
                1.0,
                10.0,
            ],
            "model__solver": [
                "lbfgs",
            ],
            "model__class_weight": [
                "balanced",
            ],
        }

    @staticmethod
    def random_forest():
        """
        Return hyperparameter grid for Random Forest.
        """

        return {
            "model__n_estimators": [
                100,
                200,
                300,
            ],
            "model__max_depth": [
                10,
                15,
                20,
            ],
            "model__min_samples_split": [
                2,
                5,
            ],
            "model__min_samples_leaf": [
                1,
                2,
            ],
            "model__class_weight": [
                "balanced",
            ],
        }

    @classmethod
    def get_grid(cls, model_name: str):
        """
        Return the hyperparameter grid for a model.
        """

        try:

            grids = {
                "Logistic Regression":
                    cls.logistic_regression(),

                "Random Forest":
                    cls.random_forest(),
            }

            if model_name not in grids:

                raise ValueError(
                    f"Unsupported model: {model_name}"
                )

            logger.info(
                f"Parameter grid generated for "
                f"{model_name}"
            )

            return grids[model_name]

        except Exception as error:

            logger.exception(error)

            raise LoanRiskException(
                error,
                sys,
            )

    @classmethod
    def available_models(cls):
        """
        Return supported model names.
        """

        return [
            "Logistic Regression",
            "Random Forest",
        ]


# ==================================================
# Backward-Compatible ParameterGrid
# ==================================================

class ParameterGrid:
    """
    Backward-compatible interface for parameter grids.

    This class is kept so existing tests and modules using
    ParameterGrid continue to work.
    """

    @staticmethod
    def logistic_regression():
        """
        Return Logistic Regression parameter grid.
        """

        return ParameterGridFactory.logistic_regression()

    @staticmethod
    def random_forest():
        """
        Return Random Forest parameter grid.
        """

        return ParameterGridFactory.random_forest()

    @classmethod
    def get_grid(cls, model_name: str):
        """
        Return parameter grid for the requested model.
        """

        return ParameterGridFactory.get_grid(
            model_name
        )

    @classmethod
    def available_models(cls):
        """
        Return supported model names.
        """

        return ParameterGridFactory.available_models()


# ==================================================
# Utility Function
# ==================================================

def expand_grid(model_name: str):
    """
    Expand a parameter dictionary into a list of
    individual parameter combinations.

    Useful for testing and inspection.
    """

    grid = ParameterGridFactory.get_grid(
        model_name
    )

    return list(
        SklearnParameterGrid(grid)
    )


# ==================================================
# Main
# ==================================================

if __name__ == "__main__":

    print("=" * 60)
    print("PARAMETER GRID MODULE")
    print("=" * 60)

    print(
        "\nAvailable Models:"
    )

    for model in ParameterGridFactory.available_models():

        print(f"- {model}")

        grid = ParameterGridFactory.get_grid(
            model
        )

        print(
            f"  Parameters: {list(grid.keys())}"
        )

        combinations = expand_grid(model)

        print(
            f"  Combinations: {len(combinations)}"
        )

    print(
        "\nParameter Grid Module Test Successful"
    )