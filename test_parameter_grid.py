"""
Production-grade Parameter Grid Test Module

Author: Ujwal
Project: Intelligent Loan Risk Assessment Platform
"""

# ==================================================
# Third-Party Imports
# ==================================================

from sklearn.model_selection import ParameterGrid as SklearnParameterGrid

# ==================================================
# Local Imports
# ==================================================

from src.parameter_grid import ParameterGrid
from src.exceptions import LoanRiskException


# ==================================================
# Test: Logistic Regression Grid
# ==================================================

def test_logistic_regression_grid():
    """
    Test Logistic Regression parameter grid.
    """

    grid = ParameterGrid.get_grid(
        "Logistic Regression"
    )

    assert isinstance(
        grid,
        dict,
    )

    assert len(grid) > 0

    combinations = list(
        SklearnParameterGrid(grid)
    )

    assert len(combinations) > 0


# ==================================================
# Test: Random Forest Grid
# ==================================================

def test_random_forest_grid():
    """
    Test Random Forest parameter grid.
    """

    grid = ParameterGrid.get_grid(
        "Random Forest"
    )

    assert isinstance(
        grid,
        dict,
    )

    assert len(grid) > 0

    combinations = list(
        SklearnParameterGrid(grid)
    )

    assert len(combinations) > 0


# ==================================================
# Test: Supported Model Names
# ==================================================

def test_supported_models():
    """
    Test supported model names.
    """

    logistic_grid = ParameterGrid.get_grid(
        "Logistic Regression"
    )

    random_forest_grid = ParameterGrid.get_grid(
        "Random Forest"
    )

    assert isinstance(
        logistic_grid,
        dict,
    )

    assert isinstance(
        random_forest_grid,
        dict,
    )


# ==================================================
# Test: Invalid Model
# ==================================================

def test_invalid_model():
    """
    Test invalid model handling.

    The ParameterGrid implementation wraps
    underlying exceptions using LoanRiskException.
    """

    try:

        ParameterGrid.get_grid(
            "Invalid Model"
        )

        assert False, (
            "Expected LoanRiskException "
            "for invalid model."
        )

    except LoanRiskException:

        assert True


# ==================================================
# Standalone Execution
# ==================================================

if __name__ == "__main__":

    print("=" * 60)
    print("PARAMETER GRID TEST MODULE")
    print("=" * 60)

    logistic_grid = ParameterGrid.get_grid(
        "Logistic Regression"
    )

    print("\nLogistic Regression Grid")

    for key, value in logistic_grid.items():

        print(
            f"{key:<30} -> {value}"
        )

    random_forest_grid = ParameterGrid.get_grid(
        "Random Forest"
    )

    print("\nRandom Forest Grid")

    for key, value in random_forest_grid.items():

        print(
            f"{key:<30} -> {value}"
        )

    print("\nParameter Grid Test Successful")