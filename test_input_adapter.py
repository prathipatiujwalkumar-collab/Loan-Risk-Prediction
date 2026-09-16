"""
Test script for InputAdapter.

Author: Ujwal
Project: Intelligent Loan Risk Assessment Platform
"""

# ==================================================
# Local Imports
# ==================================================

from src.input_adapter import InputAdapter


def main():
    """
    Test InputAdapter using a sample loan application.
    """

    # --------------------------------------------------
    # Sample API Input
    # --------------------------------------------------

    sample_input = {
        "AMT_CREDIT": 180000,
        "AMT_INCOME_TOTAL": 180000,
        "AMT_ANNUITY": 9000,
        "AMT_GOODS_PRICE": 150000,
        "DAYS_EMPLOYED": -2000,
        "DAYS_BIRTH": -12000,
        "CNT_CHILDREN": 1,
        "CNT_FAM_MEMBERS": 3,
    }

    # --------------------------------------------------
    # Initialize Adapter
    # --------------------------------------------------

    adapter = InputAdapter()

    # --------------------------------------------------
    # Adapt Input
    # --------------------------------------------------

    adapted_data = adapter.adapt(
        sample_input
    )

    # --------------------------------------------------
    # Results
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("INPUT ADAPTER TEST")
    print("=" * 60)

    print("\nOriginal Input:")
    print(sample_input)

    print("\nAdapted Data Shape:")
    print(adapted_data.shape)

    print("\nEngineered Features:")

    print(
        adapted_data[
            [
                "CREDIT_INCOME_RATIO",
                "ANNUITY_INCOME_RATIO",
                "CREDIT_GOODS_RATIO",
                "EMPLOYMENT_AGE_RATIO",
                "CHILDREN_RATIO",
                "INCOME_PER_FAMILY_MEMBER",
                "ANNUITY_CREDIT_RATIO",
                "GOODS_INCOME_RATIO",
            ]
        ]
    )

    print("\n" + "=" * 60)
    print("InputAdapter Test Successful")
    print("=" * 60)


if __name__ == "__main__":
    main()