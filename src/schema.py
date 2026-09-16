"""
Production-grade API Schemas

Author: Ujwal
Project: Intelligent Loan Risk Assessment Platform
"""

# ==================================================
# Third-Party Imports
# ==================================================

from pydantic import BaseModel, Field, ConfigDict


# ==================================================
# Loan Prediction Request
# ==================================================

class LoanPredictionRequest(BaseModel):
    """
    Request schema for loan risk prediction.

    The API accepts a simplified customer profile
    containing the features required by InputAdapter.
    """

    model_config = ConfigDict(
        populate_by_name=True
    )

    amt_credit: float = Field(
        ...,
        alias="AMT_CREDIT",
        description="Credit amount requested by the customer.",
    )

    amt_income_total: float = Field(
        ...,
        alias="AMT_INCOME_TOTAL",
        description="Total annual income of the customer.",
    )

    amt_annuity: float = Field(
        ...,
        alias="AMT_ANNUITY",
        description="Loan annuity amount.",
    )

    amt_goods_price: float = Field(
        ...,
        alias="AMT_GOODS_PRICE",
        description="Price of the goods associated with the loan.",
    )

    days_employed: int = Field(
        ...,
        alias="DAYS_EMPLOYED",
        description="Number of days employed.",
    )

    days_birth: int = Field(
        ...,
        alias="DAYS_BIRTH",
        description="Number of days since birth.",
    )

    cnt_children: int = Field(
        ...,
        alias="CNT_CHILDREN",
        description="Number of children.",
    )

    cnt_fam_members: float = Field(
        ...,
        alias="CNT_FAM_MEMBERS",
        description="Number of family members.",
    )


# ==================================================
# Loan Prediction Response
# ==================================================

class LoanPredictionResponse(BaseModel):
    """
    Response schema returned by the prediction API.
    """

    prediction: int = Field(
        ...,
        description="Predicted loan default class.",
    )

    probability: float = Field(
        ...,
        description="Probability of loan default.",
    )

    risk: str = Field(
        ...,
        description="Risk classification.",
    )