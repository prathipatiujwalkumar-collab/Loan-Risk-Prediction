"""
Production-grade FastAPI Application

Author: Ujwal
Project: Intelligent Loan Risk Assessment Platform
"""

# ==================================================
# Standard Library Imports
# ==================================================

import sys

# ==================================================
# Third-Party Imports
# ==================================================

from fastapi import FastAPI, HTTPException

# ==================================================
# Local Imports
# ==================================================

from src.logger import get_logger
from src.exceptions import LoanRiskException
from src.schema import (
    LoanPredictionRequest,
    LoanPredictionResponse,
)
from src.input_adapter import InputAdapter
from src.predictor import Predictor

# ==================================================
# Logger
# ==================================================

logger = get_logger("API")

# ==================================================
# FastAPI Application
# ==================================================

app = FastAPI(
    title="Intelligent Loan Risk Assessment Platform",
    description=(
        "Production-grade REST API for loan default "
        "risk prediction."
    ),
    version="1.0.0",
)

# ==================================================
# Initialize Components
# ==================================================

try:

    adapter = InputAdapter()
    predictor = Predictor()

    logger.info(
        "API components initialized successfully."
    )

except Exception as error:

    logger.exception(error)

    adapter = None
    predictor = None


# ==================================================
# Health Check
# ==================================================

@app.get("/")
def root():
    """
    API health check.

    Returns
    -------
    dict
        API status information.
    """

    return {
        "project": "Intelligent Loan Risk Assessment Platform",
        "status": "running",
        "version": "1.0.0",
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True,
        "version": "1.0.0"
    }

# ==================================================
# Prediction Endpoint
# ==================================================

@app.post(
    "/predict",
    response_model=LoanPredictionResponse,
)
def predict(
    request: LoanPredictionRequest,
):
    """
    Predict loan default risk for a customer.

    Parameters
    ----------
    request : LoanPredictionRequest
        Validated API request.

    Returns
    -------
    LoanPredictionResponse
        Prediction, probability and risk classification.
    """

    try:

        logger.info(
            "Prediction request received."
        )

        # ==================================================
        # Validate API Components
        # ==================================================

        if adapter is None:

            raise RuntimeError(
                "InputAdapter is not initialized."
            )

        if predictor is None:

            raise RuntimeError(
                "Predictor is not initialized."
            )

        # ==================================================
        # Convert Pydantic Request → Dictionary
        # ==================================================

        input_data = request.model_dump(
            by_alias=True
        )

        logger.info(
            f"Received input fields: "
            f"{list(input_data.keys())}"
        )

        # ==================================================
        # Input Adaptation
        #
        # API Input
        #     ↓
        # Pydantic
        #     ↓
        # Dictionary
        #     ↓
        # InputAdapter
        #     ↓
        # Feature Engineering
        #     ↓
        # Model-ready DataFrame
        # ==================================================

        adapted_data = adapter.adapt(
            input_data
        )

        logger.info(
            f"Adapted input shape: "
            f"{adapted_data.shape}"
        )

        # ==================================================
        # Load Trained Model
        # ==================================================

        if predictor.pipeline is None:

            logger.info(
                "Loading trained model..."
            )

            predictor.load_model()

        # ==================================================
        # Get Expected Training Features
        # ==================================================

        if not hasattr(
            predictor.pipeline,
            "feature_names_in_",
        ):

            raise ValueError(
                "Loaded model does not contain "
                "feature_names_in_."
            )

        expected_features = (
            predictor.pipeline.feature_names_in_
        )

        logger.info(
            f"Trained model expects "
            f"{len(expected_features)} features."
        )

        # ==================================================
        # Align API Input With Training Features
        # ==================================================

        adapted_data = adapted_data.reindex(
            columns=expected_features
        )

        logger.info(
            f"Input aligned with trained model. "
            f"Final shape: {adapted_data.shape}"
        )

        # ==================================================
        # Prediction
        # ==================================================

        prediction = predictor.predict(
            adapted_data
        )

        logger.info(
            "Class prediction generated."
        )

        # ==================================================
        # Probability Prediction
        # ==================================================

        probability = predictor.predict_proba(
            adapted_data
        )

        logger.info(
            "Prediction probability generated."
        )

        # ==================================================
        # Extract Prediction
        # ==================================================

        prediction_value = int(
            prediction[0]
        )

        # ==================================================
        # Extract Probability of Class 1
        # ==================================================

        if probability.ndim != 2:

            raise ValueError(
                "Prediction probability output "
                "has an unexpected shape."
            )

        if probability.shape[1] < 2:

            raise ValueError(
                "Model does not provide probability "
                "for class 1."
            )

        probability_value = float(
            probability[0][1]
        )

        # ==================================================
        # Risk Classification
        # ==================================================

        if probability_value >= 0.70:

            risk = "High Risk"

        elif probability_value >= 0.40:

            risk = "Medium Risk"

        else:

            risk = "Low Risk"

        # ==================================================
        # Logging
        # ==================================================

        logger.info(
            f"Prediction     : {prediction_value}"
        )

        logger.info(
            f"Probability    : {probability_value:.4f}"
        )

        logger.info(
            f"Risk            : {risk}"
        )

        # ==================================================
        # Return API Response
        # ==================================================

        return LoanPredictionResponse(
            prediction=prediction_value,
            probability=probability_value,
            risk=risk,
        )

    # ==================================================
    # Project-Specific Exception
    # ==================================================

    except LoanRiskException as error:

        logger.exception(
            "LoanRiskException during prediction."
        )

        raise HTTPException(
            status_code=500,
            detail="Loan risk prediction failed.",
        ) from error

    # ==================================================
    # Unexpected Exception
    # ==================================================

    except Exception as error:

        logger.exception(
            "Unexpected error during prediction."
        )

        raise HTTPException(
            status_code=500,
            detail="Prediction failed.",
        ) from error


# ==================================================
# Application Entry Point
# ==================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "src.api:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )
