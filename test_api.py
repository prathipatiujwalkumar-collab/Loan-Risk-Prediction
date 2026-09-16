"""
Production-grade API Test Module

Author: Ujwal
Project: Intelligent Loan Risk Assessment Platform
"""

# ==================================================
# Third-Party Imports
# ==================================================

from fastapi.testclient import TestClient

# ==================================================
# Local Imports
# ==================================================

from src.api import app


# ==================================================
# Test Client
# ==================================================

client = TestClient(app)


# ==================================================
# Test Root Endpoint
# ==================================================

def test_root_endpoint():
    """
    Test API health-check endpoint.
    """

    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["project"] == (
        "Intelligent Loan Risk Assessment Platform"
    )

    assert data["status"] == "running"

    assert data["version"] == "1.0.0"


# ==================================================
# Test Prediction Endpoint
# ==================================================

def test_prediction_endpoint():
    """
    Test successful loan-risk prediction.
    """

    payload = {
        "amt_credit": 180000,
        "amt_income_total": 180000,
        "amt_annuity": 9000,
        "amt_goods_price": 150000,
        "days_employed": -2000,
        "days_birth": -12000,
        "cnt_children": 1,
        "cnt_fam_members": 3,
    }

    response = client.post(
        "/predict",
        json=payload,
    )

    # --------------------------------------------------
    # Validate HTTP Response
    # --------------------------------------------------

    assert response.status_code == 200

    data = response.json()

    # --------------------------------------------------
    # Validate Response Structure
    # --------------------------------------------------

    assert "prediction" in data
    assert "probability" in data
    assert "risk" in data

    # --------------------------------------------------
    # Validate Prediction
    # --------------------------------------------------

    assert data["prediction"] in [0, 1]

    # --------------------------------------------------
    # Validate Probability
    # --------------------------------------------------

    assert 0.0 <= data["probability"] <= 1.0

    # --------------------------------------------------
    # Validate Risk Classification
    # --------------------------------------------------

    assert data["risk"] in [
        "Low Risk",
        "Medium Risk",
        "High Risk",
    ]


# ==================================================
# Test Invalid Request
# ==================================================

def test_invalid_prediction_request():
    """
    Test API validation with missing required features.
    """

    # Missing required fields
    payload = {
        "amt_credit": 180000,
        "amt_income_total": 180000,
    }

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422


# ==================================================
# Test Empty Request
# ==================================================

def test_empty_prediction_request():
    """
    Test API validation with an empty request body.
    """

    response = client.post(
        "/predict",
        json={},
    )

    assert response.status_code == 422


# ==================================================
# Main
# ==================================================

if __name__ == "__main__":

    print("=" * 60)
    print("API TEST MODULE")
    print("=" * 60)

    print("\nRun this test using:")
    print("pytest -v test_api.py")