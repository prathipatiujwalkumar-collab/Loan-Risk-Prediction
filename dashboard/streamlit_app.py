"""
Streamlit Dashboard
Intelligent Loan Risk Assessment Platform

Frontend for the FastAPI loan-risk prediction service.

Author: Ujwal
"""

import requests
import streamlit as st


# ============================================================
# Configuration
# ============================================================

API_URL = "http://api:8000/predict"


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="Loan Risk Assessment",
    page_icon="💳",
    layout="wide",
)


# ============================================================
# Custom Styling
# ============================================================

st.markdown(
    """
    <style>

    .main {
        padding-top: 1rem;
    }

    .title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        font-size: 1.1rem;
        color: #777;
        margin-bottom: 2rem;
    }

    .risk-card {
        padding: 1.5rem;
        border-radius: 12px;
        margin-top: 1rem;
        text-align: center;
        border: 1px solid #ddd;
    }

    .metric-title {
        font-size: 1rem;
        color: #666;
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 700;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Header
# ============================================================

st.markdown(
    '<div class="title">💳 Intelligent Loan Risk Assessment</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    "Production-grade ML-powered loan default risk prediction"
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================
# API Status
# ============================================================

with st.sidebar:

    st.header("⚙️ System")

    try:

        health_response = requests.get(
            "http://api:8000/health",
            timeout=5,
        )

        if health_response.status_code == 200:

            st.success("API Online")

        else:

            st.error("API Unavailable")

    except requests.exceptions.RequestException:

        st.error("API Offline")

    st.divider()

    st.caption(
        "FastAPI → ML Pipeline → Prediction"
    )


# ============================================================
# Input Section
# ============================================================

st.header("👤 Customer Information")

st.write(
    "Enter the customer's financial and demographic information "
    "to assess loan default risk."
)


# ------------------------------------------------------------
# Create two columns
# ------------------------------------------------------------

col1, col2 = st.columns(2)


# ============================================================
# Financial Information
# ============================================================

with col1:

    st.subheader("💰 Financial Information")

    amt_credit = st.number_input(
        "Loan Credit Amount",
        min_value=0.0,
        value=500000.0,
        step=10000.0,
        help="Credit amount requested by the customer.",
    )

    amt_income_total = st.number_input(
        "Annual Income",
        min_value=0.0,
        value=200000.0,
        step=10000.0,
        help="Total annual income of the customer.",
    )

    amt_annuity = st.number_input(
        "Loan Annuity",
        min_value=0.0,
        value=25000.0,
        step=1000.0,
        help="Annualized loan repayment amount.",
    )

    amt_goods_price = st.number_input(
        "Goods Price",
        min_value=0.0,
        value=450000.0,
        step=10000.0,
        help="Price of the goods associated with the loan.",
    )


# ============================================================
# Demographic Information
# ============================================================

with col2:

    st.subheader("👨‍👩‍👧 Customer Information")

    days_employed = st.number_input(
        "Days Employed",
        value=-2000,
        step=100,
        help=(
            "Number of days employed. "
            "Negative values follow the dataset convention."
        ),
    )

    days_birth = st.number_input(
        "Days Since Birth",
        value=-12000,
        step=100,
        help=(
            "Number of days since birth. "
            "Negative values follow the dataset convention."
        ),
    )

    cnt_children = st.number_input(
        "Number of Children",
        min_value=0,
        value=1,
        step=1,
    )

    cnt_fam_members = st.number_input(
        "Family Members",
        min_value=1.0,
        value=3.0,
        step=1.0,
    )


# ============================================================
# Prediction Button
# ============================================================

st.divider()

predict_button = st.button(
    "🔍 Assess Loan Risk",
    type="primary",
    use_container_width=True,
)


# ============================================================
# Prediction Logic
# ============================================================

if predict_button:

    # --------------------------------------------------------
    # Build API payload
    # --------------------------------------------------------

    payload = {

        "AMT_CREDIT": amt_credit,

        "AMT_INCOME_TOTAL": amt_income_total,

        "AMT_ANNUITY": amt_annuity,

        "AMT_GOODS_PRICE": amt_goods_price,

        "DAYS_EMPLOYED": days_employed,

        "DAYS_BIRTH": days_birth,

        "CNT_CHILDREN": cnt_children,

        "CNT_FAM_MEMBERS": cnt_fam_members,

    }

    # --------------------------------------------------------
    # Show processing status
    # --------------------------------------------------------

    with st.spinner(
        "Running ML prediction..."
    ):

        try:

            response = requests.post(
                API_URL,
                json=payload,
                timeout=30,
            )

            # ------------------------------------------------
            # Successful prediction
            # ------------------------------------------------

            if response.status_code == 200:

                result = response.json()

                prediction = result.get(
                    "prediction"
                )

                probability = result.get(
                    "probability"
                )

                risk = result.get(
                    "risk"
                )

                # ============================================
                # Results
                # ============================================

                st.success(
                    "Loan risk assessment completed successfully."
                )

                st.header("📊 Prediction Result")

                result_col1, result_col2, result_col3 = st.columns(
                    3
                )

                # --------------------------------------------
                # Prediction
                # --------------------------------------------

                with result_col1:

                    st.metric(
                        label="Prediction",
                        value=str(prediction),
                    )

                # --------------------------------------------
                # Probability
                # --------------------------------------------

                with result_col2:

                    st.metric(
                        label="Default Probability",
                        value=f"{probability:.2%}",
                    )

                # --------------------------------------------
                # Risk
                # --------------------------------------------

                with result_col3:

                    st.metric(
                        label="Risk Classification",
                        value=risk,
                    )

                # ============================================
                # Risk Interpretation
                # ============================================

                st.divider()

                if risk == "Low Risk":

                    st.success(
                        "🟢 Low Risk — "
                        "The model estimates a relatively low "
                        "probability of loan default."
                    )

                elif risk == "Medium Risk":

                    st.warning(
                        "🟡 Medium Risk — "
                        "The model estimates a moderate "
                        "probability of loan default."
                    )

                elif risk == "High Risk":

                    st.error(
                        "🔴 High Risk — "
                        "The model estimates a relatively high "
                        "probability of loan default."
                    )

                else:

                    st.info(
                        f"Risk classification: {risk}"
                    )

                # ============================================
                # Probability Visualization
                # ============================================

                st.subheader(
                    "Default Probability"
                )

                probability_percentage = (
                    probability * 100
                )

                st.progress(
                    min(
                        max(
                            probability,
                            0.0,
                        ),
                        1.0,
                    )
                )

                st.caption(
                    f"Estimated probability of default: "
                    f"{probability_percentage:.2f}%"
                )

                # ============================================
                # Prediction Details
                # ============================================

                with st.expander(
                    "🔎 View Prediction Details"
                ):

                    st.json(
                        {
                            "prediction": prediction,
                            "probability": probability,
                            "risk": risk,
                        }
                    )

            # ------------------------------------------------
            # API error
            # ------------------------------------------------

            else:

                st.error(
                    f"Prediction API returned "
                    f"HTTP {response.status_code}"
                )

                try:

                    error_details = response.json()

                    st.json(
                        error_details
                    )

                except ValueError:

                    st.code(
                        response.text
                    )

        # ----------------------------------------------------
        # Connection error
        # ----------------------------------------------------

        except requests.exceptions.ConnectionError:

            st.error(
                "❌ Could not connect to the FastAPI server."
            )

            st.info(
                "Make sure the Docker container is running:"
            )

            st.code(
                "docker ps"
            )

        # ----------------------------------------------------
        # Timeout
        # ----------------------------------------------------

        except requests.exceptions.Timeout:

            st.error(
                "⏱️ The prediction request timed out."
            )

        # ----------------------------------------------------
        # Other request errors
        # ----------------------------------------------------

        except requests.exceptions.RequestException as error:

            st.error(
                "❌ Request failed."
            )

            st.exception(
                error
            )

        # ----------------------------------------------------
        # Unexpected errors
        # ----------------------------------------------------

        except Exception as error:

            st.error(
                "❌ Unexpected dashboard error."
            )

            st.exception(
                error
            )


# ============================================================
# Footer
# ============================================================

st.divider()

st.caption(
    "Intelligent Loan Risk Assessment Platform | "
    "FastAPI + Scikit-Learn + XGBoost + Docker + Streamlit"
)