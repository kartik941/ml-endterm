import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Credit Risk Scoring Engine",
    page_icon="💳",
    layout="wide"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------
st.markdown("""
<style>
.main {
    background-color: #f8fafc;
}

.block-container {
    padding-top: 2rem;
}

.header-box {
    background: linear-gradient(90deg, #1d4ed8, #2563eb);
    padding: 28px;
    border-radius: 18px;
    margin-bottom: 25px;
    box-shadow: 0 8px 20px rgba(0,0,0,0.08);
}

.header-title {
    font-size: 34px;
    font-weight: 800;
    color: white;
}

.header-subtitle {
    font-size: 18px;
    color: #dbeafe;
    margin-top: 8px;
}

.result-card {
    background: white;
    padding: 20px;
    border-radius: 16px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.06);
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown("""
<div class="header-box">
    <div class="header-title">PROJECT 1 | CREDIT RISK SCORING ENGINE</div>
    <div class="header-subtitle">Domain: Retail Banking / Lending</div>
</div>
""", unsafe_allow_html=True)

st.markdown("### Predict whether a loan applicant should be Approved or Rejected using Machine Learning")

# --------------------------------------------------
# LOAD SAVED FILES
# --------------------------------------------------
try:
    with open("logistic_regression_model.pkl", "rb") as file:
        model = pickle.load(file)

    with open("scaler.pkl", "rb") as file:
        scaler = pickle.load(file)

    st.success("Model loaded successfully")

except Exception:
    st.error("Model files not found. Please keep .pkl files in the same folder as this app.")
    st.stop()

# --------------------------------------------------
# MODEL SELECTION
# --------------------------------------------------
st.markdown("---")
st.subheader("Step 1: Select Prediction Model")

selected_model = st.radio(
    "Choose ML Model",
    ["Logistic Regression", "Random Forest"],
    horizontal=True
)

if selected_model == "Random Forest":
    try:
        with open("random_forest_model.pkl", "rb") as file:
            model = pickle.load(file)
        st.info("Random Forest selected for prediction")
    except Exception:
        st.warning("Random Forest model file not found. Using Logistic Regression instead.")

else:
    st.info("Logistic Regression selected for prediction")

# --------------------------------------------------
# INPUT FORM
# --------------------------------------------------
st.markdown("---")
st.subheader("Step 2: Enter Applicant Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=80, value=30)
    income = st.number_input("Monthly Income", min_value=10000, max_value=500000, value=50000)
    loan_amount = st.number_input("Loan Amount Requested", min_value=50000, max_value=5000000, value=500000)
    credit_history = st.number_input("Credit History Length (Years)", min_value=0, max_value=40, value=5)

with col2:
    existing_loans = st.number_input("Existing Loans Count", min_value=0, max_value=10, value=1)
    past_defaults = st.number_input("Past Defaults", min_value=0, max_value=10, value=0)

    employment_type = st.selectbox(
        "Employment Type",
        ["Business", "Freelancer", "Salaried", "Self-Employed"]
    )

# --------------------------------------------------
# MANUAL ONE-HOT ENCODING
# Base category = Business
# --------------------------------------------------
emp_freelancer = 1 if employment_type == "Freelancer" else 0
emp_salaried = 1 if employment_type == "Salaried" else 0
emp_self_employed = 1 if employment_type == "Self-Employed" else 0

# --------------------------------------------------
# PREDICTION BUTTON
# --------------------------------------------------
if st.button("Generate Credit Decision", use_container_width=True):

    input_data = np.array([[ 
        age,
        income,
        loan_amount,
        credit_history,
        existing_loans,
        past_defaults,
        emp_freelancer,
        emp_salaried,
        emp_self_employed
    ]])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.markdown("---")
    st.subheader("Prediction Result")

    if prediction == 1:
        st.success(f"Loan Status: APPROVED")
    else:
        st.error(f"Loan Status: REJECTED")

    st.info(f"Approval Probability Score: {round(probability * 100, 2)}%")

        # --------------------------------------------------
    # REASONING SECTION
    # --------------------------------------------------
    st.markdown("### Why this prediction?")

    reasons = []

    if credit_history >= 10:
        reasons.append("Strong credit history improves trust and repayment confidence.")
    else:
        reasons.append("Short credit history increases uncertainty for lenders.")

    if income >= 100000:
        reasons.append("Higher monthly income supports better repayment ability.")
    else:
        reasons.append("Lower monthly income may reduce repayment strength.")

    if past_defaults >= 2:
        reasons.append("Multiple past defaults significantly increase rejection risk.")
    elif past_defaults == 1:
        reasons.append("A previous default adds moderate lending risk.")
    else:
        reasons.append("No past defaults improves approval confidence.")

    if existing_loans >= 3:
        reasons.append("Higher existing loan burden increases financial pressure.")
    else:
        reasons.append("Manageable number of existing loans supports approval chances.")

    for reason in reasons:
        st.write(f"• {reason}")

    st.markdown("---")
    st.subheader("Final Business Interpretation")

    if prediction == 1:
        st.success(
            "The applicant shows healthy financial indicators and lower repayment risk, making loan approval suitable for NBFC operations."
        )
    else:
        st.error(
            "The applicant shows higher lending risk due to weak repayment indicators, so rejecting the loan helps reduce future NPAs."
        )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("---")
st.markdown("Built using Logistic Regression + Scikit-learn + Streamlit")
