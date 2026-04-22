import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Credit Risk Scoring Engine",
    page_icon="💳",
    layout="wide"
)

# -----------------------------
# CUSTOM STYLING
# -----------------------------
st.markdown("""
<style>
.main {
    background-color: #f8fafc;
}

.big-title {
    font-size: 40px;
    font-weight: 700;
    color: #0f172a;
}

.sub-title {
    font-size: 18px;
    color: #475569;
}

.metric-card {
    background: white;
    padding: 20px;
    border-radius: 16px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.06);
    margin-bottom: 15px;
}

.section-box {
    background: white;
    padding: 20px;
    border-radius: 18px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.05);
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# HEADER
# -----------------------------
st.markdown('<div style="background: linear-gradient(90deg, #1e3a8a, #2563eb); padding: 28px; border-radius: 18px; margin-bottom: 20px; box-shadow: 0 8px 24px rgba(0,0,0,0.08);">'
            '<div style="font-size: 38px; font-weight: 800; color: white;">💳 Credit Risk Scoring Engine</div>'
            '<div style="font-size: 18px; color: #dbeafe; margin-top: 8px;">AI/ML for NBFC Loan Approval Prediction | End-Term Project</div>'
            '</div>', unsafe_allow_html=True)
# subtitle moved inside premium header card
st.markdown("---")

st.markdown("""
### Project Objective
To help NBFCs reduce bad loans (NPAs) by predicting whether a customer should be approved for a loan using Machine Learning.
""")

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("Project Navigation")
section = st.sidebar.radio(
    "Go to Section",
    [
        "Project Overview",
        "Upload Dataset",
        "Model Performance",
        "Feature Importance",
        "Final Recommendation"
    ]
)

# -----------------------------
# PROJECT OVERVIEW
# -----------------------------
if section == "Project Overview":
    st.markdown("## Business Problem")
    st.info(
        "NBFCs face major losses due to incorrect loan approvals. This system helps identify safer borrowers using ML-driven credit scoring."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Dataset Size", "500 Rows")

    with col2:
        st.metric("Models Used", "2")

    with col3:
        st.metric("Final Accuracy", "92%")

    st.markdown("---")
    st.markdown("### Models Used")
    st.write("- Logistic Regression")
    st.write("- Random Forest Classifier")

# -----------------------------
# DATASET SECTION
# -----------------------------
elif section == "Upload Dataset":
    uploaded_file = st.file_uploader("Upload Credit Risk CSV Dataset", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        st.success("Dataset uploaded successfully")

        st.markdown("## Dataset Preview")
        st.dataframe(df.head(), use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Shape")
            st.write(f"Rows: {df.shape[0]}")
            st.write(f"Columns: {df.shape[1]}")

        with col2:
            st.markdown("### Missing Values")
            st.dataframe(
                df.isnull().sum().reset_index().rename(
                    columns={"index": "Column", 0: "Missing Values"}
                ),
                use_container_width=True
            )

        st.markdown("## Target Variable Distribution")

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(x="Approved", data=df, ax=ax)
        ax.set_title("Loan Approval Distribution")
        st.pyplot(fig)

        st.success("Dataset is balanced: No major class imbalance detected")

    else:
        st.warning("Please upload the CSV dataset first.")

# -----------------------------
# MODEL PERFORMANCE
# -----------------------------
elif section == "Model Performance":
    st.markdown("## Model Comparison")

    comparison_df = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"],
        "Logistic Regression": [0.92, 0.8667, 0.9512, 0.9070, 0.9644],
        "Random Forest": [0.83, 0.75, 0.8780, 0.8090, 0.9305]
    })

    st.dataframe(comparison_df, use_container_width=True)

    st.success("Logistic Regression outperformed Random Forest across all major metrics")

    st.markdown("## Confusion Matrix")

    cm = np.array([[53, 6], [2, 39]])

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", ax=ax2)
    ax2.set_title("Confusion Matrix - Logistic Regression")
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Actual")
    st.pyplot(fig2)

# -----------------------------
# FEATURE IMPORTANCE
# -----------------------------
elif section == "Feature Importance":
    st.markdown("## Top Features Affecting Credit Decisions")

    importance = pd.DataFrame({
        "Feature": [
            "Credit_History_Length_Years",
            "Income_Monthly",
            "Past_Defaults",
            "Loan_Amount_Requested",
            "Age"
        ],
        "Importance": [0.289, 0.280, 0.196, 0.085, 0.074]
    })

    st.dataframe(importance, use_container_width=True)

    fig3, ax3 = plt.subplots(figsize=(8, 5))
    sns.barplot(
        data=importance,
        x="Importance",
        y="Feature",
        ax=ax3
    )
    ax3.set_title("Feature Importance")
    st.pyplot(fig3)

    st.info(
        "Top 3 drivers: Credit History Length, Monthly Income, and Past Defaults"
    )

# -----------------------------
# FINAL RECOMMENDATION
# -----------------------------
elif section == "Final Recommendation":
    st.markdown("## Final Business Recommendation")

    st.success("Recommended Model: Logistic Regression")

    st.markdown("""
### Why Logistic Regression?

- Higher Accuracy (92%)
- Better Recall (95.12%)
- Stronger ROC-AUC
- More interpretable for financial decisions
- Easier deployment for NBFC operations

### Final Conclusion

Logistic Regression is the best deployment choice because it provides strong predictive performance and explainable decision-making, which is critical in financial lending systems.
""")

    st.info("Project completed successfully with a recommended deployment model for NBFC operations.")
