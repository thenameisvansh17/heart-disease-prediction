"""
Heart Disease Prediction — Streamlit Web App
==============================================
Interactive live demo that loads the pre-trained model and lets a user
enter patient vitals to get an instant risk prediction, along with
model performance dashboards for recruiters/reviewers to inspect.

Run locally:
    streamlit run app.py

Deploy free on Streamlit Community Cloud (share.streamlit.io) by pointing
it at this repo's app.py — no server config needed.
"""

import json

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Load artifacts
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("models/heart_disease_model.pkl")
    feature_columns = joblib.load("models/feature_columns.pkl")
    with open("models/metadata.json") as f:
        metadata = json.load(f)
    return model, feature_columns, metadata


model, feature_columns, metadata = load_artifacts()

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.title("❤️ Heart Disease Predictor")
page = st.sidebar.radio("Navigate", ["🔮 Predict", "📊 Model Performance", "ℹ️ About Project"])

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"**Model in use:** {metadata['best_model']}  \n"
    f"**Training samples:** {metadata['n_samples']}  \n"
    f"**Features:** {metadata['n_features']}"
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    "[![GitHub](https://img.shields.io/badge/GitHub-Repo-black?logo=github)]"
    "(https://github.com/YOUR_USERNAME/heart-disease-prediction)"
)

# ---------------------------------------------------------------------------
# Page: Predict
# ---------------------------------------------------------------------------
if page == "🔮 Predict":
    st.title("❤️ Heart Disease Risk Prediction")
    st.write(
        "Enter patient details below. The model returns a probability of "
        "heart disease based on patterns learned from the UCI Cleveland "
        "Heart Disease dataset (303 patients, 13 clinical features)."
    )

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.slider("Age", 18, 100, 50)
            sex = st.selectbox("Sex", options=[1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
            cp = st.selectbox(
                "Chest Pain Type",
                options=[0, 1, 2, 3],
                format_func=lambda x: [
                    "Typical Angina",
                    "Atypical Angina",
                    "Non-anginal Pain",
                    "Asymptomatic",
                ][x],
            )
            trestbps = st.slider("Resting Blood Pressure (mm Hg)", 80, 220, 120)
            fbs = st.selectbox(
                "Fasting Blood Sugar > 120 mg/dl",
                options=[0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No",
            )

        with col2:
            chol = st.slider("Serum Cholesterol (mg/dl)", 100, 600, 200)
            restecg = st.selectbox(
                "Resting ECG Results",
                options=[0, 1, 2],
                format_func=lambda x: ["Normal", "ST-T Abnormality", "LV Hypertrophy"][x],
            )
            thalach = st.slider("Max Heart Rate Achieved", 60, 220, 150)
            exang = st.selectbox(
                "Exercise Induced Angina", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No"
            )
            oldpeak = st.slider("ST Depression (oldpeak)", 0.0, 6.5, 1.0, step=0.1)

        with col3:
            slope = st.selectbox(
                "Slope of Peak Exercise ST Segment",
                options=[0, 1, 2],
                format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x],
            )
            ca = st.selectbox("Number of Major Vessels (0-4) colored by fluoroscopy", options=[0, 1, 2, 3, 4])
            thal = st.selectbox(
                "Thalassemia",
                options=[0, 1, 2, 3],
                format_func=lambda x: ["Unknown", "Normal", "Fixed Defect", "Reversible Defect"][x],
            )

        submitted = st.form_submit_button("🔍 Predict", use_container_width=True)

    if submitted:
        input_dict = {
            "age": age,
            "sex": sex,
            "cp": cp,
            "trestbps": trestbps,
            "chol": chol,
            "fbs": fbs,
            "restecg": restecg,
            "thalach": thalach,
            "exang": exang,
            "oldpeak": oldpeak,
            "slope": slope,
            "ca": ca,
            "thal": thal,
        }
        input_df = pd.DataFrame([input_dict])[feature_columns]

        proba = model.predict_proba(input_df)[0][1]
        prediction = model.predict(input_df)[0]

        st.markdown("---")
        result_col, gauge_col = st.columns([1, 1])

        with result_col:
            if prediction == 1:
                st.error(f"⚠️ **High Risk Detected** — Estimated probability: {proba*100:.1f}%")
                st.write(
                    "The model flags patterns consistent with heart disease in this "
                    "profile. This is **not a medical diagnosis** — please consult a "
                    "cardiologist for a proper clinical evaluation."
                )
            else:
                st.success(f"✅ **Low Risk** — Estimated probability: {proba*100:.1f}%")
                st.write(
                    "The model does not find strong patterns of heart disease in this "
                    "profile. Regular checkups are still recommended."
                )

        with gauge_col:
            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=proba * 100,
                    title={"text": "Risk Probability (%)"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "darkred" if prediction == 1 else "darkgreen"},
                        "steps": [
                            {"range": [0, 40], "color": "#d4f7d4"},
                            {"range": [40, 70], "color": "#fff3cd"},
                            {"range": [70, 100], "color": "#f8d7da"},
                        ],
                    },
                )
            )
            fig.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "⚕️ Disclaimer: This tool is for educational/portfolio purposes only and "
            "must not be used for real medical decisions."
        )

# ---------------------------------------------------------------------------
# Page: Model Performance
# ---------------------------------------------------------------------------
elif page == "📊 Model Performance":
    st.title("📊 Model Performance Dashboard")

    metrics_df = pd.DataFrame(metadata["metrics"])
    st.subheader("Model Comparison (5 algorithms, cross-validated)")
    st.dataframe(
        metrics_df[
            ["model", "cv_roc_auc", "test_accuracy", "test_precision", "test_recall", "test_f1", "test_roc_auc"]
        ].style.highlight_max(
            subset=["cv_roc_auc", "test_accuracy", "test_precision", "test_recall", "test_f1", "test_roc_auc"],
            color="lightgreen",
        ),
        use_container_width=True,
    )

    st.markdown(f"**Selected model for deployment:** `{metadata['best_model']}` (highest test ROC-AUC)")

    col1, col2 = st.columns(2)
    with col1:
        st.image("images/model_comparison.png", caption="Model comparison across metrics")
        st.image("images/confusion_matrix.png", caption="Confusion matrix of the best model")
    with col2:
        st.image("images/roc_curve.png", caption="ROC curve of the best model")
        st.image("images/correlation_heatmap.png", caption="Feature correlation heatmap")

    st.subheader("Exploratory Data Analysis")
    col3, col4 = st.columns(2)
    with col3:
        st.image("images/target_distribution.png")
    with col4:
        st.image("images/age_distribution.png")

# ---------------------------------------------------------------------------
# Page: About
# ---------------------------------------------------------------------------
else:
    st.title("ℹ️ About This Project")
    st.markdown(
        """
### Heart Disease Prediction — End-to-End ML Project

**Problem statement:** Predict whether a patient is likely to have heart
disease based on 13 clinical attributes (age, cholesterol, chest pain type,
ECG results, etc.), using the widely-cited **UCI Cleveland Heart Disease
dataset** (303 patients).

**What this project demonstrates:**
- Data cleaning & exploratory data analysis (EDA)
- Feature engineering & preprocessing with `scikit-learn` Pipelines
- Training and comparing **5 classification algorithms**: Logistic
  Regression, Random Forest, Gradient Boosting, SVM, Decision Tree
- Hyperparameter tuning with `GridSearchCV` + Stratified K-Fold CV
- Model evaluation: accuracy, precision, recall, F1, ROC-AUC, confusion matrix
- Model persistence with `joblib`
- An interactive **Streamlit** web app for real-time predictions
- Clean, reproducible project structure ready for a GitHub portfolio

**Tech stack:** Python · pandas · scikit-learn · matplotlib/seaborn ·
Plotly · Streamlit

**Author:** *Add your name, LinkedIn, and GitHub link here.*

**Disclaimer:** This is an educational/portfolio project. Predictions are
not medical advice.
        """
    )
