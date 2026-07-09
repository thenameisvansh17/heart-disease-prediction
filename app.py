import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import shap

st.set_page_config(page_title="Heart Disease Risk Predictor", page_icon="🫀", layout="wide")

# ------------------------------------------------------------------
# Load artifacts
# ------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("models/heart_disease_model.pkl")          # calibrated, used for probability
    explainer_model = joblib.load("models/explainer_model.pkl")    # raw tree model, used for SHAP
    scaler = joblib.load("models/scaler.pkl")
    features = joblib.load("models/feature_columns.pkl")
    X_bg = joblib.load("models/X_train_scaled_sample.pkl")
    with open("models/metadata.json") as f:
        meta = json.load(f)
    df_raw = pd.read_csv("data/heart_clean.csv")
    df_raw.columns = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                       "thalach", "exang", "oldpeak", "slope", "target"]
    explainer = shap.TreeExplainer(explainer_model, X_bg[:100])
    return model, explainer, scaler, features, meta, df_raw

model, explainer, scaler, FEATURES, meta, df_raw = load_artifacts()

CP_LABELS = {1: "Typical Angina", 2: "Atypical Angina", 3: "Non-anginal Pain", 4: "Asymptomatic"}
RESTECG_LABELS = {0: "Normal", 1: "ST-T Wave Abnormality", 2: "Left Ventricular Hypertrophy"}
SLOPE_LABELS = {0: "Downsloping", 1: "Flat", 2: "Upsloping"}
FEATURE_DISPLAY = {
    "age": "Age", "sex": "Sex", "cp": "Chest Pain Type", "trestbps": "Resting Blood Pressure",
    "chol": "Cholesterol", "fbs": "Fasting Blood Sugar > 120", "restecg": "Resting ECG",
    "thalach": "Max Heart Rate", "exang": "Exercise-Induced Angina", "oldpeak": "ST Depression (oldpeak)",
    "slope": "ST Slope"
}

# ------------------------------------------------------------------
# Styling
# ------------------------------------------------------------------
st.markdown("""
<style>
    .risk-card {
        padding: 1.4rem 1.6rem; border-radius: 14px; margin-bottom: 1rem;
    }
    .risk-low { background: linear-gradient(135deg, #e8f8ee, #d3f2e0); border-left: 6px solid #1e9e5a; }
    .risk-mod { background: linear-gradient(135deg, #fff6e0, #ffedc2); border-left: 6px solid #d68a10; }
    .risk-high { background: linear-gradient(135deg, #fde9e9, #fad0d0); border-left: 6px solid #c62828; }
    .metric-box { text-align: center; padding: 0.6rem; }
    div[data-testid="stMetricValue"] { font-size: 1.6rem; }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🫀 Heart Disease Predictor")
    page = st.radio("Navigate", ["Predict", "Model Performance", "About Project"], label_visibility="collapsed")
    st.divider()
    st.markdown(f"**Model:** {meta['best_model']} *(calibrated)*")
    st.markdown(f"**Training data:** {meta['n_total']} real patients")
    st.markdown(f"**Test ROC-AUC:** {meta['test_roc_auc']:.3f}")
    st.markdown(f"**Test Accuracy:** {meta['test_accuracy']:.1%}")
    st.caption("Dataset: Cleveland + Hungarian + Switzerland + VA Long Beach + Statlog (deduplicated & cleaned)")
    st.divider()
    st.caption("⚠️ Educational project — not a medical device. Not medical advice.")

# ------------------------------------------------------------------
# PAGE 1 — PREDICT
# ------------------------------------------------------------------
if page == "Predict":
    st.title("🫀 Heart Disease Risk Prediction")
    st.write(
        f"Enter patient details below. The model returns a **calibrated probability** of heart "
        f"disease, learned from {meta['n_total']} real patients across five clinical studies."
    )

    with st.form("patient_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Demographics**")
            age = st.slider("Age", 18, 100, 50)
            sex = st.selectbox("Sex", ["Male", "Female"])
            cp = st.selectbox("Chest Pain Type", list(CP_LABELS.values()))
            trestbps = st.slider("Resting Blood Pressure (mm Hg)", 80, 220, 120)

        with c2:
            st.markdown("**Labs & ECG**")
            chol = st.slider("Serum Cholesterol (mg/dl)", 100, 600, 200)
            fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
            restecg = st.selectbox("Resting ECG Results", list(RESTECG_LABELS.values()))
            thalach = st.slider("Max Heart Rate Achieved", 60, 220, 150)

        with c3:
            st.markdown("**Exercise Test**")
            exang = st.selectbox("Exercise-Induced Angina", ["No", "Yes"])
            oldpeak = st.slider("ST Depression (oldpeak)", 0.0, 6.5, 1.0, 0.1)
            slope = st.selectbox("Slope of Peak Exercise ST Segment", list(SLOPE_LABELS.values()))

        submitted = st.form_submit_button("🔍 Predict Risk", use_container_width=True)

    if submitted:
        cp_val = [k for k, v in CP_LABELS.items() if v == cp][0]
        restecg_val = [k for k, v in RESTECG_LABELS.items() if v == restecg][0]
        slope_val = [k for k, v in SLOPE_LABELS.items() if v == slope][0]

        row = pd.DataFrame([{
            "age": age, "sex": 1 if sex == "Male" else 0, "cp": cp_val, "trestbps": trestbps,
            "chol": chol, "fbs": 1 if fbs == "Yes" else 0, "restecg": restecg_val,
            "thalach": thalach, "exang": 1 if exang == "Yes" else 0, "oldpeak": oldpeak,
            "slope": slope_val
        }])[FEATURES]

        row_scaled = scaler.transform(row)
        proba = model.predict_proba(row_scaled)[0, 1]
        pct = proba * 100

        if pct < 30:
            css, label, emoji = "risk-low", "Low Risk", "🟢"
        elif pct < 60:
            css, label, emoji = "risk-mod", "Moderate Risk", "🟠"
        else:
            css, label, emoji = "risk-high", "High Risk", "🔴"

        st.markdown("---")
        left, right = st.columns([1, 1.3])

        with left:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pct,
                number={"suffix": "%", "font": {"size": 46}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1},
                    "bar": {"color": "#1f2937"},
                    "steps": [
                        {"range": [0, 30], "color": "#c7f0d8"},
                        {"range": [30, 60], "color": "#ffe4a3"},
                        {"range": [60, 100], "color": "#f8c1c1"},
                    ],
                    "threshold": {"line": {"color": "black", "width": 3}, "thickness": 0.8, "value": pct},
                },
                title={"text": "Predicted Probability of Heart Disease"}
            ))
            fig.update_layout(height=320, margin=dict(l=20, r=20, t=60, b=10))
            st.plotly_chart(fig, use_container_width=True)

            st.markdown(
                f"<div class='risk-card {css}'><h3>{emoji} {label} &mdash; {pct:.1f}% chance</h3>"
                f"<p>This estimate is calibrated: among all patients this model gives roughly "
                f"a {pct:.0f}% score to, about {pct:.0f}% actually have heart disease "
                f"(based on held-out test data).</p></div>",
                unsafe_allow_html=True
            )

        with right:
            st.markdown("#### Why this prediction? (per-patient factor breakdown)")
            sv = explainer.shap_values(row_scaled, check_additivity=False)
            sv = np.array(sv).reshape(-1)
            contrib = pd.DataFrame({
                "feature": [FEATURE_DISPLAY[f] for f in FEATURES],
                "value": row.iloc[0].values,
                "impact": sv
            }).sort_values("impact", key=abs, ascending=True)

            colors = ["#c62828" if v > 0 else "#1e9e5a" for v in contrib["impact"]]
            fig2 = go.Figure(go.Bar(
                x=contrib["impact"], y=contrib["feature"], orientation="h",
                marker_color=colors,
                text=[f"{v:+.3f}" for v in contrib["impact"]], textposition="outside"
            ))
            fig2.update_layout(
                height=380, margin=dict(l=10, r=10, t=10, b=10),
                xaxis_title="← lowers risk    |    raises risk →",
                yaxis_title=None
            )
            st.plotly_chart(fig2, use_container_width=True)
            st.caption(
                "Red bars push this patient's risk **up**, green bars push it **down**, relative to "
                "the model's average prediction. Values are this patient's actual inputs."
            )

        st.markdown("#### How this patient compares to the training population")
        compare_cols = st.columns(4)
        highlights = [("age", age), ("trestbps", trestbps), ("chol", chol), ("thalach", thalach)]
        for col, (feat, val) in zip(compare_cols, highlights):
            pop_mean = df_raw[feat].mean()
            delta = val - pop_mean
            col.metric(FEATURE_DISPLAY[feat], f"{val}", f"{delta:+.0f} vs avg ({pop_mean:.0f})")

        st.caption(
            "⚠️ This tool is an educational ML portfolio project trained on public research data. "
            "It is **not** a diagnostic device and does not replace professional medical evaluation. "
            "If you have real health concerns, please consult a doctor."
        )

# ------------------------------------------------------------------
# PAGE 2 — MODEL PERFORMANCE
# ------------------------------------------------------------------
elif page == "Model Performance":
    st.title("📊 Model Performance")
    st.write(
        f"**{len(meta['leaderboard'])} algorithms** were tuned with `RandomizedSearchCV` "
        f"(5-fold stratified cross-validation) on {meta['n_train']} training patients, "
        f"then evaluated on a held-out set of {meta['n_test']} patients never seen during training."
    )

    lb = pd.DataFrame(meta["leaderboard"])
    st.markdown("#### Leaderboard (sorted by test ROC-AUC)")
    st.dataframe(
        lb.style.background_gradient(subset=["test_roc_auc"], cmap="Greens")
                .format({c: "{:.4f}" for c in lb.columns if c != "model"}),
        use_container_width=True
    )

    st.markdown(
        f"**Selected for deployment:** :green[{meta['best_model']}] — highest test ROC-AUC, "
        f"then wrapped in probability calibration (Brier score: {meta['brier_score_calibrated']:.4f}, "
        f"lower is better) so the displayed % is trustworthy, not just a confident-looking raw score."
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Model Comparison")
        st.image("images/model_comparison.png", use_container_width=True)
        st.markdown("##### Confusion Matrix")
        st.image("images/confusion_matrix.png", use_container_width=True)
    with c2:
        st.markdown("##### ROC Curve")
        st.image("images/roc_curve.png", use_container_width=True)
        st.markdown("##### Calibration Curve")
        st.image("images/calibration_curve.png", use_container_width=True)

    st.markdown("##### Global Feature Importance (permutation-based)")
    st.image("images/feature_importance.png", use_container_width=True)

    st.markdown("##### Exploratory Data Analysis")
    c3, c4 = st.columns(2)
    with c3:
        st.image("images/correlation_heatmap.png", use_container_width=True)
    with c4:
        st.image("images/age_distribution.png", use_container_width=True)

    with st.expander("Interactive: explore the training data yourself"):
        feat_x = st.selectbox("X-axis feature", FEATURES, index=0)
        feat_y = st.selectbox("Y-axis feature", FEATURES, index=7)
        fig3 = px.scatter(
            df_raw, x=feat_x, y=feat_y, color=df_raw["target"].map({0: "No Disease", 1: "Disease"}),
            color_discrete_map={"No Disease": "#1e9e5a", "Disease": "#c62828"},
            labels={feat_x: FEATURE_DISPLAY.get(feat_x, feat_x), feat_y: FEATURE_DISPLAY.get(feat_y, feat_y)},
            opacity=0.65
        )
        st.plotly_chart(fig3, use_container_width=True)

# ------------------------------------------------------------------
# PAGE 3 — ABOUT
# ------------------------------------------------------------------
else:
    st.title("ℹ️ About This Project")
    st.markdown("## Heart Disease Prediction — End-to-End ML Project")
    st.markdown(
        f"**Problem statement:** Predict whether a patient has heart disease from "
        f"{meta['n_features']} clinical attributes, using a combined, deduplicated dataset of "
        f"{meta['n_total']} real patients from five studies (Cleveland Clinic, Hungarian Institute "
        f"of Cardiology, University Hospital Zurich, University Hospital Basel, and the VA Medical "
        f"Center Long Beach)."
    )

    st.markdown("**What this project demonstrates:**")
    st.markdown("""
- Data cleaning of a known-dirty real-world dataset (duplicate removal, zero-encoded missing values imputed by group median)
- Training and hyperparameter-tuning **6 models**: Logistic Regression, Random Forest, Gradient Boosting, XGBoost, SVM, Decision Tree — plus a **stacking ensemble**
- Model selection via 5-fold stratified cross-validated ROC-AUC on held-out test data
- **Probability calibration** (`CalibratedClassifierCV`) so predicted percentages reflect true observed frequencies, verified with a reliability/calibration curve and Brier score
- **Per-patient explainability** with SHAP — every prediction shows which specific factors pushed that patient's risk up or down
- An interactive Streamlit dashboard with a live risk gauge, factor breakdown, and population comparison
    """)

    st.markdown(f"**Tech stack:** Python · pandas · scikit-learn · XGBoost · SHAP · Plotly · Streamlit")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Test ROC-AUC", f"{meta['test_roc_auc']:.3f}")
    m2.metric("Test Accuracy", f"{meta['test_accuracy']:.1%}")
    m3.metric("Test F1", f"{meta['test_f1']:.3f}")
    m4.metric("Brier Score", f"{meta['brier_score_calibrated']:.3f}")

    st.markdown("**Author:** *Add your name, LinkedIn, and GitHub link here.*")
    st.info(
        "⚠️ **Disclaimer:** This is an educational/portfolio project. Predictions are not medical "
        "advice and should never be used for real clinical decisions."
    )
