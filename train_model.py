"""
Heart Disease Prediction — Advanced Training Pipeline
Dataset: Combined Cleveland + Hungarian + Switzerland + VA Long Beach + Statlog (918 clean patients)
"""
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              roc_auc_score, roc_curve, confusion_matrix, brier_score_loss)
from xgboost import XGBClassifier

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ---------------------------------------------------------------
# 1. Load clean data
# ---------------------------------------------------------------
df = pd.read_csv("data/heart_clean.csv")
df.columns = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
              "thalach", "exang", "oldpeak", "slope", "target"]

FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope"]
X = df[FEATURES]
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# ---------------------------------------------------------------
# 2. Define candidate models + hyperparameter search spaces
# ---------------------------------------------------------------
search_spaces = {
    "Logistic Regression": (
        LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        {"C": np.logspace(-2, 2, 20), "penalty": ["l2"], "solver": ["lbfgs"]}
    ),
    "Random Forest": (
        RandomForestClassifier(random_state=RANDOM_STATE),
        {"n_estimators": [200, 300, 400, 500],
         "max_depth": [4, 6, 8, 10, None],
         "min_samples_split": [2, 4, 6],
         "min_samples_leaf": [1, 2, 4],
         "max_features": ["sqrt", "log2"]}
    ),
    "Gradient Boosting": (
        GradientBoostingClassifier(random_state=RANDOM_STATE),
        {"n_estimators": [100, 200, 300],
         "learning_rate": [0.01, 0.03, 0.05, 0.1],
         "max_depth": [2, 3, 4],
         "subsample": [0.8, 0.9, 1.0]}
    ),
    "XGBoost": (
        XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE),
        {"n_estimators": [100, 200, 300],
         "learning_rate": [0.01, 0.03, 0.05, 0.1],
         "max_depth": [2, 3, 4, 5],
         "subsample": [0.8, 0.9, 1.0],
         "colsample_bytree": [0.7, 0.85, 1.0]}
    ),
    "SVM": (
        SVC(probability=True, random_state=RANDOM_STATE),
        {"C": np.logspace(-1, 2, 15), "gamma": ["scale", "auto"], "kernel": ["rbf"]}
    ),
    "Decision Tree": (
        DecisionTreeClassifier(random_state=RANDOM_STATE),
        {"max_depth": [3, 4, 5, 6, 8], "min_samples_split": [2, 4, 6, 10],
         "min_samples_leaf": [1, 2, 4]}
    ),
}

results = []
fitted_models = {}

print("=" * 70)
print("HYPERPARAMETER TUNING (RandomizedSearchCV, 5-fold stratified CV)")
print("=" * 70)

for name, (estimator, params) in search_spaces.items():
    search = RandomizedSearchCV(
        estimator, params, n_iter=25, scoring="roc_auc",
        cv=cv, random_state=RANDOM_STATE, n_jobs=-1
    )
    search.fit(X_train_s, y_train)
    best_model = search.best_estimator_
    fitted_models[name] = best_model

    # Cross-validated metrics on train set
    cv_scores = cross_validate(
        best_model, X_train_s, y_train, cv=cv,
        scoring=["accuracy", "precision", "recall", "f1", "roc_auc"]
    )

    # Held-out test metrics
    y_pred = best_model.predict(X_test_s)
    y_proba = best_model.predict_proba(X_test_s)[:, 1]

    row = {
        "model": name,
        "best_params": search.best_params_,
        "cv_roc_auc": cv_scores["test_roc_auc"].mean(),
        "test_accuracy": accuracy_score(y_test, y_pred),
        "test_precision": precision_score(y_test, y_pred),
        "test_recall": recall_score(y_test, y_pred),
        "test_f1": f1_score(y_test, y_pred),
        "test_roc_auc": roc_auc_score(y_test, y_proba),
        "test_brier": brier_score_loss(y_test, y_proba),
    }
    results.append(row)
    print(f"\n{name}: test ROC-AUC={row['test_roc_auc']:.4f}  "
          f"acc={row['test_accuracy']:.4f}  f1={row['test_f1']:.4f}  "
          f"brier={row['test_brier']:.4f}")
    print(f"  best params: {search.best_params_}")

results_df = pd.DataFrame(results).sort_values("test_roc_auc", ascending=False).reset_index(drop=True)

# ---------------------------------------------------------------
# 3. Build a stacking ensemble on top of the tuned base models
# ---------------------------------------------------------------
print("\n" + "=" * 70)
print("STACKING ENSEMBLE")
print("=" * 70)

estimators = [(n.lower().replace(" ", "_"), m) for n, m in fitted_models.items()
              if n in ["Random Forest", "Gradient Boosting", "XGBoost", "Logistic Regression"]]

stack = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
    cv=cv, n_jobs=-1
)
stack.fit(X_train_s, y_train)
y_pred_stack = stack.predict(X_test_s)
y_proba_stack = stack.predict_proba(X_test_s)[:, 1]

stack_row = {
    "model": "Stacking Ensemble",
    "best_params": "meta: LogisticRegression",
    "cv_roc_auc": cross_validate(stack, X_train_s, y_train, cv=cv, scoring="roc_auc")["test_score"].mean(),
    "test_accuracy": accuracy_score(y_test, y_pred_stack),
    "test_precision": precision_score(y_test, y_pred_stack),
    "test_recall": recall_score(y_test, y_pred_stack),
    "test_f1": f1_score(y_test, y_pred_stack),
    "test_roc_auc": roc_auc_score(y_test, y_proba_stack),
    "test_brier": brier_score_loss(y_test, y_proba_stack),
}
print(f"Stacking Ensemble: test ROC-AUC={stack_row['test_roc_auc']:.4f}  "
      f"acc={stack_row['test_accuracy']:.4f}  f1={stack_row['test_f1']:.4f}  "
      f"brier={stack_row['test_brier']:.4f}")

fitted_models["Stacking Ensemble"] = stack
results_df = pd.concat([results_df, pd.DataFrame([stack_row])], ignore_index=True)
results_df = results_df.sort_values("test_roc_auc", ascending=False).reset_index(drop=True)

print("\n" + "=" * 70)
print("FINAL LEADERBOARD (sorted by test ROC-AUC)")
print("=" * 70)
print(results_df[["model", "cv_roc_auc", "test_accuracy", "test_f1", "test_roc_auc", "test_brier"]]
      .to_string(index=False))

# ---------------------------------------------------------------
# 4. Select best model and calibrate its probabilities
# ---------------------------------------------------------------
best_name = results_df.iloc[0]["model"]
best_raw_model = fitted_models[best_name]
print(f"\nSelected model for deployment: {best_name}")

# Calibrate probabilities (isotonic if enough data, else sigmoid) so the
# "% chance of heart disease" the app shows is a genuinely calibrated probability,
# not just a raw, possibly overconfident score.
calibrated_model = CalibratedClassifierCV(best_raw_model, method="sigmoid", cv=cv)
calibrated_model.fit(X_train_s, y_train)

y_proba_cal = calibrated_model.predict_proba(X_test_s)[:, 1]
brier_before = brier_score_loss(y_test, best_raw_model.predict_proba(X_test_s)[:, 1])
brier_after = brier_score_loss(y_test, y_proba_cal)
print(f"Brier score (lower=better calibrated): raw={brier_before:.4f} -> calibrated={brier_after:.4f}")

# ---------------------------------------------------------------
# 5. Save plots for the dashboard
# ---------------------------------------------------------------
import os
os.makedirs("images", exist_ok=True)

sns.set_style("whitegrid")

# Model comparison bar chart
plt.figure(figsize=(11, 5))
plot_df = results_df.set_index("model")[["test_accuracy", "test_f1", "test_roc_auc"]]
plot_df.plot(kind="bar", figsize=(11, 5), colormap="viridis")
plt.title("Model Comparison — Accuracy / F1 / ROC-AUC")
plt.ylabel("Score")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.savefig("images/model_comparison.png", dpi=110)
plt.close()

# ROC curve for best model
fpr, tpr, _ = roc_curve(y_test, y_proba_cal)
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f"{best_name} (AUC={roc_auc_score(y_test, y_proba_cal):.3f})", linewidth=2)
plt.plot([0, 1], [0, 1], "--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC Curve — {best_name} (calibrated)")
plt.legend()
plt.tight_layout()
plt.savefig("images/roc_curve.png", dpi=110)
plt.close()

# Confusion matrix
cm = confusion_matrix(y_test, calibrated_model.predict(X_test_s))
plt.figure(figsize=(5, 4.2))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Disease", "Disease"], yticklabels=["No Disease", "Disease"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix — {best_name}")
plt.tight_layout()
plt.savefig("images/confusion_matrix.png", dpi=110)
plt.close()

# Calibration curve (reliability diagram) — shows the probabilities are trustworthy
prob_true_raw, prob_pred_raw = calibration_curve(y_test, best_raw_model.predict_proba(X_test_s)[:, 1], n_bins=10)
prob_true_cal, prob_pred_cal = calibration_curve(y_test, y_proba_cal, n_bins=10)
plt.figure(figsize=(6, 6))
plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfectly calibrated")
plt.plot(prob_pred_raw, prob_true_raw, "o-", label="Before calibration")
plt.plot(prob_pred_cal, prob_true_cal, "o-", label="After calibration")
plt.xlabel("Predicted probability")
plt.ylabel("Observed frequency")
plt.title("Calibration Curve — is the % trustworthy?")
plt.legend()
plt.tight_layout()
plt.savefig("images/calibration_curve.png", dpi=110)
plt.close()

# Feature importance (permutation-based, model-agnostic, works for stacking too)
from sklearn.inspection import permutation_importance
perm = permutation_importance(calibrated_model, X_test_s, y_test, n_repeats=20,
                               random_state=RANDOM_STATE, scoring="roc_auc", n_jobs=-1)
feat_imp = pd.Series(perm.importances_mean, index=FEATURES).sort_values(ascending=True)
plt.figure(figsize=(8, 6))
feat_imp.plot(kind="barh", color="#c0392b")
plt.xlabel("Mean decrease in ROC-AUC when shuffled")
plt.title("Feature Importance (Permutation)")
plt.tight_layout()
plt.savefig("images/feature_importance.png", dpi=110)
plt.close()
feat_imp_dict = feat_imp.sort_values(ascending=False).to_dict()

# EDA: correlation heatmap + distributions (kept from v1, regenerated on clean data)
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("images/correlation_heatmap.png", dpi=110)
plt.close()

plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="age", hue="target", kde=True, bins=20, palette=["#3498db", "#e74c3c"])
plt.title("Age Distribution by Heart Disease Status")
plt.tight_layout()
plt.savefig("images/age_distribution.png", dpi=110)
plt.close()

# ---------------------------------------------------------------
# 6. Persist model artifacts
# ---------------------------------------------------------------
os.makedirs("models", exist_ok=True)
joblib.dump(calibrated_model, "models/heart_disease_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(FEATURES, "models/feature_columns.pkl")
# Save the raw (pre-calibration) tree-based model too, purely so the app can
# generate a fast, faithful per-patient SHAP explanation (which factors pushed
# THIS patient's risk up or down). The calibrated_model above is still what's
# used for the actual probability shown to the user.
joblib.dump(best_raw_model, "models/explainer_model.pkl")
joblib.dump(X_train_s, "models/X_train_scaled_sample.pkl")

metadata = {
    "best_model": best_name,
    "n_train": len(X_train),
    "n_test": len(X_test),
    "n_total": len(df),
    "n_features": len(FEATURES),
    "dataset": "Combined Cleveland + Hungarian + Switzerland + VA Long Beach + Statlog (deduplicated, cleaned)",
    "test_accuracy": float(results_df.iloc[0]["test_accuracy"]),
    "test_precision": float(results_df.iloc[0]["test_precision"]),
    "test_recall": float(results_df.iloc[0]["test_recall"]),
    "test_f1": float(results_df.iloc[0]["test_f1"]),
    "test_roc_auc": float(roc_auc_score(y_test, y_proba_cal)),
    "brier_score_calibrated": float(brier_after),
    "feature_importance": {k: float(v) for k, v in feat_imp_dict.items()},
    "leaderboard": results_df.drop(columns=["best_params"]).to_dict(orient="records"),
}
with open("models/metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

results_df.drop(columns=["best_params"]).to_csv("models/leaderboard.csv", index=False)

print("\nSaved: models/heart_disease_model.pkl (calibrated), scaler.pkl, feature_columns.pkl, metadata.json")
print("Saved plots to images/")
print("\nDONE.")
