"""
Heart Disease Prediction — Model Training Pipeline
====================================================
Trains and compares multiple ML models on the UCI Heart Disease (Cleveland)
dataset, picks the best performer, and saves the trained pipeline + metrics
for use in the Streamlit demo app.

Run:
    python train_model.py
"""

import json
import warnings

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

RANDOM_STATE = 42
DATA_PATH = "data/heart.csv"
MODEL_DIR = "models"
IMAGE_DIR = "images"

FEATURE_DESCRIPTIONS = {
    "age": "Age in years",
    "sex": "Sex (1 = male, 0 = female)",
    "cp": "Chest pain type (0-3)",
    "trestbps": "Resting blood pressure (mm Hg)",
    "chol": "Serum cholesterol (mg/dl)",
    "fbs": "Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)",
    "restecg": "Resting ECG results (0-2)",
    "thalach": "Maximum heart rate achieved",
    "exang": "Exercise induced angina (1 = yes, 0 = no)",
    "oldpeak": "ST depression induced by exercise relative to rest",
    "slope": "Slope of the peak exercise ST segment (0-2)",
    "ca": "Number of major vessels colored by fluoroscopy (0-4)",
    "thal": "Thalassemia (0-3)",
}


def load_data():
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip() for c in df.columns]
    return df


def run_eda(df):
    """Generate and save a handful of EDA plots for the README / notebook."""
    sns.set_theme(style="whitegrid")

    # Target distribution
    plt.figure(figsize=(5, 4))
    sns.countplot(x="target", data=df, palette=["#4C72B0", "#DD8452"])
    plt.title("Target Distribution (0 = No Disease, 1 = Disease)")
    plt.xlabel("Heart Disease")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"{IMAGE_DIR}/target_distribution.png", dpi=150)
    plt.close()

    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(f"{IMAGE_DIR}/correlation_heatmap.png", dpi=150)
    plt.close()

    # Age distribution by target
    plt.figure(figsize=(6, 4))
    sns.histplot(data=df, x="age", hue="target", kde=True, palette=["#4C72B0", "#DD8452"])
    plt.title("Age Distribution by Heart Disease Status")
    plt.tight_layout()
    plt.savefig(f"{IMAGE_DIR}/age_distribution.png", dpi=150)
    plt.close()

    print("EDA plots saved to images/")


def build_models():
    """Return a dict of {name: sklearn Pipeline} to compare."""
    models = {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", RandomForestClassifier(random_state=RANDOM_STATE)),
            ]
        ),
        "Gradient Boosting": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", GradientBoostingClassifier(random_state=RANDOM_STATE)),
            ]
        ),
        "Support Vector Machine": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", SVC(probability=True, random_state=RANDOM_STATE)),
            ]
        ),
        "Decision Tree": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", DecisionTreeClassifier(random_state=RANDOM_STATE)),
            ]
        ),
    }
    return models


PARAM_GRIDS = {
    "Logistic Regression": {"clf__C": [0.01, 0.1, 1, 10]},
    "Random Forest": {
        "clf__n_estimators": [100, 200],
        "clf__max_depth": [None, 5, 10],
        "clf__min_samples_split": [2, 5],
    },
    "Gradient Boosting": {
        "clf__n_estimators": [100, 200],
        "clf__learning_rate": [0.05, 0.1],
        "clf__max_depth": [2, 3],
    },
    "Support Vector Machine": {"clf__C": [0.1, 1, 10], "clf__kernel": ["rbf", "linear"]},
    "Decision Tree": {"clf__max_depth": [3, 5, 10, None]},
}


def train_and_compare(X_train, X_test, y_train, y_test):
    models = build_models()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    results = []
    fitted_models = {}

    for name, pipe in models.items():
        print(f"Training {name} ...")
        grid = GridSearchCV(pipe, PARAM_GRIDS[name], cv=cv, scoring="roc_auc", n_jobs=-1)
        grid.fit(X_train, y_train)
        best = grid.best_estimator_
        fitted_models[name] = best

        y_pred = best.predict(X_test)
        y_proba = best.predict_proba(X_test)[:, 1]

        results.append(
            {
                "model": name,
                "best_params": grid.best_params_,
                "cv_roc_auc": round(grid.best_score_, 4),
                "test_accuracy": round(accuracy_score(y_test, y_pred), 4),
                "test_precision": round(precision_score(y_test, y_pred), 4),
                "test_recall": round(recall_score(y_test, y_pred), 4),
                "test_f1": round(f1_score(y_test, y_pred), 4),
                "test_roc_auc": round(roc_auc_score(y_test, y_proba), 4),
            }
        )

        print(classification_report(y_test, y_pred, target_names=["No Disease", "Disease"]))

    results_df = pd.DataFrame(results).sort_values("test_roc_auc", ascending=False)
    return results_df, fitted_models


def save_comparison_plot(results_df):
    plt.figure(figsize=(8, 5))
    plot_df = results_df.set_index("model")[["test_accuracy", "test_f1", "test_roc_auc"]]
    plot_df.plot(kind="bar", ax=plt.gca(), colormap="viridis")
    plt.title("Model Comparison")
    plt.ylabel("Score")
    plt.xticks(rotation=25, ha="right")
    plt.ylim(0, 1)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{IMAGE_DIR}/model_comparison.png", dpi=150)
    plt.close()


def save_confusion_and_roc(best_model, best_name, X_test, y_test):
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_estimator(
        best_model, X_test, y_test, display_labels=["No Disease", "Disease"], cmap="Blues", ax=ax
    )
    ax.set_title(f"Confusion Matrix — {best_name}")
    plt.tight_layout()
    plt.savefig(f"{IMAGE_DIR}/confusion_matrix.png", dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_estimator(best_model, X_test, y_test, ax=ax)
    ax.set_title(f"ROC Curve — {best_name}")
    plt.tight_layout()
    plt.savefig(f"{IMAGE_DIR}/roc_curve.png", dpi=150)
    plt.close()


def main():
    df = load_data()
    run_eda(df)

    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    results_df, fitted_models = train_and_compare(X_train, X_test, y_train, y_test)
    print("\n=== Model Comparison (sorted by test ROC-AUC) ===")
    print(results_df.to_string(index=False))

    save_comparison_plot(results_df)

    best_name = results_df.iloc[0]["model"]
    best_model = fitted_models[best_name]
    print(f"\nBest model: {best_name}")

    save_confusion_and_roc(best_model, best_name, X_test, y_test)

    # Persist everything the Streamlit app needs
    joblib.dump(best_model, f"{MODEL_DIR}/heart_disease_model.pkl")
    joblib.dump(list(X.columns), f"{MODEL_DIR}/feature_columns.pkl")

    metadata = {
        "best_model": best_name,
        "feature_descriptions": FEATURE_DESCRIPTIONS,
        "metrics": results_df.to_dict(orient="records"),
        "n_samples": int(len(df)),
        "n_features": int(X.shape[1]),
        "random_state": RANDOM_STATE,
    }
    with open(f"{MODEL_DIR}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved model -> {MODEL_DIR}/heart_disease_model.pkl")
    print(f"Saved metadata -> {MODEL_DIR}/metadata.json")


if __name__ == "__main__":
    main()
