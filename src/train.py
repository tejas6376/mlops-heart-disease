
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import joblib
import json

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "heart_disease.csv"
OUTPUT_DIR = ROOT / "reports" / "training"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXPERIMENT_NAME = "Heart Disease Classification"

# These columns represent categorical values encoded as numbers.
CATEGORICAL_COLUMNS = [
    "sex", "cp", "fbs", "restecg",
    "exang", "slope", "ca", "thal"
]

SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "roc_auc": "roc_auc",
}


def build_pipeline(model):
    """Build preprocessing and model as one pipeline."""

    categorical = [
        col for col in CATEGORICAL_COLUMNS
        if col in X_train.columns
    ]

    numerical = [
        col for col in X_train.columns
        if col not in categorical
    ]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessing = ColumnTransformer([
        ("numeric", numeric_pipeline, numerical),
        ("categorical", categorical_pipeline, categorical),
    ])

    return Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", model),
    ])


def main():
    global X_train

    df = pd.read_csv(DATA_FILE)

    X = df.drop(columns=["target"])
    y = df["target"].astype(int)

    # Hold out test data before cross-validation and model selection.
    global X_test, y_test
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    # A small, documented tuning experiment.
    experiments = [
        (
            "LogisticRegression_C_0.1",
            LogisticRegression(C=0.1, max_iter=2000),
            {"model_type": "LogisticRegression", "C": 0.1},
        ),
        (
            "LogisticRegression_C_1",
            LogisticRegression(C=1.0, max_iter=2000),
            {"model_type": "LogisticRegression", "C": 1.0},
        ),
        (
            "LogisticRegression_C_10",
            LogisticRegression(C=10.0, max_iter=2000),
            {"model_type": "LogisticRegression", "C": 10.0},
        ),
        (
            "RandomForest_100",
            RandomForestClassifier(
                n_estimators=100,
                max_depth=None,
                random_state=42,
            ),
            {
                "model_type": "RandomForest",
                "n_estimators": 100,
                "max_depth": "None",
            },
        ),
        (
            "RandomForest_200_depth5",
            RandomForestClassifier(
                n_estimators=200,
                max_depth=5,
                random_state=42,
            ),
            {
                "model_type": "RandomForest",
                "n_estimators": 200,
                "max_depth": 5,
            },
        ),
    ]

    mlflow.set_experiment(EXPERIMENT_NAME)

    results = []
    fitted_pipelines = {}

    for name, estimator, parameters in experiments:
        print(f"\nRunning experiment: {name}")

        pipeline = build_pipeline(estimator)

        with mlflow.start_run(run_name=name):
            # Record parameters.
            mlflow.log_params(parameters)
            mlflow.log_param("cv_folds", 5)
            mlflow.log_param("test_size", 0.20)
            mlflow.log_param("random_state", 42)

            # Cross-validation uses the complete preprocessing pipeline.
            cv_results = cross_validate(
                pipeline,
                X_train,
                y_train,
                cv=cv,
                scoring=SCORING,
                return_train_score=False,
                n_jobs=1,
            )

            for metric in SCORING:
                scores = cv_results[f"test_{metric}"]

                mlflow.log_metric(
                    f"cv_{metric}_mean",
                    float(np.mean(scores)),
                )
                mlflow.log_metric(
                    f"cv_{metric}_std",
                    float(np.std(scores)),
                )

            # Train on the complete training split.
            pipeline.fit(X_train, y_train)

            # Keep the fitted pipeline for model selection.
            fitted_pipelines[name] = pipeline

            predictions = pipeline.predict(X_test)
            probabilities = pipeline.predict_proba(X_test)[:, 1]

            metrics = {
                "test_accuracy": accuracy_score(
                    y_test, predictions
                ),
                "test_precision": precision_score(
                    y_test, predictions, zero_division=0
                ),
                "test_recall": recall_score(
                    y_test, predictions, zero_division=0
                ),
                "test_roc_auc": roc_auc_score(
                    y_test, probabilities
                ),
            }

            mlflow.log_metrics({
                key: float(value)
                for key, value in metrics.items()
            })

            # Save the confusion matrix.
            ConfusionMatrixDisplay.from_predictions(
                y_test,
                predictions,
                display_labels=[
                    "No disease", "Disease present"
                ],
            )
            plt.title(f"Confusion Matrix: {name}")
            plt.tight_layout()

            confusion_path = OUTPUT_DIR / "confusion_matrix.png"
            plt.savefig(confusion_path, dpi=150)
            plt.close()
            mlflow.log_artifact(str(confusion_path))

            # Save the ROC curve.
            RocCurveDisplay.from_predictions(
                y_test, probabilities
            )
            plt.title(f"ROC Curve: {name}")
            plt.tight_layout()

            roc_path = OUTPUT_DIR / "roc_curve.png"
            plt.savefig(roc_path, dpi=150)
            plt.close()
            mlflow.log_artifact(str(roc_path))

            # Log the complete fitted pipeline as a model artifact.
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                artifact_path="model",
            )

            results.append({
                "experiment": name,
                "cv_roc_auc_mean": float(
                    cv_results["test_roc_auc"].mean()
                ),
                **metrics,
            })

            print("Cross-validation ROC-AUC:",
                  round(float(cv_results["test_roc_auc"].mean()), 4))
            print("Test metrics:", metrics)

    # Save the comparison table.
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        "cv_roc_auc_mean", ascending=False
    )

    results_df.to_csv(
        OUTPUT_DIR / "model_comparison.csv",
        index=False,
    )

    print("\nModel comparison, ranked by CV ROC-AUC:")
    print(results_df.to_string(index=False))

    print(f"\nResults saved in: {OUTPUT_DIR}")

    
    # Select the best configuration using cross-validation ROC-AUC.
    best_name = max(
        results,
        key=lambda result: result["cv_roc_auc_mean"]
    )["experiment"]

    best_pipeline = fitted_pipelines[best_name]

    # Save the complete pipeline.
    model_dir = ROOT / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "heart_disease_pipeline.joblib"
    joblib.dump(best_pipeline, model_path)

    # Save information about the selected model.
    best_result = next(
        result for result in results
        if result["experiment"] == best_name
    )

    metadata = {
        "selected_experiment": best_name,
        "selection_metric": "cv_roc_auc_mean",
        "metrics": best_result,
        "model_file": "heart_disease_pipeline.joblib",
    }

    with open(model_dir / "model_metadata.json", "w") as file:
        json.dump(metadata, file, indent=4)

    print(f"\nSelected model: {best_name}")
    print(f"Saved model: {model_path}")
    print(f"Saved metadata: {model_dir / 'model_metadata.json'}")



if __name__ == "__main__":
    main()
