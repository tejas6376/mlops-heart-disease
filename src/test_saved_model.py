
from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent

MODEL_FILE = ROOT / "models" / "heart_disease_pipeline.joblib"
DATA_FILE = ROOT / "data" / "heart_disease.csv"


def main():
    # Load the saved pipeline.
    model = joblib.load(MODEL_FILE)

    # Load one example, excluding the target column.
    df = pd.read_csv(DATA_FILE)
    sample = df.drop(columns=["target"]).iloc[[0]]

    # Predict without retraining.
    prediction = int(model.predict(sample)[0])
    probabilities = model.predict_proba(sample)[0]

    confidence = float(probabilities[prediction])

    print("Prediction:", prediction)
    print("Confidence:", round(confidence, 4))
    print("Model loaded and prediction completed successfully.")


if __name__ == "__main__":
    main()
