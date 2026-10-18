
from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
MODEL_FILE = ROOT / "models" / "heart_disease_pipeline.joblib"
DATA_FILE = ROOT / "data" / "heart_disease.csv"


def test_saved_model_exists():
    assert MODEL_FILE.exists()


def test_model_can_predict():
    model = joblib.load(MODEL_FILE)

    df = pd.read_csv(DATA_FILE)
    sample = df.drop(columns=["target"]).iloc[[0]]

    prediction = model.predict(sample)

    assert len(prediction) == 1
    assert prediction[0] in (0, 1)


def test_prediction_probabilities_are_valid():
    model = joblib.load(MODEL_FILE)

    df = pd.read_csv(DATA_FILE)
    sample = df.drop(columns=["target"]).iloc[[0]]

    probabilities = model.predict_proba(sample)

    assert probabilities.shape == (1, 2)
    assert (probabilities >= 0).all()
    assert (probabilities <= 1).all()
    assert abs(probabilities.sum() - 1.0) < 1e-6
