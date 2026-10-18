
from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict


ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "heart_disease_pipeline.joblib"

model = joblib.load(MODEL_PATH)
feature_names = list(model.feature_names_in_)

app = FastAPI(title="Heart Disease Prediction API")


class PatientData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    age: float
    sex: float
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: float
    thal: float


@app.get("/")
def home():
    return {"message": "Heart Disease Prediction API is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(patient: PatientData):
    try:
        values = patient.model_dump()

        # Ensure the input matches the trained model's schema.
        missing = set(feature_names) - set(values)
        extra = set(values) - set(feature_names)

        if missing or extra:
            raise HTTPException(
                status_code=422,
                detail={
                    "missing_features": sorted(missing),
                    "unexpected_features": sorted(extra),
                },
            )

        input_df = pd.DataFrame(
            [[values[name] for name in feature_names]],
            columns=feature_names,
        )

        prediction = int(model.predict(input_df)[0])
        probabilities = model.predict_proba(input_df)[0]
        confidence = float(probabilities[prediction])

        return {
            "prediction": prediction,
            "confidence": round(confidence, 4),
            "interpretation": (
                "Disease present"
                if prediction == 1
                else "No disease detected"
            ),
            "note": "This is an ML prediction, not a medical diagnosis.",
        }

    except HTTPException:
        raise
