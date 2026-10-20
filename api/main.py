from collections import deque
from datetime import datetime, timezone
from pathlib import Path
import logging
import time

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, ConfigDict


ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "heart_disease_pipeline.joblib"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("heart-disease-api")

model = joblib.load(MODEL_PATH)
feature_names = list(model.feature_names_in_)

REQUEST_COUNT = Counter(
    "api_requests_total",
    "Total API requests",
    ["method", "path", "status"],
)
REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds",
    "API request latency in seconds",
    ["method", "path"],
)
PREDICTION_COUNT = Counter(
    "heart_disease_predictions_total",
    "Model predictions by class",
    ["label"],
)

recent_logs = deque(maxlen=50)
prediction_stats = {"total": 0, "no_disease": 0, "disease_present": 0}

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


def _record_log(message: str) -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    recent_logs.appendleft(f"{timestamp} | {message}")
    logger.info(message)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - start
    path = request.url.path

    REQUEST_COUNT.labels(request.method, path, str(response.status_code)).inc()
    REQUEST_LATENCY.labels(request.method, path).observe(duration)

    if path not in {"/metrics", "/monitor", "/monitor/stats", "/health"}:
        _record_log(
            f"{request.method} {path} -> {response.status_code} ({duration:.3f}s)"
        )

    return response


@app.get("/")
def home():
    return {"message": "Heart Disease Prediction API is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/monitor/stats")
def monitor_stats():
    return {
        "predictions": prediction_stats,
        "recent_requests": list(recent_logs),
        "note": "This is an ML demonstration, not a medical diagnosis.",
    }


@app.get("/monitor", response_class=HTMLResponse)
def monitor():
    rows = "".join(f"<li>{entry}</li>" for entry in recent_logs) or "<li>No requests yet</li>"
    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <meta http-equiv="refresh" content="5" />
  <title>Heart Disease API Monitor</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; background: #f6f7fb; color: #1f2933; }}
    .cards {{ display: flex; gap: 16px; flex-wrap: wrap; }}
    .card {{ background: white; padding: 16px 20px; border-radius: 8px; min-width: 180px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }}
    h1 {{ margin-bottom: 8px; }}
    .muted {{ color: #6b7280; }}
    ul {{ background: white; padding: 16px 28px; border-radius: 8px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }}
  </style>
</head>
<body>
  <h1>Heart Disease API Monitor</h1>
  <p class="muted">Live request logs and prediction counts. Refreshing every 5 seconds.</p>
  <div class="cards">
    <div class="card"><strong>Total predictions</strong><br>{prediction_stats["total"]}</div>
    <div class="card"><strong>No disease (0)</strong><br>{prediction_stats["no_disease"]}</div>
    <div class="card"><strong>Disease present (1)</strong><br>{prediction_stats["disease_present"]}</div>
  </div>
  <h2>Recent requests</h2>
  <ul>{rows}</ul>
  <p class="muted">Prometheus metrics: <a href="/metrics">/metrics</a>. Predictions are not medical advice.</p>
</body>
</html>
"""


@app.post("/predict")
def predict(patient: PatientData):
    try:
        values = patient.model_dump()

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
        label = "disease_present" if prediction == 1 else "no_disease"

        PREDICTION_COUNT.labels(str(prediction)).inc()
        prediction_stats["total"] += 1
        prediction_stats[label] += 1
        _record_log(
            f"prediction={prediction} confidence={round(confidence, 4)}"
        )

        return {
            "prediction": prediction,
            "confidence": round(confidence, 4),
            "interpretation": (
                "Disease present" if prediction == 1 else "No disease detected"
            ),
            "note": "This is an ML prediction, not a medical diagnosis.",
        }

    except HTTPException:
        raise
    except Exception:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed")
