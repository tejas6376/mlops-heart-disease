# Heart Disease Classification MLOps Project

An end-to-end MLOps assignment using the UCI Heart Disease dataset. The project downloads and explores the data, compares scikit-learn pipelines with MLflow, saves a selected model, runs automated checks in GitHub Actions, serves predictions through FastAPI and Docker, and deploys the API to local Kubernetes.

> **Important:** This project is an educational machine-learning demonstration. Its output is not a diagnosis or medical advice.

## Repository map

| Path | Purpose |
| --- | --- |
| `src/download_data.py` | Fetches UCI dataset 45 and writes the binary target CSV |
| `src/eda.py` | Generates exploratory plots and missing-value summary |
| `src/train.py` | Runs cross-validation experiments, logs to MLflow, and saves model/evaluation artifacts |
| `src/test_saved_model.py` | Manually reloads and exercises the saved model |
| `models/` | Serialized fitted pipeline and selected-model metadata |
| `reports/` | EDA and training artifacts |
| `tests/` | Dataset and saved-model tests |
| `.github/workflows/ci.yml` | GitHub Actions training and test workflow |
| `api/main.py` | FastAPI prediction, health, monitoring, and metrics routes |
| `Dockerfile` | Container image definition |
| `k8s/` | Kubernetes Deployment and LoadBalancer Service |

## Requirements

- Python 3.9 or compatible environment for the current Docker image
- Docker Desktop with Kubernetes enabled (for the local Kubernetes section)
- `kubectl`
- Git, if cloning the repository

## Local setup

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The dataset is committed for convenience. To download a fresh copy and regenerate EDA outputs:

```bash
python src/download_data.py
python src/eda.py
```

The download requires internet access to the UCI repository. The script converts the original target (`num` values 0–4) into a binary target: 0 means no disease and values greater than 0 become 1.

## Train and track experiments

```bash
python src/train.py
```

Training uses a stratified 80/20 train/test split and five-fold stratified cross-validation on the training split. It compares three Logistic Regression regularization settings and two Random Forest configurations. Preprocessing is part of each scikit-learn pipeline: numerical fields use median imputation and scaling; categorical-coded fields use most-frequent imputation and one-hot encoding. The saved metadata identifies the selected pipeline and records its metrics.

View local MLflow runs in another terminal:

```bash
mlflow ui --backend-store-uri ./mlruns --port 5000
```

Open `http://localhost:5000`. Stop the UI with Ctrl+C when finished. `mlruns/` is intentionally ignored by Git; rerun training to recreate it.

## Automated checks

```bash
python -m pytest tests/ -v
ruff check src tests --select E9,F63,F7,F82
python src/test_saved_model.py
```

The GitHub Actions workflow runs data download, training, tests, and a limited Ruff check on pushes and pull requests targeting `main`.

## Run the API locally

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

If port 8000 is busy, select another port, for example `--port 8001`. Useful routes:

| Route | Purpose |
| --- | --- |
| `GET /` | API welcome response |
| `GET /health` | Health check used by Kubernetes probes |
| `POST /predict` | Predict from the 13 input features |
| `GET /docs` | Interactive OpenAPI/Swagger page |
| `GET /monitor` | Simple in-memory request and prediction dashboard |
| `GET /monitor/stats` | Monitoring counts and recent request log as JSON |
| `GET /metrics` | Prometheus request, latency, and prediction metrics |

Example request:

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"age":55,"sex":1,"cp":2,"trestbps":130,"chol":240,"fbs":0,"restecg":1,"thalach":150,"exang":0,"oldpeak":1.0,"slope":1,"ca":0,"thal":2}'
```

The request schema rejects additional fields. Responses include a predicted class, confidence, interpretation, and educational-use disclaimer.

## Build and run with Docker

```bash
docker build -t heart-disease-api:latest .
docker run --rm --name heart-disease-test -p 8001:8000 heart-disease-api:latest
```

Use `http://localhost:8001/health`, `http://localhost:8001/docs`, and `http://localhost:8001/monitor`. The host port is 8001 because port 8000 was previously occupied in this development setup. Stop the foreground container with Ctrl+C. To run detached, remove `--rm`, add `-d`, and stop it later with `docker stop heart-disease-test`.

## Deploy to local Kubernetes

The checked-in deployment uses the local image `heart-disease-api:latest` with `imagePullPolicy: Never`; build the image in Docker Desktop before applying the manifests. Confirm that Docker Desktop Kubernetes is the selected context:

```bash
kubectl config current-context
kubectl get nodes
docker image ls heart-disease-api
```

Deploy and verify:

```bash
docker build -t heart-disease-api:latest .
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl rollout status deployment/heart-disease-api --timeout=120s
kubectl get pods,services -o wide
```

Access the service through a local port-forward:

```bash
kubectl port-forward service/heart-disease-api 8003:8000
```

In another terminal, visit `http://localhost:8003/docs` or `http://localhost:8003/monitor`, or run:

```bash
curl -i http://localhost:8003/health
curl -s http://localhost:8003/metrics
```

The LoadBalancer Service manifest requests external service exposure, but its external IP may not be reachable from the host in every local cluster configuration. Port-forward is the verified local access path. For failures, inspect:

```bash
kubectl describe pod -l app=heart-disease-api
kubectl get events --sort-by=.lastTimestamp
kubectl logs deployment/heart-disease-api
```

## Monitoring and limitations

The API emits standard process logs to stdout, records recent requests and prediction totals in process memory, and exposes Prometheus counters/histograms at `/metrics`. Kubernetes readiness and liveness probes call `/health`; container resource requests and limits are set in the Deployment. In-memory counts reset when the process restarts and are not a persistent monitoring system. The Prometheus scrape annotations are metadata; a Prometheus server is not included in this repository.

The project uses a small public dataset and a simplified binary target transformation. Metrics describe one held-out split and cross-validation experiment; they do not establish clinical suitability, fairness, or real-world performance. Do not use this model for healthcare decisions.

## Reproducibility artifacts

- Selected model: `models/heart_disease_pipeline.joblib`
- Selection and held-out metrics: `models/model_metadata.json`
- Experiment comparison: `reports/training/model_comparison.csv`
- EDA and evaluation plots: `reports/eda/` and `reports/training/`
- MLflow local run database: generated in ignored `mlruns/` after training

