# Heart Disease Classification MLOps Assignment Report

**Student:** Tejas Mahajan  
**Project:** End-to-end MLOps pipeline for binary heart disease classification  

> This report is a submission draft. Replace bracketed fields with your details and insert genuine screenshots from your run before exporting/submitting. Model output is for educational demonstration only and is not medical advice.

## 1. Executive summary

This project implements a reproducible machine-learning workflow for the UCI Heart Disease dataset. It covers data download and exploratory analysis, preprocessing and model comparison, experiment tracking with MLflow, serialization of a selected scikit-learn pipeline, automated CI checks, a FastAPI prediction service, Docker packaging, local Kubernetes deployment, and basic application monitoring.

The saved model metadata selects Logistic Regression with `C=1` according to mean five-fold cross-validation ROC-AUC. The recorded mean CV ROC-AUC is 0.9025. On the held-out test split, the saved metadata reports accuracy 0.8852, precision 0.8387, recall 0.9286, and ROC-AUC 0.9665. These results are specific to this dataset split and should not be interpreted as clinical performance.

## 2. Problem and data

The project treats the task as binary classification: target 0 represents the dataset's no-disease class; original target values greater than 0 are mapped to class 1. `src/download_data.py` fetches UCI Machine Learning Repository dataset ID 45 using `ucimlrepo`, converts feature and target columns to numeric values, removes rows without a target, performs the binary mapping, and writes `data/heart_disease.csv`.

The repository includes a dataset CSV so training can proceed without immediately fetching the data again. Re-downloading requires network access. The source dataset is small and observational; the target transformation also combines several original disease severity values into one class.

**Evidence to insert:** screenshot of the download script's row/column and target-distribution output, plus the dataset file in the repository.

## 3. Exploratory data analysis

`src/eda.py` prints dimensions, sample rows, dtypes, descriptive statistics, duplicate count, and missingness. It writes a missing-value summary and visualizations for target balance, feature distributions, and feature/target correlations.

The generated artifacts are `reports/eda/missing_values.csv`, `class_balance.png`, `feature_histograms.png`, and `correlation_heatmap.png`. The training pipeline uses imputers to handle feature missingness without requiring rows to be discarded.

**Evidence to insert:** class balance, feature histograms, and correlation heatmap from `reports/eda/`.

## 4. Feature engineering and model development

The training script reserves a stratified 20% holdout set with random state 42. Five model configurations are compared: Logistic Regression with `C` values 0.1, 1, and 10; a 100-tree Random Forest; and a 200-tree Random Forest with maximum depth 5. Each candidate is evaluated with stratified five-fold cross-validation on the training portion.

Preprocessing and the estimator are combined in a scikit-learn `Pipeline` to ensure the same transformations are applied during training and inference. Numerical columns receive median imputation and standard scaling. Categorical-coded fields (`sex`, `cp`, `fbs`, `restecg`, `exang`, `slope`, `ca`, and `thal`) receive most-frequent imputation and one-hot encoding. The full fitted pipeline is serialized so API requests use the exact trained transformations.

## 5. Results and model selection

The model is selected using mean cross-validation ROC-AUC. Current committed artifacts report:

| Candidate | Mean CV ROC-AUC | Test accuracy | Test precision | Test recall | Test ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression, C=1 | 0.9025 | 0.8852 | 0.8387 | 0.9286 | 0.9665 |
| Logistic Regression, C=0.1 | 0.9025 | 0.8852 | 0.8387 | 0.9286 | 0.9654 |
| Logistic Regression, C=10 | 0.8995 | 0.8689 | 0.8333 | 0.8929 | 0.9632 |
| Random Forest, 100 trees | 0.8968 | 0.8689 | 0.8125 | 0.9286 | 0.9437 |
| Random Forest, 200 trees, depth 5 | 0.8931 | 0.8689 | 0.8333 | 0.8929 | 0.9426 |

The selected C=1 model slightly exceeds C=0.1 on mean CV ROC-AUC. Its metrics are recorded in `models/model_metadata.json`; the comparison table is `reports/training/model_comparison.csv`. Training also saves confusion-matrix and ROC-curve images in `reports/training/`.

## 6. Experiment tracking and model packaging

Experiments are logged to MLflow under `Heart Disease Classification`. Parameters include estimator configuration, fold count, test size, and random seed; metrics include cross-validation scores and held-out metrics. The trained model and visual artifacts are logged to MLflow. The local tracking store is created under `mlruns/` and excluded from Git; run `python src/train.py` to reproduce it and `mlflow ui --backend-store-uri ./mlruns --port 5000` to inspect it.

The selected fitted pipeline is stored at `models/heart_disease_pipeline.joblib`, with selection metadata in `models/model_metadata.json`. The repository includes `src/test_saved_model.py` to reload and exercise the artifact. Pickle-based formats such as joblib should only be loaded from trusted sources, and dependency compatibility should be maintained when moving the artifact between environments.

**Evidence to insert:** MLflow experiment page showing candidate runs and metrics; model artifact listing.

## 7. CI/CD and testing

The GitHub Actions workflow at `.github/workflows/ci.yml` runs on pushes and pull requests to `main`, sets up Python 3.11, installs dependencies, runs a limited Ruff check, downloads the dataset, trains the models, runs pytest, and uploads model/training artifacts. Tests validate dataset presence and binary target validity, plus saved model loading, prediction shape/class, and probability bounds.

This workflow is continuous integration and artifact generation; it does not publish a container image or automatically deploy the Kubernetes service. A successful run on the final commit should be captured for the submission.

**Evidence to insert:** successful Actions workflow run with commit identifier visible.

## 8. API and Docker deployment

`api/main.py` loads the saved pipeline and exposes `/`, `/health`, `/predict`, `/docs`, `/monitor`, `/monitor/stats`, and `/metrics`. Input is validated using Pydantic, and unexpected fields are forbidden. The prediction response includes the predicted class, estimated confidence, interpretation, and an explicit non-diagnostic note. Request and prediction counters are exposed using Prometheus client metrics.

The Dockerfile installs project requirements, copies the API and model artifact, and starts Uvicorn on port 8000. The image is built with `docker build -t heart-disease-api:latest .`. A local container can be exposed at host port 8001 if port 8000 is occupied.

**Evidence to insert:** Docker image/container listing, `/health` response, Swagger docs, and sample prediction response.

## 9. Kubernetes deployment and monitoring

The local Kubernetes manifests define one API replica with resource requests and limits, readiness and liveness checks against `/health`, and a `LoadBalancer` Service. The deployment references the local Docker image with `imagePullPolicy: Never`, so Docker Desktop must have the built image available before applying the manifests.

The deployment was verified in the local `docker-desktop` cluster: the rollout completed, the Pod reported `1/1 Running` with zero restarts, and the Service reported type `LoadBalancer`. Through a temporary Kubernetes port-forward, `/health`, `/metrics`, `/monitor/stats`, and `/predict` returned successfully. A sample prediction returned class 0 with confidence 0.8057. The external Service IP did not respond directly from the host, so port-forward is the verified access path.

Monitoring is intentionally lightweight: stdout logs are available through `kubectl logs`, `/monitor` shows in-memory recent requests and prediction counts, `/monitor/stats` provides JSON state, and `/metrics` exposes Prometheus-compatible counters and latency histograms. Counts reset when the API process restarts, and no Prometheus server or persistent log store is included.

**Evidence to insert:** `kubectl get pods,services -o wide`, successful rollout status, logs, `/monitor`, and `/metrics` output.

## 10. Reproduction, limitations, and conclusion

From the repository root, the core workflow is:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/download_data.py
python src/eda.py
python src/train.py
python -m pytest tests/ -v
docker build -t heart-disease-api:latest .
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl rollout status deployment/heart-disease-api --timeout=120s
```

The project demonstrates a basic end-to-end MLOps workflow, but has limitations: small dataset, simplified binary labeling, one holdout split, no external model registry, no persistent monitoring backend, and no cloud deployment. The chosen model is not clinically validated and must not be used to make care decisions. The final submission should include genuine screenshots, a short demonstration video, student/course details, and confirmation that the assignment's original rubric is met.

## Appendix: screenshot and video checklist

Capture these from your own running environment (do not fabricate or edit outputs):

1. EDA target balance and feature visualizations.
2. MLflow experiment list with all five candidate runs and metrics.
3. CI workflow success on the final commit.
4. Docker image/container and working `/health` or `/docs` page.
5. Kubernetes Pod Ready, Service, and rollout status.
6. API sample prediction, `/monitor`, and `/metrics`.

Suggested short video: show repository structure, open MLflow results, call `/predict` in Swagger, show monitoring output, and finish with Kubernetes Pod/Service status. Avoid showing local secrets, access tokens, or unrelated personal data.

