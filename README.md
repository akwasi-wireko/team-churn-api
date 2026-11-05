# Telecom Customer Churn Prediction Service

This repository contains a production-ready machine learning API for predicting telecom customer churn. It takes a reproducible scikit-learn pipeline and serves it through a validated FastAPI endpoint.

The project uses the **UCI Iranian Churn dataset (ID 563)** to perform binary classification, identifying whether a customer is at risk of churning based on 13 usage features.

## Repository Structure

The project follows a standard layout for deploying machine learning models alongside application code:

```text
team-churn-api/
├── api/
│   ├── __init__.py
│   └── main.py                     # FastAPI deployment entry point (api.main:app)
├── ml_platform/
│   └── training/
│       └── churn/
│           ├── train.py            # Model training script
│           └── models/             # Saved artifacts (pipeline.joblib, feature_names.json)
├── production_app/
│   └── tests/
│       └── test_api.py             # API tests (pytest)
├── pyproject.toml                  # Project dependencies
└── uv.lock                         # Locked dependency versions
```

## Prerequisites

This project uses `uv` for dependency management. Ensure you have Python installed and the `uv` package manager configured on your system.

## Step-by-Step Setup

### 1. Install Dependencies
Initialize your environment and install all required packages (FastAPI, scikit-learn, pandas, etc.) matching the lockfile:
```bash
uv sync --locked
```

### 2. Train the Model
Before running the API, you must train the model pipeline and save the artifacts. The training script automatically fetches the UCI dataset, trains a logistic regression model with standard scaling and median imputation, and saves the outputs to `ml_platform/training/churn/models/`.

```bash
uv run python ml_platform/training/churn/train.py
```
*Verify that `churn_pipeline.joblib` and `feature_names.json` have been generated in the `models/` directory.*

### 3. Start the FastAPI Service
Launch the API using Uvicorn. The deployment contract for this repository requires running the application from the root directory.

```bash
uv run uvicorn api.main:app --host 0.0.0.0 --port 8080
```

## API Usage

Once the server is running, you can view the automatically generated interactive OpenAPI documentation by navigating to `http://localhost:8080/docs` in your browser.

### Health Check
Verify the service is running:
```bash
curl http://localhost:8080/health
```

### Make a Prediction
The `/predictions` endpoint requires all 13 features. It utilizes Pydantic to validate the input data types and constraints before passing them to the machine learning model.

```bash
curl -X POST http://localhost:8080/predictions \
-H "Content-Type: application/json" \
-d '{
  "call_failure": 8,
  "complains": 1,
  "subscription_length": 20,
  "charge_amount": 3,
  "seconds_of_use": 2500,
  "frequency_of_use": 55,
  "frequency_of_sms": 12,
  "distinct_called_numbers": 25,
  "age_group": 3,
  "tariff_plan": 1,
  "status": 1,
  "age": 42,
  "customer_value": 1200.5
}'
```

**Expected Response:**
```json
{
  "prediction": 0,
  "churn_probability": 0.2741,
  "model_version": "week1-v1"
}
```

## Testing and Quality Checks

Before committing any changes, ensure your code passes all linting, formatting, and unit tests.

```bash
# Check formatting
uv run ruff check
uv run ruff format --check

# Run API test suite
uv run pytest
```