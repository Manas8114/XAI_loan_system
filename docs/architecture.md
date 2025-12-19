# XAI Loan System - Architecture Document

## System Overview

The XAI-driven Causal Counterfactual Loan System is a research-grade AI system for loan risk assessment with explainable AI capabilities. It provides not just predictions, but actionable, causally-valid recommendations for loan applicants.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           API Layer (FastAPI)                           │
│                    /predict  /explain  /counterfactual                  │
└────────────────────────────────┬───────────────────────────────────────┘
                                 │
         ┌───────────────────────┼───────────────────────┐
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────────┐
│   ML Models     │    │  Explainability │    │   Counterfactual Layer  │
│   XGBoost/RF    │    │   SHAP Engine   │    │  DiCE | Opt | Causal*   │
└────────┬────────┘    └────────┬────────┘    └───────────┬─────────────┘
         │                      │                         │
         └──────────────────────┼─────────────────────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │     Ethics Layer      │
                    │ Actionability | Fair  │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │      Data Layer       │
                    │ Ingestion | Preproc   │
                    └───────────────────────┘

* Primary Contribution
```

## Component Details

### 1. Data Layer

| Module | Purpose |
|--------|---------|
| `ingestion.py` | Load data from CSV or generate synthetic data |
| `preprocessing.py` | Feature engineering, scaling, encoding |
| `bias_handler.py` | Bias detection and mitigation |

**Key Features:**
- Synthetic data generation mimicking LendingClub structure
- Feature engineering (payment_to_income, credit utilization flags)
- Fairness auditing with disparate impact analysis

### 2. ML Models Layer

| Module | Purpose |
|--------|---------|
| `xgboost_model.py` | Primary XGBoost classifier |
| `random_forest.py` | Baseline Random Forest |
| `model_registry.py` | Centralized model management |

**Key Features:**
- Stratified cross-validation
- Feature importance extraction
- Model persistence and versioning

### 3. Explainability Layer

| Module | Purpose |
|--------|---------|
| `shap_engine.py` | TreeSHAP integration |
| `explanation_report.py` | Combined report generation |

**Key Features:**
- Local and global SHAP explanations
- Feature interactions
- SHAP is **diagnostic only** (not prescriptive)

### 4. Counterfactual Layer (PRIMARY CONTRIBUTION)

| Module | Purpose |
|--------|---------|
| `dice_engine.py` | DiCE baseline (correlation-based) |
| `optimization_cf.py` | Optimization baseline |
| `causal_model.py` | Structural Causal Model |
| `causal_cf.py` | **Causal counterfactual engine** |

**Key Features:**
- DAG-based causal graph
- Intervention simulation via do-calculus
- Counterfactual validity checking
- Baseline comparison framework

### 5. Ethics Layer

| Module | Purpose |
|--------|---------|
| `actionability.py` | Formal actionability scoring |
| `constraints.py` | Immutability, bounds, direction constraints |
| `feasibility.py` | Combined feasibility guard |
| `fairness.py` | Group fairness validation |

**Key Features:**
- Actionability Score formula (novel contribution)
- Automatic constraint violation fixing
- Fairness auditing across demographics

### 6. API Layer

| Module | Purpose |
|--------|---------|
| `main.py` | FastAPI application |
| `routes.py` | Endpoint handlers |
| `schemas.py` | Pydantic request/response models |

**Endpoints:**
- `POST /api/v1/predict` - Loan prediction
- `POST /api/v1/explain` - SHAP explanation
- `POST /api/v1/counterfactual` - Generate recommendations
- `POST /api/v1/compare` - Compare CF methods

## Data Flow

1. **Request** → API receives loan application features
2. **Preprocessing** → Features are transformed and scaled
3. **Prediction** → Model predicts approval probability
4. **Explanation** → SHAP provides diagnostic insight
5. **Counterfactual** → Causal engine generates recommendations
6. **Validation** → Ethics layer validates constraints and fairness
7. **Response** → Formatted explanation returned to user

## Causal Graph Structure

```
emp_length ──────┬─────────────────────────────────────┐
                 ▼                                     │
         ──→ annual_inc ──→ dti ──→ loan_status ◄─────┤
                 ▲                       ▲             │
                 │                       │             │
    fico_score ──┴─────────────────────────────────────┘
         │
         ▼
    revol_util
```

**Edges:**
- `emp_length → annual_inc` (more experience = higher income)
- `annual_inc → dti` (income affects debt-to-income ratio)
- `fico_score → revol_util` (credit score affects utilization behavior)
- `dti → loan_status` (DTI affects approval)
- `fico_score → loan_status` (credit score affects approval)

## Deployment

```bash
# Development
python -m src.api.main

# Production
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test module
pytest tests/test_counterfactual.py -v
```
