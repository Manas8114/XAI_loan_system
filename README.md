# XAI_loan_system

Explainable AI for loan underwriting — transparent credit decisions with feature attributions, counterfactuals, and regulatory compliance.

## Problem

Traditional ML credit models are black boxes. Regulators (RBI, CFPB) and customers demand explanations:
- Why was I denied?
- What would change the outcome?
- Is the model biased?

## Solution

XAI layer on top of any credit model (XGBoost, LightGBM, Neural Net):

```
┌─────────────────────────────────────────────────────────────┐
│                    XAI LOAN SYSTEM                           │
├─────────────────────────────────────────────────────────────┤
│  INPUT: Applicant features + Model prediction               │
├─────────────────────────────────────────────────────────────┤
│  EXPLANATION ENGINE                                          │
│  ├── SHAP (global + local feature importance)               │
│  ├── Counterfactuals (minimal changes for approval)         │
│  ├── LIME (local interpretable explanations)                │
│  ├── Feature Interaction (SHAP interaction values)          │
│  └── Bias Audit (demographic parity, equalized odds)        │
├─────────────────────────────────────────────────────────────┤
│  OUTPUT: Human-readable report + API for frontend           │
└─────────────────────────────────────────────────────────────┘
```

## Features

| Capability | Description |
|------------|-------------|
| **Global SHAP** | Feature importance across portfolio |
| **Local SHAP** | Per-applicant waterfall plot |
| **Counterfactuals** | "Increase income by ₹15k → Approved" |
| **Adverse Action Codes** | Regulatory-compliant denial reasons |
| **Bias Dashboard** | Disparate impact by gender, age, region |
| **Model Cards** | Standardized documentation (Google format) |

## Quick Start

```bash
pip install -r requirements.txt  # shap, lime, alibi, fairlearn, xgboost

# Explain a single prediction
python explain.py --model model.xgb --applicant applicant.json

# Batch explain portfolio
python batch_explain.py --model model.xgb --data portfolio.csv --output explanations/

# Bias audit
python bias_audit.py --model model.xgb --data portfolio.csv --sensitive gender,age
```

## API

```python
from xai_loan import LoanExplainer

explainer = LoanExplainer(model_path="model.xgb")

# Single applicant
result = explainer.explain(applicant_data)
print(result.shap_waterfall)
print(result.counterfactuals)
print(result.adverse_action_codes)

# Portfolio bias report
bias_report = explainer.audit_bias(portfolio_df, sensitive_attrs=["gender", "age"])
bias_report.save_html("bias_report.html")
```

## Compliance

- **RBI Circular** — Digital lending transparency
- **ECOA/Reg B** — Adverse action notice requirements
- **GDPR Art. 22** — Right to explanation
- **SR 11-7** — Model risk management

## Model Support

| Framework | SHAP Explainer | Notes |
|-----------|----------------|-------|
| XGBoost/LightGBM | TreeSHAP | Fast, exact |
| CatBoost | TreeSHAP | Handles categorical |
| Sklearn (RF, GBT) | TreeSHAP | |
| Neural Net (TF/PyTorch) | DeepSHAP / GradientSHAP | Approximate |
| Any (black-box) | KernelSHAP | Slow, universal |