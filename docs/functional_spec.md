# XAI Loan System - Functional Specification

## 1. System Purpose

The XAI Loan System provides:
1. **Loan Approval Prediction** - Binary classification for loan approval
2. **Interest Rate Prediction** - Multi-class classification for rate bands
3. **Diagnostic Explanations** - SHAP-based feature importance (understanding "why")
4. **Prescriptive Recommendations** - Causal counterfactuals (understanding "what to change")

## 2. Functional Requirements

### FR1: Prediction
| ID | Requirement |
|----|-------------|
| FR1.1 | System shall predict loan approval probability |
| FR1.2 | System shall classify interest rate into bands (Low/Medium/High/Very High) |
| FR1.3 | System shall provide confidence scores for predictions |

### FR2: Explanation
| ID | Requirement |
|----|-------------|
| FR2.1 | System shall compute SHAP values for each prediction |
| FR2.2 | System shall identify top positive and negative contributing features |
| FR2.3 | System shall generate human-readable diagnostic insights |

### FR3: Counterfactual Generation
| ID | Requirement |
|----|-------------|
| FR3.1 | System shall generate 1-10 counterfactual alternatives |
| FR3.2 | System shall support three methods: DiCE, Optimization, Causal |
| FR3.3 | Causal method shall respect causal dependencies |
| FR3.4 | System shall compute actionability scores for each counterfactual |

### FR4: Constraint Enforcement
| ID | Requirement |
|----|-------------|
| FR4.1 | System shall enforce immutability constraints (age, race) |
| FR4.2 | System shall enforce bound constraints (realistic ranges) |
| FR4.3 | System shall enforce direction constraints (increasing only) |

### FR5: Fairness
| ID | Requirement |
|----|-------------|
| FR5.1 | System shall compute disparate impact ratios |
| FR5.2 | System shall audit recommendations across demographic groups |
| FR5.3 | System shall flag potential fairness violations |

## 3. API Specification

### 3.1 Prediction Endpoint

**POST** `/api/v1/predict`

**Request:**
```json
{
  "annual_inc": 65000,
  "dti": 18.5,
  "emp_length": 5,
  "loan_amnt": 15000,
  "fico_score": 680,
  "revol_util": 45
}
```

**Response:**
```json
{
  "approval_probability": 0.72,
  "approved": true,
  "interest_rate_band": "Medium",
  "interest_rate_probabilities": {
    "Low": 0.15,
    "Medium": 0.45,
    "High": 0.30,
    "Very High": 0.10
  },
  "confidence": 0.72
}
```

### 3.2 Explanation Endpoint

**POST** `/api/v1/explain`

**Request:**
```json
{
  "features": { ... },
  "top_k": 5
}
```

**Response:**
```json
{
  "prediction": {...},
  "base_value": 0.45,
  "top_positive_features": [
    {"feature": "fico_score", "value": 720, "shap_value": 0.15}
  ],
  "top_negative_features": [
    {"feature": "dti", "value": 28, "shap_value": -0.08}
  ],
  "diagnostic_insight": "Your credit score positively influenced..."
}
```

### 3.3 Counterfactual Endpoint

**POST** `/api/v1/counterfactual`

**Request:**
```json
{
  "features": { ... },
  "target_outcome": "approved",
  "num_counterfactuals": 3,
  "method": "causal",
  "min_actionability": 0.3
}
```

**Response:**
```json
{
  "success": true,
  "method": "Causal",
  "original_prediction": {"approval_probability": 0.35},
  "counterfactuals": [
    {
      "option_number": 1,
      "actionability_score": 0.75,
      "actionability_level": "High",
      "causal_valid": true,
      "changes": [
        {
          "feature": "dti",
          "current_value": 28,
          "recommended_value": 22,
          "change_direction": "decrease"
        }
      ]
    }
  ],
  "mean_actionability": 0.68,
  "causal_validity_rate": 1.0
}
```

## 4. Actionability Score Specification

### Formula
```
Actionability(cf) = Σᵢ wᵢ · aᵢ(cf)

where aᵢ(cf) = (1 - effort_costᵢ) × feasibilityᵢ × time_factorᵢ × direction_penaltyᵢ
```

### Components
| Component | Description | Range |
|-----------|-------------|-------|
| effort_cost | Difficulty of change | 0-1 (higher = harder) |
| feasibility | Within realistic bounds | 0 or 1 |
| time_factor | Decay for time to achieve | 0-1 |
| direction_penalty | Respects direction constraint | 0 or 1 |

### Levels
| Score Range | Level |
|-------------|-------|
| ≥ 0.70 | Highly Actionable |
| 0.50 - 0.69 | Moderately Actionable |
| 0.30 - 0.49 | Challenging |
| < 0.30 | Difficult |

## 5. Feature Constraints

### Immutable Features
- Age-related
- Race/ethnicity
- Gender
- Protected attributes

### Bound Constraints
| Feature | Min | Max |
|---------|-----|-----|
| annual_inc | 10,000 | 500,000 |
| dti | 0 | 50 |
| fico_score | 300 | 850 |
| revol_util | 0 | 100 |
| emp_length | 0 | 40 |

### Direction Constraints
| Feature | Direction |
|---------|-----------|
| fico_score | increase_only |
| annual_inc | increase_only |
| delinq_2yrs | decrease_only |
| pub_rec | decrease_only |

## 6. Causal Model Specification

### Variables
- Employment length (years)
- Annual income ($)
- Debt-to-income ratio (%)
- FICO credit score
- Revolving utilization (%)
- Loan status (approved/rejected)

### Causal Edges
1. emp_length → annual_inc
2. annual_inc → dti
3. fico_score → revol_util
4. dti → loan_status
5. fico_score → loan_status
6. emp_length → loan_status
7. revol_util → loan_status

### Intervention Rules
When intervening on feature X:
1. Set X to target value
2. Propagate effects to descendants using structural equations
3. Validate resulting counterfactual

## 7. Non-Functional Requirements

| ID | Requirement | Target |
|----|-------------|--------|
| NFR1 | API response time | < 2 seconds |
| NFR2 | CF generation time | < 5 seconds |
| NFR3 | Model accuracy (ROC-AUC) | > 0.75 |
| NFR4 | Causal validity rate | > 90% |
| NFR5 | Mean actionability | > 0.50 |
