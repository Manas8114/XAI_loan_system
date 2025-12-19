# XAI Loan System Frontend

## Overview

This is the research demonstration interface for the XAI-driven Causal Counterfactual Loan Risk Assessment System.

**Important Note:** This frontend is designed for **academic evaluation and demonstration** only. It does NOT:
- Store user data
- Require authentication
- Make real-world loan decisions
- Provide financial advice

## Technology Stack

- **Framework**: Streamlit
- **Backend**: FastAPI REST API
- **Communication**: JSON over HTTP

## Running the Frontend

### Prerequisites

1. Install Streamlit:
```bash
pip install streamlit requests
```

2. Start the backend API (in the project root):
```bash
python -m src.api.main
```

3. Run the Streamlit app:
```bash
cd frontend
streamlit run app.py
```

The interface will open at `http://localhost:8501`

## UI Structure

### 1. Header Section
- Project title with academic positioning
- Clear disclaimer about research purpose

### 2. Applicant Input Panel (Left)
- Hypothetical applicant profile fields
- Numeric validation with reasonable bounds
- No personal identifiers collected

### 3. Prediction Output (Right)
- Approval probability
- Risk category (Low/Medium/High)
- Interest rate band

### 4. Counterfactual Recommendations (Core Feature)
- Top 3 actionable recommendations
- Actionability scores with progress bars
- Method comparison toggle:
  - **Causal (Primary)** - Our contribution
  - DiCE (Baseline)
  - Optimization (Baseline)

### 5. SHAP Explanation (Collapsed)
- Diagnostic feature attribution
- Clearly labeled as explanatory, not prescriptive

### 6. Footer
- Ethics disclaimer
- Dataset acknowledgment
- Academic positioning

## Design Principles

- Clean, neutral color palette
- Professional typography
- No gamification
- Accessibility-friendly
- Minimal animations

## Language Guidelines

The interface uses:
- "Risk estimation" (not "loan decision")
- "Recommendation" (not "approval granted")
- "Decision support" (not "guaranteed outcome")

## Research Intent

This interface is designed to:
1. Demonstrate causal counterfactual reasoning visually
2. Be presented to university review panels
3. Enable evaluation of XAI research contributions
