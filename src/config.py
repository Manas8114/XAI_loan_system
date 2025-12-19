"""
XAI Loan System Configuration

Central configuration for all system components including paths, model parameters,
feature constraints, and causal graph structure.
"""

import os
from pathlib import Path
from typing import Dict, List, Any, Literal
from dataclasses import dataclass, field

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"
MODELS_DIR = BASE_DIR / "models"

# Create directories if they don't exist
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, CACHE_DIR, MODELS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# =============================================================================
# DATA CONFIGURATION
# =============================================================================

# LendingClub dataset URL (Kaggle)
DATASET_URL = "https://www.kaggle.com/datasets/wordsforthewise/lending-club"

# Sample size for development (use None for full dataset)
SAMPLE_SIZE = 100000

# Random seed for reproducibility
RANDOM_SEED = 42

# Train-test split ratio
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1

# =============================================================================
# FEATURE CONFIGURATION
# =============================================================================

# Target columns
TARGET_APPROVAL = "loan_status"  # Binary: 0=Rejected, 1=Approved
TARGET_INTEREST_RATE = "int_rate_band"  # Multi-class: Low, Medium, High, Very High

# Interest rate bands
INTEREST_RATE_BANDS = {
    "Low": (0, 8),
    "Medium": (8, 12),
    "High": (12, 18),
    "Very High": (18, 100)
}

# Feature lists
NUMERICAL_FEATURES = [
    "annual_inc",
    "dti",
    "emp_length",
    "loan_amnt",
    "installment",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "delinq_2yrs",
    "inq_last_6mths",
    "mort_acc",
    "pub_rec_bankruptcies",
    "fico_score",
]

CATEGORICAL_FEATURES = [
    "term",
    "grade",
    "sub_grade",
    "home_ownership",
    "verification_status",
    "purpose",
    "addr_state",
]

# Protected attributes (for fairness analysis)
PROTECTED_ATTRIBUTES = ["addr_state"]  # Can add more if available

# =============================================================================
# FEATURE CONSTRAINTS (for counterfactual generation)
# =============================================================================

@dataclass
class FeatureConstraints:
    """Constraints for counterfactual feature modifications."""
    
    # Immutable features - cannot be changed in counterfactuals
    immutable: List[str] = field(default_factory=lambda: [
        "term",  # Loan term is fixed
        "addr_state",  # Location is fixed
        "pub_rec_bankruptcies",  # Historical record
        "pub_rec",  # Historical record
    ])
    
    # Feature bounds (min, max)
    bounds: Dict[str, tuple] = field(default_factory=lambda: {
        "annual_inc": (0, 500000),
        "dti": (0, 50),
        "emp_length": (0, 40),
        "loan_amnt": (1000, 40000),
        "installment": (0, 2000),
        "open_acc": (0, 50),
        "revol_bal": (0, 500000),
        "revol_util": (0, 100),
        "total_acc": (0, 100),
        "delinq_2yrs": (0, 20),
        "inq_last_6mths": (0, 10),
        "mort_acc": (0, 20),
        "fico_score": (300, 850),
    })
    
    # Direction constraints: "increase_only", "decrease_only", or "both"
    direction: Dict[str, str] = field(default_factory=lambda: {
        "annual_inc": "increase_only",
        "emp_length": "increase_only",
        "fico_score": "increase_only",
        "dti": "decrease_only",
        "revol_util": "decrease_only",
        "delinq_2yrs": "decrease_only",
        "inq_last_6mths": "decrease_only",
    })


FEATURE_CONSTRAINTS = FeatureConstraints()

# =============================================================================
# ACTIONABILITY CONFIGURATION
# =============================================================================

# Feature weights for actionability score
ACTIONABILITY_WEIGHTS = {
    "annual_inc": 0.15,
    "emp_length": 0.10,
    "dti": 0.20,
    "fico_score": 0.25,
    "delinq_2yrs": 0.10,
    "open_acc": 0.05,
    "revol_util": 0.10,
    "loan_amnt": 0.05,
}

# Effort costs for feature changes (normalized 0-1)
# Higher = more difficult to achieve
EFFORT_COSTS = {
    "annual_inc": {
        "10%": 0.4,
        "25%": 0.6,
        "50%": 0.8,
        "100%": 0.95,
    },
    "emp_length": {
        "+1yr": 0.9,  # Requires actual time
        "+2yr": 0.95,
    },
    "dti": {
        "-5%": 0.3,
        "-10%": 0.5,
        "-20%": 0.7,
    },
    "fico_score": {
        "+25": 0.3,
        "+50": 0.5,
        "+100": 0.7,
    },
    "revol_util": {
        "-10%": 0.2,
        "-25%": 0.4,
        "-50%": 0.6,
    },
}

# Time factors (decay based on time to achieve - months)
TIME_FACTOR_DECAY = 0.1  # Decay rate per month

# =============================================================================
# CAUSAL GRAPH CONFIGURATION
# =============================================================================

# Causal edges: (parent, child)
# Based on domain knowledge and credit risk literature
CAUSAL_EDGES = [
    ("emp_length", "annual_inc"),        # Tenure increases income
    ("annual_inc", "dti"),                # Income affects DTI
    ("annual_inc", "fico_score"),         # Income enables better credit
    ("annual_inc", "revol_bal"),          # Income affects available credit
    ("emp_length", "fico_score"),         # Stability improves credit
    ("dti", "loan_status"),               # DTI affects approval
    ("fico_score", "loan_status"),        # Credit score affects approval
    ("fico_score", "int_rate"),           # Credit score affects rate
    ("delinq_2yrs", "fico_score"),        # Delinquencies hurt credit
    ("delinq_2yrs", "loan_status"),       # Delinquencies affect approval
    ("revol_util", "fico_score"),         # Utilization affects credit
    ("loan_amnt", "dti"),                 # Loan amount affects DTI
    ("loan_amnt", "installment"),         # Loan amount determines payment
    ("installment", "dti"),               # Payment affects DTI
]

# Structural equation coefficients (estimated from data)
# Format: child -> {parent: coefficient}
# These are placeholders - actual values estimated during training
STRUCTURAL_EQUATIONS = {
    "annual_inc": {
        "emp_length": 2500,  # $2500 per year of employment
        "intercept": 45000,
    },
    "dti": {
        "annual_inc": -0.00002,  # Higher income reduces DTI
        "loan_amnt": 0.0003,
        "intercept": 15,
    },
    "fico_score": {
        "annual_inc": 0.0005,
        "emp_length": 2,
        "delinq_2yrs": -15,
        "revol_util": -0.5,
        "intercept": 680,
    },
}

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

# XGBoost parameters for loan approval
XGBOOST_APPROVAL_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "max_depth": 6,
    "learning_rate": 0.1,
    "n_estimators": 200,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}

# XGBoost parameters for interest rate prediction
XGBOOST_INTEREST_PARAMS = {
    "objective": "multi:softprob",
    "eval_metric": "mlogloss",
    "max_depth": 5,
    "learning_rate": 0.1,
    "n_estimators": 150,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "num_class": 4,  # Low, Medium, High, Very High
}

# Random Forest parameters (baseline)
RF_PARAMS = {
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_split": 5,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}

# =============================================================================
# COUNTERFACTUAL CONFIGURATION
# =============================================================================

# DiCE parameters
DICE_CONFIG = {
    "num_counterfactuals": 5,
    "diversity_weight": 1.0,
    "proximity_weight": 0.5,
    "method": "random",  # Can be "random", "genetic", or "kdtree"
}

# Optimization-based CF parameters
OPTIMIZATION_CF_CONFIG = {
    "learning_rate": 0.01,
    "max_iterations": 1000,
    "lambda_distance": 0.1,
    "lambda_validity": 1.0,
    "convergence_threshold": 1e-6,
}

# Causal CF parameters
CAUSAL_CF_CONFIG = {
    "max_interventions": 4,  # Maximum features to modify
    "actionability_threshold": 0.5,
    "propagate_effects": True,
}

# =============================================================================
# API CONFIGURATION
# =============================================================================

API_CONFIG = {
    "title": "XAI Loan Counterfactual API",
    "description": "API for loan prediction and counterfactual explanations",
    "version": "1.0.0",
    "host": "0.0.0.0",
    "port": 8000,
}

# =============================================================================
# SHAP CONFIGURATION
# =============================================================================

SHAP_CONFIG = {
    "background_samples": 100,
    "max_samples_for_plot": 500,
    "cache_enabled": True,
}

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
}
