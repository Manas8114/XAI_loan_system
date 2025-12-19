"""
Pydantic Schemas for API Request/Response Models
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from enum import Enum


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    models_loaded: bool
    

class LoanFeatures(BaseModel):
    """Input features for loan prediction."""
    annual_inc: float = Field(..., description="Annual income", ge=0)
    dti: float = Field(..., description="Debt-to-income ratio", ge=0, le=100)
    emp_length: float = Field(0, description="Employment length in years", ge=0)
    loan_amnt: float = Field(..., description="Loan amount requested", ge=0)
    installment: Optional[float] = Field(None, description="Monthly installment")
    open_acc: Optional[int] = Field(None, description="Number of open accounts", ge=0)
    pub_rec: Optional[int] = Field(0, description="Number of public records", ge=0)
    revol_bal: Optional[float] = Field(None, description="Revolving balance", ge=0)
    revol_util: Optional[float] = Field(None, description="Revolving utilization %", ge=0, le=100)
    total_acc: Optional[int] = Field(None, description="Total accounts", ge=0)
    delinq_2yrs: Optional[int] = Field(0, description="Delinquencies in past 2 years", ge=0)
    inq_last_6mths: Optional[int] = Field(0, description="Inquiries in last 6 months", ge=0)
    mort_acc: Optional[int] = Field(0, description="Mortgage accounts", ge=0)
    pub_rec_bankruptcies: Optional[int] = Field(0, description="Public record bankruptcies", ge=0)
    fico_score: float = Field(..., description="FICO credit score", ge=300, le=850)
    term: Optional[str] = Field(" 36 months", description="Loan term")
    grade: Optional[str] = Field(None, description="Loan grade")
    home_ownership: Optional[str] = Field("RENT", description="Home ownership status")
    verification_status: Optional[str] = Field("Not Verified", description="Verification status")
    purpose: Optional[str] = Field("debt_consolidation", description="Loan purpose")
    addr_state: Optional[str] = Field("CA", description="State")
    
    class Config:
        json_schema_extra = {
            "example": {
                "annual_inc": 65000,
                "dti": 18.5,
                "emp_length": 5,
                "loan_amnt": 15000,
                "fico_score": 680,
                "revol_util": 45,
                "delinq_2yrs": 0,
            }
        }


class PredictionResponse(BaseModel):
    """Response from /predict endpoint."""
    approval_probability: float = Field(..., description="Probability of approval")
    approved: bool = Field(..., description="Whether loan would be approved")
    interest_rate_band: str = Field(..., description="Predicted interest rate band")
    interest_rate_probabilities: Dict[str, float] = Field(
        ..., description="Probabilities for each rate band"
    )
    confidence: float = Field(..., description="Model confidence score")


class ExplanationRequest(BaseModel):
    """Request for /explain endpoint."""
    features: LoanFeatures
    top_k: int = Field(10, description="Number of top features to return", ge=1, le=50)
    include_interactions: bool = Field(False, description="Include feature interactions")


class FeatureContribution(BaseModel):
    """SHAP contribution for a single feature."""
    feature: str
    value: float
    shap_value: float
    impact_direction: str


class ExplanationResponse(BaseModel):
    """Response from /explain endpoint."""
    prediction: Dict[str, Any]
    base_value: float
    top_positive_features: List[FeatureContribution]
    top_negative_features: List[FeatureContribution]
    diagnostic_insight: str


class CounterfactualMethod(str, Enum):
    """Available counterfactual generation methods."""
    DICE = "dice"
    OPTIMIZATION = "optimization"
    CAUSAL = "causal"
    ALL = "all"


class CounterfactualRequest(BaseModel):
    """Request for /counterfactual endpoint."""
    features: LoanFeatures
    target_outcome: str = Field("approved", description="Desired outcome: 'approved' or 'lower_rate'")
    num_counterfactuals: int = Field(5, description="Number of CFs to generate", ge=1, le=10)
    method: CounterfactualMethod = Field(CounterfactualMethod.CAUSAL, description="CF generation method")
    min_actionability: float = Field(0.3, description="Minimum actionability score", ge=0, le=1)


class FeatureChange(BaseModel):
    """A single feature change in a counterfactual."""
    feature: str
    current_value: float
    recommended_value: float
    change_direction: str
    change_magnitude: float
    effort_level: str


class CounterfactualExplanation(BaseModel):
    """A single counterfactual explanation."""
    option_number: int
    actionability_score: float
    actionability_level: str
    causal_valid: bool
    changes: List[FeatureChange]
    estimated_outcome: str
    l1_distance: float
    sparsity: int


class CounterfactualResponse(BaseModel):
    """Response from /counterfactual endpoint."""
    success: bool
    method: str
    original_prediction: Dict[str, Any]
    counterfactuals: List[CounterfactualExplanation]
    num_generated: int
    mean_actionability: float
    causal_validity_rate: Optional[float]
    comparison: Optional[Dict[str, Any]] = None


class ComparisonRequest(BaseModel):
    """Request to compare all CF methods."""
    features: LoanFeatures
    num_counterfactuals: int = Field(5, ge=1, le=10)


class ComparisonResponse(BaseModel):
    """Response comparing all CF methods."""
    methods: List[str]
    metrics: Dict[str, Dict[str, Any]]
    best_method: str
    recommendation: str
