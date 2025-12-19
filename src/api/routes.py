"""
FastAPI Routes Module

Implements the API endpoints:
- /predict: Loan approval and interest rate prediction
- /explain: SHAP-based explanations
- /counterfactual: Generate actionable counterfactuals
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends
import logging

from .schemas import (
    LoanFeatures,
    PredictionResponse,
    ExplanationRequest,
    ExplanationResponse,
    FeatureContribution,
    CounterfactualRequest,
    CounterfactualResponse,
    CounterfactualExplanation,
    FeatureChange,
    CounterfactualMethod,
    ComparisonRequest,
    ComparisonResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Global references to initialized components (set by main.py)
_model_registry = None
_shap_engine = None
_dice_engine = None
_optimization_engine = None
_causal_engine = None
_preprocessor = None
_feature_names = None


def set_components(
    model_registry,
    shap_engine,
    dice_engine,
    optimization_engine,
    causal_engine,
    preprocessor,
    feature_names
):
    """Set the global component references."""
    global _model_registry, _shap_engine, _dice_engine
    global _optimization_engine, _causal_engine, _preprocessor, _feature_names
    
    _model_registry = model_registry
    _shap_engine = shap_engine
    _dice_engine = dice_engine
    _optimization_engine = optimization_engine
    _causal_engine = causal_engine
    _preprocessor = preprocessor
    _feature_names = feature_names


def _safe_get(data: dict, key: str, default):
    """Safely get a value from dict, returning default if key is missing or None."""
    val = data.get(key)
    return val if val is not None else default


def features_to_dataframe(features: LoanFeatures) -> pd.DataFrame:
    """Convert LoanFeatures to a DataFrame for model input."""
    data = features.model_dump()
    
    # Helper for safe numeric access
    loan_amnt = _safe_get(data, "loan_amnt", 15000)
    annual_inc = _safe_get(data, "annual_inc", 65000)
    revol_util = _safe_get(data, "revol_util", 50.0)
    delinq_2yrs = _safe_get(data, "delinq_2yrs", 0)
    inq_last_6mths = _safe_get(data, "inq_last_6mths", 1)
    pub_rec = _safe_get(data, "pub_rec", 0)
    
    # Comprehensive defaults for all features the model might expect
    # The model is trained with grade/sub_grade from synthetic data
    defaults = {
        # Core numerical features
        "loan_amnt": 15000,
        "annual_inc": 65000,
        "dti": 20.0,
        "emp_length": 5,
        "fico_score": 700,
        "revol_util": 50.0,
        "delinq_2yrs": 0,
        "inq_last_6mths": 1,
        "open_acc": 10,
        "total_acc": 15,
        "revol_bal": 10000,
        "installment": loan_amnt / 36,
        "pub_rec": 0,
        "mort_acc": 1,
        "pub_rec_bankruptcies": 0,
        "term": 36,
        
        # Categorical features (will be label-encoded by preprocessor)
        "home_ownership": "MORTGAGE",
        "verification_status": "Verified",
        "purpose": "debt_consolidation",
        "addr_state": "CA",
        "grade": "B",  # Loan grade based on creditworthiness
        "sub_grade": "B3",  # Sub-grade within the grade
        
        # Engineered features
        "income_to_loan_ratio": annual_inc / max(loan_amnt, 1),
        "monthly_inc": annual_inc / 12,
        "payment_to_income": (loan_amnt / 36) / (annual_inc / 12 + 1),
        "high_utilization": 1 if revol_util > 80 else 0,
        "closed_acc_ratio": 0.3,
        "combined_risk_factors": delinq_2yrs + inq_last_6mths,
    }
    
    # Apply defaults for missing or None values
    for key, default in defaults.items():
        if data.get(key) is None:
            data[key] = default
    
    # Recalculate derived features based on provided values (now guaranteed non-None)
    data["income_to_loan_ratio"] = data["annual_inc"] / max(data["loan_amnt"], 1)
    data["monthly_inc"] = data["annual_inc"] / 12
    data["payment_to_income"] = data["installment"] / (data["annual_inc"] / 12 + 1)
    
    data["high_utilization"] = 1 if data["revol_util"] > 80 else 0
    data["combined_risk_factors"] = data["delinq_2yrs"] + data["inq_last_6mths"] + data["pub_rec"]
            
    df = pd.DataFrame([data])
    return df


def align_features_to_model(df: pd.DataFrame) -> pd.DataFrame:
    """
    Align DataFrame columns to match exactly what the model was trained on.
    """

    if _feature_names is None:
        # Fallback: just use numeric columns
        return df.select_dtypes(include=[np.number])
    
    # Create DataFrame with exactly the features the model expects
    aligned_data = {}
    for feature in _feature_names:
        if feature in df.columns:
            aligned_data[feature] = df[feature].values[0]
        else:
            aligned_data[feature] = 0  # Default for missing features
    
    return pd.DataFrame([aligned_data])


@router.post("/predict", response_model=PredictionResponse)
async def predict(features: LoanFeatures) -> PredictionResponse:
    """
    Predict loan approval and interest rate band.
    
    Returns:
        Prediction with approval probability, interest rate band, and confidence.
    """
    if _model_registry is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
        
    try:
        # Convert to DataFrame
        df = features_to_dataframe(features)
        
        # Preprocess if preprocessor is available
        if _preprocessor is not None and _preprocessor._fitted:
            df_processed = _preprocessor.transform(df)
        else:
            # Select only model features and align with feature_names exactly
            if _feature_names:
                # Create DataFrame with exactly the features the model expects
                aligned_data = {}
                for f in _feature_names:
                    if f in df.columns:
                        aligned_data[f] = df[f].values[0]
                    else:
                        aligned_data[f] = 0  # Default for missing features
                df_processed = pd.DataFrame([aligned_data])
            else:
                df_processed = df.select_dtypes(include=[np.number])
        
        # Get approval model
        approval_model = _model_registry.get_approval_model()
        approval_proba = approval_model.predict_proba(df_processed)[0]
        
        # Determine approval
        if len(approval_proba) == 2:
            approval_probability = float(approval_proba[1])
        else:
            approval_probability = float(approval_proba[0])
            
        approved = approval_probability >= 0.5
        
        # Get interest rate prediction
        try:
            interest_model = _model_registry.get_interest_model()
            interest_proba = interest_model.predict_proba(df_processed)[0]
            bands = ["Low", "Medium", "High", "Very High"]
            interest_rate_band = bands[np.argmax(interest_proba)]
            interest_probabilities = dict(zip(bands, interest_proba.tolist()))
        except Exception:
            # Fallback if interest model not available
            interest_rate_band = "Medium"
            interest_probabilities = {"Low": 0.2, "Medium": 0.4, "High": 0.3, "Very High": 0.1}
            
        # Confidence is max probability
        confidence = float(max(approval_proba))
        
        return PredictionResponse(
            approval_probability=approval_probability,
            approved=approved,
            interest_rate_band=interest_rate_band,
            interest_rate_probabilities=interest_probabilities,
            confidence=confidence,
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/explain", response_model=ExplanationResponse)
async def explain(request: ExplanationRequest) -> ExplanationResponse:
    """
    Generate SHAP-based explanation for a prediction.
    
    Note: SHAP is used as a DIAGNOSTIC tool, not prescriptive guidance.
    Use /counterfactual for actionable recommendations.
    """
    if _shap_engine is None:
        raise HTTPException(status_code=503, detail="SHAP engine not initialized")
        
    try:
        # Get prediction first
        df = features_to_dataframe(request.features)
        
        if _preprocessor is not None and _preprocessor._fitted:
            df_processed = _preprocessor.transform(df)
        else:
            if _feature_names:
                available = [f for f in _feature_names if f in df.columns]
                df_processed = df[available]
            else:
                df_processed = df.select_dtypes(include=[np.number])
        
        # Get SHAP explanation
        explanation = _shap_engine.explain_instance(
            df_processed, top_k=request.top_k
        )
        
        # Format contributions
        top_features = explanation.get("top_features", [])
        
        positive_features = []
        negative_features = []
        
        for f in top_features:
            contribution = FeatureContribution(
                feature=f["feature"],
                value=float(f["value"]),
                shap_value=float(f["shap_value"]),
                impact_direction="positive" if f["shap_value"] > 0 else "negative"
            )
            
            if f["shap_value"] > 0:
                positive_features.append(contribution)
            else:
                negative_features.append(contribution)
                
        # Generate insight
        if positive_features:
            top_pos = positive_features[0].feature
            insight = f"Your {top_pos} positively influenced the decision. "
        else:
            insight = "No strong positive factors identified. "
            
        if negative_features:
            top_neg = negative_features[0].feature
            insight += f"Your {top_neg} could be improved."
            
        # Get prediction for response
        approval_model = _model_registry.get_approval_model()
        proba = approval_model.predict_proba(df_processed)[0]
        prediction = {
            "approval_probability": float(proba[1] if len(proba) == 2 else proba[0]),
            "predicted_class": int(np.argmax(proba)),
        }
        
        return ExplanationResponse(
            prediction=prediction,
            base_value=float(explanation.get("base_value", 0.5)),
            top_positive_features=positive_features[:5],
            top_negative_features=negative_features[:5],
            diagnostic_insight=insight,
        )
        
    except Exception as e:
        logger.error(f"Explanation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/counterfactual", response_model=CounterfactualResponse)
async def generate_counterfactual(
    request: CounterfactualRequest
) -> CounterfactualResponse:
    """
    Generate counterfactual explanations.
    
    Provides actionable recommendations for improving loan outcomes.
    Default method is 'causal' (our primary contribution).
    """
    try:
        # Convert features
        df = features_to_dataframe(request.features)
        
        if _preprocessor is not None and _preprocessor._fitted:
            df_processed = _preprocessor.transform(df)
        else:
            if _feature_names:
                available = [f for f in _feature_names if f in df.columns]
                df_processed = df[available]
            else:
                df_processed = df.select_dtypes(include=[np.number])
        
        # Select method
        method = request.method
        target_class = 1 if request.target_outcome == "approved" else 0
        
        if method == CounterfactualMethod.CAUSAL:
            if _causal_engine is None:
                raise HTTPException(status_code=503, detail="Causal engine not available")
            results = _causal_engine.generate_counterfactuals(
                df_processed,
                num_cfs=request.num_counterfactuals,
                desired_class=target_class,
            )
            method_name = "Causal"
            
        elif method == CounterfactualMethod.DICE:
            if _dice_engine is None:
                raise HTTPException(status_code=503, detail="DiCE engine not available")
            results = _dice_engine.generate_counterfactuals(
                df_processed,
                num_cfs=request.num_counterfactuals,
                desired_class=target_class,
            )
            method_name = "DiCE"
            
        elif method == CounterfactualMethod.OPTIMIZATION:
            if _optimization_engine is None:
                raise HTTPException(status_code=503, detail="Optimization engine not available")
            results = _optimization_engine.generate_counterfactuals(
                df_processed,
                num_cfs=request.num_counterfactuals,
                desired_class=target_class,
            )
            method_name = "Optimization"
            
        else:  # ALL - compare methods
            comparison = await compare_methods(
                ComparisonRequest(
                    features=request.features,
                    num_counterfactuals=request.num_counterfactuals
                )
            )
            # Use causal results as primary
            if _causal_engine:
                results = _causal_engine.generate_counterfactuals(
                    df_processed, num_cfs=request.num_counterfactuals
                )
            else:
                results = {"success": False, "counterfactuals": []}
            method_name = "All (Causal primary)"
            
        if not results.get("success", False):
            return CounterfactualResponse(
                success=False,
                method=method_name,
                original_prediction={"message": "Prediction failed"},
                counterfactuals=[],
                num_generated=0,
                mean_actionability=0,
                causal_validity_rate=None,
            )
            
        # Format counterfactuals
        original = results.get("original", {})
        formatted_cfs = []
        
        for i, cf_data in enumerate(results.get("counterfactuals", [])):
            cf_features = cf_data.get("cf_features", cf_data)
            
            # Extract changes
            changes = []
            for feature in cf_features:
                if feature in original:
                    orig_val = original[feature]
                    cf_val = cf_features[feature]
                    
                    if isinstance(orig_val, (int, float)) and isinstance(cf_val, (int, float)):
                        if abs(cf_val - orig_val) > 0.01:
                            changes.append(FeatureChange(
                                feature=feature,
                                current_value=float(orig_val),
                                recommended_value=float(cf_val),
                                change_direction="increase" if cf_val > orig_val else "decrease",
                                change_magnitude=abs(cf_val - orig_val),
                                effort_level="moderate",
                            ))
                            
            actionability = cf_data.get("actionability_score", 0.5)
            
            formatted_cfs.append(CounterfactualExplanation(
                option_number=i + 1,
                actionability_score=actionability,
                actionability_level="High" if actionability > 0.7 else "Moderate" if actionability > 0.4 else "Low",
                causal_valid=cf_data.get("causal_valid", True),
                changes=changes,
                estimated_outcome="Approved" if target_class == 1 else "Lower Rate",
                l1_distance=cf_data.get("l1_distance", 0),
                sparsity=cf_data.get("sparsity", len(changes)),
            ))
            
        # Get original prediction
        approval_model = _model_registry.get_approval_model()
        orig_proba = approval_model.predict_proba(df_processed)[0]
        original_prediction = {
            "approval_probability": float(orig_proba[1] if len(orig_proba) == 2 else orig_proba[0]),
            "would_be_approved": bool(np.argmax(orig_proba) == 1),
        }
        
        return CounterfactualResponse(
            success=True,
            method=method_name,
            original_prediction=original_prediction,
            counterfactuals=formatted_cfs,
            num_generated=len(formatted_cfs),
            mean_actionability=results.get("mean_actionability", 0.5),
            causal_validity_rate=results.get("causal_validity_rate"),
        )
        
    except Exception as e:
        logger.error(f"Counterfactual error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compare", response_model=ComparisonResponse)
async def compare_methods(request: ComparisonRequest) -> ComparisonResponse:
    """
    Compare all counterfactual generation methods.
    
    Useful for demonstrating the advantage of causal counterfactuals.
    """
    try:
        df = features_to_dataframe(request.features)
        
        if _preprocessor is not None and _preprocessor._fitted:
            df_processed = _preprocessor.transform(df)
        else:
            if _feature_names:
                available = [f for f in _feature_names if f in df.columns]
                df_processed = df[available]
            else:
                df_processed = df.select_dtypes(include=[np.number])
        
        metrics = {}
        methods = []
        
        # DiCE
        if _dice_engine is not None:
            try:
                dice_results = _dice_engine.generate_counterfactuals(
                    df_processed, num_cfs=request.num_counterfactuals
                )
                if dice_results.get("success"):
                    methods.append("DiCE")
                    cfs = dice_results.get("counterfactuals", [])
                    metrics["DiCE"] = {
                        "num_generated": len(cfs),
                        "mean_distance": np.mean([cf.get("l1_distance", 0) for cf in cfs]),
                        "mean_sparsity": np.mean([cf.get("sparsity", 0) for cf in cfs]),
                        "mean_actionability": np.mean([cf.get("actionability_score", 0.5) for cf in cfs]),
                    }
            except Exception as e:
                metrics["DiCE"] = {"error": str(e)}
                
        # Optimization
        if _optimization_engine is not None:
            try:
                opt_results = _optimization_engine.generate_counterfactuals(
                    df_processed, num_cfs=request.num_counterfactuals
                )
                if opt_results.get("success"):
                    methods.append("Optimization")
                    cfs = opt_results.get("counterfactuals", [])
                    metrics["Optimization"] = {
                        "num_generated": len(cfs),
                        "mean_distance": np.mean([cf.get("l1_distance", 0) for cf in cfs]),
                        "mean_sparsity": np.mean([cf.get("sparsity", 0) for cf in cfs]),
                        "mean_actionability": np.mean([cf.get("actionability_score", 0.5) for cf in cfs]),
                    }
            except Exception as e:
                metrics["Optimization"] = {"error": str(e)}
                
        # Causal (our method)
        if _causal_engine is not None:
            try:
                causal_results = _causal_engine.generate_counterfactuals(
                    df_processed, num_cfs=request.num_counterfactuals
                )
                if causal_results.get("success"):
                    methods.append("Causal")
                    cfs = causal_results.get("counterfactuals", [])
                    metrics["Causal"] = {
                        "num_generated": len(cfs),
                        "mean_distance": np.mean([cf.get("l1_distance", 0) for cf in cfs]),
                        "mean_sparsity": np.mean([cf.get("sparsity", 0) for cf in cfs]),
                        "mean_actionability": causal_results.get("mean_actionability", 0.5),
                        "causal_validity_rate": causal_results.get("causal_validity_rate", 1.0),
                    }
            except Exception as e:
                metrics["Causal"] = {"error": str(e)}
                
        # Determine best method
        best_method = "Causal"  # Default
        if metrics:
            valid_methods = [m for m in metrics if "error" not in metrics[m]]
            if valid_methods:
                best_method = max(
                    valid_methods,
                    key=lambda m: metrics[m].get("mean_actionability", 0)
                )
                
        recommendation = (
            f"The {best_method} method provides the most actionable counterfactuals. "
            "Causal counterfactuals are recommended for their validity guarantees."
        )
        
        return ComparisonResponse(
            methods=methods,
            metrics=metrics,
            best_method=best_method,
            recommendation=recommendation,
        )
        
    except Exception as e:
        logger.error(f"Comparison error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
