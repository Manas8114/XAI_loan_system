"""
SHAP Engine Module

Provides SHAP-based explanations for model predictions.
Uses TreeSHAP for efficient computation on tree-based models.
"""

import numpy as np
import pandas as pd
import shap
from typing import Optional, Dict, Any, List, Union
import logging
import joblib
from pathlib import Path

from ..config import SHAP_CONFIG, CACHE_DIR

logger = logging.getLogger(__name__)


class SHAPEngine:
    """
    SHAP-based explanation engine.
    
    Provides:
    - Global feature importance
    - Local (instance-level) explanations
    - Feature interaction effects
    - Explanation caching for efficiency
    
    Note: SHAP is used as a DIAGNOSTIC tool, not the primary explanation.
    Counterfactuals are the main decision-support output.
    """
    
    def __init__(
        self,
        model: Any,
        X_background: Optional[pd.DataFrame] = None,
        feature_names: Optional[List[str]] = None
    ):
        """
        Initialize the SHAP engine.
        
        Args:
            model: Trained model (XGBoost, RF, or sklearn-compatible)
            X_background: Background data for SHAP computation
            feature_names: List of feature names
        """
        self.model = model
        self.feature_names = feature_names
        self.explainer: Optional[shap.Explainer] = None
        self.background_data: Optional[pd.DataFrame] = None
        self.shap_values_cache: Dict[str, np.ndarray] = {}
        self._initialized = False
        
        if X_background is not None:
            self.initialize(X_background)
            
    def initialize(self, X_background: pd.DataFrame):
        """
        Initialize the SHAP explainer with background data.
        
        Args:
            X_background: Background dataset for SHAP computation
        """
        logger.info("Initializing SHAP explainer...")
        
        # Sample background data if too large
        n_samples = SHAP_CONFIG.get("background_samples", 100)
        if len(X_background) > n_samples:
            self.background_data = X_background.sample(
                n=n_samples, random_state=42
            )
        else:
            self.background_data = X_background
            
        # Store feature names
        if self.feature_names is None:
            self.feature_names = list(X_background.columns)
            
        # Create TreeExplainer for tree-based models
        try:
            # Try TreeExplainer first (fastest for tree models)
            if hasattr(self.model, 'model'):
                # Wrapper model (our XGBoostLoanModel)
                self.explainer = shap.TreeExplainer(self.model.model)
            else:
                self.explainer = shap.TreeExplainer(self.model)
        except Exception as e:
            logger.warning(f"TreeExplainer failed, falling back to Explainer: {e}")
            # Fallback to general explainer
            predict_fn = (
                self.model.predict_proba
                if hasattr(self.model, 'predict_proba')
                else self.model.predict
            )
            self.explainer = shap.Explainer(predict_fn, self.background_data)
            
        self._initialized = True
        logger.info("SHAP explainer initialized successfully")
        
    def compute_shap_values(
        self,
        X: pd.DataFrame,
        use_cache: bool = True
    ) -> np.ndarray:
        """
        Compute SHAP values for given samples.
        
        Args:
            X: Samples to explain
            use_cache: Whether to use cached values if available
            
        Returns:
            SHAP values array (n_samples x n_features)
        """
        if not self._initialized:
            raise ValueError("SHAP engine not initialized. Call initialize() first.")
            
        # Check cache
        cache_key = str(hash(X.values.tobytes()))
        if use_cache and cache_key in self.shap_values_cache:
            return self.shap_values_cache[cache_key]
            
        # Compute SHAP values
        shap_output = self.explainer(X)
        
        # Handle SHAP Explanation objects (from shap.Explainer)
        if hasattr(shap_output, 'values'):
            shap_values = shap_output.values
            # For multi-output (binary classification), extract positive class
            if len(shap_values.shape) == 3:
                # Shape is (n_samples, n_features, n_classes)
                shap_values = shap_values[:, :, 1]  # Positive class
            elif len(shap_values.shape) == 2 and shap_values.shape[1] == 2:
                # Sometimes shape is (n_samples, 2) - wrong, should be (n_samples, n_features)
                # This means we got class probabilities not feature values
                # Recompute using shap_values method if available
                try:
                    shap_values = self.explainer.shap_values(X)
                    if isinstance(shap_values, list) and len(shap_values) == 2:
                        shap_values = shap_values[1]
                except:
                    pass
        else:
            # Handle list output (from TreeExplainer.shap_values)
            shap_values = shap_output
            if isinstance(shap_values, list):
                if len(shap_values) == 2:
                    shap_values = shap_values[1]  # Positive class
                else:
                    shap_values = np.array(shap_values)
                
        # Ensure numpy array
        shap_values = np.asarray(shap_values)
        
        # Cache results
        if SHAP_CONFIG.get("cache_enabled", True):
            self.shap_values_cache[cache_key] = shap_values
            
        return shap_values
    
    def explain_instance(
        self,
        instance: pd.DataFrame,
        top_k: int = 10
    ) -> Dict[str, Any]:
        """
        Generate local explanation for a single instance.
        
        Args:
            instance: Single row DataFrame to explain
            top_k: Number of top features to return
            
        Returns:
            Dictionary with explanation details
        """
        if len(instance) != 1:
            raise ValueError("explain_instance expects a single row DataFrame")
            
        shap_values = self.compute_shap_values(instance)
        
        # Ensure shap_values is 1D
        shap_values = np.asarray(shap_values)
        while len(shap_values.shape) > 1:
            shap_values = shap_values[0]
        shap_values = shap_values.flatten()
            
        # Create feature contribution DataFrame
        # Ensure values are 1-dimensional
        feature_values = instance.iloc[0].values
        if hasattr(feature_values, 'flatten'):
            feature_values = feature_values.flatten()
        # Convert any nested arrays to scalars
        feature_values = [float(v) if isinstance(v, (np.ndarray, list)) and np.asarray(v).size == 1 
                         else (float(v) if isinstance(v, (int, float, np.number)) else 0.0) 
                         for v in feature_values]
        
        # Ensure shap_values matches feature count
        if len(shap_values) != len(self.feature_names):
            logger.warning(f"SHAP values length ({len(shap_values)}) != features ({len(self.feature_names)}). Adjusting...")
            if len(shap_values) > len(self.feature_names):
                shap_values = shap_values[:len(self.feature_names)]
            else:
                shap_values = np.pad(shap_values, (0, len(self.feature_names) - len(shap_values)), constant_values=0)
        
        contributions = pd.DataFrame({
            "feature": self.feature_names,
            "value": feature_values,
            "shap_value": shap_values.tolist(),  # Ensure it's a plain list
            "abs_shap": np.abs(shap_values).tolist()
        })
        
        contributions = contributions.sort_values("abs_shap", ascending=False)
        
        # Get base value (expected value)
        if hasattr(self.explainer, "expected_value"):
            base_value = self.explainer.expected_value
            if isinstance(base_value, (list, np.ndarray)):
                base_value = base_value[1] if len(base_value) == 2 else base_value[0]
        else:
            base_value = 0.5
            
        # Get prediction
        if hasattr(self.model, 'predict_proba'):
            if hasattr(self.model, 'model'):
                prediction = self.model.predict_proba(instance)[0]
            else:
                prediction = self.model.predict_proba(instance)[0]
            prediction = prediction[1] if len(prediction) == 2 else prediction
        else:
            prediction = self.model.predict(instance)[0]
            
        explanation = {
            "base_value": float(base_value),
            "prediction": float(prediction) if isinstance(prediction, (int, float, np.number)) else prediction.tolist(),
            "top_features": contributions.head(top_k).to_dict("records"),
            "all_contributions": contributions.to_dict("records"),
            "sum_shap_values": float(shap_values.sum()),
        }
        
        return explanation
    
    def global_importance(
        self,
        X: pd.DataFrame,
        max_samples: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Compute global feature importance using mean absolute SHAP values.
        
        Args:
            X: Dataset to compute importance over
            max_samples: Max samples to use (for efficiency)
            
        Returns:
            DataFrame with feature importance scores
        """
        max_samples = max_samples or SHAP_CONFIG.get("max_samples_for_plot", 500)
        
        if len(X) > max_samples:
            X = X.sample(n=max_samples, random_state=42)
            
        shap_values = self.compute_shap_values(X, use_cache=False)
        
        # Compute mean absolute SHAP value per feature
        importance = np.abs(shap_values).mean(axis=0)
        
        importance_df = pd.DataFrame({
            "feature": self.feature_names,
            "importance": importance,
            "importance_normalized": importance / importance.sum()
        })
        
        return importance_df.sort_values("importance", ascending=False)
    
    def compute_interactions(
        self,
        X: pd.DataFrame,
        max_samples: int = 100
    ) -> Dict[str, np.ndarray]:
        """
        Compute SHAP interaction values.
        
        This is computationally expensive, so limited samples are used.
        
        Args:
            X: Samples to compute interactions for
            max_samples: Maximum samples to use
            
        Returns:
            Dictionary with interaction matrices
        """
        if len(X) > max_samples:
            X = X.sample(n=max_samples, random_state=42)
            
        try:
            # TreeExplainer supports interaction values
            if isinstance(self.explainer, shap.TreeExplainer):
                interaction_values = self.explainer.shap_interaction_values(X)
                
                if isinstance(interaction_values, list):
                    interaction_values = interaction_values[1]  # Positive class
                    
                # Compute mean interaction strengths
                mean_interactions = np.abs(interaction_values).mean(axis=0)
                
                return {
                    "interaction_matrix": mean_interactions,
                    "feature_names": self.feature_names,
                }
        except Exception as e:
            logger.warning(f"Interaction values computation failed: {e}")
            
        return {"error": "Interaction values not available for this model type"}
    
    def compare_with_counterfactual(
        self,
        original: pd.DataFrame,
        counterfactual: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Compare SHAP explanations between original and counterfactual.
        
        This helps understand the relationship between SHAP importance
        and counterfactual changes.
        
        Args:
            original: Original sample
            counterfactual: Counterfactual sample
            
        Returns:
            Comparison dictionary
        """
        original_shap = self.compute_shap_values(original)[0]
        cf_shap = self.compute_shap_values(counterfactual)[0]
        
        # Feature changes
        changes = (counterfactual.values - original.values)[0]
        
        comparison = pd.DataFrame({
            "feature": self.feature_names,
            "original_value": original.iloc[0].values,
            "cf_value": counterfactual.iloc[0].values,
            "change": changes,
            "original_shap": original_shap,
            "cf_shap": cf_shap,
            "shap_change": cf_shap - original_shap,
        })
        
        # Find features where high SHAP importance aligns with CF changes
        comparison["shap_importance"] = np.abs(comparison["original_shap"])
        comparison["changed"] = comparison["change"].abs() > 0.01
        
        return {
            "comparison_table": comparison.to_dict("records"),
            "aligned_changes": comparison[
                comparison["changed"] & (comparison["shap_importance"] > 0.1)
            ]["feature"].tolist(),
            "correlation": np.corrcoef(
                np.abs(comparison["original_shap"]),
                np.abs(comparison["change"])
            )[0, 1] if len(comparison) > 1 else 0,
        }
    
    def save(self, path: Optional[Path] = None):
        """Save SHAP engine state."""
        path = path or CACHE_DIR / "shap_engine.joblib"
        
        state = {
            "background_data": self.background_data,
            "feature_names": self.feature_names,
            "expected_value": getattr(self.explainer, "expected_value", None),
        }
        
        joblib.dump(state, path)
        logger.info(f"SHAP engine saved to {path}")
        
    def clear_cache(self):
        """Clear the SHAP values cache."""
        self.shap_values_cache.clear()
        logger.info("SHAP cache cleared")
