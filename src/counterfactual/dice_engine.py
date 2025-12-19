"""
DiCE Counterfactual Engine (Baseline 1)

Implements Diverse Counterfactual Explanations using the DiCE library.
This is a BASELINE method for comparison with our causal approach.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List, Union
import logging
import warnings

# DiCE imports
try:
    import dice_ml
    from dice_ml import Dice
    DICE_AVAILABLE = True
except ImportError:
    DICE_AVAILABLE = False
    warnings.warn("dice-ml not installed. DiCE functionality will be limited.")

from ..config import DICE_CONFIG, FEATURE_CONSTRAINTS

logger = logging.getLogger(__name__)


class DiCEEngine:
    """
    DiCE-based counterfactual generator (Baseline 1).
    
    Uses the DiCE library to generate diverse counterfactual explanations.
    This is correlation-based and does NOT consider causal relationships.
    
    Used for comparison with our causal counterfactual approach.
    """
    
    def __init__(
        self,
        model: Any,
        data: pd.DataFrame,
        outcome_name: str = "loan_status",
        continuous_features: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None
    ):
        """
        Initialize the DiCE engine.
        
        Args:
            model: Trained model for predictions
            data: Training data for DiCE
            outcome_name: Name of the outcome column
            continuous_features: List of continuous feature names
            categorical_features: List of categorical feature names
        """
        self.model = model
        self.data = data
        self.outcome_name = outcome_name
        self.dice_explainer: Optional[Any] = None
        self._initialized = False
        
        # Determine feature types
        if continuous_features is None:
            self.continuous_features = data.select_dtypes(
                include=[np.number]
            ).columns.tolist()
            if outcome_name in self.continuous_features:
                self.continuous_features.remove(outcome_name)
        else:
            self.continuous_features = continuous_features
            
        if categorical_features is None:
            self.categorical_features = data.select_dtypes(
                include=['object', 'category']
            ).columns.tolist()
            if outcome_name in self.categorical_features:
                self.categorical_features.remove(outcome_name)
        else:
            self.categorical_features = categorical_features
            
    def initialize(self):
        """Initialize the DiCE explainer."""
        if not DICE_AVAILABLE:
            raise ImportError("dice-ml is not installed. Run: pip install dice-ml")
            
        logger.info("Initializing DiCE explainer...")
        
        # Create DiCE data object
        feature_list = self.continuous_features + self.categorical_features
        
        # Prepare data with outcome
        dice_data = self.data.copy()
        if self.outcome_name not in dice_data.columns:
            # Add dummy outcome for DiCE initialization
            dice_data[self.outcome_name] = 1
            
        # Create DiCE data interface
        self.dice_data = dice_ml.Data(
            dataframe=dice_data,
            continuous_features=self.continuous_features,
            outcome_name=self.outcome_name
        )
        
        # Create model interface
        if hasattr(self.model, 'model'):
            # Our wrapper model
            model_obj = self.model.model
        else:
            model_obj = self.model
            
        self.dice_model = dice_ml.Model(
            model=model_obj,
            backend="sklearn"
        )
        
        # Create DiCE explainer
        self.dice_explainer = Dice(
            self.dice_data,
            self.dice_model,
            method=DICE_CONFIG.get("method", "random")
        )
        
        self._initialized = True
        logger.info("DiCE explainer initialized successfully")
        
    def generate_counterfactuals(
        self,
        instance: pd.DataFrame,
        num_cfs: int = 5,
        desired_class: Union[int, str] = 1,
        features_to_vary: Optional[List[str]] = None,
        permitted_range: Optional[Dict[str, List]] = None
    ) -> Dict[str, Any]:
        """
        Generate counterfactual explanations for an instance.
        
        Args:
            instance: Single row DataFrame to explain
            num_cfs: Number of counterfactuals to generate
            desired_class: Target class for counterfactuals
            features_to_vary: List of features that can be modified
            permitted_range: Allowed ranges for features
            
        Returns:
            Dictionary with counterfactuals and metadata
        """
        if not self._initialized:
            self.initialize()
            
        # Apply immutability constraints
        if features_to_vary is None:
            features_to_vary = [
                f for f in self.continuous_features + self.categorical_features
                if f not in FEATURE_CONSTRAINTS.immutable
            ]
            
        # Apply bound constraints
        if permitted_range is None:
            permitted_range = {}
            for feature, bounds in FEATURE_CONSTRAINTS.bounds.items():
                if feature in features_to_vary:
                    permitted_range[feature] = list(bounds)
                    
        try:
            # Generate counterfactuals
            dice_exp = self.dice_explainer.generate_counterfactuals(
                query_instances=instance,
                total_CFs=num_cfs,
                desired_class=desired_class,
                features_to_vary=features_to_vary,
                permitted_range=permitted_range if permitted_range else None
            )
            
            # Extract counterfactuals
            cf_df = dice_exp.cf_examples_list[0].final_cfs_df
            
            if cf_df is None or len(cf_df) == 0:
                return {
                    "success": False,
                    "message": "No counterfactuals found",
                    "counterfactuals": [],
                    "original": instance.to_dict("records")[0],
                }
                
            # Compute metrics
            counterfactuals = []
            for idx, row in cf_df.iterrows():
                cf = row.to_dict()
                # Remove outcome column from CF
                cf.pop(self.outcome_name, None)
                
                # Calculate distance metrics
                cf_features = {k: v for k, v in cf.items() 
                              if k in instance.columns}
                distance = self._compute_distance(instance, cf_features)
                sparsity = self._compute_sparsity(instance, cf_features)
                
                counterfactuals.append({
                    "cf_features": cf_features,
                    "l1_distance": distance["l1"],
                    "l2_distance": distance["l2"],
                    "sparsity": sparsity,
                    "n_changes": sparsity,
                })
                
            return {
                "success": True,
                "method": "DiCE",
                "counterfactuals": counterfactuals,
                "original": instance.to_dict("records")[0],
                "num_generated": len(counterfactuals),
                "diversity": self._compute_diversity(counterfactuals),
            }
            
        except Exception as e:
            logger.error(f"DiCE generation failed: {e}")
            return {
                "success": False,
                "message": str(e),
                "counterfactuals": [],
                "original": instance.to_dict("records")[0],
            }
            
    def _compute_distance(
        self,
        original: pd.DataFrame,
        cf: Dict[str, Any]
    ) -> Dict[str, float]:
        """Compute distance metrics between original and counterfactual."""
        distances = {"l1": 0.0, "l2": 0.0}
        
        for feature in self.continuous_features:
            if feature in cf and feature in original.columns:
                orig_val = original[feature].iloc[0]
                cf_val = cf[feature]
                diff = abs(cf_val - orig_val)
                distances["l1"] += diff
                distances["l2"] += diff ** 2
                
        distances["l2"] = np.sqrt(distances["l2"])
        return distances
    
    def _compute_sparsity(
        self,
        original: pd.DataFrame,
        cf: Dict[str, Any]
    ) -> int:
        """Count number of features that changed."""
        n_changes = 0
        
        for feature in cf:
            if feature in original.columns:
                orig_val = original[feature].iloc[0]
                cf_val = cf[feature]
                
                # Check if changed (with tolerance for floats)
                if isinstance(orig_val, (int, float)) and isinstance(cf_val, (int, float)):
                    if abs(cf_val - orig_val) > 0.01:
                        n_changes += 1
                elif orig_val != cf_val:
                    n_changes += 1
                    
        return n_changes
    
    def _compute_diversity(
        self,
        counterfactuals: List[Dict[str, Any]]
    ) -> float:
        """Compute diversity score across counterfactuals."""
        if len(counterfactuals) < 2:
            return 0.0
            
        # Compute pairwise distances
        distances = []
        for i in range(len(counterfactuals)):
            for j in range(i + 1, len(counterfactuals)):
                cf1 = counterfactuals[i]["cf_features"]
                cf2 = counterfactuals[j]["cf_features"]
                
                dist = 0
                for feature in cf1:
                    if feature in cf2:
                        v1 = cf1[feature]
                        v2 = cf2[feature]
                        if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                            dist += abs(v2 - v1)
                        elif v1 != v2:
                            dist += 1
                            
                distances.append(dist)
                
        return np.mean(distances) if distances else 0.0
