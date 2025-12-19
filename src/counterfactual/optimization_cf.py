"""
Optimization-based Counterfactual Engine (Baseline 2)

Generates counterfactuals using gradient-based optimization.
This is a BASELINE method for comparison with our causal approach.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List, Callable
from scipy.optimize import minimize
import logging

from ..config import OPTIMIZATION_CF_CONFIG, FEATURE_CONSTRAINTS

logger = logging.getLogger(__name__)


class OptimizationCFEngine:
    """
    Optimization-based counterfactual generator (Baseline 2).
    
    Uses gradient-free optimization to find minimal changes that flip
    the prediction. This method is correlation-based and does NOT
    consider causal relationships.
    
    Used for comparison with our causal counterfactual approach.
    """
    
    def __init__(
        self,
        model: Any,
        feature_names: List[str],
        feature_ranges: Optional[Dict[str, tuple]] = None
    ):
        """
        Initialize the optimization-based CF engine.
        
        Args:
            model: Trained model with predict_proba method
            feature_names: List of feature names
            feature_ranges: Optional dict of (min, max) for each feature
        """
        self.model = model
        self.feature_names = feature_names
        self.feature_ranges = feature_ranges or FEATURE_CONSTRAINTS.bounds
        self.config = OPTIMIZATION_CF_CONFIG.copy()
        
    def generate_counterfactuals(
        self,
        instance: pd.DataFrame,
        num_cfs: int = 5,
        desired_class: int = 1,
        target_probability: float = 0.6
    ) -> Dict[str, Any]:
        """
        Generate counterfactual explanations using optimization.
        
        Args:
            instance: Single row DataFrame to explain
            num_cfs: Number of counterfactuals to generate
            desired_class: Target class (0 or 1)
            target_probability: Minimum probability for target class
            
        Returns:
            Dictionary with counterfactuals and metadata
        """
        original = instance.values.flatten().astype(float)
        counterfactuals = []
        
        # Get mutable feature indices
        mutable_indices = self._get_mutable_indices()
        
        for i in range(num_cfs * 3):  # Try more times to get diverse CFs
            if len(counterfactuals) >= num_cfs:
                break
                
            # Random initialization with perturbation
            x0 = original.copy()
            perturbation = np.random.normal(0, 0.1, len(x0))
            x0[mutable_indices] += perturbation[mutable_indices]
            
            # Optimize
            result = self._optimize(
                x0=x0,
                original=original,
                target_class=desired_class,
                target_prob=target_probability,
                mutable_indices=mutable_indices
            )
            
            if result is not None:
                # Check if this CF is diverse enough from existing ones
                if self._is_diverse(result, counterfactuals, min_distance=0.1):
                    counterfactuals.append(result)
                    
        if not counterfactuals:
            return {
                "success": False,
                "message": "Optimization failed to find valid counterfactuals",
                "counterfactuals": [],
                "original": dict(zip(self.feature_names, original)),
            }
            
        # Format results
        formatted_cfs = []
        for cf in counterfactuals:
            cf_dict = dict(zip(self.feature_names, cf))
            distance = self._compute_distance(original, cf)
            sparsity = self._compute_sparsity(original, cf)
            
            formatted_cfs.append({
                "cf_features": cf_dict,
                "l1_distance": distance["l1"],
                "l2_distance": distance["l2"],
                "sparsity": sparsity,
                "n_changes": sparsity,
            })
            
        return {
            "success": True,
            "method": "Optimization",
            "counterfactuals": formatted_cfs,
            "original": dict(zip(self.feature_names, original)),
            "num_generated": len(formatted_cfs),
            "diversity": self._compute_diversity(formatted_cfs),
        }
        
    def _optimize(
        self,
        x0: np.ndarray,
        original: np.ndarray,
        target_class: int,
        target_prob: float,
        mutable_indices: List[int]
    ) -> Optional[np.ndarray]:
        """Run optimization to find a counterfactual."""
        
        def objective(x):
            # Distance penalty
            distance_loss = np.sum((x - original) ** 2)
            
            # Prediction loss
            x_df = pd.DataFrame([x], columns=self.feature_names)
            try:
                if hasattr(self.model, 'model'):
                    proba = self.model.predict_proba(x_df)[0][target_class]
                else:
                    proba = self.model.predict_proba(x_df)[0][target_class]
            except Exception:
                return 1e10
                
            validity_loss = max(0, target_prob - proba) ** 2
            
            # Combined loss
            lambda_dist = self.config.get("lambda_distance", 0.1)
            lambda_valid = self.config.get("lambda_validity", 1.0)
            
            return lambda_dist * distance_loss + lambda_valid * validity_loss
        
        # Bounds
        bounds = []
        for i, feature in enumerate(self.feature_names):
            if i in mutable_indices:
                if feature in self.feature_ranges:
                    bounds.append(self.feature_ranges[feature])
                else:
                    # Use data range with some margin
                    val = original[i]
                    bounds.append((val - abs(val) * 0.5, val + abs(val) * 0.5))
            else:
                # Immutable: fix to original value
                bounds.append((original[i], original[i]))
                
        try:
            result = minimize(
                objective,
                x0,
                method="L-BFGS-B",
                bounds=bounds,
                options={
                    "maxiter": self.config.get("max_iterations", 1000),
                    "ftol": self.config.get("convergence_threshold", 1e-6),
                }
            )
            
            # Verify the result achieves target
            x_result = result.x
            x_df = pd.DataFrame([x_result], columns=self.feature_names)
            
            if hasattr(self.model, 'model'):
                proba = self.model.predict_proba(x_df)[0][target_class]
            else:
                proba = self.model.predict_proba(x_df)[0][target_class]
                
            if proba >= target_prob:
                return x_result
            else:
                return None
                
        except Exception as e:
            logger.debug(f"Optimization iteration failed: {e}")
            return None
            
    def _get_mutable_indices(self) -> List[int]:
        """Get indices of mutable features."""
        immutable = FEATURE_CONSTRAINTS.immutable
        return [
            i for i, f in enumerate(self.feature_names)
            if f not in immutable
        ]
        
    def _is_diverse(
        self,
        new_cf: np.ndarray,
        existing_cfs: List[np.ndarray],
        min_distance: float
    ) -> bool:
        """Check if new CF is diverse from existing ones."""
        if not existing_cfs:
            return True
            
        for cf in existing_cfs:
            dist = np.sqrt(np.sum((new_cf - cf) ** 2))
            if dist < min_distance:
                return False
                
        return True
        
    def _compute_distance(
        self,
        original: np.ndarray,
        cf: np.ndarray
    ) -> Dict[str, float]:
        """Compute distance metrics."""
        diff = np.abs(cf - original)
        return {
            "l1": float(np.sum(diff)),
            "l2": float(np.sqrt(np.sum(diff ** 2))),
        }
        
    def _compute_sparsity(
        self,
        original: np.ndarray,
        cf: np.ndarray,
        threshold: float = 0.01
    ) -> int:
        """Count number of changed features."""
        return int(np.sum(np.abs(cf - original) > threshold))
        
    def _compute_diversity(
        self,
        counterfactuals: List[Dict[str, Any]]
    ) -> float:
        """Compute diversity across counterfactuals."""
        if len(counterfactuals) < 2:
            return 0.0
            
        distances = []
        for i in range(len(counterfactuals)):
            for j in range(i + 1, len(counterfactuals)):
                cf1 = np.array(list(counterfactuals[i]["cf_features"].values()))
                cf2 = np.array(list(counterfactuals[j]["cf_features"].values()))
                dist = np.sqrt(np.sum((cf1 - cf2) ** 2))
                distances.append(dist)
                
        return float(np.mean(distances))
