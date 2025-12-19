"""
Causal Counterfactual Engine (PRIMARY CONTRIBUTION)

Generates counterfactual explanations that respect causal dependencies.
This is our main contribution - causally-constrained counterfactuals.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List, Tuple
from scipy.optimize import minimize, differential_evolution
import logging

from .causal_model import CausalModel
from ..config import (
    FEATURE_CONSTRAINTS,
    CAUSAL_CF_CONFIG,
    ACTIONABILITY_WEIGHTS,
    EFFORT_COSTS,
)
from ..ethics.actionability import ActionabilityScorer

logger = logging.getLogger(__name__)


class CausalCFEngine:
    """
    Causal Counterfactual Generator (PRIMARY CONTRIBUTION).
    
    Generates counterfactual explanations that:
    1. Respect causal dependencies (propagate interventions through SCM)
    2. Satisfy actionability constraints
    3. Minimize changes while achieving target prediction
    
    This is our main contribution, compared against DiCE and Optimization baselines.
    """
    
    def __init__(
        self,
        model: Any,
        causal_model: CausalModel,
        feature_names: List[str],
        actionability_scorer: Optional[ActionabilityScorer] = None
    ):
        """
        Initialize the causal CF engine.
        
        Args:
            model: Trained prediction model
            causal_model: Structural Causal Model defining relationships
            feature_names: List of feature names
            actionability_scorer: Optional scorer for actionability
        """
        self.model = model
        self.causal_model = causal_model
        self.feature_names = feature_names
        self.actionability_scorer = actionability_scorer or ActionabilityScorer()
        self.config = CAUSAL_CF_CONFIG.copy()
        
    def generate_counterfactuals(
        self,
        instance: pd.DataFrame,
        num_cfs: int = 5,
        desired_class: int = 1,
        target_probability: float = 0.6,
        max_interventions: Optional[int] = None,
        relaxed_mode: bool = False
    ) -> Dict[str, Any]:
        """
        Generate causally-valid counterfactual explanations.
        
        Args:
            instance: Single row DataFrame to explain
            num_cfs: Number of counterfactuals to generate
            desired_class: Target class (0 or 1)
            target_probability: Minimum probability for target class
            max_interventions: Maximum number of features to intervene on
            relaxed_mode: If True, allow bounded downstream proxy changes with penalty
            
        Returns:
            Dictionary with counterfactuals and metadata
        """
        max_interventions = max_interventions or self.config.get("max_interventions", 4)
        original = instance.iloc[0].to_dict()
        
        # Get modifiable features (considering causal structure)
        modifiable_features = self._get_modifiable_features()
        
        # Find root causes - features that should be intervened on
        # (features without parents among modifiable features)
        intervention_candidates = self._get_intervention_candidates(modifiable_features)
        
        # In relaxed mode, also allow downstream features as candidates
        if relaxed_mode:
            intervention_candidates = modifiable_features.copy()
        
        counterfactuals = []
        attempted = 0
        max_attempts = num_cfs * 5
        
        while len(counterfactuals) < num_cfs and attempted < max_attempts:
            attempted += 1
            
            # Select features to intervene on (randomly for diversity)
            n_interventions = np.random.randint(1, min(max_interventions + 1, len(intervention_candidates) + 1))
            selected_features = np.random.choice(
                intervention_candidates,
                size=n_interventions,
                replace=False
            ).tolist()
            
            # Generate intervention values
            result = self._optimize_interventions(
                original=original,
                intervention_features=selected_features,
                target_class=desired_class,
                target_prob=target_probability,
                relaxed_mode=relaxed_mode
            )
            
            if result is not None:
                cf, interventions = result
                
                # Validate causal consistency
                validation = self.causal_model.validate_counterfactual(original, cf)
                
                # In relaxed mode, accept partial violations with penalty
                if relaxed_mode:
                    # Count violations (proxy penalty)
                    violations = validation.get("violations", [])
                    proxy_penalty = len(violations) * 0.1  # 10% penalty per violation
                    is_valid = len(violations) <= 2  # Allow up to 2 proxy changes
                else:
                    is_valid = validation["valid"]
                    proxy_penalty = 0.0
                
                if is_valid:
                    # Check diversity
                    if self._is_diverse(cf, counterfactuals):
                        # Compute actionability
                        actionability = self.actionability_scorer.score(
                            original=original,
                            counterfactual=cf
                        )
                        
                        # Apply proxy penalty in relaxed mode
                        adjusted_actionability = max(0, actionability["total_score"] - proxy_penalty)
                        
                        counterfactuals.append({
                            "cf_features": cf,
                            "interventions": interventions,
                            "causal_valid": validation["valid"],
                            "relaxed_valid": is_valid,
                            "proxy_penalty": proxy_penalty,
                            "actionability_score": adjusted_actionability,
                            "actionability_details": actionability,
                        })
                        
        if not counterfactuals:
            return {
                "success": False,
                "message": "Failed to generate counterfactuals",
                "counterfactuals": [],
                "original": original,
            }
            
        # Compute metrics for each CF
        formatted_cfs = []
        for cf_data in counterfactuals:
            cf = cf_data["cf_features"]
            distance = self._compute_distance(original, cf)
            sparsity = self._compute_sparsity(original, cf)
            
            formatted_cfs.append({
                "cf_features": cf,
                "l1_distance": distance["l1"],
                "l2_distance": distance["l2"],
                "sparsity": sparsity,
                "n_changes": sparsity,
                "interventions": cf_data["interventions"],
                "causal_valid": cf_data["causal_valid"],
                "relaxed_valid": cf_data.get("relaxed_valid", cf_data["causal_valid"]),
                "proxy_penalty": cf_data.get("proxy_penalty", 0.0),
                "actionability_score": cf_data["actionability_score"],
                "actionability_details": cf_data["actionability_details"],
            })
            
        # Sort by actionability score
        formatted_cfs.sort(key=lambda x: x["actionability_score"], reverse=True)
        
        mode_name = "Causal-Relaxed" if relaxed_mode else "Causal"
        
        return {
            "success": True,
            "method": mode_name,
            "relaxed_mode": relaxed_mode,
            "counterfactuals": formatted_cfs,
            "original": original,
            "num_generated": len(formatted_cfs),
            "diversity": self._compute_diversity(formatted_cfs),
            "causal_validity_rate": sum(1 for cf in formatted_cfs if cf["causal_valid"]) / len(formatted_cfs),
            "mean_actionability": np.mean([cf["actionability_score"] for cf in formatted_cfs]),
        }
        
    def _get_modifiable_features(self) -> List[str]:
        """Get features that can be modified (not immutable)."""
        immutable = FEATURE_CONSTRAINTS.immutable
        return [f for f in self.feature_names if f not in immutable]
    
    def _get_intervention_candidates(self, modifiable: List[str]) -> List[str]:
        """
        Get candidate features for intervention.
        
        Prioritizes features that are root causes (no modifiable parents)
        or have high impact on downstream variables.
        """
        candidates = []
        
        for feature in modifiable:
            parents = self.causal_model.get_parents(feature)
            modifiable_parents = [p for p in parents if p in modifiable]
            
            # Prefer features with no modifiable parents (root causes)
            if not modifiable_parents:
                candidates.append(feature)
            # Also include features with children (for propagation effects)
            elif self.causal_model.get_children(feature):
                candidates.append(feature)
                
        # If no candidates, use all modifiable features
        if not candidates:
            candidates = modifiable
            
        return candidates
    
    def _optimize_interventions(
        self,
        original: Dict[str, float],
        intervention_features: List[str],
        target_class: int,
        target_prob: float,
        relaxed_mode: bool = False
    ) -> Optional[Tuple[Dict[str, float], Dict[str, float]]]:
        """
        Optimize intervention values to achieve target prediction.
        
        Uses the causal model to propagate effects of interventions.
        In relaxed mode, applies less strict actionability penalty.
        """
        
        def objective(intervention_values):
            # Create intervention dict
            interventions = dict(zip(intervention_features, intervention_values))
            
            # Propagate through causal model
            cf_values = self.causal_model.intervene(original, interventions)
            
            # Create feature vector for prediction
            cf_vector = np.array([
                cf_values.get(f, original.get(f, 0))
                for f in self.feature_names
            ]).reshape(1, -1)
            
            cf_df = pd.DataFrame(cf_vector, columns=self.feature_names)
            
            # Get prediction
            try:
                if hasattr(self.model, 'model'):
                    proba = self.model.predict_proba(cf_df)[0][target_class]
                else:
                    proba = self.model.predict_proba(cf_df)[0][target_class]
            except Exception:
                return 1e10
                
            # Validity loss
            validity_loss = max(0, target_prob - proba) ** 2
            
            # Distance loss (prefer minimal changes)
            distance_loss = sum(
                (interventions[f] - original.get(f, 0)) ** 2
                for f in intervention_features
            )
            
            # Actionability penalty
            actionability = self.actionability_scorer.score(original, cf_values)
            actionability_loss = (1 - actionability["total_score"]) ** 2
            
            return validity_loss * 10 + distance_loss * 0.01 + actionability_loss * 0.5
        
        # Get bounds for intervention features
        bounds = []
        for f in intervention_features:
            if f in FEATURE_CONSTRAINTS.bounds:
                bounds.append(FEATURE_CONSTRAINTS.bounds[f])
            else:
                # Use original value +/- 50%
                orig = original.get(f, 0)
                margin = max(abs(orig) * 0.5, 1)
                bounds.append((orig - margin, orig + margin))
                
        # Check direction constraints
        for i, f in enumerate(intervention_features):
            if f in FEATURE_CONSTRAINTS.direction:
                direction = FEATURE_CONSTRAINTS.direction[f]
                orig = original.get(f, 0)
                
                if direction == "increase_only":
                    bounds[i] = (orig, bounds[i][1])
                elif direction == "decrease_only":
                    bounds[i] = (bounds[i][0], orig)
                    
        # Initial guess
        x0 = np.array([original.get(f, 0) for f in intervention_features])
        
        try:
            # Try differential evolution for global optimization
            result = differential_evolution(
                objective,
                bounds=bounds,
                maxiter=100,
                tol=1e-5,
                seed=np.random.randint(0, 10000)
            )
            
            if result.fun < 0.5:  # Reasonable loss
                interventions = dict(zip(intervention_features, result.x))
                cf_values = self.causal_model.intervene(original, interventions)
                
                # Verify prediction
                cf_vector = np.array([
                    cf_values.get(f, original.get(f, 0))
                    for f in self.feature_names
                ]).reshape(1, -1)
                
                cf_df = pd.DataFrame(cf_vector, columns=self.feature_names)
                
                if hasattr(self.model, 'model'):
                    proba = self.model.predict_proba(cf_df)[0][target_class]
                else:
                    proba = self.model.predict_proba(cf_df)[0][target_class]
                    
                if proba >= target_prob:
                    # Round to reasonable precision
                    cf_values = {k: round(v, 4) if isinstance(v, float) else v 
                                for k, v in cf_values.items()}
                    return cf_values, interventions
                    
        except Exception as e:
            logger.debug(f"Optimization failed: {e}")
            
        return None
    
    def _is_diverse(
        self,
        new_cf: Dict[str, float],
        existing_cfs: List[Dict[str, Any]],
        min_distance: float = 0.1
    ) -> bool:
        """Check if new CF is sufficiently diverse from existing ones."""
        if not existing_cfs:
            return True
            
        for cf_data in existing_cfs:
            cf = cf_data["cf_features"]
            # Compute L2 distance
            dist = sum(
                (new_cf.get(f, 0) - cf.get(f, 0)) ** 2
                for f in self.feature_names
            ) ** 0.5
            
            if dist < min_distance:
                return False
                
        return True
    
    def _compute_distance(
        self,
        original: Dict[str, float],
        cf: Dict[str, float]
    ) -> Dict[str, float]:
        """Compute distance metrics."""
        l1 = sum(
            abs(cf.get(f, 0) - original.get(f, 0))
            for f in self.feature_names
        )
        l2 = sum(
            (cf.get(f, 0) - original.get(f, 0)) ** 2
            for f in self.feature_names
        ) ** 0.5
        
        return {"l1": float(l1), "l2": float(l2)}
    
    def _compute_sparsity(
        self,
        original: Dict[str, float],
        cf: Dict[str, float],
        threshold: float = 0.01
    ) -> int:
        """Count number of changed features."""
        return sum(
            1 for f in self.feature_names
            if abs(cf.get(f, 0) - original.get(f, 0)) > threshold
        )
    
    def _compute_diversity(
        self,
        counterfactuals: List[Dict[str, Any]]
    ) -> float:
        """Compute diversity score."""
        if len(counterfactuals) < 2:
            return 0.0
            
        distances = []
        for i in range(len(counterfactuals)):
            for j in range(i + 1, len(counterfactuals)):
                cf1 = counterfactuals[i]["cf_features"]
                cf2 = counterfactuals[j]["cf_features"]
                
                dist = sum(
                    (cf1.get(f, 0) - cf2.get(f, 0)) ** 2
                    for f in self.feature_names
                ) ** 0.5
                distances.append(dist)
                
        return float(np.mean(distances))
    
    def compare_with_baseline(
        self,
        instance: pd.DataFrame,
        dice_results: Dict[str, Any],
        optimization_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compare causal CFs with baseline methods.
        
        Args:
            instance: Original instance
            dice_results: Results from DiCE engine
            optimization_results: Results from Optimization engine
            
        Returns:
            Comparison metrics
        """
        original = instance.iloc[0].to_dict()
        
        comparison = {
            "methods": ["DiCE", "Optimization", "Causal"],
            "metrics": {}
        }
        
        all_results = {
            "DiCE": dice_results,
            "Optimization": optimization_results,
            "Causal": self.generate_counterfactuals(instance)
        }
        
        for method, results in all_results.items():
            if not results.get("success", False):
                comparison["metrics"][method] = {"error": "Generation failed"}
                continue
                
            cfs = results.get("counterfactuals", [])
            
            if not cfs:
                comparison["metrics"][method] = {"error": "No counterfactuals"}
                continue
                
            # Compute metrics
            causal_validity_scores = []
            for cf_data in cfs:
                cf = cf_data.get("cf_features", cf_data)
                validation = self.causal_model.validate_counterfactual(original, cf)
                causal_validity_scores.append(1 if validation["valid"] else 0)
                
            comparison["metrics"][method] = {
                "num_counterfactuals": len(cfs),
                "mean_l1_distance": np.mean([cf.get("l1_distance", 0) for cf in cfs]),
                "mean_l2_distance": np.mean([cf.get("l2_distance", 0) for cf in cfs]),
                "mean_sparsity": np.mean([cf.get("sparsity", cf.get("n_changes", 0)) for cf in cfs]),
                "causal_validity_rate": np.mean(causal_validity_scores),
                "mean_actionability": np.mean([
                    cf.get("actionability_score", 0.5) for cf in cfs
                ]),
                "diversity": results.get("diversity", 0),
            }
            
        return comparison
