"""
Feasibility Guard Module

Combined guard layer that validates counterfactuals for both
constraints and actionability before returning to users.
"""

from typing import Dict, Any, List, Optional
import logging

from .constraints import ConstraintValidator
from .actionability import ActionabilityScorer
from .fairness import FairnessValidator

logger = logging.getLogger(__name__)


class FeasibilityGuard:
    """
    Comprehensive feasibility guard for counterfactual explanations.
    
    Combines:
    - Constraint validation (immutability, bounds, direction)
    - Actionability assessment
    - Fairness checking
    
    This is the final filter before counterfactuals are returned to users.
    """
    
    def __init__(
        self,
        constraint_validator: Optional[ConstraintValidator] = None,
        actionability_scorer: Optional[ActionabilityScorer] = None,
        fairness_validator: Optional[FairnessValidator] = None
    ):
        """
        Initialize the feasibility guard.
        
        Args:
            constraint_validator: Optional custom constraint validator
            actionability_scorer: Optional custom actionability scorer
            fairness_validator: Optional custom fairness validator
        """
        self.constraint_validator = constraint_validator or ConstraintValidator()
        self.actionability_scorer = actionability_scorer or ActionabilityScorer()
        self.fairness_validator = fairness_validator or FairnessValidator()
        
    def check(
        self,
        original: Dict[str, float],
        counterfactual: Dict[str, float],
        min_actionability: float = 0.3,
        protected_attr: Optional[str] = None,
        protected_value: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive feasibility check on a counterfactual.
        
        Args:
            original: Original feature values
            counterfactual: Proposed counterfactual values
            min_actionability: Minimum actionability score required
            protected_attr: Optional protected attribute for fairness check
            protected_value: Value of protected attribute for this instance
            
        Returns:
            Feasibility assessment with pass/fail and details
        """
        # Check constraints
        constraint_result = self.constraint_validator.validate(original, counterfactual)
        
        # Check actionability
        actionability_result = self.actionability_scorer.score(original, counterfactual)
        
        # Determine overall pass/fail
        passes_constraints = constraint_result["valid"]
        passes_actionability = actionability_result["total_score"] >= min_actionability
        
        overall_pass = passes_constraints and passes_actionability
        
        # Build result
        result = {
            "feasible": overall_pass,
            "passes_constraints": passes_constraints,
            "passes_actionability": passes_actionability,
            "constraint_validation": constraint_result,
            "actionability": actionability_result,
            "recommendations": [],
        }
        
        # Add recommendations based on failures
        if not passes_constraints:
            for violation in constraint_result["violations"]:
                result["recommendations"].append({
                    "type": "constraint_fix",
                    "message": violation["message"],
                })
                
        if not passes_actionability:
            result["recommendations"].append({
                "type": "actionability",
                "message": f"Actionability score ({actionability_result['total_score']:.2f}) "
                          f"is below minimum threshold ({min_actionability}). "
                          f"Consider smaller or more realistic changes.",
            })
            
        return result
    
    def filter_counterfactuals(
        self,
        original: Dict[str, float],
        counterfactuals: List[Dict[str, float]],
        min_actionability: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Filter a list of counterfactuals, returning only feasible ones.
        
        Args:
            original: Original feature values
            counterfactuals: List of proposed counterfactuals
            min_actionability: Minimum actionability score
            
        Returns:
            List of feasible counterfactuals with their scores
        """
        feasible_cfs = []
        
        for cf in counterfactuals:
            result = self.check(original, cf, min_actionability)
            
            if result["feasible"]:
                feasible_cfs.append({
                    "counterfactual": cf,
                    "actionability_score": result["actionability"]["total_score"],
                    "actionability_level": result["actionability"]["level"],
                    "n_changes": result["actionability"]["n_changes"],
                })
                
        # Sort by actionability score (highest first)
        feasible_cfs.sort(key=lambda x: x["actionability_score"], reverse=True)
        
        return feasible_cfs
    
    def auto_fix(
        self,
        original: Dict[str, float],
        counterfactual: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Attempt to automatically fix constraint violations.
        
        Args:
            original: Original feature values
            counterfactual: Counterfactual with potential violations
            
        Returns:
            Fixed counterfactual with validation results
        """
        # Fix constraint violations
        fixed_cf = self.constraint_validator.fix_violations(original, counterfactual)
        
        # Validate the fixed version
        validation = self.check(original, fixed_cf)
        
        # Compute what changed
        changes_made = []
        for feature in counterfactual:
            if feature in fixed_cf:
                if counterfactual[feature] != fixed_cf[feature]:
                    changes_made.append({
                        "feature": feature,
                        "original_cf_value": counterfactual[feature],
                        "fixed_value": fixed_cf[feature],
                        "reason": "Constraint violation fix",
                    })
                    
        return {
            "fixed_counterfactual": fixed_cf,
            "changes_made": changes_made,
            "n_changes": len(changes_made),
            "validation": validation,
        }
    
    def rank_counterfactuals(
        self,
        original: Dict[str, float],
        counterfactuals: List[Dict[str, float]],
        weights: Optional[Dict[str, float]] = None
    ) -> List[Dict[str, Any]]:
        """
        Rank counterfactuals by a weighted combination of factors.
        
        Args:
            original: Original feature values
            counterfactuals: List of counterfactuals to rank
            weights: Optional weights for ranking factors
                     (actionability, proximity, sparsity)
                     
        Returns:
            Ranked list of counterfactuals with scores
        """
        weights = weights or {
            "actionability": 0.5,
            "proximity": 0.3,
            "sparsity": 0.2,
        }
        
        ranked = []
        
        for cf in counterfactuals:
            result = self.check(original, cf)
            
            # Compute proximity (inverse of L1 distance, normalized)
            l1_dist = sum(
                abs(cf.get(f, 0) - original.get(f, 0))
                for f in cf
            )
            proximity = 1 / (1 + l1_dist / len(cf))
            
            # Compute sparsity (fewer changes = better)
            n_changes = sum(
                1 for f in cf
                if abs(cf.get(f, 0) - original.get(f, 0)) > 0.01
            )
            sparsity = 1 / (1 + n_changes)
            
            # Combined score
            actionability = result["actionability"]["total_score"]
            combined_score = (
                weights["actionability"] * actionability +
                weights["proximity"] * proximity +
                weights["sparsity"] * sparsity
            )
            
            ranked.append({
                "counterfactual": cf,
                "combined_score": combined_score,
                "actionability_score": actionability,
                "proximity_score": proximity,
                "sparsity_score": sparsity,
                "n_changes": n_changes,
                "feasible": result["feasible"],
            })
            
        # Sort by combined score
        ranked.sort(key=lambda x: x["combined_score"], reverse=True)
        
        return ranked
