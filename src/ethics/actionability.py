"""
Actionability Scorer Module

Computes the formal Actionability Score for counterfactual recommendations.
This is a KEY NOVEL CONTRIBUTION of our research.
"""

import numpy as np
from typing import Dict, Any, List, Optional
import logging

from ..config import (
    ACTIONABILITY_WEIGHTS,
    EFFORT_COSTS,
    FEATURE_CONSTRAINTS,
    TIME_FACTOR_DECAY,
)

logger = logging.getLogger(__name__)


class ActionabilityScorer:
    """
    Computes actionability scores for counterfactual explanations.
    
    The Actionability Score quantifies how feasible a counterfactual
    recommendation is for an applicant to achieve.
    
    Formula:
        Actionability(cf) = Σᵢ wᵢ · aᵢ(cf)
        
        aᵢ(cf) = (1 - effort_costᵢ) × feasibilityᵢ × time_factorᵢ × direction_penaltyᵢ
        
    Components:
        - effort_cost: How difficult is the change (0-1, higher = harder)
        - feasibility: Whether the change is within realistic bounds (0 or 1)
        - time_factor: Decay based on time to achieve the change
        - direction_penalty: 0 if impossible direction, 1 otherwise
        
    This is a NOVEL CONTRIBUTION of our research.
    """
    
    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        effort_costs: Optional[Dict[str, Dict[str, float]]] = None
    ):
        """
        Initialize the actionability scorer.
        
        Args:
            weights: Feature weights for actionability score
            effort_costs: Effort cost mappings per feature
        """
        self.weights = weights or ACTIONABILITY_WEIGHTS
        self.effort_costs = effort_costs or EFFORT_COSTS
        self.constraints = FEATURE_CONSTRAINTS
        
    def score(
        self,
        original: Dict[str, float],
        counterfactual: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Compute the actionability score for a counterfactual.
        
        Args:
            original: Original feature values
            counterfactual: Counterfactual feature values
            
        Returns:
            Dictionary with total score and per-feature breakdown
        """
        feature_scores = {}
        weighted_sum = 0.0
        total_weight = 0.0
        
        for feature in counterfactual:
            if feature not in original:
                continue
                
            orig_val = original[feature]
            cf_val = counterfactual[feature]
            change = cf_val - orig_val
            
            # Skip unchanged features
            if abs(change) < 0.01:
                continue
                
            # Get weight for this feature
            weight = self.weights.get(feature, 0.1)
            
            # Compute per-feature actionability
            effort_cost = self._compute_effort_cost(feature, orig_val, cf_val)
            feasibility = self._check_feasibility(feature, cf_val)
            time_factor = self._compute_time_factor(feature, change)
            direction_penalty = self._check_direction(feature, change)
            
            # Combined actionability for this feature
            feature_actionability = (
                (1 - effort_cost) * feasibility * time_factor * direction_penalty
            )
            
            feature_scores[feature] = {
                "original_value": orig_val,
                "counterfactual_value": cf_val,
                "change": change,
                "change_percent": (change / (abs(orig_val) + 1e-10)) * 100,
                "effort_cost": effort_cost,
                "feasibility": feasibility,
                "time_factor": time_factor,
                "direction_penalty": direction_penalty,
                "feature_actionability": feature_actionability,
                "weight": weight,
                "weighted_contribution": weight * feature_actionability,
            }
            
            weighted_sum += weight * feature_actionability
            total_weight += weight
            
        # Normalize total score
        if total_weight > 0:
            total_score = weighted_sum / total_weight
        else:
            total_score = 1.0  # No changes = fully actionable
            
        # Determine actionability level
        if total_score >= 0.7:
            level = "Highly Actionable"
        elif total_score >= 0.5:
            level = "Moderately Actionable"
        elif total_score >= 0.3:
            level = "Challenging"
        else:
            level = "Difficult"
            
        return {
            "total_score": float(total_score),
            "level": level,
            "n_changes": len(feature_scores),
            "feature_scores": feature_scores,
            "weighted_sum": float(weighted_sum),
            "total_weight": float(total_weight),
        }
    
    def _compute_effort_cost(
        self,
        feature: str,
        orig_val: float,
        cf_val: float
    ) -> float:
        """
        Compute the effort cost for changing a feature.
        
        Args:
            feature: Feature name
            orig_val: Original value
            cf_val: Counterfactual value
            
        Returns:
            Effort cost between 0 (easy) and 1 (very hard)
        """
        change = cf_val - orig_val
        pct_change = abs(change) / (abs(orig_val) + 1e-10)
        
        # Check if we have predefined effort costs for this feature
        if feature in self.effort_costs:
            feature_costs = self.effort_costs[feature]
            
            # Find the applicable cost tier
            if feature == "annual_inc":
                if pct_change <= 0.1:
                    return feature_costs.get("10%", 0.4)
                elif pct_change <= 0.25:
                    return feature_costs.get("25%", 0.6)
                elif pct_change <= 0.5:
                    return feature_costs.get("50%", 0.8)
                else:
                    return feature_costs.get("100%", 0.95)
                    
            elif feature == "dti":
                if abs(change) <= 5:
                    return feature_costs.get("-5%", 0.3)
                elif abs(change) <= 10:
                    return feature_costs.get("-10%", 0.5)
                else:
                    return feature_costs.get("-20%", 0.7)
                    
            elif feature == "fico_score":
                if abs(change) <= 25:
                    return feature_costs.get("+25", 0.3)
                elif abs(change) <= 50:
                    return feature_costs.get("+50", 0.5)
                else:
                    return feature_costs.get("+100", 0.7)
                    
            elif feature == "emp_length":
                if abs(change) <= 1:
                    return feature_costs.get("+1yr", 0.9)
                else:
                    return feature_costs.get("+2yr", 0.95)
                    
            elif feature == "revol_util":
                if abs(change) <= 10:
                    return feature_costs.get("-10%", 0.2)
                elif abs(change) <= 25:
                    return feature_costs.get("-25%", 0.4)
                else:
                    return feature_costs.get("-50%", 0.6)
                    
        # Default effort cost based on percentage change
        return min(0.95, pct_change * 0.5 + 0.1)
    
    def _check_feasibility(
        self,
        feature: str,
        value: float
    ) -> float:
        """
        Check if a value is within feasible bounds.
        
        Returns:
            1.0 if feasible, 0.0 if infeasible
        """
        if feature in self.constraints.bounds:
            min_val, max_val = self.constraints.bounds[feature]
            if value < min_val or value > max_val:
                return 0.0
                
        return 1.0
    
    def _compute_time_factor(
        self,
        feature: str,
        change: float
    ) -> float:
        """
        Compute time decay factor based on expected time to achieve change.
        
        Returns:
            Time factor between 0 and 1 (higher = achievable sooner)
        """
        # Estimate time in months to achieve the change
        if feature == "emp_length":
            # Employment length requires actual time
            months = abs(change) * 12  # years to months
            return np.exp(-TIME_FACTOR_DECAY * months)
            
        elif feature == "fico_score":
            # Credit score improvement takes 6-12 months
            months = abs(change) / 10  # ~10 points per month improvement
            return np.exp(-TIME_FACTOR_DECAY * months)
            
        elif feature == "delinq_2yrs":
            # Delinquencies fall off after 2 years
            months = abs(change) * 12
            return np.exp(-TIME_FACTOR_DECAY * months)
            
        elif feature == "annual_inc":
            # Income changes can happen quickly (new job) or slowly
            pct_change = abs(change) / (abs(change) + 1e-10)
            if pct_change > 0.5:
                months = 12  # Major career change
            elif pct_change > 0.25:
                months = 6  # New job
            else:
                months = 3  # Raise or promotion
            return np.exp(-TIME_FACTOR_DECAY * months)
            
        elif feature == "dti":
            # DTI can be reduced by paying off debt
            months = abs(change) / 2  # ~2% per month with effort
            return np.exp(-TIME_FACTOR_DECAY * months)
            
        # Default: moderate time factor
        return 0.8
    
    def _check_direction(
        self,
        feature: str,
        change: float
    ) -> float:
        """
        Check if change direction is feasible.
        
        Returns:
            1.0 if direction is feasible, 0.0 if impossible
        """
        if feature not in self.constraints.direction:
            return 1.0
            
        direction = self.constraints.direction[feature]
        
        if direction == "increase_only" and change < 0:
            # Cannot decrease this feature
            return 0.0
        elif direction == "decrease_only" and change > 0:
            # Cannot increase this feature
            return 0.0
            
        return 1.0
    
    def get_improvement_suggestions(
        self,
        original: Dict[str, float],
        counterfactual: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """
        Generate actionable improvement suggestions from counterfactual.
        
        Args:
            original: Original values
            counterfactual: Counterfactual values
            
        Returns:
            List of prioritized improvement suggestions
        """
        score_result = self.score(original, counterfactual)
        suggestions = []
        
        for feature, details in score_result["feature_scores"].items():
            if details["feature_actionability"] > 0.3:
                suggestion = {
                    "feature": feature,
                    "action": self._format_action(feature, details["change"]),
                    "from_value": details["original_value"],
                    "to_value": details["counterfactual_value"],
                    "difficulty": self._effort_to_difficulty(details["effort_cost"]),
                    "estimated_time": self._estimate_time(feature, details["change"]),
                    "actionability": details["feature_actionability"],
                    "priority": details["weighted_contribution"],
                }
                suggestions.append(suggestion)
                
        # Sort by actionability (highest first)
        suggestions.sort(key=lambda x: x["actionability"], reverse=True)
        
        return suggestions
    
    def _format_action(self, feature: str, change: float) -> str:
        """Format human-readable action description."""
        direction = "Increase" if change > 0 else "Decrease"
        
        feature_names = {
            "annual_inc": "annual income",
            "dti": "debt-to-income ratio",
            "fico_score": "credit score",
            "emp_length": "employment length",
            "revol_util": "credit utilization",
            "delinq_2yrs": "recent delinquencies",
        }
        
        feature_name = feature_names.get(feature, feature)
        return f"{direction} your {feature_name}"
    
    def _effort_to_difficulty(self, effort: float) -> str:
        """Convert effort cost to difficulty label."""
        if effort < 0.3:
            return "Easy"
        elif effort < 0.5:
            return "Moderate"
        elif effort < 0.7:
            return "Challenging"
        else:
            return "Difficult"
    
    def _estimate_time(self, feature: str, change: float) -> str:
        """Estimate time to achieve the change."""
        if feature == "emp_length":
            years = abs(change)
            return f"{years:.0f} year(s)"
        elif feature == "fico_score":
            months = abs(change) / 10
            return f"{max(1, months):.0f} month(s)"
        elif feature == "annual_inc":
            return "3-12 months"
        elif feature == "dti":
            months = abs(change) / 2
            return f"{max(1, months):.0f} month(s)"
        else:
            return "Variable"
