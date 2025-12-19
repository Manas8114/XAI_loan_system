"""
Constraint Validator Module

Validates counterfactuals against feature constraints (immutability, bounds, direction).
"""

from typing import Dict, Any, List, Optional
import logging

from ..config import FEATURE_CONSTRAINTS

logger = logging.getLogger(__name__)


class ConstraintValidator:
    """
    Validates counterfactuals against defined constraints.
    
    Enforces:
    - Immutability constraints (some features cannot change)
    - Bound constraints (features must stay within realistic ranges)
    - Direction constraints (some features can only increase/decrease)
    """
    
    def __init__(self, constraints: Optional[Any] = None):
        """
        Initialize the constraint validator.
        
        Args:
            constraints: FeatureConstraints object or None for defaults
        """
        self.constraints = constraints or FEATURE_CONSTRAINTS
        
    def validate(
        self,
        original: Dict[str, float],
        counterfactual: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Validate a counterfactual against all constraints.
        
        Args:
            original: Original feature values
            counterfactual: Proposed counterfactual values
            
        Returns:
            Validation result with details
        """
        violations = []
        warnings = []
        
        # Check immutability
        immutability_result = self._check_immutability(original, counterfactual)
        violations.extend(immutability_result["violations"])
        
        # Check bounds
        bounds_result = self._check_bounds(counterfactual)
        violations.extend(bounds_result["violations"])
        warnings.extend(bounds_result["warnings"])
        
        # Check direction
        direction_result = self._check_direction(original, counterfactual)
        violations.extend(direction_result["violations"])
        
        return {
            "valid": len(violations) == 0,
            "n_violations": len(violations),
            "n_warnings": len(warnings),
            "violations": violations,
            "warnings": warnings,
            "details": {
                "immutability": immutability_result,
                "bounds": bounds_result,
                "direction": direction_result,
            }
        }
    
    def _check_immutability(
        self,
        original: Dict[str, float],
        counterfactual: Dict[str, float]
    ) -> Dict[str, Any]:
        """Check that immutable features are not changed."""
        violations = []
        
        for feature in self.constraints.immutable:
            if feature in original and feature in counterfactual:
                orig_val = original[feature]
                cf_val = counterfactual[feature]
                
                if abs(cf_val - orig_val) > 0.001:
                    violations.append({
                        "type": "immutability",
                        "feature": feature,
                        "original_value": orig_val,
                        "counterfactual_value": cf_val,
                        "message": f"Feature '{feature}' is immutable and cannot be changed"
                    })
                    
        return {
            "checked": len(self.constraints.immutable),
            "violations": violations,
        }
    
    def _check_bounds(
        self,
        counterfactual: Dict[str, float]
    ) -> Dict[str, Any]:
        """Check that features are within realistic bounds."""
        violations = []
        warnings = []
        
        for feature, (min_val, max_val) in self.constraints.bounds.items():
            if feature in counterfactual:
                value = counterfactual[feature]
                
                if value < min_val:
                    violations.append({
                        "type": "bounds_lower",
                        "feature": feature,
                        "value": value,
                        "min_allowed": min_val,
                        "message": f"Feature '{feature}' value {value} is below minimum {min_val}"
                    })
                elif value > max_val:
                    violations.append({
                        "type": "bounds_upper",
                        "feature": feature,
                        "value": value,
                        "max_allowed": max_val,
                        "message": f"Feature '{feature}' value {value} exceeds maximum {max_val}"
                    })
                elif value < min_val + 0.1 * (max_val - min_val):
                    # Near lower bound - warning
                    warnings.append({
                        "type": "near_lower_bound",
                        "feature": feature,
                        "message": f"Feature '{feature}' is near its minimum value"
                    })
                elif value > max_val - 0.1 * (max_val - min_val):
                    # Near upper bound - warning
                    warnings.append({
                        "type": "near_upper_bound",
                        "feature": feature,
                        "message": f"Feature '{feature}' is near its maximum value"
                    })
                    
        return {
            "checked": len(self.constraints.bounds),
            "violations": violations,
            "warnings": warnings,
        }
    
    def _check_direction(
        self,
        original: Dict[str, float],
        counterfactual: Dict[str, float]
    ) -> Dict[str, Any]:
        """Check that feature changes respect direction constraints."""
        violations = []
        
        for feature, direction in self.constraints.direction.items():
            if feature in original and feature in counterfactual:
                orig_val = original[feature]
                cf_val = counterfactual[feature]
                change = cf_val - orig_val
                
                if direction == "increase_only" and change < -0.001:
                    violations.append({
                        "type": "direction",
                        "feature": feature,
                        "direction_constraint": direction,
                        "actual_change": change,
                        "message": f"Feature '{feature}' can only increase, but decreased by {-change}"
                    })
                elif direction == "decrease_only" and change > 0.001:
                    violations.append({
                        "type": "direction",
                        "feature": feature,
                        "direction_constraint": direction,
                        "actual_change": change,
                        "message": f"Feature '{feature}' can only decrease, but increased by {change}"
                    })
                    
        return {
            "checked": len(self.constraints.direction),
            "violations": violations,
        }
    
    def fix_violations(
        self,
        original: Dict[str, float],
        counterfactual: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Attempt to fix constraint violations in a counterfactual.
        
        Args:
            original: Original feature values
            counterfactual: Counterfactual with potential violations
            
        Returns:
            Fixed counterfactual
        """
        fixed = counterfactual.copy()
        
        # Fix immutability violations
        for feature in self.constraints.immutable:
            if feature in original:
                fixed[feature] = original[feature]
                
        # Fix bound violations
        for feature, (min_val, max_val) in self.constraints.bounds.items():
            if feature in fixed:
                fixed[feature] = max(min_val, min(max_val, fixed[feature]))
                
        # Fix direction violations
        for feature, direction in self.constraints.direction.items():
            if feature in original and feature in fixed:
                orig_val = original[feature]
                cf_val = fixed[feature]
                
                if direction == "increase_only" and cf_val < orig_val:
                    fixed[feature] = orig_val
                elif direction == "decrease_only" and cf_val > orig_val:
                    fixed[feature] = orig_val
                    
        return fixed
