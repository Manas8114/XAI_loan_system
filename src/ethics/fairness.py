"""
Fairness Validator Module

Ensures counterfactual recommendations are fair across demographic groups.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
import logging

from ..config import PROTECTED_ATTRIBUTES

logger = logging.getLogger(__name__)


class FairnessValidator:
    """
    Validates fairness of counterfactual explanations across demographics.
    
    Ensures that:
    - Counterfactual recommendations don't systematically differ by group
    - "Effort" required is comparable across protected groups
    - No disparate treatment in recommendations
    """
    
    def __init__(self, protected_attributes: Optional[List[str]] = None):
        """
        Initialize the fairness validator.
        
        Args:
            protected_attributes: List of protected attribute names
        """
        self.protected_attributes = protected_attributes or PROTECTED_ATTRIBUTES
        
    def validate_single(
        self,
        original: Dict[str, float],
        counterfactual: Dict[str, float],
        actionability_score: float,
        group: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Validate fairness for a single counterfactual.
        
        Args:
            original: Original feature values
            counterfactual: Counterfactual values
            actionability_score: Pre-computed actionability score
            group: Demographic group identifier
            
        Returns:
            Fairness validation result
        """
        # Compute effort metrics
        total_change = 0
        n_changes = 0
        
        for feature in counterfactual:
            if feature in original:
                change = abs(counterfactual[feature] - original[feature])
                if change > 0.01:
                    total_change += change
                    n_changes += 1
                    
        return {
            "group": group,
            "actionability_score": actionability_score,
            "total_change_magnitude": total_change,
            "n_changes_required": n_changes,
        }
    
    def validate_batch(
        self,
        originals: List[Dict[str, float]],
        counterfactuals: List[Dict[str, float]],
        actionability_scores: List[float],
        groups: List[str]
    ) -> Dict[str, Any]:
        """
        Validate fairness across a batch of counterfactuals by group.
        
        Args:
            originals: List of original feature dicts
            counterfactuals: List of counterfactual dicts
            actionability_scores: List of actionability scores
            groups: List of group identifiers
            
        Returns:
            Batch fairness analysis
        """
        # Group results
        group_metrics = {}
        
        for orig, cf, score, group in zip(originals, counterfactuals, actionability_scores, groups):
            if group not in group_metrics:
                group_metrics[group] = {
                    "actionability_scores": [],
                    "change_magnitudes": [],
                    "n_changes": [],
                }
                
            result = self.validate_single(orig, cf, score, group)
            group_metrics[group]["actionability_scores"].append(score)
            group_metrics[group]["change_magnitudes"].append(result["total_change_magnitude"])
            group_metrics[group]["n_changes"].append(result["n_changes_required"])
            
        # Compute group statistics
        group_stats = {}
        for group, metrics in group_metrics.items():
            group_stats[group] = {
                "n_samples": len(metrics["actionability_scores"]),
                "mean_actionability": np.mean(metrics["actionability_scores"]),
                "std_actionability": np.std(metrics["actionability_scores"]),
                "mean_change_magnitude": np.mean(metrics["change_magnitudes"]),
                "mean_n_changes": np.mean(metrics["n_changes"]),
            }
            
        # Compute disparity metrics
        actionability_values = [s["mean_actionability"] for s in group_stats.values()]
        change_values = [s["mean_change_magnitude"] for s in group_stats.values()]
        
        if len(actionability_values) >= 2:
            actionability_disparity = max(actionability_values) - min(actionability_values)
            change_disparity = max(change_values) - min(change_values) if change_values else 0
        else:
            actionability_disparity = 0
            change_disparity = 0
            
        # Determine if fair
        is_fair = actionability_disparity < 0.2  # Allow 20% difference
        
        return {
            "is_fair": is_fair,
            "group_statistics": group_stats,
            "actionability_disparity": actionability_disparity,
            "change_magnitude_disparity": change_disparity,
            "fairness_threshold": 0.2,
            "recommendation": self._generate_recommendation(
                is_fair, actionability_disparity, group_stats
            ),
        }
    
    def _generate_recommendation(
        self,
        is_fair: bool,
        disparity: float,
        group_stats: Dict[str, Dict]
    ) -> str:
        """Generate fairness recommendation."""
        if is_fair:
            return "Counterfactual recommendations appear fair across groups."
            
        # Find disadvantaged group
        min_group = min(
            group_stats.keys(),
            key=lambda g: group_stats[g]["mean_actionability"]
        )
        
        return (
            f"Potential fairness concern: Group '{min_group}' receives less actionable "
            f"recommendations (disparity: {disparity:.3f}). Consider reviewing "
            f"feature constraints or reweighting actionability for this group."
        )
    
    def compute_counterfactual_fairness(
        self,
        original: Dict[str, float],
        counterfactual: Dict[str, float],
        protected_attr: str,
        alternative_values: List[Any]
    ) -> Dict[str, Any]:
        """
        Test counterfactual fairness by checking if recommendation
        would change under different protected attribute values.
        
        Args:
            original: Original features
            counterfactual: Generated counterfactual
            protected_attr: Protected attribute to test
            alternative_values: Alternative values for the protected attribute
            
        Returns:
            Counterfactual fairness analysis
        """
        results = {
            "protected_attribute": protected_attr,
            "original_value": original.get(protected_attr),
            "alternative_analyses": [],
            "is_counterfactually_fair": True,
        }
        
        # The counterfactual for each alternative group should be similar
        # (This is a simplified check - full counterfactual fairness requires
        # generating new counterfactuals under alternative world scenarios)
        
        original_changes = set(
            f for f in counterfactual
            if f in original and abs(counterfactual[f] - original[f]) > 0.01
        )
        
        for alt_value in alternative_values:
            if alt_value == original.get(protected_attr):
                continue
                
            results["alternative_analyses"].append({
                "alternative_value": alt_value,
                "note": "Full counterfactual fairness requires regenerating CF under alternative world",
                "features_changed": list(original_changes),
            })
            
        return results
    
    def audit_explanations(
        self,
        explanations: List[Dict[str, Any]],
        group_column: str = "group"
    ) -> Dict[str, Any]:
        """
        Audit a batch of explanations for systematic bias.
        
        Args:
            explanations: List of explanation dictionaries containing
                          original, counterfactual, score, and group info
            group_column: Key for group identifier in each explanation
            
        Returns:
            Comprehensive audit report
        """
        if not explanations:
            return {"error": "No explanations to audit"}
            
        # Extract data
        groups = [e.get(group_column, "unknown") for e in explanations]
        
        # Group explanations
        by_group = {}
        for exp, group in zip(explanations, groups):
            if group not in by_group:
                by_group[group] = []
            by_group[group].append(exp)
            
        audit = {
            "total_explanations": len(explanations),
            "n_groups": len(by_group),
            "groups": list(by_group.keys()),
            "group_counts": {g: len(exps) for g, exps in by_group.items()},
            "group_analyses": {},
        }
        
        for group, exps in by_group.items():
            scores = [e.get("actionability_score", 0.5) for e in exps]
            n_changes_list = [e.get("n_changes", 0) for e in exps]
            
            audit["group_analyses"][group] = {
                "count": len(exps),
                "mean_actionability": float(np.mean(scores)),
                "std_actionability": float(np.std(scores)),
                "mean_n_changes": float(np.mean(n_changes_list)),
            }
            
        # Compute overall fairness
        all_means = [a["mean_actionability"] for a in audit["group_analyses"].values()]
        audit["max_disparity"] = max(all_means) - min(all_means) if len(all_means) >= 2 else 0
        audit["passes_fairness_check"] = audit["max_disparity"] < 0.2
        
        return audit
