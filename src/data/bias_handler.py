"""
Bias Handler Module

Identifies and mitigates bias in loan data to ensure fair model training
and counterfactual generation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from collections import defaultdict

from ..config import PROTECTED_ATTRIBUTES

logger = logging.getLogger(__name__)


class BiasHandler:
    """
    Handles bias detection and mitigation in loan data.
    
    Implements:
    - Disparate impact ratio calculation
    - Demographic parity analysis
    - Reweighting for bias mitigation
    - Fairness audit reports
    """
    
    def __init__(self, protected_attributes: Optional[List[str]] = None):
        """
        Initialize the bias handler.
        
        Args:
            protected_attributes: List of protected attribute column names.
        """
        self.protected_attributes = protected_attributes or PROTECTED_ATTRIBUTES
        self.sample_weights: Optional[np.ndarray] = None
        self.fairness_metrics: Dict[str, Any] = {}
        
    def compute_disparate_impact(
        self,
        df: pd.DataFrame,
        outcome_col: str,
        protected_attr: str,
        favorable_outcome: Any = 1
    ) -> Dict[str, float]:
        """
        Compute disparate impact ratio for a protected attribute.
        
        Disparate impact ratio = P(favorable|unprivileged) / P(favorable|privileged)
        A ratio < 0.8 is typically considered discriminatory (80% rule).
        
        Args:
            df: DataFrame with data
            outcome_col: Name of the outcome column
            protected_attr: Name of the protected attribute column
            favorable_outcome: Value representing favorable outcome
            
        Returns:
            Dictionary with disparate impact metrics
        """
        if protected_attr not in df.columns:
            logger.warning(f"Protected attribute '{protected_attr}' not found in data")
            return {}
            
        if outcome_col not in df.columns:
            raise ValueError(f"Outcome column '{outcome_col}' not found in data")
            
        # Get unique groups
        groups = df[protected_attr].unique()
        
        if len(groups) < 2:
            logger.warning(f"Only one group found for '{protected_attr}'")
            return {}
            
        # Calculate favorable outcome rate for each group
        group_rates = {}
        for group in groups:
            group_mask = df[protected_attr] == group
            favorable_mask = df[outcome_col] == favorable_outcome
            rate = (group_mask & favorable_mask).sum() / group_mask.sum()
            group_rates[group] = rate
            
        # Find privileged group (highest rate) and unprivileged
        privileged_group = max(group_rates, key=group_rates.get)
        privileged_rate = group_rates[privileged_group]
        
        # Compute disparate impact for each unprivileged group
        disparate_impacts = {}
        for group, rate in group_rates.items():
            if group != privileged_group and privileged_rate > 0:
                di_ratio = rate / privileged_rate
                disparate_impacts[f"{group}_vs_{privileged_group}"] = {
                    "ratio": di_ratio,
                    "unprivileged_rate": rate,
                    "privileged_rate": privileged_rate,
                    "passes_80_rule": di_ratio >= 0.8,
                }
                
        return {
            "protected_attribute": protected_attr,
            "privileged_group": privileged_group,
            "group_rates": group_rates,
            "disparate_impacts": disparate_impacts,
        }
    
    def compute_demographic_parity(
        self,
        df: pd.DataFrame,
        outcome_col: str,
        protected_attr: str
    ) -> Dict[str, float]:
        """
        Compute demographic parity metrics.
        
        Demographic parity requires that the probability of favorable outcome
        is the same across all groups.
        
        Args:
            df: DataFrame with data
            outcome_col: Name of the outcome column
            protected_attr: Name of the protected attribute column
            
        Returns:
            Dictionary with demographic parity metrics
        """
        if protected_attr not in df.columns:
            return {}
            
        groups = df[protected_attr].unique()
        group_rates = {}
        
        for group in groups:
            group_mask = df[protected_attr] == group
            rate = df.loc[group_mask, outcome_col].mean()
            group_rates[group] = rate
            
        # Calculate parity difference (max - min)
        max_rate = max(group_rates.values())
        min_rate = min(group_rates.values())
        parity_difference = max_rate - min_rate
        
        return {
            "protected_attribute": protected_attr,
            "group_rates": group_rates,
            "parity_difference": parity_difference,
            "satisfies_parity": parity_difference < 0.1,  # 10% threshold
        }
    
    def compute_reweighting(
        self,
        df: pd.DataFrame,
        outcome_col: str,
        protected_attr: str,
        favorable_outcome: Any = 1
    ) -> np.ndarray:
        """
        Compute sample weights to mitigate bias through reweighting.
        
        This implements the reweighting pre-processing technique that adjusts
        sample weights to ensure demographic parity.
        
        Args:
            df: DataFrame with data
            outcome_col: Name of the outcome column
            protected_attr: Name of the protected attribute column
            favorable_outcome: Value representing favorable outcome
            
        Returns:
            Array of sample weights
        """
        n = len(df)
        weights = np.ones(n)
        
        if protected_attr not in df.columns:
            logger.warning(f"Protected attribute '{protected_attr}' not found")
            return weights
            
        # Calculate expected probabilities under fairness
        groups = df[protected_attr].unique()
        favorable_mask = df[outcome_col] == favorable_outcome
        
        # Overall favorable rate
        p_favorable = favorable_mask.mean()
        p_unfavorable = 1 - p_favorable
        
        for group in groups:
            group_mask = df[protected_attr] == group
            p_group = group_mask.mean()
            
            # Joint probabilities (observed)
            p_group_favorable = (group_mask & favorable_mask).mean()
            p_group_unfavorable = (group_mask & ~favorable_mask).mean()
            
            # Expected joint probabilities under independence
            expected_favorable = p_group * p_favorable
            expected_unfavorable = p_group * p_unfavorable
            
            # Compute weights
            if p_group_favorable > 0:
                weights[group_mask & favorable_mask] = expected_favorable / p_group_favorable
            if p_group_unfavorable > 0:
                weights[group_mask & ~favorable_mask] = expected_unfavorable / p_group_unfavorable
                
        # Normalize weights to sum to n
        weights = weights * (n / weights.sum())
        
        self.sample_weights = weights
        return weights
    
    def generate_fairness_audit(
        self,
        df: pd.DataFrame,
        outcome_col: str
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive fairness audit report.
        
        Args:
            df: DataFrame with data
            outcome_col: Name of the outcome column
            
        Returns:
            Dictionary containing the full fairness audit
        """
        audit = {
            "n_samples": len(df),
            "protected_attributes": self.protected_attributes,
            "outcome_distribution": df[outcome_col].value_counts().to_dict(),
            "attribute_analyses": {},
        }
        
        for attr in self.protected_attributes:
            if attr not in df.columns:
                continue
                
            attr_analysis = {
                "disparate_impact": self.compute_disparate_impact(
                    df, outcome_col, attr
                ),
                "demographic_parity": self.compute_demographic_parity(
                    df, outcome_col, attr
                ),
                "group_sizes": df[attr].value_counts().to_dict(),
            }
            
            audit["attribute_analyses"][attr] = attr_analysis
            
        # Overall fairness summary
        audit["summary"] = self._summarize_fairness(audit)
        
        self.fairness_metrics = audit
        return audit
    
    def _summarize_fairness(self, audit: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize fairness audit results."""
        issues = []
        warnings = []
        
        for attr, analysis in audit["attribute_analyses"].items():
            # Check disparate impact
            di_data = analysis.get("disparate_impact", {})
            for comparison, metrics in di_data.get("disparate_impacts", {}).items():
                if not metrics.get("passes_80_rule", True):
                    issues.append(
                        f"Disparate impact violation for {attr}: "
                        f"{comparison} has ratio {metrics['ratio']:.3f}"
                    )
                elif metrics.get("ratio", 1.0) < 0.9:
                    warnings.append(
                        f"Near disparate impact for {attr}: "
                        f"{comparison} has ratio {metrics['ratio']:.3f}"
                    )
                    
            # Check demographic parity
            dp_data = analysis.get("demographic_parity", {})
            if not dp_data.get("satisfies_parity", True):
                warnings.append(
                    f"Demographic parity gap for {attr}: "
                    f"difference = {dp_data.get('parity_difference', 0):.3f}"
                )
                
        return {
            "n_issues": len(issues),
            "n_warnings": len(warnings),
            "issues": issues,
            "warnings": warnings,
            "overall_status": "PASS" if len(issues) == 0 else "FAIL",
        }
    
    def check_counterfactual_fairness(
        self,
        original: pd.DataFrame,
        counterfactual: pd.DataFrame,
        protected_attr: str
    ) -> Dict[str, Any]:
        """
        Check if counterfactual recommendations are fair across groups.
        
        Verifies that the "effort" required to achieve favorable outcomes
        is similar across protected groups.
        
        Args:
            original: Original sample features
            counterfactual: Counterfactual features
            protected_attr: Protected attribute to check
            
        Returns:
            Dictionary with counterfactual fairness metrics
        """
        if protected_attr not in original.columns:
            return {}
            
        # Calculate change magnitude for each sample
        numerical_cols = original.select_dtypes(include=[np.number]).columns
        changes = (counterfactual[numerical_cols] - original[numerical_cols]).abs()
        total_change = changes.sum(axis=1)
        
        # Compare average change across groups
        groups = original[protected_attr].unique()
        group_changes = {}
        
        for group in groups:
            group_mask = original[protected_attr] == group
            group_changes[group] = total_change[group_mask].mean()
            
        # Calculate fairness metrics
        max_change = max(group_changes.values())
        min_change = min(group_changes.values())
        
        return {
            "protected_attribute": protected_attr,
            "group_average_changes": group_changes,
            "change_disparity": max_change - min_change,
            "disparity_ratio": min_change / max_change if max_change > 0 else 1.0,
            "is_fair": (max_change - min_change) < 0.5 * min_change,
        }
