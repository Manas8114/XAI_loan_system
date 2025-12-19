"""
Explanation Report Module

Generates comprehensive explanation reports combining SHAP diagnostics
with counterfactual recommendations.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)


class ExplanationReport:
    """
    Generates structured explanation reports for loan decisions.
    
    Combines:
    - Model prediction and confidence
    - SHAP-based diagnostic insights
    - Counterfactual recommendations (primary)
    - Actionability assessment
    """
    
    def __init__(self):
        """Initialize the explanation report generator."""
        self.report: Dict[str, Any] = {}
        
    def generate(
        self,
        prediction: Dict[str, Any],
        shap_explanation: Dict[str, Any],
        counterfactuals: List[Dict[str, Any]],
        actionability_scores: List[float],
        original_features: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive explanation report.
        
        Args:
            prediction: Model prediction results
            shap_explanation: SHAP-based explanation
            counterfactuals: List of counterfactual alternatives
            actionability_scores: Actionability score for each CF
            original_features: Original input features
            
        Returns:
            Complete explanation report
        """
        self.report = {
            "timestamp": datetime.now().isoformat(),
            "summary": self._generate_summary(prediction),
            "prediction_details": prediction,
            "diagnostic_explanation": self._format_shap_explanation(shap_explanation),
            "action_recommendations": self._format_counterfactuals(
                counterfactuals, actionability_scores, original_features
            ),
            "interpretation_guide": self._generate_interpretation_guide(),
        }
        
        return self.report
    
    def _generate_summary(self, prediction: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive summary of the decision."""
        approved = prediction.get("approved", False)
        confidence = prediction.get("confidence", 0)
        
        if approved:
            status = "APPROVED"
            message = "Your loan application meets the approval criteria."
        else:
            status = "NEEDS IMPROVEMENT"
            message = "Your application requires some improvements for approval."
            
        return {
            "status": status,
            "message": message,
            "confidence_level": self._confidence_to_text(confidence),
            "interest_rate_band": prediction.get("interest_rate_band", "Unknown"),
        }
    
    def _confidence_to_text(self, confidence: float) -> str:
        """Convert confidence score to human-readable text."""
        if confidence >= 0.9:
            return "Very High"
        elif confidence >= 0.75:
            return "High"
        elif confidence >= 0.6:
            return "Moderate"
        elif confidence >= 0.4:
            return "Low"
        else:
            return "Very Low"
    
    def _format_shap_explanation(
        self,
        shap_explanation: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Format SHAP explanation for the report."""
        top_features = shap_explanation.get("top_features", [])
        
        # Categorize features by impact direction
        positive_factors = []
        negative_factors = []
        
        for feature in top_features:
            feature_info = {
                "name": feature["feature"],
                "value": feature["value"],
                "impact": abs(feature["shap_value"]),
            }
            
            if feature["shap_value"] > 0:
                positive_factors.append(feature_info)
            else:
                negative_factors.append(feature_info)
                
        return {
            "role": "DIAGNOSTIC (for understanding, not prescription)",
            "positive_factors": positive_factors[:5],
            "negative_factors": negative_factors[:5],
            "key_insight": self._generate_shap_insight(positive_factors, negative_factors),
        }
    
    def _generate_shap_insight(
        self,
        positive: List[Dict],
        negative: List[Dict]
    ) -> str:
        """Generate a natural language insight from SHAP values."""
        if not positive and not negative:
            return "Unable to determine key factors."
            
        insights = []
        
        if positive:
            top_pos = positive[0]["name"]
            insights.append(f"Your {top_pos} positively contributed to your application.")
            
        if negative:
            top_neg = negative[0]["name"]
            insights.append(f"Your {top_neg} is an area that could be improved.")
            
        return " ".join(insights)
    
    def _format_counterfactuals(
        self,
        counterfactuals: List[Dict[str, Any]],
        actionability_scores: List[float],
        original: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """Format counterfactual recommendations."""
        if not counterfactuals:
            return []
            
        recommendations = []
        
        for i, (cf, score) in enumerate(zip(counterfactuals, actionability_scores)):
            # Extract changes
            changes = []
            for feature, new_value in cf.items():
                if feature in original.columns:
                    old_value = original[feature].iloc[0]
                    if abs(new_value - old_value) > 0.01:
                        change_pct = ((new_value - old_value) / (abs(old_value) + 1e-10)) * 100
                        changes.append({
                            "feature": feature,
                            "current_value": round(old_value, 2),
                            "recommended_value": round(new_value, 2),
                            "change_direction": "increase" if new_value > old_value else "decrease",
                            "change_percent": round(change_pct, 1),
                        })
                        
            recommendation = {
                "option_number": i + 1,
                "actionability_score": round(score, 3),
                "actionability_level": self._score_to_level(score),
                "required_changes": changes,
                "n_changes": len(changes),
                "estimated_outcome": "Approved" if cf.get("predicted_outcome", 1) == 1 else "Improved Rate",
            }
            
            recommendations.append(recommendation)
            
        # Sort by actionability score
        recommendations.sort(key=lambda x: x["actionability_score"], reverse=True)
        
        return recommendations
    
    def _score_to_level(self, score: float) -> str:
        """Convert actionability score to level."""
        if score >= 0.7:
            return "Highly Actionable"
        elif score >= 0.5:
            return "Moderately Actionable"
        elif score >= 0.3:
            return "Challenging"
        else:
            return "Difficult"
    
    def _generate_interpretation_guide(self) -> Dict[str, str]:
        """Generate interpretation guidance."""
        return {
            "diagnostic_explanation": (
                "The diagnostic section shows which factors most influenced "
                "the model's decision. This helps you understand WHY the "
                "decision was made, but does not prescribe specific actions."
            ),
            "action_recommendations": (
                "The recommendations section provides specific, actionable "
                "changes you could make to improve your outcome. Each option "
                "shows an Actionability Score indicating how realistic the "
                "changes are to achieve."
            ),
            "actionability_scores": (
                "Scores range from 0 to 1, where higher scores indicate "
                "changes that are more feasible to achieve in practice. "
                "Highly Actionable (0.7+) options are recommended."
            ),
        }
    
    def to_json(self) -> str:
        """Export report as JSON string."""
        return json.dumps(self.report, indent=2, default=str)
    
    def to_markdown(self) -> str:
        """Export report as Markdown."""
        md = []
        
        md.append("# Loan Application Explanation Report\n")
        md.append(f"*Generated: {self.report.get('timestamp', 'Unknown')}*\n")
        
        # Summary
        summary = self.report.get("summary", {})
        md.append(f"## Summary\n")
        md.append(f"**Status:** {summary.get('status', 'Unknown')}\n")
        md.append(f"**Message:** {summary.get('message', '')}\n")
        md.append(f"**Confidence:** {summary.get('confidence_level', '')}\n\n")
        
        # Diagnostic
        diagnostic = self.report.get("diagnostic_explanation", {})
        md.append("## Diagnostic Analysis (Understanding the Decision)\n")
        md.append(f"*{diagnostic.get('role', '')}*\n\n")
        
        if diagnostic.get("positive_factors"):
            md.append("### Positive Factors\n")
            for f in diagnostic["positive_factors"]:
                md.append(f"- **{f['name']}**: {f['value']} (Impact: {f['impact']:.3f})\n")
            md.append("\n")
            
        if diagnostic.get("negative_factors"):
            md.append("### Areas for Improvement\n")
            for f in diagnostic["negative_factors"]:
                md.append(f"- **{f['name']}**: {f['value']} (Impact: {f['impact']:.3f})\n")
            md.append("\n")
            
        # Recommendations
        recommendations = self.report.get("action_recommendations", [])
        if recommendations:
            md.append("## Recommended Actions\n")
            for rec in recommendations:
                md.append(f"### Option {rec['option_number']} ")
                md.append(f"(Actionability: {rec['actionability_level']})\n")
                md.append(f"**Score:** {rec['actionability_score']}\n\n")
                
                for change in rec["required_changes"]:
                    md.append(f"- {change['feature']}: {change['current_value']} → ")
                    md.append(f"{change['recommended_value']} ")
                    md.append(f"({change['change_direction']} {abs(change['change_percent'])}%)\n")
                md.append("\n")
                
        return "".join(md)
