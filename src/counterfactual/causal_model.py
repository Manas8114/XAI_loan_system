"""
Causal Model Module

Defines the Structural Causal Model (SCM) for loan features.
This is the foundation for our PRIMARY contribution: causal counterfactuals.
"""

import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Any, Optional, Tuple, Callable
import logging
from dataclasses import dataclass, field

from ..config import CAUSAL_EDGES, STRUCTURAL_EQUATIONS

logger = logging.getLogger(__name__)


@dataclass
class StructuralEquation:
    """Represents a structural equation for a variable."""
    variable: str
    parents: List[str]
    coefficients: Dict[str, float]
    intercept: float = 0.0
    noise_std: float = 0.0
    transform: Optional[Callable] = None  # Optional non-linear transform
    
    def evaluate(
        self,
        parent_values: Dict[str, float],
        noise: float = 0.0
    ) -> float:
        """Evaluate the structural equation."""
        value = self.intercept
        
        for parent, coef in self.coefficients.items():
            if parent in parent_values:
                value += coef * parent_values[parent]
                
        value += noise * self.noise_std
        
        if self.transform is not None:
            value = self.transform(value)
            
        return value


class CausalModel:
    """
    Structural Causal Model for loan features.
    
    Defines:
    - Causal graph (DAG) between features
    - Structural equations relating variables
    - Intervention simulation
    - Counterfactual validity checking
    
    This is the core component for our causal counterfactual approach.
    """
    
    def __init__(
        self,
        edges: Optional[List[Tuple[str, str]]] = None,
        structural_equations: Optional[Dict[str, Dict]] = None
    ):
        """
        Initialize the causal model.
        
        Args:
            edges: List of (parent, child) tuples defining causal relationships
            structural_equations: Dict of variable -> {parent: coef, intercept: val}
        """
        self.edges = edges or CAUSAL_EDGES
        self.graph = nx.DiGraph()
        self.graph.add_edges_from(self.edges)
        
        # Validate DAG (no cycles)
        if not nx.is_directed_acyclic_graph(self.graph):
            raise ValueError("Causal graph contains cycles - must be a DAG")
            
        # Store structural equations
        self.equations: Dict[str, StructuralEquation] = {}
        self._build_structural_equations(structural_equations or STRUCTURAL_EQUATIONS)
        
        # Compute topological order for propagation
        self.topological_order = list(nx.topological_sort(self.graph))
        
        logger.info(f"Causal model initialized with {len(self.edges)} edges")
        
    def _build_structural_equations(self, eq_dict: Dict[str, Dict]):
        """Build StructuralEquation objects from dictionary."""
        for variable, params in eq_dict.items():
            parents = list(self.graph.predecessors(variable))
            coefficients = {k: v for k, v in params.items() if k != "intercept"}
            intercept = params.get("intercept", 0.0)
            
            self.equations[variable] = StructuralEquation(
                variable=variable,
                parents=parents,
                coefficients=coefficients,
                intercept=intercept
            )
            
    def get_parents(self, node: str) -> List[str]:
        """Get parent nodes (direct causes) of a node."""
        if node not in self.graph:
            return []
        return list(self.graph.predecessors(node))
    
    def get_children(self, node: str) -> List[str]:
        """Get child nodes (direct effects) of a node."""
        if node not in self.graph:
            return []
        return list(self.graph.successors(node))
    
    def get_ancestors(self, node: str) -> List[str]:
        """Get all ancestor nodes (indirect causes) of a node."""
        if node not in self.graph:
            return []
        return list(nx.ancestors(self.graph, node))
    
    def get_descendants(self, node: str) -> List[str]:
        """Get all descendant nodes (indirect effects) of a node."""
        if node not in self.graph:
            return []
        return list(nx.descendants(self.graph, node))
    
    def intervene(
        self,
        observation: Dict[str, float],
        interventions: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Perform do-calculus intervention and propagate effects.
        
        This simulates do(X=x) - setting a variable to a value and
        computing downstream effects through structural equations.
        
        Args:
            observation: Original observed values
            interventions: Dict of variable -> new_value to intervene on
            
        Returns:
            Dictionary with all variable values after intervention
        """
        result = observation.copy()
        
        # Apply interventions
        for var, value in interventions.items():
            result[var] = value
            
        # Propagate effects in topological order
        for var in self.topological_order:
            if var in interventions:
                # Intervened variable - keep the intervention value
                continue
                
            if var in self.equations:
                # Compute value from parents
                parent_values = {p: result.get(p, 0) for p in self.equations[var].parents}
                result[var] = self.equations[var].evaluate(parent_values)
                
        return result
    
    def validate_counterfactual(
        self,
        original: Dict[str, float],
        counterfactual: Dict[str, float],
        tolerance: float = 0.1
    ) -> Dict[str, Any]:
        """
        Validate if a counterfactual is causally consistent.
        
        Checks whether the counterfactual respects the causal graph by:
        1. Identifying which features changed (interventions)
        2. Simulating the interventions through the SCM
        3. Checking if the CF values match the simulated values
        
        Args:
            original: Original observation
            counterfactual: Proposed counterfactual
            tolerance: Relative tolerance for value comparison
            
        Returns:
            Validation result with details
        """
        # Identify changes (interventions)
        interventions = {}
        for var in original:
            if var in counterfactual:
                orig_val = original[var]
                cf_val = counterfactual[var]
                if abs(cf_val - orig_val) > tolerance * max(abs(orig_val), 1):
                    interventions[var] = cf_val
                    
        if not interventions:
            return {
                "valid": True,
                "reason": "No changes detected",
                "interventions": {},
                "violations": [],
            }
            
        # Find root interventions (those without causal parents among interventions)
        root_interventions = {}
        for var in interventions:
            ancestor_interventions = [
                a for a in self.get_ancestors(var) if a in interventions
            ]
            if not ancestor_interventions:
                root_interventions[var] = interventions[var]
                
        # Simulate from root interventions
        simulated = self.intervene(original, root_interventions)
        
        # Check for violations
        violations = []
        for var, cf_val in counterfactual.items():
            if var in simulated:
                sim_val = simulated[var]
                if abs(cf_val - sim_val) > tolerance * max(abs(sim_val), 1):
                    violations.append({
                        "variable": var,
                        "counterfactual_value": cf_val,
                        "expected_value": sim_val,
                        "difference": abs(cf_val - sim_val),
                    })
                    
        return {
            "valid": len(violations) == 0,
            "reason": "Causally consistent" if not violations else "Causal violations detected",
            "interventions": interventions,
            "root_interventions": root_interventions,
            "violations": violations,
            "simulated_values": simulated,
        }
    
    def estimate_from_data(
        self,
        data: pd.DataFrame,
        target_variables: Optional[List[str]] = None
    ):
        """
        Estimate structural equation coefficients from data.
        
        Uses linear regression to estimate coefficients for each
        endogenous variable based on its parents.
        
        Args:
            data: DataFrame with observed data
            target_variables: Variables to estimate equations for
                              (defaults to all endogenous variables)
        """
        from sklearn.linear_model import LinearRegression
        
        if target_variables is None:
            target_variables = [v for v in self.graph.nodes() 
                               if self.graph.in_degree(v) > 0]
            
        for var in target_variables:
            parents = self.get_parents(var)
            
            if not parents or var not in data.columns:
                continue
                
            available_parents = [p for p in parents if p in data.columns]
            
            if not available_parents:
                continue
                
            X = data[available_parents].dropna()
            y = data.loc[X.index, var]
            
            # Fit linear regression
            reg = LinearRegression()
            reg.fit(X, y)
            
            # Store coefficients
            coefficients = dict(zip(available_parents, reg.coef_))
            
            self.equations[var] = StructuralEquation(
                variable=var,
                parents=parents,
                coefficients=coefficients,
                intercept=float(reg.intercept_)
            )
            
        logger.info(f"Estimated structural equations for {len(target_variables)} variables")
        
    def to_adjacency_matrix(self) -> Tuple[np.ndarray, List[str]]:
        """Convert graph to adjacency matrix."""
        nodes = list(self.graph.nodes())
        matrix = nx.to_numpy_array(self.graph, nodelist=nodes)
        return matrix, nodes
    
    def visualize_graph(self) -> Dict[str, Any]:
        """Get graph data for visualization."""
        return {
            "nodes": list(self.graph.nodes()),
            "edges": list(self.graph.edges()),
            "node_degrees": dict(self.graph.degree()),
            "topological_order": self.topological_order,
        }
    
    def get_causal_path(self, source: str, target: str) -> List[List[str]]:
        """Get all causal paths from source to target."""
        if source not in self.graph or target not in self.graph:
            return []
            
        try:
            paths = list(nx.all_simple_paths(self.graph, source, target))
            return paths
        except nx.NetworkXNoPath:
            return []
