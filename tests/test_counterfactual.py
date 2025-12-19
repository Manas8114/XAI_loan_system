"""
Tests for Counterfactual Engines
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import DataIngestionService, DataPreprocessor
from src.models import XGBoostLoanModel
from src.counterfactual import CausalModel, CausalCFEngine, OptimizationCFEngine
from src.ethics import ActionabilityScorer


@pytest.fixture
def trained_model():
    """Create a trained model for testing."""
    ingestion = DataIngestionService()
    data = ingestion.load_data(use_synthetic=True, sample_size=1000)
    
    preprocessor = DataPreprocessor()
    X, y = preprocessor.fit_transform(data)
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(X, y)
    
    model = XGBoostLoanModel(task="approval")
    model.fit(X_train, y_train, verbose=False)
    
    return {
        "model": model,
        "X_train": X_train,
        "X_test": X_test,
        "feature_names": list(X_train.columns),
        "data": data,
    }


class TestCausalModel:
    """Tests for Structural Causal Model."""
    
    def test_initialization(self):
        """Test SCM initialization."""
        scm = CausalModel()
        
        assert len(scm.edges) > 0
        assert len(scm.topological_order) > 0
        
    def test_get_parents_children(self):
        """Test graph traversal methods."""
        scm = CausalModel()
        
        # annual_inc has parents (emp_length)
        parents = scm.get_parents("annual_inc")
        assert isinstance(parents, list)
        
        # dti has children (loan_status)
        children = scm.get_children("dti")
        assert isinstance(children, list)
        
    def test_intervene(self):
        """Test intervention simulation."""
        scm = CausalModel()
        
        original = {
            "emp_length": 5,
            "annual_inc": 60000,
            "dti": 20,
            "fico_score": 680,
        }
        
        interventions = {"emp_length": 10}
        
        result = scm.intervene(original, interventions)
        
        assert result["emp_length"] == 10  # Intervention value
        # annual_inc should be affected if in equations
        assert "annual_inc" in result
        
    def test_validate_counterfactual(self):
        """Test counterfactual validation."""
        scm = CausalModel()
        
        original = {"emp_length": 5, "annual_inc": 60000, "dti": 20}
        counterfactual = {"emp_length": 10, "annual_inc": 72500, "dti": 18}
        
        validation = scm.validate_counterfactual(original, counterfactual)
        
        assert "valid" in validation
        assert "interventions" in validation


class TestOptimizationCFEngine:
    """Tests for optimization-based counterfactual engine."""
    
    def test_initialization(self, trained_model):
        """Test engine initialization."""
        engine = OptimizationCFEngine(
            model=trained_model["model"],
            feature_names=trained_model["feature_names"],
        )
        
        assert engine.model is not None
        assert len(engine.feature_names) > 0
        
    def test_generate_counterfactuals(self, trained_model):
        """Test counterfactual generation."""
        engine = OptimizationCFEngine(
            model=trained_model["model"],
            feature_names=trained_model["feature_names"],
        )
        
        instance = trained_model["X_test"].iloc[[0]]
        
        results = engine.generate_counterfactuals(
            instance,
            num_cfs=3,
            desired_class=1,
        )
        
        assert "success" in results
        assert "counterfactuals" in results
        assert "original" in results


class TestCausalCFEngine:
    """Tests for causal counterfactual engine."""
    
    def test_initialization(self, trained_model):
        """Test engine initialization."""
        scm = CausalModel()
        scorer = ActionabilityScorer()
        
        engine = CausalCFEngine(
            model=trained_model["model"],
            causal_model=scm,
            feature_names=trained_model["feature_names"],
            actionability_scorer=scorer,
        )
        
        assert engine.model is not None
        assert engine.causal_model is not None
        
    def test_generate_counterfactuals(self, trained_model):
        """Test causal counterfactual generation."""
        scm = CausalModel()
        # Use preprocessed data with numeric values for estimation
        X_with_target = trained_model["X_train"].copy()
        X_with_target['loan_status'] = 0  # Dummy target values
        scm.estimate_from_data(X_with_target)
        
        scorer = ActionabilityScorer()
        
        engine = CausalCFEngine(
            model=trained_model["model"],
            causal_model=scm,
            feature_names=trained_model["feature_names"],
            actionability_scorer=scorer,
        )
        
        instance = trained_model["X_test"].iloc[[0]]
        
        results = engine.generate_counterfactuals(
            instance,
            num_cfs=3,
        )
        
        assert "success" in results
        assert "counterfactuals" in results
        
        if results["success"] and results["counterfactuals"]:
            cf = results["counterfactuals"][0]
            assert "actionability_score" in cf
            assert "causal_valid" in cf


class TestActionabilityScorer:
    """Tests for actionability scoring."""
    
    def test_score_calculation(self):
        """Test actionability score calculation."""
        scorer = ActionabilityScorer()
        
        original = {"annual_inc": 60000, "dti": 25, "fico_score": 650}
        counterfactual = {"annual_inc": 70000, "dti": 20, "fico_score": 700}
        
        result = scorer.score(original, counterfactual)
        
        assert "total_score" in result
        assert "level" in result
        assert "feature_scores" in result
        assert 0 <= result["total_score"] <= 1
        
    def test_improvement_suggestions(self):
        """Test improvement suggestion generation."""
        scorer = ActionabilityScorer()
        
        original = {"annual_inc": 60000, "dti": 25}
        counterfactual = {"annual_inc": 70000, "dti": 18}
        
        suggestions = scorer.get_improvement_suggestions(original, counterfactual)
        
        assert isinstance(suggestions, list)
        if suggestions:
            assert "feature" in suggestions[0]
            assert "action" in suggestions[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
