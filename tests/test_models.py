"""
Tests for ML Models
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import DataIngestionService, DataPreprocessor
from src.models import XGBoostLoanModel, RandomForestLoanModel


@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    ingestion = DataIngestionService()
    data = ingestion.load_data(use_synthetic=True, sample_size=1000)
    
    preprocessor = DataPreprocessor()
    X, y = preprocessor.fit_transform(data)
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(X, y)
    
    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "preprocessor": preprocessor,
    }


class TestXGBoostModel:
    """Tests for XGBoost model."""
    
    def test_model_initialization(self):
        """Test model can be initialized."""
        model = XGBoostLoanModel(task="approval")
        assert model.task == "approval"
        assert model._fitted is False
        
    def test_model_training(self, sample_data):
        """Test model training."""
        model = XGBoostLoanModel(task="approval")
        model.fit(sample_data["X_train"], sample_data["y_train"], verbose=False)
        
        assert model._fitted is True
        assert len(model.feature_names) > 0
        
    def test_model_prediction(self, sample_data):
        """Test model predictions."""
        model = XGBoostLoanModel(task="approval")
        model.fit(sample_data["X_train"], sample_data["y_train"], verbose=False)
        
        predictions = model.predict(sample_data["X_test"])
        
        assert len(predictions) == len(sample_data["X_test"])
        assert set(predictions).issubset({0, 1})
        
    def test_model_probabilities(self, sample_data):
        """Test probability predictions."""
        model = XGBoostLoanModel(task="approval")
        model.fit(sample_data["X_train"], sample_data["y_train"], verbose=False)
        
        probas = model.predict_proba(sample_data["X_test"])
        
        assert probas.shape == (len(sample_data["X_test"]), 2)
        assert np.allclose(probas.sum(axis=1), 1.0)
        
    def test_model_evaluation(self, sample_data):
        """Test model evaluation."""
        model = XGBoostLoanModel(task="approval")
        model.fit(sample_data["X_train"], sample_data["y_train"], verbose=False)
        
        metrics = model.evaluate(sample_data["X_test"], sample_data["y_test"])
        
        assert "roc_auc" in metrics
        assert "accuracy" in metrics
        assert metrics["roc_auc"] >= 0.5  # Better than random
        
    def test_feature_importance(self, sample_data):
        """Test feature importance extraction."""
        model = XGBoostLoanModel(task="approval")
        model.fit(sample_data["X_train"], sample_data["y_train"], verbose=False)
        
        importance = model.get_feature_importance()
        
        assert len(importance) > 0
        assert "feature" in importance.columns
        assert "importance" in importance.columns


class TestRandomForestModel:
    """Tests for Random Forest model."""
    
    def test_model_training(self, sample_data):
        """Test RF model training."""
        model = RandomForestLoanModel(task="approval")
        model.fit(sample_data["X_train"], sample_data["y_train"])
        
        assert model._fitted is True
        
    def test_model_prediction(self, sample_data):
        """Test RF predictions."""
        model = RandomForestLoanModel(task="approval")
        model.fit(sample_data["X_train"], sample_data["y_train"])
        
        predictions = model.predict(sample_data["X_test"])
        
        assert len(predictions) == len(sample_data["X_test"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
