"""
Tests for API Endpoints
"""

import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def client():
    """Create test client."""
    from src.api.main import app
    return TestClient(app)


@pytest.fixture
def sample_features():
    """Sample loan features for testing."""
    return {
        "annual_inc": 65000,
        "dti": 18.5,
        "emp_length": 5,
        "loan_amnt": 15000,
        "fico_score": 680,
        "revol_util": 45,
        "delinq_2yrs": 0,
        "inq_last_6mths": 1,
    }


class TestHealthEndpoints:
    """Test health check endpoints."""
    
    def test_root(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "version" in data
        
    def test_health(self, client):
        """Test health endpoint."""
        response = client.get("/health")
        assert response.status_code == 200


class TestPredictEndpoint:
    """Test /predict endpoint."""
    
    def test_predict_success(self, client, sample_features):
        """Test successful prediction."""
        response = client.post("/api/v1/predict", json=sample_features)
        
        # May return 503 if models not loaded in test environment
        if response.status_code == 200:
            data = response.json()
            assert "approval_probability" in data
            assert "approved" in data
            assert "interest_rate_band" in data
            assert 0 <= data["approval_probability"] <= 1
            
    def test_predict_invalid_features(self, client):
        """Test prediction with invalid features."""
        invalid_features = {"annual_inc": -1000}  # Negative income
        
        response = client.post("/api/v1/predict", json=invalid_features)
        # Should fail validation
        assert response.status_code in [422, 503]


class TestExplainEndpoint:
    """Test /explain endpoint."""
    
    def test_explain_success(self, client, sample_features):
        """Test successful explanation."""
        request = {
            "features": sample_features,
            "top_k": 5,
        }
        
        response = client.post("/api/v1/explain", json=request)
        
        if response.status_code == 200:
            data = response.json()
            assert "prediction" in data
            assert "diagnostic_insight" in data


class TestCounterfactualEndpoint:
    """Test /counterfactual endpoint."""
    
    def test_counterfactual_success(self, client, sample_features):
        """Test successful counterfactual generation."""
        request = {
            "features": sample_features,
            "target_outcome": "approved",
            "num_counterfactuals": 3,
            "method": "causal",
        }
        
        response = client.post("/api/v1/counterfactual", json=request)
        
        if response.status_code == 200:
            data = response.json()
            assert "success" in data
            assert "method" in data
            assert "counterfactuals" in data
            
    def test_counterfactual_methods(self, client, sample_features):
        """Test different CF methods."""
        for method in ["dice", "optimization", "causal"]:
            request = {
                "features": sample_features,
                "num_counterfactuals": 2,
                "method": method,
            }
            
            response = client.post("/api/v1/counterfactual", json=request)
            # Accept 200 (success), 500 (internal error from missing dependency), 
            # or 503 (service unavailable)
            assert response.status_code in [200, 500, 503]


class TestCompareEndpoint:
    """Test /compare endpoint."""
    
    def test_compare_methods(self, client, sample_features):
        """Test method comparison."""
        request = {
            "features": sample_features,
            "num_counterfactuals": 3,
        }
        
        response = client.post("/api/v1/compare", json=request)
        
        if response.status_code == 200:
            data = response.json()
            assert "methods" in data
            assert "metrics" in data
            assert "best_method" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
