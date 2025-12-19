"""
Model Registry Module

Centralized management for multiple models (approval, interest rate).
Handles model loading, saving, and selection.
"""

import logging
from typing import Dict, Optional, Any, Union
from pathlib import Path
import json

from .xgboost_model import XGBoostLoanModel
from .random_forest import RandomForestLoanModel
from ..config import MODELS_DIR

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Registry for managing multiple trained models.
    
    Provides:
    - Centralized model storage
    - Easy model switching
    - Model comparison utilities
    """
    
    def __init__(self):
        """Initialize the model registry."""
        self.models: Dict[str, Union[XGBoostLoanModel, RandomForestLoanModel]] = {}
        self.default_model_type = "xgboost"
        self.metadata: Dict[str, Any] = {}
        
    def register(
        self,
        name: str,
        model: Union[XGBoostLoanModel, RandomForestLoanModel]
    ):
        """
        Register a trained model.
        
        Args:
            name: Unique name for the model (e.g., "xgboost_approval")
            model: Trained model instance
        """
        self.models[name] = model
        logger.info(f"Registered model: {name}")
        
    def get(
        self,
        name: str
    ) -> Union[XGBoostLoanModel, RandomForestLoanModel]:
        """
        Get a registered model by name.
        
        Args:
            name: Name of the model to retrieve
            
        Returns:
            The requested model
        """
        if name not in self.models:
            raise KeyError(f"Model '{name}' not found. Available: {list(self.models.keys())}")
        return self.models[name]
    
    def get_approval_model(
        self,
        model_type: Optional[str] = None
    ) -> Union[XGBoostLoanModel, RandomForestLoanModel]:
        """Get the loan approval model."""
        model_type = model_type or self.default_model_type
        name = f"{model_type}_approval"
        return self.get(name)
    
    def get_interest_model(
        self,
        model_type: Optional[str] = None
    ) -> Union[XGBoostLoanModel, RandomForestLoanModel]:
        """Get the interest rate prediction model."""
        model_type = model_type or self.default_model_type
        name = f"{model_type}_interest_rate"
        return self.get(name)
    
    def list_models(self) -> Dict[str, Dict[str, Any]]:
        """List all registered models with their info."""
        return {
            name: model.get_model_info() if hasattr(model, 'get_model_info') else {"name": name}
            for name, model in self.models.items()
        }
    
    def compare_models(
        self,
        task: str,
        X_test,
        y_test
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compare all models for a specific task.
        
        Args:
            task: Either "approval" or "interest_rate"
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary with evaluation metrics for each model
        """
        comparisons = {}
        
        for name, model in self.models.items():
            if task in name:
                try:
                    metrics = model.evaluate(X_test, y_test)
                    comparisons[name] = {
                        "roc_auc": metrics.get("roc_auc"),
                        "accuracy": metrics.get("accuracy"),
                        "f1": metrics.get("f1"),
                    }
                except Exception as e:
                    logger.error(f"Error evaluating {name}: {e}")
                    comparisons[name] = {"error": str(e)}
                    
        return comparisons
    
    def save_all(self, directory: Optional[Path] = None):
        """Save all registered models."""
        directory = directory or MODELS_DIR
        directory.mkdir(parents=True, exist_ok=True)
        
        for name, model in self.models.items():
            model_path = directory / f"{name}.joblib"
            model.save(model_path)
            
        # Save registry metadata
        meta_path = directory / "registry_metadata.json"
        with open(meta_path, "w") as f:
            json.dump({
                "models": list(self.models.keys()),
                "default_model_type": self.default_model_type,
            }, f, indent=2)
            
        logger.info(f"Saved {len(self.models)} models to {directory}")
        
    def load_all(self, directory: Optional[Path] = None):
        """Load all models from directory."""
        directory = directory or MODELS_DIR
        
        # Load registry metadata
        meta_path = directory / "registry_metadata.json"
        if meta_path.exists():
            with open(meta_path, "r") as f:
                meta = json.load(f)
                self.default_model_type = meta.get("default_model_type", "xgboost")
                model_names = meta.get("models", [])
        else:
            # Discover models from files
            model_names = [p.stem for p in directory.glob("*.joblib")]
            
        for name in model_names:
            model_path = directory / f"{name}.joblib"
            if not model_path.exists():
                continue
                
            # Determine model type
            if "xgboost" in name:
                task = "approval" if "approval" in name else "interest_rate"
                model = XGBoostLoanModel(task=task)
            else:
                task = "approval" if "approval" in name else "interest_rate"
                model = RandomForestLoanModel(task=task)
                
            model.load(model_path)
            self.models[name] = model
            
        logger.info(f"Loaded {len(self.models)} models from {directory}")


# Global registry instance
_registry: Optional[ModelRegistry] = None


def get_registry() -> ModelRegistry:
    """Get or create the global model registry."""
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry
