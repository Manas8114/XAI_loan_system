"""
Random Forest Model Module

Baseline model for comparison with XGBoost.
Provides interpretable tree-based predictions.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import logging
import joblib
from pathlib import Path

from ..config import RF_PARAMS, MODELS_DIR, RANDOM_SEED

logger = logging.getLogger(__name__)


class RandomForestLoanModel:
    """
    Random Forest model for loan prediction (baseline).
    
    Used for comparison with XGBoost to validate model selection.
    """
    
    def __init__(
        self,
        task: str = "approval",
        params: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the Random Forest model.
        
        Args:
            task: Either "approval" or "interest_rate"
            params: Optional custom parameters
        """
        self.task = task
        self.params = params or RF_PARAMS.copy()
        self.model: Optional[RandomForestClassifier] = None
        self.feature_names: List[str] = []
        self.feature_importances_: Optional[np.ndarray] = None
        self._fitted = False
        
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> "RandomForestLoanModel":
        """
        Train the Random Forest model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Optional validation features (unused, for API compatibility)
            y_val: Optional validation labels (unused)
            
        Returns:
            Self for method chaining
        """
        self.feature_names = list(X_train.columns)
        
        # Initialize and fit model
        self.model = RandomForestClassifier(**self.params)
        
        logger.info(f"Training Random Forest model for {self.task}...")
        self.model.fit(X_train, y_train)
        
        # Store feature importances
        self.feature_importances_ = self.model.feature_importances_
        
        self._fitted = True
        logger.info(f"Model training complete. Features: {len(self.feature_names)}")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self._fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities."""
        if not self._fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model.predict_proba(X)
    
    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, Any]:
        """Evaluate model on test data."""
        if not self._fitted:
            raise ValueError("Model not fitted. Call fit() first.")
            
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)
        
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="weighted"),
            "recall": recall_score(y_test, y_pred, average="weighted"),
            "f1": f1_score(y_test, y_pred, average="weighted"),
        }
        
        # ROC-AUC
        if self.task == "approval":
            metrics["roc_auc"] = roc_auc_score(y_test, y_proba[:, 1])
        else:
            try:
                metrics["roc_auc"] = roc_auc_score(
                    y_test, y_proba, multi_class="ovr", average="weighted"
                )
            except ValueError:
                metrics["roc_auc"] = None
                
        metrics["confusion_matrix"] = confusion_matrix(y_test, y_pred).tolist()
        metrics["classification_report"] = classification_report(
            y_test, y_pred, output_dict=True
        )
        
        return metrics
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance scores."""
        if not self._fitted:
            raise ValueError("Model not fitted. Call fit() first.")
            
        importance_df = pd.DataFrame({
            "feature": self.feature_names,
            "importance": self.feature_importances_
        })
        
        return importance_df.sort_values("importance", ascending=False)
    
    def save(self, path: Optional[Path] = None):
        """Save the model to disk."""
        if path is None:
            path = MODELS_DIR / f"random_forest_{self.task}.joblib"
            
        state = {
            "model": self.model,
            "task": self.task,
            "params": self.params,
            "feature_names": self.feature_names,
            "feature_importances_": self.feature_importances_,
            "_fitted": self._fitted,
        }
        
        joblib.dump(state, path)
        logger.info(f"Model saved to {path}")
        
    def load(self, path: Optional[Path] = None):
        """Load a model from disk."""
        if path is None:
            path = MODELS_DIR / f"random_forest_{self.task}.joblib"
            
        if not path.exists():
            raise FileNotFoundError(f"Model not found at {path}")
            
        state = joblib.load(path)
        
        self.model = state["model"]
        self.task = state["task"]
        self.params = state["params"]
        self.feature_names = state["feature_names"]
        self.feature_importances_ = state["feature_importances_"]
        self._fitted = state["_fitted"]
        
        logger.info(f"Model loaded from {path}")
