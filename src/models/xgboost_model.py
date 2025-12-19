"""
XGBoost Model Module

Primary model for loan approval classification and interest rate prediction.
Includes hyperparameter optimization and model evaluation.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Tuple, List, Union
import xgboost as xgb
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
import logging
import joblib
from pathlib import Path

from ..config import (
    XGBOOST_APPROVAL_PARAMS,
    XGBOOST_INTEREST_PARAMS,
    MODELS_DIR,
    RANDOM_SEED,
)

logger = logging.getLogger(__name__)


class XGBoostLoanModel:
    """
    XGBoost model for loan prediction tasks.
    
    Supports:
    - Binary classification (loan approval)
    - Multi-class classification (interest rate bands)
    - Hyperparameter optimization
    - Cross-validation
    - Feature importance extraction
    """
    
    def __init__(
        self,
        task: str = "approval",
        params: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the XGBoost model.
        
        Args:
            task: Either "approval" for binary classification or
                  "interest_rate" for multi-class classification
            params: Optional custom parameters. If None, uses config defaults.
        """
        self.task = task
        
        if params is None:
            if task == "approval":
                self.params = XGBOOST_APPROVAL_PARAMS.copy()
            else:
                self.params = XGBOOST_INTEREST_PARAMS.copy()
        else:
            self.params = params.copy()
            
        self.model: Optional[xgb.XGBClassifier] = None
        self.feature_names: List[str] = []
        self.feature_importances_: Optional[np.ndarray] = None
        self.training_history: Dict[str, Any] = {}
        self._fitted = False
        
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        early_stopping_rounds: int = 20,
        verbose: bool = True
    ) -> "XGBoostLoanModel":
        """
        Train the XGBoost model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Optional validation features
            y_val: Optional validation labels
            early_stopping_rounds: Rounds for early stopping
            verbose: Whether to print training progress
            
        Returns:
            Self for method chaining
        """
        self.feature_names = list(X_train.columns)
        
        # Initialize model
        self.model = xgb.XGBClassifier(**self.params)
        
        # Prepare evaluation set
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))
            
        # Fit model
        logger.info(f"Training XGBoost model for {self.task}...")
        
        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            verbose=verbose
        )
        
        # Store feature importances
        self.feature_importances_ = self.model.feature_importances_
        
        # Store training history
        self.training_history = {
            "n_samples": len(X_train),
            "n_features": len(self.feature_names),
            "best_iteration": getattr(self.model, "best_iteration", None),
        }
        
        self._fitted = True
        logger.info(f"Model training complete. Features: {len(self.feature_names)}")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features to predict on
            
        Returns:
            Array of predicted labels
        """
        if not self._fitted:
            raise ValueError("Model not fitted. Call fit() first.")
            
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            X: Features to predict on
            
        Returns:
            Array of class probabilities
        """
        if not self._fitted:
            raise ValueError("Model not fitted. Call fit() first.")
            
        return self.model.predict_proba(X)
    
    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, Any]:
        """
        Evaluate model on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of evaluation metrics
        """
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
        
        # ROC-AUC (handle binary vs multi-class)
        if self.task == "approval":
            metrics["roc_auc"] = roc_auc_score(y_test, y_proba[:, 1])
        else:
            try:
                metrics["roc_auc"] = roc_auc_score(
                    y_test, y_proba, multi_class="ovr", average="weighted"
                )
            except ValueError:
                metrics["roc_auc"] = None
                
        # Confusion matrix
        metrics["confusion_matrix"] = confusion_matrix(y_test, y_pred).tolist()
        
        # Classification report
        metrics["classification_report"] = classification_report(
            y_test, y_pred, output_dict=True
        )
        
        logger.info(f"Evaluation complete. ROC-AUC: {metrics.get('roc_auc', 'N/A')}")
        
        return metrics
    
    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_folds: int = 5
    ) -> Dict[str, Any]:
        """
        Perform cross-validation.
        
        Args:
            X: Features
            y: Labels
            n_folds: Number of CV folds
            
        Returns:
            Dictionary with CV results
        """
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
        
        # Create a fresh model for CV
        model = xgb.XGBClassifier(**self.params)
        
        # Score using ROC-AUC
        scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
        
        results = {
            "n_folds": n_folds,
            "scores": scores.tolist(),
            "mean_score": float(scores.mean()),
            "std_score": float(scores.std()),
        }
        
        logger.info(f"CV Results - Mean ROC-AUC: {results['mean_score']:.4f} "
                    f"(+/- {results['std_score']:.4f})")
        
        return results
    
    def get_feature_importance(
        self,
        importance_type: str = "gain"
    ) -> pd.DataFrame:
        """
        Get feature importance scores.
        
        Args:
            importance_type: Type of importance ("gain", "weight", "cover")
            
        Returns:
            DataFrame with feature names and importance scores
        """
        if not self._fitted:
            raise ValueError("Model not fitted. Call fit() first.")
            
        # Get importance from booster
        booster = self.model.get_booster()
        importance = booster.get_score(importance_type=importance_type)
        
        # Map feature names - handle both "f0" format and actual feature names
        importance_list = []
        for k, v in importance.items():
            # Check if key is in "f{index}" format
            if k.startswith("f") and k[1:].isdigit():
                idx = int(k[1:])
                if idx < len(self.feature_names):
                    feature_name = self.feature_names[idx]
                else:
                    feature_name = k  # Keep original if index out of range
            else:
                # XGBoost returned actual feature name
                feature_name = k
            importance_list.append({"feature": feature_name, "importance": v})
        
        importance_df = pd.DataFrame(importance_list)
        
        if len(importance_df) > 0:
            importance_df = importance_df.sort_values("importance", ascending=False)
            
        return importance_df
    
    def save(self, path: Optional[Path] = None):
        """Save the model to disk."""
        if path is None:
            path = MODELS_DIR / f"xgboost_{self.task}.joblib"
            
        state = {
            "model": self.model,
            "task": self.task,
            "params": self.params,
            "feature_names": self.feature_names,
            "feature_importances_": self.feature_importances_,
            "training_history": self.training_history,
            "_fitted": self._fitted,
        }
        
        joblib.dump(state, path)
        logger.info(f"Model saved to {path}")
        
    def load(self, path: Optional[Path] = None):
        """Load a model from disk."""
        if path is None:
            path = MODELS_DIR / f"xgboost_{self.task}.joblib"
            
        if not path.exists():
            raise FileNotFoundError(f"Model not found at {path}")
            
        state = joblib.load(path)
        
        self.model = state["model"]
        self.task = state["task"]
        self.params = state["params"]
        self.feature_names = state["feature_names"]
        self.feature_importances_ = state["feature_importances_"]
        self.training_history = state["training_history"]
        self._fitted = state["_fitted"]
        
        logger.info(f"Model loaded from {path}")
        
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        return {
            "task": self.task,
            "algorithm": "XGBoost",
            "params": self.params,
            "n_features": len(self.feature_names),
            "feature_names": self.feature_names,
            "fitted": self._fitted,
            "training_history": self.training_history,
        }
