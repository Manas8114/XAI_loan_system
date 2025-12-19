"""
Data Preprocessing Module

Handles feature engineering, encoding, scaling, and preparation
of data for ML models.
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import train_test_split
import logging
import joblib
from pathlib import Path

from ..config import (
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_APPROVAL,
    TARGET_INTEREST_RATE,
    INTEREST_RATE_BANDS,
    RANDOM_SEED,
    TEST_SIZE,
    VALIDATION_SIZE,
    CACHE_DIR,
)

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Comprehensive data preprocessing pipeline for loan data.
    
    Handles:
    - Missing value imputation
    - Feature engineering
    - Encoding (label and one-hot)
    - Scaling
    - Train/test splitting
    """
    
    def __init__(self):
        """Initialize the preprocessor with default components."""
        self.numerical_imputer: Optional[SimpleImputer] = None
        self.categorical_imputer: Optional[SimpleImputer] = None
        self.scaler: Optional[StandardScaler] = None
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.onehot_encoder: Optional[OneHotEncoder] = None
        self.feature_names: List[str] = []
        self.target_encoder: Optional[LabelEncoder] = None
        self._fitted = False
        
    def fit_transform(
        self,
        df: pd.DataFrame,
        target_col: str = TARGET_APPROVAL
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Fit the preprocessor and transform the data.
        
        Args:
            df: Input DataFrame
            target_col: Name of the target column
            
        Returns:
            Tuple of (processed features DataFrame, target Series)
        """
        logger.info("Starting preprocessing pipeline...")
        
        # Make a copy to avoid modifying original
        df = df.copy()
        
        # Clean and prepare data
        df = self._clean_data(df)
        
        # Engineer features
        df = self._engineer_features(df)
        
        # Prepare target variable
        y = self._prepare_target(df, target_col)
        
        # Select features for modeling
        X = self._select_features(df)
        
        # Handle missing values
        X = self._impute_missing(X, fit=True)
        
        # Encode categorical variables
        X = self._encode_categorical(X, fit=True)
        
        # Scale numerical features
        X = self._scale_features(X, fit=True)
        
        self._fitted = True
        self.feature_names = list(X.columns)
        
        logger.info(f"Preprocessing complete. Shape: {X.shape}")
        return X, y
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted preprocessor.
        
        Args:
            df: Input DataFrame to transform
            
        Returns:
            Processed DataFrame
        """
        if not self._fitted:
            raise ValueError("Preprocessor not fitted. Call fit_transform first.")
            
        df = df.copy()
        df = self._clean_data(df)
        df = self._engineer_features(df)
        X = self._select_features(df)
        X = self._impute_missing(X, fit=False)
        X = self._encode_categorical(X, fit=False)
        X = self._scale_features(X, fit=False)
        
        return X
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean raw data - handle formatting issues."""
        # Clean term column if present
        if "term" in df.columns:
            df["term"] = df["term"].astype(str).str.extract(r"(\d+)").astype(float)
            
        # Clean employment length
        if "emp_length" in df.columns:
            if df["emp_length"].dtype == object:
                df["emp_length"] = df["emp_length"].replace({
                    "< 1 year": 0,
                    "1 year": 1,
                    "10+ years": 10,
                }).astype(str).str.extract(r"(\d+)").astype(float)
                
        # Clean interest rate
        if "int_rate" in df.columns:
            if df["int_rate"].dtype == object:
                df["int_rate"] = df["int_rate"].str.rstrip("%").astype(float)
                
        # Clean revol_util
        if "revol_util" in df.columns:
            if df["revol_util"].dtype == object:
                df["revol_util"] = df["revol_util"].str.rstrip("%").astype(float)
        
        # Create fico_score from fico_range_high and fico_range_low (real LendingClub format)
        if "fico_range_high" in df.columns and "fico_range_low" in df.columns:
            df["fico_score"] = (df["fico_range_high"] + df["fico_range_low"]) / 2
        elif "fico_range_high" in df.columns:
            df["fico_score"] = df["fico_range_high"]
        elif "fico_range_low" in df.columns:
            df["fico_score"] = df["fico_range_low"]
                
        return df
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create derived features."""
        # Income-to-loan ratio
        if "annual_inc" in df.columns and "loan_amnt" in df.columns:
            df["income_to_loan_ratio"] = df["annual_inc"] / (df["loan_amnt"] + 1)
            
        # Monthly income
        if "annual_inc" in df.columns:
            df["monthly_inc"] = df["annual_inc"] / 12
            
        # Payment-to-income ratio
        if "installment" in df.columns and "monthly_inc" in df.columns:
            df["payment_to_income"] = df["installment"] / (df["monthly_inc"] + 1)
            
        # Credit utilization risk (high utilization indicator)
        if "revol_util" in df.columns:
            df["high_utilization"] = (df["revol_util"] > 80).astype(int)
            
        # Account age approximation
        if "total_acc" in df.columns and "open_acc" in df.columns:
            df["closed_acc_ratio"] = 1 - (df["open_acc"] / (df["total_acc"] + 1))
            
        # Risk factors combined
        risk_cols = ["delinq_2yrs", "pub_rec", "pub_rec_bankruptcies", "inq_last_6mths"]
        existing_risk_cols = [c for c in risk_cols if c in df.columns]
        if existing_risk_cols:
            df["combined_risk_factors"] = df[existing_risk_cols].sum(axis=1)
            
        # FICO score bins
        if "fico_score" in df.columns:
            df["fico_category"] = pd.cut(
                df["fico_score"],
                bins=[0, 580, 670, 740, 800, 850],
                labels=["Poor", "Fair", "Good", "Very Good", "Exceptional"]
            )
            
        # Interest rate bands (if not already present)
        if "int_rate" in df.columns and "int_rate_band" not in df.columns:
            df["int_rate_band"] = pd.cut(
                df["int_rate"],
                bins=[0, 8, 12, 18, 100],
                labels=["Low", "Medium", "High", "Very High"]
            )
            
        return df
    
    def _prepare_target(self, df: pd.DataFrame, target_col: str) -> pd.Series:
        """Prepare target variable for modeling."""
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")
            
        y = df[target_col].copy()
        
        # Binary encoding for loan_status
        if target_col == "loan_status" or target_col == TARGET_APPROVAL:
            # Map to binary: 1 = Good (Fully Paid), 0 = Bad (Default/Charged Off)
            good_statuses = ["Fully Paid", "Current", "In Grace Period", 1, "1"]
            y = y.apply(lambda x: 1 if x in good_statuses else 0)
            
        # Encode interest rate bands
        elif target_col == "int_rate_band" or target_col == TARGET_INTEREST_RATE:
            if self.target_encoder is None:
                self.target_encoder = LabelEncoder()
                y = pd.Series(
                    self.target_encoder.fit_transform(y.astype(str)),
                    index=y.index
                )
            else:
                y = pd.Series(
                    self.target_encoder.transform(y.astype(str)),
                    index=y.index
                )
                
        return y
    
    def _select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select features for modeling."""
        # Start with configured features
        all_features = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
        
        # Add engineered features
        engineered = [
            "income_to_loan_ratio",
            "monthly_inc",
            "payment_to_income",
            "high_utilization",
            "closed_acc_ratio",
            "combined_risk_factors",
        ]
        
        # Filter to only existing columns
        available_features = [f for f in all_features + engineered if f in df.columns]
        
        # Exclude target columns
        exclude_cols = ["loan_status", "int_rate_band", "int_rate", "fico_category"]
        available_features = [f for f in available_features if f not in exclude_cols]
        
        return df[available_features].copy()
    
    def _impute_missing(self, X: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Impute missing values."""
        # Identify column types
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        
        # Numerical imputation (median strategy)
        if num_cols:
            if fit:
                self.numerical_imputer = SimpleImputer(strategy="median")
                X[num_cols] = self.numerical_imputer.fit_transform(X[num_cols])
            elif self.numerical_imputer is not None:
                X[num_cols] = self.numerical_imputer.transform(X[num_cols])
                
        # Categorical imputation (most frequent)
        if cat_cols:
            if fit:
                self.categorical_imputer = SimpleImputer(strategy="most_frequent")
                X[cat_cols] = self.categorical_imputer.fit_transform(X[cat_cols])
            elif self.categorical_imputer is not None:
                X[cat_cols] = self.categorical_imputer.transform(X[cat_cols])
                
        return X
    
    def _encode_categorical(self, X: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Encode categorical variables using label encoding."""
        cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        
        for col in cat_cols:
            if fit:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.label_encoders[col] = le
            elif col in self.label_encoders:
                le = self.label_encoders[col]
                # Handle unseen categories
                X[col] = X[col].astype(str).apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )
                
        return X
    
    def _scale_features(self, X: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Scale numerical features."""
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        if not num_cols:
            return X
            
        if fit:
            self.scaler = StandardScaler()
            X[num_cols] = self.scaler.fit_transform(X[num_cols])
        elif self.scaler is not None:
            X[num_cols] = self.scaler.transform(X[num_cols])
            
        return X
    
    def split_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = TEST_SIZE,
        val_size: float = VALIDATION_SIZE
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Split data into train, validation, and test sets.
        
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=RANDOM_SEED, stratify=y
        )
        
        # Second split: separate validation set from training
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=RANDOM_SEED, stratify=y_temp
        )
        
        logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def save(self, path: Optional[Path] = None):
        """Save the fitted preprocessor."""
        path = path or CACHE_DIR / "preprocessor.joblib"
        
        state = {
            "numerical_imputer": self.numerical_imputer,
            "categorical_imputer": self.categorical_imputer,
            "scaler": self.scaler,
            "label_encoders": self.label_encoders,
            "target_encoder": self.target_encoder,
            "feature_names": self.feature_names,
            "_fitted": self._fitted,
        }
        
        joblib.dump(state, path)
        logger.info(f"Preprocessor saved to {path}")
        
    def load(self, path: Optional[Path] = None):
        """Load a fitted preprocessor."""
        path = path or CACHE_DIR / "preprocessor.joblib"
        
        if not path.exists():
            raise FileNotFoundError(f"Preprocessor not found at {path}")
            
        state = joblib.load(path)
        
        self.numerical_imputer = state["numerical_imputer"]
        self.categorical_imputer = state["categorical_imputer"]
        self.scaler = state["scaler"]
        self.label_encoders = state["label_encoders"]
        self.target_encoder = state["target_encoder"]
        self.feature_names = state["feature_names"]
        self._fitted = state["_fitted"]
        
        logger.info(f"Preprocessor loaded from {path}")
        
    def inverse_transform_sample(self, X_scaled: pd.DataFrame) -> pd.DataFrame:
        """
        Inverse transform a scaled sample back to original scale.
        Useful for counterfactual interpretation.
        """
        if self.scaler is None:
            return X_scaled
            
        X_original = X_scaled.copy()
        num_cols = X_scaled.select_dtypes(include=[np.number]).columns.tolist()
        
        # Only inverse transform the columns that were originally scaled
        scaled_cols = [c for c in num_cols if c in self.feature_names]
        
        if scaled_cols:
            X_original[scaled_cols] = self.scaler.inverse_transform(X_original[scaled_cols])
            
        return X_original
    
    def get_feature_info(self) -> Dict[str, Any]:
        """Get information about processed features."""
        return {
            "feature_names": self.feature_names,
            "n_features": len(self.feature_names),
            "categorical_mappings": {
                col: dict(zip(range(len(le.classes_)), le.classes_))
                for col, le in self.label_encoders.items()
            },
            "scaler_mean": list(self.scaler.mean_) if self.scaler else None,
            "scaler_std": list(self.scaler.scale_) if self.scaler else None,
        }
