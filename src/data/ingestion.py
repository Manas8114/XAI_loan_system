"""
Data Ingestion Service

Handles loading, caching, and initial validation of the LendingClub dataset.
Supports both local files and synthetic data generation for testing.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import logging
import json
from datetime import datetime

from ..config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    SAMPLE_SIZE,
    RANDOM_SEED,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_APPROVAL,
    INTEREST_RATE_BANDS,
)

logger = logging.getLogger(__name__)


class DataIngestionService:
    """
    Service for loading and validating loan data.
    
    Supports:
    - Loading from local CSV files
    - Generating synthetic data for testing
    - Data validation and quality checks
    - Caching processed data
    """
    
    def __init__(self, data_path: Optional[Path] = None):
        """
        Initialize the data ingestion service.
        
        Args:
            data_path: Optional path to the raw data file.
                       If None, will look in RAW_DATA_DIR or generate synthetic data.
        """
        self.data_path = data_path
        self.raw_data: Optional[pd.DataFrame] = None
        self.metadata: Dict[str, Any] = {}
        
    def load_data(
        self, 
        use_synthetic: bool = False,
        sample_size: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Load the loan dataset.
        
        Args:
            use_synthetic: If True, generate synthetic data instead of loading real data.
            sample_size: Number of samples to load/generate. Defaults to config SAMPLE_SIZE.
            
        Returns:
            DataFrame with the loan data.
        """
        sample_size = sample_size or SAMPLE_SIZE
        
        if use_synthetic:
            logger.info(f"Generating synthetic dataset with {sample_size} samples")
            self.raw_data = self._generate_synthetic_data(sample_size)
        else:
            self.raw_data = self._load_from_file(sample_size)
            
        self._compute_metadata()
        self._validate_data()
        
        return self.raw_data
    
    def _load_from_file(self, sample_size: int) -> pd.DataFrame:
        """Load data from CSV file."""
        # Try to find data file
        possible_paths = [
            self.data_path,
            RAW_DATA_DIR / "loan_data.csv",
            RAW_DATA_DIR / "lending_club.csv",
            RAW_DATA_DIR / "accepted_2007_to_2018Q4.csv",
        ]
        
        data_file = None
        for path in possible_paths:
            if path and Path(path).exists():
                data_file = path
                break
                
        if data_file is None:
            logger.warning("No data file found. Generating synthetic data.")
            return self._generate_synthetic_data(sample_size)
            
        logger.info(f"Loading data from {data_file}")
        
        # Load with sampling for large files
        df = pd.read_csv(data_file, nrows=sample_size, low_memory=False)
        
        return df
    
    def _generate_synthetic_data(self, n_samples: int) -> pd.DataFrame:
        """
        Generate realistic synthetic loan data for testing.
        
        This creates data with realistic correlations mirroring
        the LendingClub dataset structure.
        """
        np.random.seed(RANDOM_SEED)
        
        # Generate base features with realistic distributions
        data = {}
        
        # Employment length (0-40 years, right-skewed)
        data["emp_length"] = np.clip(
            np.random.exponential(5, n_samples),
            0, 40
        ).astype(int)
        
        # Annual income (log-normal distribution)
        # Correlated with employment length
        base_income = np.random.lognormal(10.8, 0.5, n_samples)
        emp_bonus = data["emp_length"] * 2500
        data["annual_inc"] = np.clip(base_income + emp_bonus, 20000, 500000)
        
        # Loan amount (normal distribution, capped)
        data["loan_amnt"] = np.clip(
            np.random.normal(15000, 8000, n_samples),
            1000, 40000
        )
        
        # Term (36 or 60 months)
        data["term"] = np.random.choice([" 36 months", " 60 months"], n_samples, p=[0.7, 0.3])
        
        # Installment (calculated from loan amount and term)
        term_months = np.where(data["term"] == " 36 months", 36, 60)
        avg_rate = 0.12  # Average interest rate for calculation
        data["installment"] = (data["loan_amnt"] * (avg_rate/12) * 
                               (1 + avg_rate/12)**term_months) / \
                              ((1 + avg_rate/12)**term_months - 1)
        
        # DTI (debt-to-income ratio)
        existing_debt = np.random.exponential(500, n_samples)
        total_monthly_debt = data["installment"] + existing_debt
        monthly_income = data["annual_inc"] / 12
        data["dti"] = np.clip((total_monthly_debt / monthly_income) * 100, 0, 50)
        
        # Credit-related features
        # Delinquencies (most have 0, some have more)
        data["delinq_2yrs"] = np.random.choice(
            [0, 1, 2, 3, 4, 5],
            n_samples,
            p=[0.7, 0.15, 0.08, 0.04, 0.02, 0.01]
        )
        
        # Inquiries in last 6 months
        data["inq_last_6mths"] = np.random.choice(
            range(11),
            n_samples,
            p=[0.3, 0.25, 0.2, 0.1, 0.05, 0.04, 0.03, 0.015, 0.01, 0.003, 0.002]
        )
        
        # FICO score (influenced by other factors)
        base_fico = np.random.normal(700, 50, n_samples)
        fico_adjustments = (
            -15 * data["delinq_2yrs"] +
            -5 * data["inq_last_6mths"] +
            2 * data["emp_length"] +
            0.0003 * data["annual_inc"] +
            -0.3 * data["dti"]
        )
        data["fico_score"] = np.clip(base_fico + fico_adjustments, 300, 850).astype(int)
        
        # Revolving balance and utilization
        data["revol_bal"] = np.clip(
            np.random.exponential(10000, n_samples),
            0, 200000
        )
        credit_limit = data["revol_bal"] / np.random.uniform(0.1, 0.9, n_samples)
        data["revol_util"] = np.clip((data["revol_bal"] / credit_limit) * 100, 0, 100)
        
        # Account-related features
        data["open_acc"] = np.clip(np.random.poisson(10, n_samples), 1, 50)
        data["total_acc"] = data["open_acc"] + np.random.poisson(5, n_samples)
        data["mort_acc"] = np.random.choice(
            range(10),
            n_samples,
            p=[0.4, 0.3, 0.15, 0.08, 0.04, 0.015, 0.01, 0.003, 0.001, 0.001]
        )
        
        # Public records
        data["pub_rec"] = np.random.choice(
            [0, 1, 2, 3],
            n_samples,
            p=[0.85, 0.1, 0.04, 0.01]
        )
        data["pub_rec_bankruptcies"] = np.random.choice(
            [0, 1, 2],
            n_samples,
            p=[0.92, 0.07, 0.01]
        )
        
        # Categorical features
        data["grade"] = self._assign_grades(data["fico_score"])
        data["sub_grade"] = [f"{g}{np.random.randint(1, 6)}" for g in data["grade"]]
        
        data["home_ownership"] = np.random.choice(
            ["RENT", "OWN", "MORTGAGE", "OTHER"],
            n_samples,
            p=[0.4, 0.1, 0.45, 0.05]
        )
        
        data["verification_status"] = np.random.choice(
            ["Verified", "Source Verified", "Not Verified"],
            n_samples,
            p=[0.35, 0.35, 0.3]
        )
        
        data["purpose"] = np.random.choice(
            ["debt_consolidation", "credit_card", "home_improvement", 
             "major_purchase", "small_business", "car", "medical", "other"],
            n_samples,
            p=[0.45, 0.25, 0.08, 0.06, 0.05, 0.05, 0.03, 0.03]
        )
        
        # US states (simplified)
        states = ["CA", "NY", "TX", "FL", "IL", "PA", "OH", "GA", "NC", "MI"]
        data["addr_state"] = np.random.choice(states, n_samples)
        
        # Interest rate (based on grade and other factors)
        base_rates = {"A": 7, "B": 10, "C": 13, "D": 17, "E": 21, "F": 24, "G": 27}
        data["int_rate"] = np.array([
            base_rates[g] + np.random.normal(0, 1.5) 
            for g in data["grade"]
        ])
        data["int_rate"] = np.clip(data["int_rate"], 5, 30)
        
        # Loan status (target variable)
        # Probability of default based on features
        default_prob = self._calculate_default_probability(data)
        data["loan_status"] = np.where(
            np.random.random(n_samples) > default_prob,
            "Fully Paid",
            "Charged Off"
        )
        
        df = pd.DataFrame(data)
        
        # Add interest rate bands
        df["int_rate_band"] = pd.cut(
            df["int_rate"],
            bins=[0, 8, 12, 18, 100],
            labels=["Low", "Medium", "High", "Very High"]
        )
        
        logger.info(f"Generated synthetic dataset with {len(df)} samples")
        return df
    
    def _assign_grades(self, fico_scores: np.ndarray) -> np.ndarray:
        """Assign loan grades based on FICO scores."""
        grades = np.empty(len(fico_scores), dtype=object)
        grades[fico_scores >= 750] = "A"
        grades[(fico_scores >= 700) & (fico_scores < 750)] = "B"
        grades[(fico_scores >= 670) & (fico_scores < 700)] = "C"
        grades[(fico_scores >= 640) & (fico_scores < 670)] = "D"
        grades[(fico_scores >= 600) & (fico_scores < 640)] = "E"
        grades[(fico_scores >= 550) & (fico_scores < 600)] = "F"
        grades[fico_scores < 550] = "G"
        return grades
    
    def _calculate_default_probability(self, data: Dict[str, np.ndarray]) -> np.ndarray:
        """Calculate probability of default based on features."""
        # Logistic-like calculation
        log_odds = (
            -3.0 +  # Base log-odds (most loans are paid)
            0.05 * data["dti"] +
            -0.005 * data["fico_score"] +
            0.3 * data["delinq_2yrs"] +
            0.15 * data["inq_last_6mths"] +
            0.2 * data["pub_rec"] +
            0.01 * data["revol_util"] +
            -0.00001 * data["annual_inc"]
        )
        prob = 1 / (1 + np.exp(-log_odds))
        return np.clip(prob, 0.05, 0.5)
    
    def _compute_metadata(self):
        """Compute metadata about the loaded dataset."""
        if self.raw_data is None:
            return
            
        self.metadata = {
            "n_samples": len(self.raw_data),
            "n_features": len(self.raw_data.columns),
            "columns": list(self.raw_data.columns),
            "dtypes": {col: str(dtype) for col, dtype in self.raw_data.dtypes.items()},
            "missing_values": self.raw_data.isnull().sum().to_dict(),
            "load_timestamp": datetime.now().isoformat(),
        }
        
        # Target distribution
        if "loan_status" in self.raw_data.columns:
            self.metadata["target_distribution"] = (
                self.raw_data["loan_status"].value_counts().to_dict()
            )
    
    def _validate_data(self):
        """Validate the loaded data."""
        if self.raw_data is None:
            raise ValueError("No data loaded")
            
        # Check for required columns
        required_cols = ["loan_status"]
        missing_cols = [col for col in required_cols if col not in self.raw_data.columns]
        
        if missing_cols:
            logger.warning(f"Missing required columns: {missing_cols}")
            
        # Check for extreme missing values
        missing_pct = self.raw_data.isnull().mean()
        high_missing = missing_pct[missing_pct > 0.5]
        
        if len(high_missing) > 0:
            logger.warning(f"Columns with >50% missing values: {list(high_missing.index)}")
            
        logger.info(f"Data validation complete. Shape: {self.raw_data.shape}")
    
    def save_processed_data(self, df: pd.DataFrame, filename: str = "processed_data.csv"):
        """Save processed data to disk."""
        output_path = PROCESSED_DATA_DIR / filename
        df.to_csv(output_path, index=False)
        
        # Save metadata
        meta_path = PROCESSED_DATA_DIR / f"{filename.replace('.csv', '_metadata.json')}"
        with open(meta_path, "w") as f:
            json.dump(self.metadata, f, indent=2)
            
        logger.info(f"Saved processed data to {output_path}")
        
    def load_processed_data(self, filename: str = "processed_data.csv") -> pd.DataFrame:
        """Load previously processed data."""
        input_path = PROCESSED_DATA_DIR / filename
        
        if not input_path.exists():
            raise FileNotFoundError(f"Processed data not found at {input_path}")
            
        df = pd.read_csv(input_path)
        logger.info(f"Loaded processed data from {input_path}")
        
        return df
    
    def get_data_quality_report(self) -> Dict[str, Any]:
        """Generate a data quality report."""
        if self.raw_data is None:
            raise ValueError("No data loaded")
            
        report = {
            "shape": self.raw_data.shape,
            "memory_usage_mb": self.raw_data.memory_usage(deep=True).sum() / 1024**2,
            "missing_values": {
                "total": int(self.raw_data.isnull().sum().sum()),
                "by_column": self.raw_data.isnull().sum().to_dict(),
            },
            "duplicates": int(self.raw_data.duplicated().sum()),
            "numeric_stats": {},
            "categorical_stats": {},
        }
        
        # Numeric column statistics
        for col in self.raw_data.select_dtypes(include=[np.number]).columns:
            report["numeric_stats"][col] = {
                "mean": float(self.raw_data[col].mean()),
                "std": float(self.raw_data[col].std()),
                "min": float(self.raw_data[col].min()),
                "max": float(self.raw_data[col].max()),
                "median": float(self.raw_data[col].median()),
            }
            
        # Categorical column statistics
        for col in self.raw_data.select_dtypes(include=["object", "category"]).columns:
            value_counts = self.raw_data[col].value_counts()
            report["categorical_stats"][col] = {
                "n_unique": int(self.raw_data[col].nunique()),
                "top_values": value_counts.head(5).to_dict(),
            }
            
        return report
