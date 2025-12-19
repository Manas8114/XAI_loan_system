"""
Model Training Script

Trains models and saves them for the API to use.
"""

import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import DataIngestionService, DataPreprocessor
from src.models import XGBoostLoanModel, RandomForestLoanModel, get_registry
from src.config import MODELS_DIR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Main training function."""
    logger.info("=" * 60)
    logger.info("XAI Loan System - Model Training")
    logger.info("=" * 60)
    
    # Step 1: Load data
    logger.info("\n[1/5] Loading data...")
    ingestion = DataIngestionService()
    data = ingestion.load_data(use_synthetic=False, sample_size=100000)  # Use real data
    logger.info(f"Loaded {len(data)} samples with {len(data.columns)} features")
    
    # Data quality report
    quality = ingestion.get_data_quality_report()
    logger.info(f"Memory usage: {quality['memory_usage_mb']:.2f} MB")
    logger.info(f"Missing values: {quality['missing_values']['total']}")
    
    # Step 2: Preprocess
    logger.info("\n[2/5] Preprocessing data...")
    preprocessor = DataPreprocessor()
    X, y = preprocessor.fit_transform(data, target_col="loan_status")
    logger.info(f"Processed features shape: {X.shape}")
    logger.info(f"Target distribution: {y.value_counts().to_dict()}")
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(X, y)
    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Save preprocessor
    preprocessor.save()
    logger.info("Preprocessor saved")
    
    # Step 3: Train XGBoost (Primary)
    logger.info("\n[3/5] Training XGBoost model...")
    xgb_model = XGBoostLoanModel(task="approval")
    xgb_model.fit(X_train, y_train, X_val, y_val, verbose=False)
    
    # Evaluate
    xgb_metrics = xgb_model.evaluate(X_test, y_test)
    logger.info(f"XGBoost ROC-AUC: {xgb_metrics['roc_auc']:.4f}")
    logger.info(f"XGBoost Accuracy: {xgb_metrics['accuracy']:.4f}")
    logger.info(f"XGBoost F1: {xgb_metrics['f1']:.4f}")
    
    # Cross-validation
    cv_results = xgb_model.cross_validate(X, y, n_folds=5)
    logger.info(f"XGBoost CV ROC-AUC: {cv_results['mean_score']:.4f} (+/- {cv_results['std_score']:.4f})")
    
    # Feature importance
    importance = xgb_model.get_feature_importance()
    logger.info("Top 5 features by importance:")
    for _, row in importance.head(5).iterrows():
        logger.info(f"  - {row['feature']}: {row['importance']:.4f}")
    
    # Save model
    xgb_model.save()
    logger.info("XGBoost model saved")
    
    # Step 4: Train Random Forest (Baseline)
    logger.info("\n[4/5] Training Random Forest baseline...")
    rf_model = RandomForestLoanModel(task="approval")
    rf_model.fit(X_train, y_train)
    
    rf_metrics = rf_model.evaluate(X_test, y_test)
    logger.info(f"RandomForest ROC-AUC: {rf_metrics['roc_auc']:.4f}")
    logger.info(f"RandomForest Accuracy: {rf_metrics['accuracy']:.4f}")
    
    rf_model.save()
    logger.info("Random Forest model saved")
    
    # Step 5: Register models
    logger.info("\n[5/5] Registering models...")
    registry = get_registry()
    registry.register("xgboost_approval", xgb_model)
    registry.register("random_forest_approval", rf_model)
    registry.save_all()
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Models saved to: {MODELS_DIR}")
    logger.info("\nModel Comparison:")
    logger.info(f"  XGBoost ROC-AUC:      {xgb_metrics['roc_auc']:.4f}")
    logger.info(f"  RandomForest ROC-AUC: {rf_metrics['roc_auc']:.4f}")
    logger.info(f"  Improvement:          {(xgb_metrics['roc_auc'] - rf_metrics['roc_auc'])*100:.2f}%")
    
    return xgb_metrics, rf_metrics


if __name__ == "__main__":
    main()
