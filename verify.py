"""Quick verification script."""
import sys
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("XAI Loan System - Quick Verification")
print("=" * 60)

# Test 1: Imports
print("\n[1/5] Testing imports...")
try:
    from src.data import DataIngestionService, DataPreprocessor
    from src.models import XGBoostLoanModel
    from src.counterfactual import CausalModel, CausalCFEngine, OptimizationCFEngine
    from src.ethics import ActionabilityScorer, ConstraintValidator
    from src.explainability import SHAPEngine
    print("[OK] All imports successful")
except Exception as e:
    print(f"[FAIL] Import error: {e}")
    sys.exit(1)

# Test 2: Data loading
print("\n[2/5] Testing data ingestion...")
try:
    ingestion = DataIngestionService()
    data = ingestion.load_data(use_synthetic=True, sample_size=1000)
    print(f"[OK] Loaded {len(data)} samples with {len(data.columns)} features")
except Exception as e:
    print(f"[FAIL] Data error: {e}")
    sys.exit(1)

# Test 3: Preprocessing
print("\n[3/5] Testing preprocessing...")
try:
    preprocessor = DataPreprocessor()
    X, y = preprocessor.fit_transform(data)
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(X, y)
    print(f"[OK] Processed features: {X.shape}")
    print(f"[OK] Train/Val/Test split: {len(X_train)}/{len(X_val)}/{len(X_test)}")
except Exception as e:
    print(f"[FAIL] Preprocessing error: {e}")
    sys.exit(1)

# Test 4: Model training
print("\n[4/5] Testing model training...")
try:
    model = XGBoostLoanModel(task="approval")
    model.fit(X_train, y_train, verbose=False)
    metrics = model.evaluate(X_test, y_test)
    print(f"[OK] Model trained successfully")
    print(f"[OK] ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"[OK] Accuracy: {metrics['accuracy']:.4f}")
except Exception as e:
    print(f"[FAIL] Model error: {e}")
    sys.exit(1)

# Test 5: Counterfactual generation
print("\n[5/5] Testing counterfactual generation...")
try:
    # Causal model - use preprocessed data
    causal_model = CausalModel()
    # Create a DataFrame with preprocessed features for estimation
    X_with_target = X.copy()
    X_with_target['loan_status'] = y.values
    causal_model.estimate_from_data(X_with_target)
    
    # Actionability scorer
    scorer = ActionabilityScorer()
    
    # Causal CF engine
    causal_engine = CausalCFEngine(
        model=model,
        causal_model=causal_model,
        feature_names=list(X_train.columns),
        actionability_scorer=scorer,
    )
    
    # Generate counterfactuals
    instance = X_test.iloc[[0]]
    results = causal_engine.generate_counterfactuals(instance, num_cfs=3)
    
    if results["success"]:
        print(f"[OK] Generated {len(results['counterfactuals'])} counterfactuals")
        if results['counterfactuals']:
            cf = results['counterfactuals'][0]
            print(f"[OK] Top CF actionability: {cf['actionability_score']:.3f}")
            print(f"[OK] Causal validity: {cf['causal_valid']}")
    else:
        print(f"[WARN] CF generation returned no results (may need more iterations)")
except Exception as e:
    print(f"[FAIL] Counterfactual error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("VERIFICATION COMPLETE")
print("=" * 60)
