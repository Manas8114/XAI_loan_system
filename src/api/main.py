"""
FastAPI Main Application

Entry point for the XAI Loan Counterfactual API.
Initializes all components and starts the server.
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from .routes import router, set_components
from .schemas import HealthResponse
from ..config import API_CONFIG, LOGGING_CONFIG

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOGGING_CONFIG["level"]),
    format=LOGGING_CONFIG["format"]
)
logger = logging.getLogger(__name__)

# Global state
_initialized = False


def initialize_components():
    """Initialize all ML components."""
    global _initialized
    
    if _initialized:
        return True
        
    try:
        logger.info("Initializing XAI Loan System components...")
        
        # Import components
        from ..data import DataIngestionService, DataPreprocessor
        from ..models import XGBoostLoanModel, ModelRegistry, get_registry
        from ..explainability import SHAPEngine
        from ..counterfactual import DiCEEngine, OptimizationCFEngine, CausalCFEngine, CausalModel
        from ..ethics import ActionabilityScorer
        
        # Load or generate data
        logger.info("Loading data...")
        ingestion = DataIngestionService()
        data = ingestion.load_data(use_synthetic=True, sample_size=10000)
        
        # Preprocess
        logger.info("Preprocessing data...")
        preprocessor = DataPreprocessor()
        X, y = preprocessor.fit_transform(data)
        X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(X, y)
        
        # Train models
        logger.info("Training models...")
        registry = get_registry()
        
        # Approval model
        approval_model = XGBoostLoanModel(task="approval")
        approval_model.fit(X_train, y_train, X_val, y_val, verbose=False)
        registry.register("xgboost_approval", approval_model)
        
        # Interest rate model (using synthetic target)
        # In real scenario, this would use int_rate_band target
        interest_model = XGBoostLoanModel(task="interest_rate")
        interest_model.fit(X_train, y_train, X_val, y_val, verbose=False)
        registry.register("xgboost_interest_rate", interest_model)
        
        # SHAP Engine
        logger.info("Initializing SHAP engine...")
        shap_engine = SHAPEngine(approval_model, X_train)
        shap_engine.initialize(X_train)
        
        # Counterfactual engines
        logger.info("Initializing counterfactual engines...")
        feature_names = list(X_train.columns)
        
        # DiCE - use preprocessed data that matches model expectations
        dice_data = X_train.copy()
        dice_data["loan_status"] = y_train.values
        dice_engine = DiCEEngine(
            model=approval_model,
            data=dice_data,
            outcome_name="loan_status",
        )
        
        # Optimization
        optimization_engine = OptimizationCFEngine(
            model=approval_model,
            feature_names=feature_names,
        )
        
        # Causal
        causal_model = CausalModel()
        # Use preprocessed data (numeric) for SCM estimation, not raw data with strings
        X_with_target = X_train.copy()
        X_with_target['loan_status'] = y_train.values  # Add numeric target
        causal_model.estimate_from_data(X_with_target)
        
        actionability_scorer = ActionabilityScorer()
        
        causal_engine = CausalCFEngine(
            model=approval_model,
            causal_model=causal_model,
            feature_names=feature_names,
            actionability_scorer=actionability_scorer,
        )
        
        # Set components in routes
        set_components(
            model_registry=registry,
            shap_engine=shap_engine,
            dice_engine=dice_engine,
            optimization_engine=optimization_engine,
            causal_engine=causal_engine,
            preprocessor=preprocessor,
            feature_names=feature_names,
        )
        
        _initialized = True
        logger.info("All components initialized successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        import traceback
        traceback.print_exc()
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    # Startup
    logger.info("Starting XAI Loan Counterfactual API...")
    success = initialize_components()
    if not success:
        logger.warning("Components initialized with errors. Some features may be unavailable.")
    yield
    # Shutdown
    logger.info("Shutting down API...")


# Create FastAPI app
app = FastAPI(
    title=API_CONFIG["title"],
    description=API_CONFIG["description"],
    version=API_CONFIG["version"],
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router, prefix="/api/v1")


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - health check."""
    return HealthResponse(
        status="healthy",
        version=API_CONFIG["version"],
        models_loaded=_initialized,
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if _initialized else "initializing",
        version=API_CONFIG["version"],
        models_loaded=_initialized,
    )


def run_server():
    """Run the API server."""
    uvicorn.run(
        "src.api.main:app",
        host=API_CONFIG["host"],
        port=API_CONFIG["port"],
        reload=True,
    )


if __name__ == "__main__":
    run_server()
