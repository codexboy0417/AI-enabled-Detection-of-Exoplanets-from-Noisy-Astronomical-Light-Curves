from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
import uvicorn
import sys
from pathlib import Path

# Add backend directory to path to resolve local module imports
sys.path.append(str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting AstroPulse API server...")
    
    # Pre-load models
    try:
        from .models import get_model_manager
        manager = get_model_manager()
        if manager.models_loaded:
            logger.info("Models loaded successfully")
        else:
            logger.warning("Models not found, will use random weights")
    except Exception as e:
        logger.warning(f"Could not pre-load models: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down AstroPulse API server...")


app = FastAPI(
    title="AstroPulse Exoplanet Detection API",
    description="AI-powered exoplanet transit detection from noisy light curves",
    version="1.0.0",
    lifespan=lifespan
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import and include routers
from .routes import inference, training, scenarios

app.include_router(inference.router)
app.include_router(training.router, prefix="/api/v1")
app.include_router(scenarios.router)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "AstroPulse Exoplanet Detection API",
        "version": "1.0.0",
        "description": "AI-powered exoplanet transit detection from noisy astronomical light curves",
        "docs": "/docs",
        "health": "/api/v1/health"
    }


if __name__ == "__main__":
    uvicorn.run(
        "backend.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )