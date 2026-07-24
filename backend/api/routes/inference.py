from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
import logging

from ..schemas import (
    LightCurveInput, DenoiseRequest, DenoiseResponse,
    ClassifyRequest, ClassifyResponse,
    PipelineRequest, PipelineResponse,
    ModelInfoResponse, HealthResponse
)
from ..models import get_model_manager, reload_models

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["inference"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    manager = get_model_manager()
    return HealthResponse(
        status="healthy" if manager.models_loaded else "degraded",
        timestamp=__import__('datetime').datetime.utcnow().isoformat(),
        models_loaded=manager.models_loaded,
        device=manager.device
    )


@router.get("/models/info", response_model=ModelInfoResponse)
async def model_info():
    """Get model architecture and loading info"""
    manager = get_model_manager()
    info = manager.get_model_info()
    return ModelInfoResponse(**info)


@router.post("/models/reload")
async def reload_models(seq_len: int = 200):
    """Force reload models (useful after training)"""
    try:
        manager = reload_models(seq_len=seq_len)
        return {"status": "success", "models_loaded": manager.models_loaded}
    except Exception as e:
        logger.error(f"Model reload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/denoise", response_model=DenoiseResponse)
async def denoise_lightcurve(request: DenoiseRequest):
    """Denoise a light curve using the DAE"""
    manager = get_model_manager()
    
    if not manager.models_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        denoised, metrics = manager.denoise(
            flux=np.array(request.flux),
            time=np.array(request.time) if request.time else None
        )
        
        return DenoiseResponse(
            original_flux=request.flux[:manager.seq_len],
            denoised_flux=denoised.tolist(),
            time=request.time[:manager.seq_len] if request.time else None,
            metrics=metrics
        )
    except Exception as e:
        logger.error(f"Denoising failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/classify", response_model=ClassifyResponse)
async def classify_lightcurve(request: ClassifyRequest):
    """Classify a (denoised) light curve for transit probability"""
    manager = get_model_manager()
    
    if not manager.models_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        prob, metrics = manager.classify(np.array(request.flux))
        
        return ClassifyResponse(
            transit_probability=prob,
            is_transit=metrics["is_transit"],
            confidence=metrics["confidence"],
            time=request.time[:manager.seq_len] if request.time else None
        )
    except Exception as e:
        logger.error(f"Classification failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/pipeline", response_model=PipelineResponse)
async def full_pipeline(request: PipelineRequest):
    """Run full pipeline: denoise -> classify"""
    manager = get_model_manager()
    
    if not manager.models_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        results = manager.full_pipeline(
            flux=np.array(request.flux),
            time=np.array(request.time) if request.time else None
        )
        
        return PipelineResponse(
            original_flux=results["original_flux"],
            denoised_flux=results["denoised_flux"],
            time=results.get("time"),
            transit_probability=results["transit_probability"],
            is_transit=results["classification"]["is_transit"],
            confidence=results["classification"]["confidence"],
            denoise_metrics=results["denoise_metrics"],
            classification=results["classification"],
            scenario_id=request.scenario_id
        )
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Import numpy here to avoid circular imports
import numpy as np