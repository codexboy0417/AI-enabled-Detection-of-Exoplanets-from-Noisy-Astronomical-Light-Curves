from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class LightCurveInput(BaseModel):
    """Input light curve data"""
    flux: List[float] = Field(..., description="Flux measurements", min_length=10)
    time: Optional[List[float]] = Field(None, description="Time stamps (optional)")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class DenoiseRequest(BaseModel):
    """Request for denoising only"""
    flux: List[float]
    time: Optional[List[float]] = None


class DenoiseResponse(BaseModel):
    """Response from denoising"""
    original_flux: List[float]
    denoised_flux: List[float]
    time: Optional[List[float]] = None
    metrics: Dict[str, float]


class ClassifyRequest(BaseModel):
    """Request for classification (expects denoised flux)"""
    flux: List[float]
    time: Optional[List[float]] = None


class ClassifyResponse(BaseModel):
    """Response from classification"""
    transit_probability: float
    is_transit: bool
    confidence: float
    time: Optional[List[float]] = None


class PipelineRequest(BaseModel):
    """Request for full pipeline"""
    flux: List[float]
    time: Optional[List[float]] = None
    scenario_id: Optional[str] = None


class PipelineResponse(BaseModel):
    """Response from full pipeline"""
    original_flux: List[float]
    denoised_flux: List[float]
    time: Optional[List[float]] = None
    transit_probability: float
    is_transit: bool
    confidence: float
    denoise_metrics: Dict[str, float]
    classification: Dict[str, Any]
    scenario_id: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class ModelInfoResponse(BaseModel):
    """Model information"""
    device: str
    sequence_length: int
    models_loaded: bool
    dae_parameters: int
    classifier_parameters: int
    dae_architecture: str
    classifier_architecture: str


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    models_loaded: bool
    device: str


class TrainingStatus(BaseModel):
    """Training progress status"""
    status: str  # "idle", "training", "completed", "failed", "stopped"
    epoch: int
    total_epochs: int
    train_loss: float
    val_loss: float
    train_acc: Optional[float] = None
    val_acc: Optional[float] = None
    message: str


class TrainingConfig(BaseModel):
    """Training configuration"""
    model_type: str = "both"  # "dae", "classifier", "both"
    epochs_dae: int = 15
    epochs_clf: int = 15
    batch_size: int = 64
    learning_rate: float = 0.001
    seq_len: int = 200
    num_samples: int = 4000


class ScenarioInfo(BaseModel):
    """Exoplanet scenario information"""
    id: str
    name: str
    type: str
    description: str
    metadata: Dict[str, Any]