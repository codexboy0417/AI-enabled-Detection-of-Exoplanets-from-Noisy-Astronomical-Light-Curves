import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import logging

from model import DenoisingAutoencoder1D, TransitClassifier1D

logger = logging.getLogger(__name__)

class ModelManager:
    """Manages loading and inference for DAE and Classifier models"""
    
    def __init__(self, seq_len: int = 200, device: Optional[str] = None):
        self.seq_len = seq_len
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dae: Optional[DenoisingAutoencoder1D] = None
        self.classifier: Optional[TransitClassifier1D] = None
        self.models_loaded = False
        
    def load_models(self, dae_path: str = "models/dae.pt", clf_path: str = "models/classifier.pt") -> bool:
        """Load both models from disk"""
        try:
            logger.info(f"Loading models on device: {self.device}")
            
            # Initialize architectures
            self.dae = DenoisingAutoencoder1D(seq_len=self.seq_len).to(self.device)
            self.classifier = TransitClassifier1D(seq_len=self.seq_len).to(self.device)
            
            # Load weights if files exist
            dae_loaded = False
            clf_loaded = False
            
            if Path(dae_path).exists():
                state_dict = torch.load(dae_path, map_location=self.device)
                self.dae.load_state_dict(state_dict)
                self.dae.eval()
                dae_loaded = True
                logger.info(f"DAE loaded from {dae_path}")
            else:
                logger.warning(f"DAE weights not found at {dae_path}, using random initialization")
                
            if Path(clf_path).exists():
                state_dict = torch.load(clf_path, map_location=self.device)
                self.classifier.load_state_dict(state_dict)
                self.classifier.eval()
                clf_loaded = True
                logger.info(f"Classifier loaded from {clf_path}")
            else:
                logger.warning(f"Classifier weights not found at {clf_path}, using random initialization")
            
            self.models_loaded = dae_loaded and clf_loaded
            return self.models_loaded
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            return False
    
    def preprocess_lightcurve(self, flux: np.ndarray, time: Optional[np.ndarray] = None) -> torch.Tensor:
        """Preprocess raw light curve for model input"""
        # Ensure correct length
        if len(flux) != self.seq_len:
            # Resample or pad/truncate
            if len(flux) > self.seq_len:
                flux = flux[:self.seq_len]
            else:
                flux = np.pad(flux, (0, self.seq_len - len(flux)), mode='edge')
        
        # Normalize to median 1.0
        median_flux = np.median(flux)
        if median_flux > 0:
            flux = flux / median_flux
            
        # Sigma clipping for outliers (positive only)
        std_flux = np.std(flux)
        outlier_mask = flux > (1.0 + 3.0 * std_flux)
        flux[outlier_mask] = 1.0
        
        # Convert to tensor: (1, 1, seq_len)
        tensor = torch.tensor(flux, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        return tensor.to(self.device)
    
    def denoise(self, flux: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Run DAE denoising on light curve"""
        if self.dae is None:
            raise RuntimeError("DAE not loaded")
            
        input_tensor = self.preprocess_lightcurve(flux)
        
        with torch.no_grad():
            denoised = self.dae(input_tensor)
            
        denoised_np = denoised.squeeze().cpu().numpy()
        
        # Compute metrics
        residual = flux[:self.seq_len] - denoised_np
        metrics = {
            "mse": float(np.mean(residual**2)),
            "max_residual": float(np.max(np.abs(residual))),
            "snr_improvement": float(np.std(flux[:self.seq_len]) / (np.std(residual) + 1e-8))
        }
        
        return denoised_np, metrics
    
    def classify(self, flux: np.ndarray) -> Tuple[float, Dict[str, Any]]:
        """Run transit classification on (optionally denoised) light curve"""
        if self.classifier is None:
            raise RuntimeError("Classifier not loaded")
            
        input_tensor = self.preprocess_lightcurve(flux)
        
        with torch.no_grad():
            prob = self.classifier(input_tensor).item()
            
        metrics = {
            "probability": prob,
            "is_transit": prob > 0.5,
            "confidence": float(abs(prob - 0.5) * 2)  # 0 to 1 scale
        }
        
        return prob, metrics
    
    def full_pipeline(self, flux: np.ndarray, time: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Run complete pipeline: denoise -> classify"""
        results = {"original_flux": flux[:self.seq_len].tolist()}
        
        if time is not None:
            results["time"] = time[:self.seq_len].tolist()
        
        # Step 1: Denoise
        denoised, denoise_metrics = self.denoise(flux)
        results["denoised_flux"] = denoised.tolist()
        results["denoise_metrics"] = denoise_metrics
        
        # Step 2: Classify
        prob, clf_metrics = self.classify(denoised)
        results["transit_probability"] = prob
        results["classification"] = clf_metrics
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model architecture info"""
        dae_params = sum(p.numel() for p in self.dae.parameters()) if self.dae else 0
        clf_params = sum(p.numel() for p in self.classifier.parameters()) if self.classifier else 0
        
        return {
            "device": self.device,
            "sequence_length": self.seq_len,
            "models_loaded": self.models_loaded,
            "dae_parameters": dae_params,
            "classifier_parameters": clf_params,
            "dae_architecture": "Conv1D Encoder-Decoder (3 layers each)",
            "classifier_architecture": "Conv1D + GlobalAvgPool + Linear (4 conv layers)"
        }


# Global instance
_model_manager: Optional[ModelManager] = None

def get_model_manager(seq_len: int = 200) -> ModelManager:
    """Get or create global model manager"""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager(seq_len=seq_len)
        _model_manager.load_models()
    return _model_manager

def reload_models(seq_len: int = 200) -> ModelManager:
    """Force reload models"""
    global _model_manager
    _model_manager = ModelManager(seq_len=seq_len)
    _model_manager.load_models()
    return _model_manager