from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, Any
import logging
import asyncio
from pathlib import Path

from ..schemas import TrainingConfig, TrainingStatus

router = APIRouter(prefix="/api/v1", tags=["training"])

# Global training state
_training_state = {
    "status": "idle",
    "epoch": 0,
    "total_epochs": 0,
    "train_loss": 0.0,
    "val_loss": 0.0,
    "train_acc": None,
    "val_acc": None,
    "message": "",
    "config": None
}

logger = logging.getLogger(__name__)


@router.get("/training/status", response_model=TrainingStatus)
async def get_training_status():
    """Get current training status"""
    return TrainingStatus(**_training_state)


@router.post("/training/start")
async def start_training(config: TrainingConfig, background_tasks: BackgroundTasks):
    """Start model training in background"""
    global _training_state
    
    if _training_state["status"] == "training":
        raise HTTPException(status_code=400, detail="Training already in progress")
    
    # Reset state
    _training_state = {
        "status": "training",
        "epoch": 0,
        "total_epochs": config.epochs_dae + config.epochs_clf,
        "train_loss": 0.0,
        "val_loss": 0.0,
        "train_acc": None,
        "val_acc": None,
        "message": "Starting training...",
        "config": config.dict()
    }
    
    # Start background training
    background_tasks.add_task(run_training_background, config)
    
    return {"status": "started", "message": "Training started in background"}


@router.post("/training/stop")
async def stop_training():
    """Stop current training"""
    global _training_state
    
    if _training_state["status"] != "training":
        raise HTTPException(status_code=400, detail="No training in progress")
    
    _training_state["status"] = "stopped"
    _training_state["message"] = "Training stopped by user"
    
    return {"status": "stopped", "message": "Training stop requested"}


async def run_training_background(config: TrainingConfig):
    """Run training in background task"""
    global _training_state
    
    try:
        import sys
        sys.path.append(str(Path(__file__).parent.parent.parent))
        
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset
        import numpy as np
        from model import DenoisingAutoencoder1D, TransitClassifier1D
        
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Generate synthetic data
        def generate_synthetic_data(num_samples=2000, seq_len=200):
            clean_curves = []
            noisy_curves = []
            labels = []
            
            for i in range(num_samples):
                clean = np.ones(seq_len)
                label = 1 if i < num_samples // 2 else 0
                
                if label == 1:
                    transit_depth = np.random.uniform(0.005, 0.02)
                    duration = np.random.randint(15, 40)
                    center = np.random.randint(50, 150)
                    t_start = center - duration // 2
                    t_end = center + duration // 2
                    
                    for idx in range(t_start, t_end):
                        if 0 <= idx < seq_len:
                            dist = abs(idx - center)
                            half_dur = duration / 2
                            if dist > half_dur * 0.8:
                                factor = (half_dur - dist) / (half_dur * 0.2)
                                clean[idx] -= transit_depth * factor
                            else:
                                clean[idx] -= transit_depth
                else:
                    scenario = np.random.choice(["clean", "sine_var", "eclipsing_binary"])
                    if scenario == "sine_var":
                        freq = np.random.uniform(2, 6)
                        amp = np.random.uniform(0.003, 0.01)
                        clean += amp * np.sin(np.linspace(0, freq * 2 * np.pi, seq_len))
                    elif scenario == "eclipsing_binary":
                        dip_depth = np.random.uniform(0.03, 0.1)
                        center = np.random.randint(70, 130)
                        width = np.random.randint(10, 25)
                        for idx in range(center - width, center + width):
                            if 0 <= idx < seq_len:
                                factor = 1.0 - (abs(idx - center) / width)
                                clean[idx] -= dip_depth * factor
                
                white_noise = np.random.normal(0, np.random.uniform(0.002, 0.008), seq_len)
                t = np.linspace(0, 1, seq_len)
                pink_noise = 0.002 * np.sin(2 * np.pi * t * np.random.uniform(1, 3))
                noisy = clean + white_noise + pink_noise
                
                clean_curves.append(clean)
                noisy_curves.append(noisy)
                labels.append(label)
            
            clean_curves = np.array(clean_curves, dtype=np.float32)[:, np.newaxis, :]
            noisy_curves = np.array(noisy_curves, dtype=np.float32)[:, np.newaxis, :]
            labels = np.array(labels, dtype=np.float32)[:, np.newaxis]
            
            return torch.tensor(noisy_curves), torch.tensor(clean_curves), torch.tensor(labels)
        
        seq_len = config.seq_len
        _training_state["message"] = "Generating synthetic data..."
        
        noisy, clean, labels = generate_synthetic_data(num_samples=config.num_samples, seq_len=seq_len)
        dataset = TensorDataset(noisy, clean, labels)
        
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
        
        # Initialize models
        dae = DenoisingAutoencoder1D(seq_len=seq_len).to(device)
        classifier = TransitClassifier1D(seq_len=seq_len).to(device)
        
        criterion_dae = nn.MSELoss()
        criterion_clf = nn.BCELoss()
        optimizer_dae = optim.Adam(dae.parameters(), lr=config.learning_rate)
        optimizer_clf = optim.Adam(classifier.parameters(), lr=config.learning_rate)
        
        total_epochs = config.epochs_dae + config.epochs_clf
        _training_state["total_epochs"] = total_epochs
        
        # Train DAE
        if config.model_type in ["dae", "both"]:
            _training_state["message"] = f"Training DAE (1/{2 if config.model_type == 'both' else 1})"
            for epoch in range(config.epochs_dae):
                if _training_state["status"] != "training":
                    return
                    
                _training_state["epoch"] = epoch + 1
                
                dae.train()
                train_loss = 0.0
                for batch_noisy, batch_clean, _ in train_loader:
                    batch_noisy = batch_noisy.to(device)
                    batch_clean = batch_clean.to(device)
                    
                    optimizer_dae.zero_grad()
                    outputs = dae(batch_noisy)
                    loss = criterion_dae(outputs, batch_clean)
                    loss.backward()
                    optimizer_dae.step()
                    train_loss += loss.item() * batch_noisy.size(0)
                
                train_loss /= len(train_loader.dataset)
                
                # Validation
                dae.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch_noisy, batch_clean, _ in val_loader:
                        batch_noisy = batch_noisy.to(device)
                        batch_clean = batch_clean.to(device)
                        outputs = dae(batch_noisy)
                        loss = criterion_dae(outputs, batch_clean)
                        val_loss += loss.item() * batch_noisy.size(0)
                val_loss /= len(val_loader.dataset)
                
                _training_state["train_loss"] = train_loss
                _training_state["val_loss"] = val_loss
                _training_state["message"] = f"DAE Epoch {epoch+1}/{config.epochs_dae} | Train: {train_loss:.6f} | Val: {val_loss:.6f}"
                
                await asyncio.sleep(0.1)  # Yield control
            
            # Save DAE
            import os
            os.makedirs("models", exist_ok=True)
            torch.save(dae.state_dict(), "models/dae.pt")
        
        # Train Classifier
        if config.model_type in ["classifier", "both"]:
            _training_state["message"] = f"Training Classifier ({2 if config.model_type == 'both' else 1}/{2 if config.model_type == 'both' else 1})"
            
            if config.model_type == "both":
                # Use trained DAE for denoising
                dae.eval()
            
            for epoch in range(config.epochs_clf):
                if _training_state["status"] != "training":
                    return
                    
                _training_state["epoch"] = config.epochs_dae + epoch + 1
                
                classifier.train()
                train_loss = 0.0
                correct = 0
                total = 0
                
                for batch_noisy, _, batch_labels in train_loader:
                    batch_noisy = batch_noisy.to(device)
                    batch_labels = batch_labels.to(device)
                    
                    if config.model_type == "both":
                        with torch.no_grad():
                            batch_denoised = dae(batch_noisy)
                    else:
                        batch_denoised = batch_noisy
                    
                    optimizer_clf.zero_grad()
                    preds = classifier(batch_denoised)
                    loss = criterion_clf(preds, batch_labels)
                    loss.backward()
                    optimizer_clf.step()
                    
                    train_loss += loss.item() * batch_noisy.size(0)
                    pred_classes = (preds >= 0.5).float()
                    correct += (pred_classes == batch_labels).sum().item()
                    total += batch_noisy.size(0)
                
                train_loss /= len(train_loader.dataset)
                train_acc = correct / total
                
                # Validation
                classifier.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for batch_noisy, _, batch_labels in val_loader:
                        batch_noisy = batch_noisy.to(device)
                        batch_labels = batch_labels.to(device)
                        
                        if config.model_type == "both":
                            batch_denoised = dae(batch_noisy)
                        else:
                            batch_denoised = batch_noisy
                        
                        preds = classifier(batch_denoised)
                        loss = criterion_clf(preds, batch_labels)
                        
                        val_loss += loss.item() * batch_noisy.size(0)
                        pred_classes = (preds >= 0.5).float()
                        val_correct += (pred_classes == batch_labels).sum().item()
                        val_total += batch_noisy.size(0)
                
                val_loss /= len(val_loader.dataset)
                val_acc = val_correct / val_total
                
                _training_state["train_loss"] = train_loss
                _training_state["val_loss"] = val_loss
                _training_state["train_acc"] = train_acc
                _training_state["val_acc"] = val_acc
                _training_state["message"] = f"CLF Epoch {epoch+1}/{config.epochs_clf} | Loss: {train_loss:.4f} | Acc: {train_acc*100:.2f}% | Val: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%"
                
                await asyncio.sleep(0.1)
            
            # Save Classifier
            torch.save(classifier.state_dict(), "models/classifier.pt")
        
        _training_state["status"] = "completed"
        _training_state["message"] = "Training completed successfully!"
        logger.info("Training completed")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        _training_state["status"] = "failed"
        _training_state["message"] = f"Training failed: {str(e)}"