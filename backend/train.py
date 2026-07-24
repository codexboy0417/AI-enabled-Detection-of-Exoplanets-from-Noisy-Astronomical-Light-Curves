import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from model import DenoisingAutoencoder1D, TransitClassifier1D

def generate_synthetic_data(num_samples=2000, seq_len=200):
    """
    Generates synthetic light curves for training.
    Classes:
    - 1: Transit Signal (Exoplanet)
    - 0: Non-Transit (Stellar noise, variable stars, eclipsing binaries without transit profile)
    """
    print(f"[*] Generating {num_samples} synthetic light curves for training...")
    
    clean_curves = []
    noisy_curves = []
    labels = []
    
    # 50% transits, 50% non-transits
    for i in range(num_samples):
        # 1. Base clean flux curve (flat line at 1.0)
        clean = np.ones(seq_len)
        
        # Determine class label
        label = 1 if i < num_samples // 2 else 0
        
        if label == 1:
            # Inject exoplanet transit (periodic U-shape dip)
            transit_depth = np.random.uniform(0.005, 0.02)  # 0.5% to 2% dip
            duration = np.random.randint(15, 40)            # transit duration in steps
            center = np.random.randint(50, 150)             # position of transit center
            
            t_start = center - duration // 2
            t_end = center + duration // 2
            
            # Create transit dip using a trapezoidal model
            for idx in range(t_start, t_end):
                if idx >= 0 and idx < seq_len:
                    # Edge smoothing
                    dist_from_center = abs(idx - center)
                    half_dur = duration / 2
                    if dist_from_center > half_dur * 0.8:
                        # Smooth entry/exit
                        factor = (half_dur - dist_from_center) / (half_dur * 0.2)
                        clean[idx] -= transit_depth * factor
                    else:
                        clean[idx] -= transit_depth
        else:
            # Generate false positives (stellar variability or eclipsing binary-like V-shape dips)
            scenario = np.random.choice(["clean", "sine_var", "eclipsing_binary"])
            if scenario == "sine_var":
                # Regular stellar pulsation/variability
                freq = np.random.uniform(2, 6)
                amp = np.random.uniform(0.003, 0.01)
                clean += amp * np.sin(np.linspace(0, freq * 2 * np.pi, seq_len))
            elif scenario == "eclipsing_binary":
                # Eclipsing binary V-shape dip (very sharp, deep, no flat bottom)
                dip_depth = np.random.uniform(0.03, 0.1)  # much deeper than planet transit
                center = np.random.randint(70, 130)
                width = np.random.randint(10, 25)
                for idx in range(center - width, center + width):
                    if idx >= 0 and idx < seq_len:
                        factor = 1.0 - (abs(idx - center) / width)
                        clean[idx] -= dip_depth * factor
                        
        # 2. Add noise to clean curve to create the noisy input
        # White noise
        white_noise = np.random.normal(0, np.random.uniform(0.002, 0.008), seq_len)
        # Low frequency pink noise (stellar activity)
        t = np.linspace(0, 1, seq_len)
        pink_noise = 0.002 * np.sin(2 * np.pi * t * np.random.uniform(1, 3))
        
        noisy = clean + white_noise + pink_noise
        
        clean_curves.append(clean)
        noisy_curves.append(noisy)
        labels.append(label)
        
    clean_curves = np.array(clean_curves, dtype=np.float32)[:, np.newaxis, :] # Shape (N, 1, seq_len)
    noisy_curves = np.array(noisy_curves, dtype=np.float32)[:, np.newaxis, :] # Shape (N, 1, seq_len)
    labels = np.array(labels, dtype=np.float32)[:, np.newaxis]                 # Shape (N, 1)
    
    return torch.tensor(noisy_curves), torch.tensor(clean_curves), torch.tensor(labels)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[i] Using training device: {device}")
    
    # Hyperparameters
    batch_size = 64
    epochs_dae = 15
    epochs_clf = 15
    seq_len = 200
    
    # 1. Load Data
    noisy, clean, labels = generate_synthetic_data(num_samples=4000, seq_len=seq_len)
    dataset = TensorDataset(noisy, clean, labels)
    
    # Split into train/validation (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 2. Initialize Models
    dae = DenoisingAutoencoder1D(seq_len=seq_len).to(device)
    classifier = TransitClassifier1D(seq_len=seq_len).to(device)
    
    # 3. Train Denoising Autoencoder
    print("\n" + "="*50)
    print("Stage 1: Training Denoising Autoencoder (DAE)")
    print("="*50)
    
    criterion_dae = nn.MSELoss()
    optimizer_dae = optim.Adam(dae.parameters(), lr=1e-3)
    
    for epoch in range(epochs_dae):
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
        
        print(f"Epoch {epoch+1:02d}/{epochs_dae:02d} | Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f}")
        
    # Save DAE weights
    os.makedirs("models", exist_ok=True)
    torch.save(dae.state_dict(), "models/dae.pt")
    print("[+] DAE weights saved to models/dae.pt")
    
    # 4. Train Classifier (using denoised curves as inputs)
    print("\n" + "="*50)
    print("Stage 2: Training Transit Classifier (1D CNN)")
    print("="*50)
    
    criterion_clf = nn.BCELoss()
    optimizer_clf = optim.Adam(classifier.parameters(), lr=1e-3)
    
    # Freeze DAE for feature input extraction
    dae.eval()
    
    for epoch in range(epochs_clf):
        classifier.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for batch_noisy, _, batch_labels in train_loader:
            batch_noisy = batch_noisy.to(device)
            batch_labels = batch_labels.to(device)
            
            # Pass noisy curves through trained DAE to get denoised representations
            with torch.no_grad():
                batch_denoised = dae(batch_noisy)
                
            optimizer_clf.zero_grad()
            preds = classifier(batch_denoised)
            loss = criterion_clf(preds, batch_labels)
            loss.backward()
            optimizer_clf.step()
            
            train_loss += loss.item() * batch_noisy.size(0)
            
            # Calculate accuracy
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
                
                batch_denoised = dae(batch_noisy)
                preds = classifier(batch_denoised)
                loss = criterion_clf(preds, batch_labels)
                
                val_loss += loss.item() * batch_noisy.size(0)
                pred_classes = (preds >= 0.5).float()
                val_correct += (pred_classes == batch_labels).sum().item()
                val_total += batch_noisy.size(0)
                
        val_loss /= len(val_loader.dataset)
        val_acc = val_correct / val_total
        
        print(f"Epoch {epoch+1:02d}/{epochs_clf:02d} | Loss: {train_loss:.4f} | Acc: {train_acc*100:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")
        
    # Save Classifier weights
    torch.save(classifier.state_dict(), "models/classifier.pt")
    print("[+] Classifier weights saved to models/classifier.pt")
    print("[*] Training Pipeline Complete!")

if __name__ == "__main__":
    train()
