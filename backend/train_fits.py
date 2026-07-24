import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from model import DenoisingAutoencoder1D, TransitClassifier1D

def main():
    print("==================================================")
    print("PyTorch Training Loop (Large Scale FITS Ingestion)")
    print("==================================================")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[i] Using computation device: {device}")
    
    dataset_path = "data/preprocessed_dataset.npz"
    if not os.path.exists(dataset_path):
        print(f"[!] Preprocessed dataset not found at {dataset_path}.")
        print("    Please run: python backend/preprocess_fits.py first.")
        return
        
    print(f"[*] Loading preprocessed binary dataset: {dataset_path}...")
    dataset_raw = np.load(dataset_path)
    
    X = torch.tensor(dataset_raw["X"]) # Shape: (N, 1, seq_len)
    Y = torch.tensor(dataset_raw["Y"]) # Shape: (N, 1, seq_len)
    labels = torch.tensor(dataset_raw["labels"]) # Shape: (N, 1)
    
    num_samples = len(X)
    seq_len = X.shape[2]
    
    print(f"[+] Loaded {num_samples} samples | Sequence Length: {seq_len}")
    
    # Create PyTorch datasets
    dataset = TensorDataset(X, Y, labels)
    
    # 80/20 train/val split
    train_size = int(0.8 * num_samples)
    val_size = num_samples - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    batch_size = 128 # larger batch size for bulk training
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 1. Initialize Models (adjusting for sequence length)
    dae = DenoisingAutoencoder1D(seq_len=seq_len).to(device)
    classifier = TransitClassifier1D(seq_len=seq_len).to(device)
    
    # 2. Train Denoising Autoencoder
    print("\n" + "="*50)
    print("Stage 1: Training Denoising Autoencoder (DAE)")
    print("="*50)
    
    epochs_dae = 10
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
        
    os.makedirs("models", exist_ok=True)
    torch.save(dae.state_dict(), "models/dae_fits.pt")
    print("[+] DAE weights saved to models/dae_fits.pt")
    
    # 3. Train Classifier
    print("\n" + "="*50)
    print("Stage 2: Training Transit Classifier (1D CNN)")
    print("="*50)
    
    epochs_clf = 10
    criterion_clf = nn.BCELoss()
    optimizer_clf = optim.Adam(classifier.parameters(), lr=1e-3)
    
    dae.eval() # Freeze DAE
    
    for epoch in range(epochs_clf):
        classifier.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for batch_noisy, _, batch_labels in train_loader:
            batch_noisy = batch_noisy.to(device)
            batch_labels = batch_labels.to(device)
            
            # Pass through DAE
            with torch.no_grad():
                batch_denoised = dae(batch_noisy)
                
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
        
    torch.save(classifier.state_dict(), "models/classifier_fits.pt")
    print("[+] Classifier weights saved to models/classifier_fits.pt")
    print("[*] Bulk Training Pipeline Complete!")

if __name__ == "__main__":
    main()
