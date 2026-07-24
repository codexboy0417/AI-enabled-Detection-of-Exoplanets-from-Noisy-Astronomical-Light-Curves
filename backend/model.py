import torch
import torch.nn as nn

class DenoisingAutoencoder1D(nn.Module):
    """
    1D Denoising Autoencoder for Light Curves.
    Takes a noisy light curve segment of shape (batch_size, 1, seq_len)
    and outputs a denoised, reconstructed segment of the same shape.
    """
    def __init__(self, seq_len=200):
        super(DenoisingAutoencoder1D, self).__init__()
        self.seq_len = seq_len
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3),  # (B, 16, L/2)
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2), # (B, 32, L/4)
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1), # (B, 64, L/8)
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), # (B, 32, L/4)
            nn.ReLU(),
            nn.ConvTranspose1d(32, 16, kernel_size=5, stride=2, padding=2, output_padding=1), # (B, 16, L/2)
            nn.ReLU(),
            nn.ConvTranspose1d(16, 1, kernel_size=7, stride=2, padding=3, output_padding=1),  # (B, 1, L)
        )
        
    def forward(self, x):
        # input shape: (B, 1, seq_len)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class TransitClassifier1D(nn.Module):
    """
    1D Convolutional Neural Network Classifier.
    Takes a denoised light curve segment of shape (batch_size, 1, seq_len)
    and outputs the probability of a planet transit event (value between 0 and 1).
    """
    def __init__(self, seq_len=200):
        super(TransitClassifier1D, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1) # Global Average Pooling -> (B, 128, 1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # input shape: (B, 1, seq_len)
        x = self.features(x)
        x = x.view(x.size(0), -1) # Flatten -> (B, 128)
        x = self.classifier(x)     # Output -> (B, 1)
        return x

if __name__ == "__main__":
    # Quick sanity check
    x = torch.randn(8, 1, 200)
    
    dae = DenoisingAutoencoder1D(seq_len=200)
    clf = TransitClassifier1D(seq_len=200)
    
    dae_out = dae(x)
    clf_out = clf(dae_out)
    
    print("Sanity Check Passed:")
    print(f"Input Shape:       {x.shape}")
    print(f"DAE Output Shape:  {dae_out.shape}")
    print(f"Classifier Output: {clf_out.shape}")
