# -*- coding: utf-8 -*-
import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# ------------------------- 1. Custom Dataset -------------------------

class TimeSeriesClsDataset(Dataset):
    def __init__(self, sequences, labels=None):
        """
        Custom PyTorch Dataset for Time Series Classification.
        
        Args:
            sequences (np.array): Input data of shape (N_samples, Window_Size, 1)
            labels (np.array, optional): Class labels for each sample. None during inference.
        """
        # Convert numpy arrays to PyTorch tensors (float32 for data, long for integer labels)
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        if labels is not None:
            self.labels = torch.tensor(labels, dtype=torch.long)
        else:
            self.labels = None

    def __len__(self):
        # Returns the total number of samples
        return len(self.sequences)

    def __getitem__(self, idx):
        # Retrieves the i-th sample (and label if it exists)
        if self.labels is not None:
            return self.sequences[idx], self.labels[idx]
        else:
            return self.sequences[idx]


# ------------------------- 2. FCN Model Architecture -------------------------

class FCN(nn.Module):
    def __init__(self, num_classes, input_channels=1):
        """
        Implements the FCN architecture from Wang et al. (2017).
        Structure: 3 Conv blocks (Conv -> BN -> ReLU) + Global Avg Pooling + Linear
        Ref: https://arxiv.org/pdf/1611.06455.pdf
        """
        super(FCN, self).__init__()
        
        # Block 1: 128 filters, kernel size 8
        # padding='same' keeps the time length consistent
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels=input_channels, out_channels=128, kernel_size=8, padding='same'),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        
        # Block 2: 256 filters, kernel size 5
        self.block2 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, padding='same'),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        
        # Block 3: 128 filters, kernel size 3
        self.block3 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding='same'),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        
        # Final Classifier: Linear layer mapping 128 features -> num_classes
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # Input x shape: (Batch, Window_Size, 1)
        # PyTorch Conv1d needs: (Batch, Channels, Window_Size)
        x = x.permute(0, 2, 1)
        
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        # Global Average Pooling: Average across the time dimension
        # This reduces (Batch, 128, Window_Size) -> (Batch, 128)
        x = x.mean(dim=-1) 
        
        return self.fc(x)

# -------------------------------------------------------------------
# ------------------------- 4. Training & Main Loop -------------------------

def train(args):
    # 1. Επιλογή Συσκευών (Filtering)
    # Μετατρέπουμε το string "DV_device2,..." σε λίστα ['DV_device2', ...]
    target_devices = args.devices.split(',') if args.devices else None
    print(f"[INFO] Training ONLY on devices: {target_devices}")
    
    # 2. Φόρτωση Δεδομένων
    feature_data, labels = load_labeled_data(args.data_dir, args.labels_csv, args.feature_col, target_devices)
    
    if len(feature_data) == 0:
        print("[ERROR] No data loaded. Check paths and device names.")
        return

    # 3. Δημιουργία Παραθύρων (Sliding Windows)
    X, y = create_sliding_windows(feature_data, labels, args.window_size)
    print(f"[INFO] Created {len(X)} windows of size {args.window_size}")
    
    # 4. Διαχωρισμός Train / Validation (80% - 20%)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Dataloaders
    train_loader = DataLoader(TimeSeriesClsDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(TimeSeriesClsDataset(X_val, y_val),   batch_size=args.batch_size, shuffle=False)
    
    # 5. Αρχικοποίηση Μοντέλου
    num_classes = len(np.unique(labels))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Classes: {num_classes}, Device: {device}")
    
    model = FCN(num_classes=num_classes, input_channels=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    # 6. Training Loop
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        
        # Εκτύπωση προόδου ανά 5 εποχές
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{args.epochs} | Loss: {avg_loss:.4f}")

    # 7. Αποθήκευση & Αναφορά
    torch.save(model.state_dict(), args.save_path)
    print(f"\n[DONE] Model saved to: {args.save_path}")
    
    # Γρήγορο Validation Report
    model.eval()
    preds, true_labels = [], []
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            out = model(inputs)
            preds.extend(out.argmax(dim=1).cpu().numpy())
            true_labels.extend(targets.numpy())
            
    print("\nClassification Report (Validation):")
    print(classification_report(true_labels, preds))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True, help="Path to folder with device CSVs (e.g. PCA/)")
    parser.add_argument('--labels_csv', required=True, help="Path to combined.csv (labels)")
    parser.add_argument('--feature_col', default='pc1_scaled', help="Column name to use as input")
    
    # ΕΔΩ ορίζουμε ποιες συσκευές θέλουμε by default
    parser.add_argument('--devices', default="DV_device2,DV_device3,DV_device4", help="Comma separated list of devices")
    
    parser.add_argument('--window_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--save_path', default='cnn_model.pt')

    args = parser.parse_args()
    train(args)