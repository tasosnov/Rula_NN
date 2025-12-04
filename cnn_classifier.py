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
# ------------------------- 3. Data Processing Helpers -------------------------

def create_sliding_windows(data, targets, window_size):
    """
    Creates sliding windows from 1D arrays.
    """
    X, y = [], []
    # Loop until len(data) - window_size
    for i in range(len(data) - window_size + 1):
        window = data[i : i + window_size]
        X.append(window.reshape(-1, 1)) # Add feature dimension (Window, 1)
        
        if targets is not None:
            # Assign label of the last timestamp in the window
            y.append(targets[i + window_size - 1])
            
    return np.array(X), np.array(y) if targets is not None else None


def load_labeled_data(data_dir, labels_csv_path, feature_col, selected_devices=None):
    """
    Loads data from CSVs and merges with labels from K-Means.
    Automatically handles 'Wide' format (e.g., columns like 'ΔVCE_epoch_2').
    """
    print(f"[INFO] Loading labels from: {labels_csv_path}")
    labels_df = pd.read_csv(labels_csv_path)
    
    # Filter specific devices if requested (partial matching)
    if selected_devices:
        print(f"[INFO] Filtering for devices containing: {selected_devices}")
        mask = labels_df['device_id'].apply(lambda x: any(dev in x for dev in selected_devices))
        labels_df = labels_df[mask]
    
    all_features = []
    all_labels = []

    # Get unique device IDs from the labels file
    unique_devices = labels_df['device_id'].unique()
    
    for dev_id in unique_devices:
        # Clean filename: remove suffix to match Raw Data filename (e.g. DV_device2.csv)
        clean_name = dev_id.replace("_features_pca_scores", "")
        filename = f"{clean_name}.csv"
        file_path = os.path.join(data_dir, filename)
        
        if not os.path.exists(file_path):
            print(f"[WARN] File not found: {file_path}. Skipping...")
            continue
            
        print(f"[INFO] Processing {clean_name}...")
        
        try:
            # Read CSV
            df = pd.read_csv(file_path)
            
            # --- AUTO-FIX: Handle Wide Format ---
            # Check if columns contain 'epoch' strings like 'ΔVCE_epoch_2'
            epoch_cols = [c for c in df.columns if 'epoch' in c]
            
            if len(epoch_cols) > 0 and 'cluster_global' not in df.columns:
                print(f"   -> Detected wide format ({len(epoch_cols)} epoch columns). Reshaping...")
                
                # Add a row index to preserve time order during melt
                df['row_id'] = range(len(df))
                
                # Melt: Turn columns into rows
                # var_name='epoch_str' will hold 'ΔVCE_epoch_2'
                # value_name='value' will hold the sensor reading
                df_long = df.melt(id_vars=['row_id'], value_vars=epoch_cols, 
                                  var_name='epoch_str', value_name='value')
                
                # Extract epoch number using regex (find digits in the string)
                df_long['epoch'] = df_long['epoch_str'].str.extract(r'(\d+)').astype(float)
                
                # Drop invalid rows and sort by epoch then time (row_id)
                df_long = df_long.dropna(subset=['epoch'])
                df_long['epoch'] = df_long['epoch'].astype(int)
                df_long = df_long.sort_values(by=['epoch', 'row_id'])
                
                # Update main dataframe to be the transformed one
                df = df_long
                
                # Override feature_col to 'value' because that's what we created
                input_feature = 'value'
            else:
                input_feature = feature_col
            # ------------------------------------

            # Merge with labels
            # Labels DF has: device_id, epoch, cluster_global
            device_labels = labels_df[labels_df['device_id'] == dev_id]
            
            # Merge on epoch
            merged = pd.merge(df, device_labels[['epoch', 'cluster_global']], on='epoch', how='inner')
            
            if len(merged) == 0:
                print(f"[WARN] No matching epochs found for {clean_name} after merge.")
                continue
            
            # Append data
            all_features.extend(merged[input_feature].values)
            all_labels.extend(merged['cluster_global'].values)
            
        except Exception as e:
            print(f"[ERROR] Failed processing {filename}: {e}")
            continue

    return np.array(all_features), np.array(all_labels)
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