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
import matplotlib.pyplot as plt
from tqdm import tqdm
# Import για Mixed Precision
from torch.cuda.amp import autocast, GradScaler

# ------------------------- 1. Custom Dataset -------------------------
class TimeSeriesClsDataset(Dataset):
    def __init__(self, sequences, labels=None):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        if labels is not None:
            self.labels = torch.tensor(labels, dtype=torch.long)
        else:
            self.labels = None

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.sequences[idx], self.labels[idx]
        else:
            return self.sequences[idx]

# ------------------------- 2. FCN Model Architecture -------------------------
class FCN(nn.Module):
    def __init__(self, num_classes, input_channels=1):
        super(FCN, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels=input_channels, out_channels=128, kernel_size=8, padding='same'),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, padding='same'),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        self.block3 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding='same'),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.mean(dim=-1) 
        return self.fc(x)

# ------------------------- 3. Data Processing Helpers -------------------------
def create_sliding_windows(data, targets, window_size, step=10):
    """
    Δημιουργεί παράθυρα με βήμα (step) για μείωση της μνήμης.
    """
    X, y = [], []
    # Χρήση του step στο range: range(start, stop, step)
    for i in range(0, len(data) - window_size + 1, step):
        window = data[i : i + window_size]
        X.append(window.reshape(-1, 1)) 
        if targets is not None:
            y.append(targets[i + window_size - 1])
            
    return np.array(X), np.array(y) if targets is not None else None

def load_labeled_data(data_dir, labels_csv_path, feature_col, selected_devices=None):
    print(f"[INFO] Loading labels from: {labels_csv_path}")
    labels_df = pd.read_csv(labels_csv_path)
    
    if selected_devices:
        print(f"[INFO] Filtering for devices containing: {selected_devices}")
        mask = labels_df['device_id'].apply(lambda x: any(dev in x for dev in selected_devices))
        labels_df = labels_df[mask]
    
    all_features = []
    all_labels = []
    unique_devices = labels_df['device_id'].unique()
    
    for dev_id in unique_devices:
        clean_name = dev_id.replace("_features_pca_scores", "")
        filename = f"{clean_name}.csv"
        file_path = os.path.join(data_dir, filename)
        
        if not os.path.exists(file_path):
            print(f"[WARN] File not found: {file_path}. Skipping...")
            continue
            
        print(f"[INFO] Processing {clean_name}...")
        try:
            df = pd.read_csv(file_path)
            epoch_cols = [c for c in df.columns if 'epoch' in c]
            
            if len(epoch_cols) > 0 and 'cluster_global' not in df.columns:
                # Wide format detection logic
                df['row_id'] = range(len(df))
                df_long = df.melt(id_vars=['row_id'], value_vars=epoch_cols, 
                                  var_name='epoch_str', value_name='value')
                df_long['epoch'] = df_long['epoch_str'].str.extract(r'(\d+)').astype(float)
                df_long = df_long.dropna(subset=['epoch'])
                df_long['epoch'] = df_long['epoch'].astype(int)
                df_long = df_long.sort_values(by=['epoch', 'row_id'])
                df = df_long
                input_feature = 'value'
            else:
                input_feature = feature_col

            device_labels = labels_df[labels_df['device_id'] == dev_id]
            merged = pd.merge(df, device_labels[['epoch', 'cluster_global']], on='epoch', how='inner')
            
            if len(merged) == 0:
                print(f"[WARN] No matching epochs found for {clean_name}")
                continue
            
            all_features.extend(merged[input_feature].values)
            all_labels.extend(merged['cluster_global'].values)
            
        except Exception as e:
            print(f"[ERROR] Failed processing {filename}: {e}")
            continue

    return np.array(all_features), np.array(all_labels)

# ------------------------- 4. Training Logic -------------------------
def train(args):
    # 1. Φόρτωση
    target_devices = args.devices.split(',') if args.devices else None
    feature_data, labels = load_labeled_data(args.data_dir, args.labels_csv, args.feature_col, target_devices)
    
    if len(feature_data) == 0:
        print("[ERROR] No data loaded.")
        return

    # 2. Δημιουργία Windows με STEP
    print(f"[INFO] Creating sliding windows (step={args.step})...")
    X, y = create_sliding_windows(feature_data, labels, args.window_size, step=args.step)
    print(f"[INFO] Successfully created {len(X)} windows.")

    # 3. Split
    print("[INFO] Splitting data...")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # --- FIX 1: Workers = 0 για να μην κρασάρει στα Windows ---
    workers = 0 

    train_loader = DataLoader(TimeSeriesClsDataset(X_train, y_train), 
                              batch_size=args.batch_size, 
                              shuffle=True, 
                              num_workers=workers, 
                              pin_memory=True)

    val_loader = DataLoader(TimeSeriesClsDataset(X_val, y_val), 
                            batch_size=args.batch_size, 
                            shuffle=False, 
                            num_workers=workers, 
                            pin_memory=True)
    
    # 4. Model Setup
    num_classes = len(np.unique(labels))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Classes: {num_classes}, Device: {device}")
    
    model = FCN(num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    # --- FIX 2: Mixed Precision Scaler ---
    scaler = GradScaler()

    loss_history, acc_history = [], []

    print("\n[INFO] Starting training... (Press Ctrl+C to stop)")
    
    for epoch in range(args.epochs):
        model.train()
        correct = 0
        total = 0
        running_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", unit="batch")
        
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            # --- FIX 2: Autocast ---
            with autocast():
                output = model(inputs)
                loss = criterion(output, targets)
            
            # Scaling loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        loss_history.append(epoch_loss)
        acc_history.append(epoch_acc)
        
        print(f"Epoch {epoch+1} DONE | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}%")

    # 6. Save & Plot
    torch.save(model.state_dict(), args.save_path)
    print(f"\n[DONE] Model saved to: {args.save_path}")
    
    plt.figure(figsize=(10, 4))
    plt.subplot(1,2,1); plt.plot(loss_history, label='Loss'); plt.legend()
    plt.subplot(1,2,2); plt.plot(acc_history, label='Accuracy', color='orange'); plt.legend()
    plt.savefig("training_progress.png")
    print("[INFO] Plot saved as 'training_progress.png'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--labels_csv', required=True)
    parser.add_argument('--feature_col', default='value')
    parser.add_argument('--devices', default="DV_device2,DV_device3,DV_device4")
    parser.add_argument('--window_size', type=int, default=64)
    parser.add_argument('--step', type=int, default=10, help="Sliding window step to reduce memory")
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--save_path', default='cnn_model.pt')

    args = parser.parse_args()
    train(args)