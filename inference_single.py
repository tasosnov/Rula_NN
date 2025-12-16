import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
import re

# --- Model Definition (Must match training architecture) ---
class FCN(nn.Module):
    def __init__(self, num_classes, input_channels=1):
        super(FCN, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels=input_channels, out_channels=128, kernel_size=8, padding='same'),
            nn.BatchNorm1d(128), nn.ReLU())
        self.block2 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, padding='same'),
            nn.BatchNorm1d(256), nn.ReLU())
        self.block3 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding='same'),
            nn.BatchNorm1d(128), nn.ReLU())
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.mean(dim=-1)
        return self.fc(x)

def load_model(model_path, device):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    state_dict = torch.load(model_path, map_location=device)
    num_classes = state_dict['fc.weight'].shape[0]
    
    model = FCN(num_classes=num_classes)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, num_classes

def preprocess_wide_format(df):
    """
    Detects columns like 'ΔVCE_epoch_X', melts them into a single time series,
    and sorts by epoch. Matches the training logic.
    """
    epoch_cols = [c for c in df.columns if 'epoch' in c]
    
    if not epoch_cols:
        raise ValueError("No columns containing 'epoch' found in CSV.")

    print(f"[INFO] Detected {len(epoch_cols)} epoch columns. Processing...")

    df['row_id'] = range(len(df))
    
    # Melt: Transform columns to rows
    df_long = df.melt(id_vars=['row_id'], value_vars=epoch_cols, 
                      var_name='epoch_str', value_name='value')
    
    # Extract epoch number from string
    df_long['epoch'] = df_long['epoch_str'].str.extract(r'(\d+)').astype(float).astype(int)
    
    # Sort by epoch to create the correct time sequence
    df_long = df_long.sort_values(by=['epoch', 'row_id'])
    
    return df_long['value'].values, df_long['epoch'].values

def run_inference(model_path, csv_path, target_epoch_index, window_size):
    """
    target_epoch_index: 0-based index (0 = First Epoch, 1 = Second Epoch...)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # 1. Load Data & Preprocess
    try:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV not found: {csv_path}")
            
        df = pd.read_csv(csv_path)
        time_series, epochs = preprocess_wide_format(df)
        
        # --- LOGIC: Map 0-based Index to Actual Epoch Label ---
        unique_epochs = np.sort(np.unique(epochs))
        
        if target_epoch_index < 0 or target_epoch_index >= len(unique_epochs):
            print(f"[ERROR] Target index {target_epoch_index} is out of range.")
            print(f"       Available range: 0 to {len(unique_epochs) - 1}")
            return

        # Direct 0-based indexing
        actual_epoch_label = unique_epochs[target_epoch_index]
        print(f"[INFO] Index {target_epoch_index} corresponds to Column/Epoch: {int(actual_epoch_label)}")
        # --------------------------------------------------

    except Exception as e:
        print(f"[ERROR] Data processing failed: {e}")
        return

    # 2. Extract Window
    indices = np.where(epochs == actual_epoch_label)[0]
    
    if len(indices) == 0:
        print(f"[ERROR] No data found for epoch {actual_epoch_label}.")
        return
    
    # Take the window ending at the last point of this epoch
    end_idx = indices[-1]
    start_idx = end_idx - window_size + 1

    if start_idx < 0:
        print(f"[WARN] Not enough history. Padding with first value.")
        window = time_series[0 : end_idx + 1]
        pad_width = window_size - len(window)
        window = np.pad(window, (pad_width, 0), mode='edge')
    else:
        window = time_series[start_idx : end_idx + 1]

    # 3. Load Model & Predict
    try:
        model, num_classes = load_model(model_path, device)
        
        input_tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0).unsqueeze(2).to(device)
        
        with torch.no_grad():
            logits = model(input_tensor)
            probs = F.softmax(logits, dim=1)
            
        logits = logits.cpu().numpy().flatten()
        probs = probs.cpu().numpy().flatten()
        prediction = np.argmax(probs)

        print("\n" + "="*50)
        print(f"Inference for Epoch Index: {target_epoch_index} (Label: {int(actual_epoch_label)})")
        print("="*50)
        print(f"{'Cluster':<10} | {'Logit Score':<15} | {'Probability':<15}")
        print("-" * 50)
        for i in range(num_classes):
            mark = "<-- PREDICTED" if i == prediction else ""
            print(f"Cluster {i:<2} | {logits[i]:<15.4f} | {probs[i]:<15.4f} {mark}")
        print("="*50 + "\n")
        
    except Exception as e:
        print(f"[ERROR] Inference failed: {e}")

if __name__ == "__main__":
    # --- Configuration ---
    MODEL_PATH = "cnn_model.pt"
    DATA_PATH = "Raw_Data/DV_device5.csv" 
    
    # 0 = First Epoch, 1 = Second Epoch, etc.
    TARGET_EPOCH =   70 
    
    WINDOW_SIZE = 64
    # ---------------------

    run_inference(MODEL_PATH, DATA_PATH, TARGET_EPOCH, WINDOW_SIZE)