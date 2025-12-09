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
    # Identify epoch columns (containing 'epoch')
    epoch_cols = [c for c in df.columns if 'epoch' in c]
    
    if not epoch_cols:
        raise ValueError("No columns containing 'epoch' found in CSV. Cannot process wide format.")

    print(f"[INFO] Detected {len(epoch_cols)} epoch columns (e.g., {epoch_cols[0]}...). Processing...")

    # Create a row identifier to keep multiple features separate if they exist
    df['row_id'] = range(len(df))
    
    # Melt: Transform columns to rows
    df_long = df.melt(id_vars=['row_id'], value_vars=epoch_cols, 
                      var_name='epoch_str', value_name='value')
    
    # Extract epoch number from string (e.g., 'ΔVCE_epoch_2' -> 2)
    df_long['epoch'] = df_long['epoch_str'].str.extract(r'(\d+)').astype(float).astype(int)
    
    # Sort by epoch to create the correct time sequence
    df_long = df_long.sort_values(by=['epoch', 'row_id'])
    
    # Return the single 'value' column as the time series
    return df_long['value'].values, df_long['epoch'].values

def run_inference(model_path, csv_path, target_epoch, window_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # 1. Load Data & Preprocess
    try:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV not found: {csv_path}")
            
        df = pd.read_csv(csv_path)
        
        # Transform wide format (columns) to long format (time series)
        time_series, epochs = preprocess_wide_format(df)
        
        print(f"[INFO] Time series constructed. Total length: {len(time_series)}")
        
    except Exception as e:
        print(f"[ERROR] Data processing failed: {e}")
        return

    # 2. Extract Window for Target Epoch
    # Find the index in the array where epoch == target_epoch
    # Note: If multiple rows exist per epoch, this takes the last one effectively due to padding logic below
    indices = np.where(epochs == target_epoch)[0]
    
    if len(indices) == 0:
        print(f"[ERROR] Epoch {target_epoch} not found in data. Max epoch is {epochs.max()}.")
        return
    
    # We take the end index of the target epoch
    end_idx = indices[-1]
    start_idx = end_idx - window_size + 1

    if start_idx < 0:
        print(f"[WARN] Not enough history for window size {window_size}. Padding with first value.")
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
        print(f"Inference for Epoch: {target_epoch}")
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
    # Change this to your actual raw data file with the columns ΔVCE_epoch_...
    DATA_PATH = "Raw_Data/DV_device2.csv" 
    TARGET_EPOCH = 50   # The epoch number (from the column name) you want to test
    WINDOW_SIZE = 64    # Must match training
    # ---------------------

    run_inference(MODEL_PATH, DATA_PATH, TARGET_EPOCH, WINDOW_SIZE)