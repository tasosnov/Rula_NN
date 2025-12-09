import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
import argparse

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
    # Infer number of classes from the final fully connected layer weights
    num_classes = state_dict['fc.weight'].shape[0]
    
    model = FCN(num_classes=num_classes)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, num_classes

def get_window_data(csv_path, target_index, window_size, column=None):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Data file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    
    # Determine feature column
    if column:
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in CSV.")
        data = df[column].values
    else:
        # Default to the first numeric column found
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            raise ValueError("No numeric columns found in CSV.")
        data = df[numeric_cols[0]].values

    # Handle window slicing with padding if necessary
    start_idx = target_index - window_size + 1
    
    if start_idx < 0:
        # Pad with the first value if we are at the beginning of the series
        window = data[0 : target_index + 1]
        pad_width = window_size - len(window)
        window = np.pad(window, (pad_width, 0), mode='edge')
    else:
        window = data[start_idx : target_index + 1]

    return window

def run_inference(model_path, csv_path, target_index, window_size, column=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # Load Model
    try:
        model, num_classes = load_model(model_path, device)
        print(f"[INFO] Model loaded. Detected classes: {num_classes}")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return

    # Load Data Window
    try:
        window_data = get_window_data(csv_path, target_index, window_size, column)
        print(f"[INFO] Loaded window ending at index {target_index} (Size: {window_data.shape})")
    except Exception as e:
        print(f"[ERROR] Failed to load data: {e}")
        return

    # Prepare Tensor (Batch=1, Length=window_size, Channels=1)
    input_tensor = torch.tensor(window_data, dtype=torch.float32).unsqueeze(0).unsqueeze(2).to(device)

    # Inference
    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1)

    # Output Results
    logits = logits.cpu().numpy().flatten()
    probs = probs.cpu().numpy().flatten()
    prediction = np.argmax(probs)

    print("\nInference Results")
    print("-" * 50)
    print(f"{'Cluster':<10} | {'Logit Score':<15} | {'Probability':<15}")
    print("-" * 50)
    for i in range(num_classes):
        mark = "(*)" if i == prediction else ""
        print(f"{i:<10} | {logits[i]:<15.4f} | {probs[i]:<15.4f} {mark}")
    print("-" * 50)
    print(f"Predicted Cluster: {prediction}")

if __name__ == "__main__":
    # --- Configuration ---
    MODEL_PATH = "cnn_model.pt"
    DATA_PATH = "Raw_Data/DV_device5.csv" # Example path, change as needed
    TARGET_INDEX = 50              # The index (row) in the CSV to classify
    WINDOW_SIZE = 64                      # Must match training window size
    COLUMN_NAME = None                    # Specify column name if known, else None
    # ---------------------

    run_inference(MODEL_PATH, DATA_PATH, TARGET_INDEX, WINDOW_SIZE, COLUMN_NAME)