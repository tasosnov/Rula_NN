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