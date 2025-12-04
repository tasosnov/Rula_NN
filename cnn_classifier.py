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

# -------------------------------------------------------------------