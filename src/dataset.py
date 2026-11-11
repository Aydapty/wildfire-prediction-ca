# src/dataset.py
import torch
from torch.utils.data import Dataset
import numpy as np

class WildfireTensorDataset(Dataset):
    def __init__(self, x_path, y_path):
        # Load pre-extracted numpy tensors
        self.X = np.load(x_path).astype(np.float32)
        self.y = np.load(y_path).astype(np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # Return PyTorch FloatTensors
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx]).unsqueeze(0)