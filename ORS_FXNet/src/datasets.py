
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class WindowedExchangeRateDataset(Dataset):
    def __init__(self, X, y, macro_idx=None, tech_idx=None):

        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)
        self.macro_idx = macro_idx
        self.tech_idx = tech_idx

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sample = {"sequence": self.X[idx], "target": self.y[idx]}
        if self.macro_idx is not None:
            sample["macro"] = self.X[idx, -1, self.macro_idx]
        if self.tech_idx is not None:
            sample["technical"] = self.X[idx, -1, self.tech_idx]
        return sample


def make_dataloader(dataset, batch_size=64, shuffle=False):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)
