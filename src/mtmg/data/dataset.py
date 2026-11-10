from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class WindowDataset(Dataset):
    """Windows of shape (3, T) of token ids, loaded from a .npy file."""

    def __init__(self, path: str | Path):
        self.x = np.load(path, mmap_mode="r")

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, i: int) -> torch.Tensor:
        return torch.from_numpy(np.asarray(self.x[i], dtype=np.int64))
