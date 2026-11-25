"""PyTorch Dataset wrapper for Landslide4Sense."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from uag.data.landslide4sense import Landslide4SenseDataset


class TorchL4SDataset(Dataset):
    """Wrap Landslide4SenseDataset for PyTorch with joint geometric augmentation."""

    def __init__(self, dataset: Landslide4SenseDataset, augment: bool = False) -> None:
        self.dataset = dataset
        self.augment = bool(augment)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        img, mask = self.dataset[idx]
        img_t = torch.from_numpy(
            np.ascontiguousarray(img.transpose(2, 0, 1))
        ).to(dtype=torch.float32)

        if mask is None:
            mask_t = torch.zeros(
                (1, img_t.shape[1], img_t.shape[2]),
                dtype=torch.float32,
            )
        else:
            mask_t = torch.from_numpy(
                np.ascontiguousarray(mask.transpose(2, 0, 1))
            ).to(dtype=torch.float32)

        if self.augment:
            if torch.rand(()) < 0.5:
                img_t = torch.flip(img_t, dims=(-1,))
                mask_t = torch.flip(mask_t, dims=(-1,))
            if torch.rand(()) < 0.5:
                img_t = torch.flip(img_t, dims=(-2,))
                mask_t = torch.flip(mask_t, dims=(-2,))
            k = int(torch.randint(0, 4, ()).item())
            if k > 0:
                img_t = torch.rot90(img_t, k=k, dims=(-2, -1))
                mask_t = torch.rot90(mask_t, k=k, dims=(-2, -1))

        return img_t, mask_t, idx
